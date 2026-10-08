"""likeness: curate, pull, enhance and train on photos of yourself, without hoarding them.

Run it through scripts/likeness, which works from any folder and adds the optional
packages each command needs. Typical flow, on the Mac with the Photos library and exo:

    likeness libraries                                # Photos libraries here and on drives
    likeness people                                   # find your exact name in Photos
    likeness select --task me --person "Your Name"    # rank photos, nothing downloaded
    likeness pull --task me                           # download just the chosen originals
    likeness identity --task me                       # face reference for the identity check
    likeness enhance --recipe polish --identity-task me photo.jpg
    likeness train-prepare --task me --trigger-word ohwx --subject-class man
    likeness train-run --task me                      # mflux LoRA training
    likeness train-export --task me                   # keep just the LoRA adapter
    likeness release --task me                        # delete pulled photos + checkpoints
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import final

from exo.likeness.enhance import (
    EnhancementEffects,
    RestorationOptions,
    enhance_image,
)
from exo.likeness.exo_client import DEFAULT_EDIT_MODEL, ExoImageEditor, ExoRequestError
from exo.likeness.identity import (
    InsightFaceEmbedder,
    ReferenceIdentityScorer,
    build_reference,
)
from exo.likeness.models import IdentityReference, SelectionSettings, TaskManifest
from exo.likeness.photo_library import (
    PhotosLibraryUnavailableError,
    default_library_search_roots,
    find_candidates,
    find_photo_libraries,
    list_people,
    open_library,
    pull_originals,
)
from exo.likeness.recipes import RECIPES, EditStep, Recipe
from exo.likeness.restoration import InsufficientMemoryError, SeedVR2Restorer
from exo.likeness.selection import select_photos
from exo.likeness.texture import DEFAULT_MINIMUM_ACUTANCE
from exo.likeness.training import export_latest_adapter, write_dataset
from exo.likeness.workspace import (
    TaskWorkspace,
    default_likeness_home,
    folder_size_bytes,
    human_size,
    list_workspaces,
)


@final
class _Arguments(argparse.Namespace):
    command: str
    home: Path
    library: Path | None
    task: str
    person: str
    since: str | None
    years: float
    target: int
    show: int
    trigger_word: str
    subject_class: str
    quantize: int | None
    recipe: str
    images: list[Path]
    from_task: str | None
    identity_task: str | None
    output_directory: Path | None
    edit_model: str
    exo_url: str
    launch_model: bool
    megapixels: float | None
    final_short_edge: int | None
    quality: str | None
    seed: int
    keep_intermediates: bool
    seedvr2_model: str
    low_ram: bool
    min_free_memory_gb: float
    texture_strength: float
    sharp_threshold: float


def _workspace(arguments: _Arguments, task_name: str | None = None) -> TaskWorkspace:
    return TaskWorkspace(arguments.home, task_name or arguments.task)


def _earliest_capture(arguments: _Arguments) -> datetime | None:
    if arguments.since:
        return datetime.fromisoformat(arguments.since).astimezone()
    if arguments.years <= 0:
        return None
    return datetime.now().astimezone() - timedelta(days=365.25 * arguments.years)


def command_libraries(_arguments: _Arguments) -> None:
    roots = default_library_search_roots()
    drives = [root for root in roots if root.parent == Path("/Volumes")]
    print("Mounted drives: " + (", ".join(drive.name for drive in drives) or "none"))
    libraries = find_photo_libraries(roots)
    if not libraries:
        print(
            "No Photos libraries found. If yours is on an external drive that is not "
            "listed above, mount it (diskutil list external, then diskutil mountDisk diskN)."
        )
        if "SSH_CONNECTION" in os.environ:
            print(
                "Over SSH, macOS hides drive contents unless System Settings > General > "
                'Sharing > Remote Login (i) > "Allow full disk access for remote users" is on.'
            )
        return
    for library in libraries:
        database = library / "database" / "Photos.sqlite"
        updated = (
            f"last updated {datetime.fromtimestamp(database.stat().st_mtime):%Y-%m-%d}"
            if database.exists()
            else "no database found inside"
        )
        print(f'  "{library}"  ({updated})')
    print('Use one with: likeness people --library "<path>"')


def command_people(arguments: _Arguments) -> None:
    for name, count in list_people(open_library(arguments.library)):
        print(f"{count:6d}  {name}")


def command_select(arguments: _Arguments) -> None:
    workspace = _workspace(arguments)
    settings = SelectionSettings(
        person_name=arguments.person,
        library_path=None
        if arguments.library is None
        else str(arguments.library.expanduser().resolve()),
        earliest_capture=_earliest_capture(arguments),
        target_count=arguments.target,
    )
    candidates = find_candidates(open_library(arguments.library), arguments.person)
    if not candidates:
        print(
            f"No photos tagged {arguments.person!r}. Check the exact name with "
            "`likeness people`, and tag yourself in Photos > People."
        )
        return
    result = select_photos(candidates, settings)
    # Keep photos already pulled for this task so re-selecting does not copy them again.
    previously_pulled = (
        workspace.load_manifest().pulled_files
        if workspace.manifest_path.exists()
        else {}
    )
    selected_uuids = {photo.uuid for photo in result.selected}
    workspace.save_manifest(
        TaskManifest(
            task_name=workspace.task_name,
            created_at=datetime.now().astimezone(),
            settings=settings,
            selected=result.selected,
            pulled_files={
                uuid: filename
                for uuid, filename in previously_pulled.items()
                if uuid in selected_uuids
                and (workspace.originals_directory / filename).exists()
            },
        )
    )

    local_uuids = {
        candidate.uuid for candidate in candidates if candidate.is_original_local
    }
    to_download = sum(1 for photo in result.selected if photo.uuid not in local_uuids)
    framings = Counter(photo.framing for photo in result.selected)
    angles = Counter(photo.head_angle for photo in result.selected)
    print(f"Photos tagged {arguments.person!r}: {len(candidates)}")
    print(f"Usable: {result.eligible_count}")
    for reason, count in sorted(
        result.rejection_counts.items(), key=lambda item: -item[1]
    ):
        print(f"  skipped {count:5d}  {reason}")
    print(
        f"Selected {len(result.selected)}: "
        + ", ".join(f"{name} {count}" for name, count in framings.items())
        + " | "
        + ", ".join(f"{name} {count}" for name, count in angles.items())
        + f" | favorites {sum(photo.is_favorite for photo in result.selected)}"
    )
    if arguments.library is None:
        print(
            f"Originals already on this Mac: {len(result.selected) - to_download}; "
            f"to download from iCloud: {to_download}"
        )
    else:
        print(
            f"Originals present in {arguments.library}: {len(result.selected) - to_download}; "
            f"missing from it (cannot be pulled): {to_download}"
        )
    for photo in result.selected[: arguments.show]:
        print(
            f"  {photo.ranking_score:.3f}  {photo.captured_at:%Y-%m-%d}  "
            f"{photo.framing:<8} {photo.head_angle:<19} {photo.uuid}"
        )
    print(f"Manifest: {workspace.manifest_path}")
    print(f"Next: likeness pull --task {workspace.task_name}")


def command_pull(arguments: _Arguments) -> None:
    workspace = _workspace(arguments)
    manifest = workspace.load_manifest()
    already_pulled = {
        uuid
        for uuid, filename in manifest.pulled_files.items()
        if (workspace.originals_directory / filename).exists()
    }
    wanted = [
        photo.uuid for photo in manifest.selected if photo.uuid not in already_pulled
    ]
    pulled = {uuid: manifest.pulled_files[uuid] for uuid in already_pulled}
    library = arguments.library or (
        Path(manifest.settings.library_path) if manifest.settings.library_path else None
    )
    if wanted:
        print(
            f"Pulling {len(wanted)} originals "
            + (
                "(downloading from iCloud where needed)..."
                if library is None
                else f"from {library}..."
            )
        )
        for uuid, path in pull_originals(
            wanted, workspace.originals_directory, library
        ).items():
            pulled[uuid] = path.name
    workspace.save_manifest(manifest.model_copy(update={"pulled_files": pulled}))
    missing = len(manifest.selected) - len(pulled)
    print(
        f"{len(pulled)} photos ready in {workspace.originals_directory} "
        f"({human_size(folder_size_bytes(workspace.originals_directory))})"
        + (f"; {missing} could not be exported" if missing else "")
    )


def command_identity(arguments: _Arguments) -> None:
    workspace = _workspace(arguments)
    paths = workspace.pulled_paths()
    if not paths:
        raise SystemExit("No pulled photos. Run `likeness pull` for this task first.")
    embedder = InsightFaceEmbedder()
    embeddings: list[list[float]] = []
    for path in paths:
        embedding = embedder.embed_largest_face(path)
        if embedding is not None:
            embeddings.append(embedding)
    reference, discarded = build_reference(embeddings)
    workspace.save_identity(
        IdentityReference(
            embedder_name=embedder.name,
            embedding=reference,
            source_face_count=len(embeddings) - discarded,
            discarded_face_count=discarded,
            created_at=datetime.now().astimezone(),
        )
    )
    print(
        f"Identity reference built from {len(embeddings) - discarded} faces "
        f"({discarded} discarded as not matching, {len(paths) - len(embeddings)} photos with no detectable face)."
    )
    print(
        f"Saved to {workspace.identity_path}; it is kept when the photos are released."
    )


def command_train_prepare(arguments: _Arguments) -> None:
    from exo.likeness.imaging import PillowImageOperations

    workspace = _workspace(arguments)
    config_path, image_count = write_dataset(
        manifest=workspace.load_manifest(),
        workspace=workspace,
        trigger_word=arguments.trigger_word,
        subject_class=arguments.subject_class,
        writer=PillowImageOperations(),
        quantize=arguments.quantize,
    )
    print(f"Dataset of {image_count} captioned images written next to {config_path}")
    print(f"Train with: likeness train-run --task {workspace.task_name}")
    print(f"Then: likeness train-export --task {workspace.task_name}")


def command_train_export(arguments: _Arguments) -> None:
    workspace = _workspace(arguments)
    adapter = export_latest_adapter(workspace)
    print(f"LoRA adapter: {adapter} ({human_size(adapter.stat().st_size)})")
    print(
        "Use it with: uvx --from mflux==0.17.5 mflux-generate-z-image-turbo --lora-paths "
        f'{adapter} --prompt "a photo of {arguments.trigger_word or "<trigger>"} ..."'
    )
    print(
        f"Free the photos and checkpoints with: likeness release --task {workspace.task_name}"
    )


def command_release(arguments: _Arguments) -> None:
    workspace = _workspace(arguments)
    freed = workspace.release()
    print(
        f"Freed {human_size(freed)} from {workspace.directory}. Kept: manifest, identity, "
        "LoRA adapters and outputs. Re-pull the same photos any time with `likeness pull`."
    )


def command_status(arguments: _Arguments) -> None:
    workspaces = list_workspaces(arguments.home)
    if not workspaces:
        print(f"No tasks under {arguments.home}")
        return
    for workspace in workspaces:
        usage = workspace.disk_usage()
        details = ", ".join(
            f"{name} {human_size(size)}" for name, size in usage.items()
        )
        print(
            f"{workspace.task_name}: {human_size(sum(usage.values()))} ({details or 'empty'})"
        )


def command_recipes(_arguments: _Arguments) -> None:
    for recipe in RECIPES.values():
        print(f"{recipe.name:<12} {recipe.style:<10} {recipe.summary}")


def _with_quality(recipe: Recipe, quality: str | None) -> Recipe:
    if quality is None:
        return recipe
    return recipe.model_copy(
        update={
            "steps": [
                step.model_copy(update={"quality": quality})
                if isinstance(step, EditStep)
                else step
                for step in recipe.steps
            ]
        }
    )


def command_enhance(arguments: _Arguments) -> None:
    from exo.likeness.imaging import PillowImageOperations

    if arguments.recipe not in RECIPES:
        raise SystemExit(
            f"Unknown recipe {arguments.recipe!r}; see `likeness recipes`."
        )
    recipe = _with_quality(RECIPES[arguments.recipe], arguments.quality)
    inputs = (
        _workspace(arguments, arguments.from_task).pulled_paths()
        if arguments.from_task
        else arguments.images
    )
    if not inputs:
        raise SystemExit("No input images. Pass image paths or --from-task.")

    identity = None
    if arguments.identity_task:
        reference = _workspace(arguments, arguments.identity_task).load_identity()
        embedder = InsightFaceEmbedder()
        if reference.embedder_name != embedder.name:
            raise SystemExit(
                f"The identity reference was built with {reference.embedder_name}; "
                f"rebuild it with `likeness identity --task {arguments.identity_task}`."
            )
        identity = ReferenceIdentityScorer(embedder, reference.embedding)
    elif recipe.style == "realistic":
        print(
            "Note: no --identity-task given, so edits will not be checked against your face."
        )

    editor = ExoImageEditor(base_url=arguments.exo_url, model_id=arguments.edit_model)
    if any(isinstance(step, EditStep) for step in recipe.steps):
        editor.ensure_model_running(
            launch_if_missing=arguments.launch_model,
            ready_timeout_seconds=3600.0,
            on_progress=print,
        )
    effects = EnhancementEffects(
        editor=editor,
        restorer=SeedVR2Restorer(
            model="seedvr2-7b"
            if arguments.seedvr2_model == "seedvr2-7b"
            else "seedvr2-3b",
            low_ram=arguments.low_ram,
            minimum_free_gigabytes=arguments.min_free_memory_gb,
        ),
        images=PillowImageOperations(),
        identity=identity,
    )
    output_root = arguments.output_directory or (
        _workspace(arguments, arguments.identity_task).outputs_directory
        if arguments.identity_task
        else Path.cwd() / "likeness-outputs"
    )
    for source in inputs:
        output_directory = output_root / f"{source.stem}-{recipe.name}"
        print(f"Enhancing {source.name} with {recipe.name}...")
        report = enhance_image(
            source=source,
            recipe=recipe,
            output_directory=output_directory,
            effects=effects,
            base_seed=arguments.seed,
            target_megapixels=arguments.megapixels or recipe.working_megapixels,
            final_short_edge=arguments.final_short_edge,
            keep_intermediates=arguments.keep_intermediates,
            restoration=RestorationOptions(
                texture_strength=arguments.texture_strength,
                minimum_acutance=arguments.sharp_threshold
                if arguments.sharp_threshold > 0
                else None,
            ),
        )
        rejected = sum(1 for step in report.steps if not step.accepted)
        similarity = (
            ""
            if report.source_similarity is None or report.final_similarity is None
            else f" | likeness {report.source_similarity:.2f} -> {report.final_similarity:.2f}"
        )
        print(
            f"  {report.final_output}{similarity}"
            + (
                f" | {rejected} edit attempt(s) rejected for drifting"
                if rejected
                else ""
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="likeness",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--home", type=Path, default=default_likeness_home())
    commands = parser.add_subparsers(dest="command", required=True)

    def task_command(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--task", required=True, help="Task name, e.g. 'me'")
        return command

    commands.add_parser(
        "libraries", help="Find Photos libraries on this Mac and mounted drives"
    )
    people = commands.add_parser("people", help="List named people in Photos")
    people.add_argument("--library", type=Path, default=None)

    select = task_command("select", "Rank photos of a person (no downloads)")
    select.add_argument(
        "--person", required=True, help="Name exactly as tagged in Photos"
    )
    select.add_argument(
        "--since", default=None, help="Earliest capture date, YYYY-MM-DD"
    )
    select.add_argument(
        "--years",
        type=float,
        default=3.0,
        help="Default window when --since is not given (0 = all)",
    )
    select.add_argument("--target", type=int, default=60)
    select.add_argument("--show", type=int, default=15)
    select.add_argument("--library", type=Path, default=None)

    pull = task_command("pull", "Download the selected originals")
    pull.add_argument("--library", type=Path, default=None)

    task_command("identity", "Build the face reference used by the identity check")

    prepare = task_command("train-prepare", "Write an mflux LoRA dataset and config")
    prepare.add_argument(
        "--trigger-word", required=True, help="Rare token for you, e.g. 'ohwx'"
    )
    prepare.add_argument(
        "--subject-class", default="person", help="e.g. man, woman, person"
    )
    prepare.add_argument("--quantize", type=int, choices=[4, 8], default=None)

    export = task_command(
        "train-export", "Copy the newest trained LoRA adapter into lora/"
    )
    export.add_argument("--trigger-word", default="")

    task_command("release", "Delete pulled photos, datasets and checkpoints")
    commands.add_parser("status", help="Disk use per task")
    commands.add_parser("recipes", help="List enhancement recipes")

    enhance = commands.add_parser("enhance", help="Enhance photos with a recipe")
    enhance.add_argument("images", nargs="*", type=Path)
    enhance.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    enhance.add_argument(
        "--from-task", default=None, help="Enhance a task's pulled photos"
    )
    enhance.add_argument(
        "--identity-task",
        default=None,
        help="Task whose identity reference guards realistic edits",
    )
    enhance.add_argument(
        "--output-dir", dest="output_directory", type=Path, default=None
    )
    enhance.add_argument("--edit-model", default=DEFAULT_EDIT_MODEL)
    enhance.add_argument("--exo-url", default="http://localhost:52415")
    enhance.add_argument(
        "--launch-model",
        action="store_true",
        help="Launch the edit model in exo if it is not running",
    )
    enhance.add_argument(
        "--megapixels",
        type=float,
        default=None,
        help=(
            "Working size; photos are reduced to this, never enlarged (default: 1 MP "
            "for recipes that edit, 6 MP for clarity; 0 keeps the native size)"
        ),
    )
    enhance.add_argument(
        "--final-short-edge",
        type=int,
        default=None,
        help=(
            "Enlarge to this short edge at the end (default: 2048 for recipes that "
            "edit, none for clarity; 0 skips)"
        ),
    )
    enhance.add_argument("--quality", choices=["low", "medium", "high"], default=None)
    enhance.add_argument("--seed", type=int, default=7)
    enhance.add_argument("--keep-intermediates", action="store_true")
    enhance.add_argument(
        "--seedvr2-model", choices=["seedvr2-3b", "seedvr2-7b"], default="seedvr2-3b"
    )
    enhance.add_argument(
        "--low-ram",
        action="store_true",
        help="Run SeedVR2 in mflux's low-memory mode (slower)",
    )
    enhance.add_argument(
        "--min-free-memory-gb",
        type=float,
        default=32.0,
        help="Refuse to start SeedVR2 with less free memory than this (0 disables)",
    )
    enhance.add_argument(
        "--texture-strength",
        type=float,
        default=RestorationOptions().texture_strength,
        help=(
            "How much of the photo's own fine texture and color to blend back into "
            "SeedVR2's output: 0 keeps SeedVR2's, 1 only the photo's (default: 0.5)"
        ),
    )
    enhance.add_argument(
        "--sharp-threshold",
        type=float,
        default=DEFAULT_MINIMUM_ACUTANCE,
        help=(
            "Skip restoring photos whose face is at least this sharp (acutance, "
            f"default: {DEFAULT_MINIMUM_ACUTANCE}); 0 always restores. "
            "report.json records each photo's measurement"
        ),
    )
    return parser


COMMANDS = {
    "libraries": command_libraries,
    "people": command_people,
    "select": command_select,
    "pull": command_pull,
    "identity": command_identity,
    "train-prepare": command_train_prepare,
    "train-export": command_train_export,
    "release": command_release,
    "status": command_status,
    "recipes": command_recipes,
    "enhance": command_enhance,
}


def main() -> None:
    arguments = build_parser().parse_args(namespace=_Arguments())
    try:
        COMMANDS[arguments.command](arguments)
    except (
        PhotosLibraryUnavailableError,
        InsufficientMemoryError,
        ExoRequestError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
