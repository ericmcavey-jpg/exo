"""Per-task working folders: pulled photos are temporary, results and manifests are kept.

Layout under the likeness home (~/.exo/likeness, or $EXO_LIKENESS_HOME):

    <task>/manifest.json   which photos the task uses (kept)
    <task>/identity.json   averaged face embedding (kept, a few KB)
    <task>/originals/      full-resolution photos pulled from Photos (released)
    <task>/training/       dataset and checkpoints (released)
    <task>/lora/           exported LoRA adapter (kept)
    <task>/outputs/        enhanced images (kept)
"""

import os
import re
import shutil
from pathlib import Path
from typing import Final, final

from exo.likeness.models import IdentityReference, PhotoUuid, TaskManifest

TASK_NAME_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RELEASED_FOLDERS: Final = ("originals", "training")


def default_likeness_home() -> Path:
    configured = os.environ.get("EXO_LIKENESS_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".exo" / "likeness"


def folder_size_bytes(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


@final
class TaskWorkspace:
    def __init__(self, home: Path, task_name: str) -> None:
        if not TASK_NAME_PATTERN.fullmatch(task_name):
            raise ValueError(
                f"Invalid task name {task_name!r}: use lowercase letters, digits, '-' or '_'."
            )
        self.task_name = task_name
        self.directory = home / task_name

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def identity_path(self) -> Path:
        return self.directory / "identity.json"

    @property
    def originals_directory(self) -> Path:
        return self.directory / "originals"

    @property
    def training_directory(self) -> Path:
        return self.directory / "training"

    @property
    def lora_directory(self) -> Path:
        return self.directory / "lora"

    @property
    def outputs_directory(self) -> Path:
        return self.directory / "outputs"

    def save_manifest(self, manifest: TaskManifest) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(self.manifest_path, manifest.model_dump_json(indent=2))

    def load_manifest(self) -> TaskManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest for task {self.task_name!r}. Run `likeness select` first."
            )
        return TaskManifest.model_validate_json(self.manifest_path.read_text())

    def save_identity(self, identity: IdentityReference) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(self.identity_path, identity.model_dump_json())

    def load_identity(self) -> IdentityReference:
        if not self.identity_path.exists():
            raise FileNotFoundError(
                f"No identity reference for task {self.task_name!r}. "
                "Run `likeness identity` first."
            )
        return IdentityReference.model_validate_json(self.identity_path.read_text())

    def pulled_paths(self) -> list[Path]:
        manifest = self.load_manifest()
        paths = [
            self.originals_directory / filename
            for filename in manifest.pulled_files.values()
        ]
        return [path for path in paths if path.exists()]

    def release(self) -> int:
        """Delete pulled photos, datasets and checkpoints. Returns bytes freed."""
        freed = 0
        for name in RELEASED_FOLDERS:
            folder = self.directory / name
            freed += folder_size_bytes(folder)
            shutil.rmtree(folder, ignore_errors=True)
        if self.manifest_path.exists():
            manifest = self.load_manifest()
            self.save_manifest(manifest.model_copy(update={"pulled_files": {}}))
        return freed

    def exclude(self, identifiers: list[str], reason: str) -> list[PhotoUuid]:
        """Rule photos out of the task for good and delete their pulled copies.

        Identifiers are photo UUIDs or unique prefixes of at least 8 characters (the
        pulled file names). A full UUID may also name a photo the task has never
        selected, so it can be ruled out before `select` ever picks it. Photos stay
        in your Photos library; `select` just never picks them again. Returns the
        UUIDs excluded.
        """
        manifest = self.load_manifest()
        known = {photo.uuid for photo in manifest.selected} | set(manifest.pulled_files)
        uuids: list[PhotoUuid] = []
        for identifier in identifiers:
            if FULL_UUID_PATTERN.fullmatch(identifier.strip()):
                uuids.append(PhotoUuid(identifier.strip().upper()))
                continue
            uuids.append(_resolve_one(identifier, known, self.task_name))
        for uuid in uuids:
            filename = manifest.pulled_files.get(uuid)
            if filename is not None:
                (self.originals_directory / filename).unlink(missing_ok=True)
        self.save_manifest(
            manifest.model_copy(
                update={
                    "selected": [
                        photo for photo in manifest.selected if photo.uuid not in uuids
                    ],
                    "pulled_files": {
                        uuid: filename
                        for uuid, filename in manifest.pulled_files.items()
                        if uuid not in uuids
                    },
                    "excluded": manifest.excluded | {uuid: reason for uuid in uuids},
                }
            )
        )
        return uuids

    def adopt(self, source: "TaskWorkspace", identifiers: list[str]) -> list[PhotoUuid]:
        """Add photos selected and pulled in another task (e.g. a scan) to this one.

        Copies their selection records and pulled originals; refuses photos this task
        has excluded. Returns the UUIDs added.
        """
        manifest = self.load_manifest()
        source_manifest = source.load_manifest()
        source_photos = {photo.uuid: photo for photo in source_manifest.selected}
        uuids = [
            _resolve_one(identifier, set(source_photos), source.task_name)
            for identifier in identifiers
        ]
        refused = [uuid for uuid in uuids if uuid in manifest.excluded]
        if refused:
            raise ValueError(
                f"Task {self.task_name!r} has excluded {', '.join(refused)}; not adding."
            )
        self.originals_directory.mkdir(parents=True, exist_ok=True)
        selected = list(manifest.selected)
        pulled = dict(manifest.pulled_files)
        already = {photo.uuid for photo in selected}
        for uuid in uuids:
            filename = source_manifest.pulled_files.get(uuid)
            if filename is None:
                raise ValueError(
                    f"{uuid} was not pulled in task {source.task_name!r}; run `likeness pull` there first."
                )
            shutil.copy2(
                source.originals_directory / filename,
                self.originals_directory / filename,
            )
            pulled[uuid] = filename
            if uuid not in already:
                selected.append(source_photos[uuid])
        self.save_manifest(
            manifest.model_copy(update={"selected": selected, "pulled_files": pulled})
        )
        return uuids

    def disk_usage(self) -> dict[str, int]:
        return {
            child.name: folder_size_bytes(child)
            for child in sorted(self.directory.iterdir())
            if child.is_dir()
        }


FULL_UUID_PATTERN: Final = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)


def _resolve_one(identifier: str, known: set[PhotoUuid], task_name: str) -> PhotoUuid:
    """The one known UUID that `identifier` (8+ leading characters) names."""
    prefix = identifier.strip().upper()
    matches = [uuid for uuid in known if uuid.upper().startswith(prefix)]
    if len(prefix) < 8 or len(matches) != 1:
        raise ValueError(
            f"{identifier!r} does not identify exactly one photo in task "
            f"{task_name!r}; use at least 8 characters of a pulled file name."
        )
    return matches[0]


def list_workspaces(home: Path) -> list[TaskWorkspace]:
    if not home.exists():
        return []
    return [
        TaskWorkspace(home, child.name)
        for child in sorted(home.iterdir())
        if child.is_dir() and TASK_NAME_PATTERN.fullmatch(child.name)
    ]


def _write_atomically(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def human_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
