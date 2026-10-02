import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exo.likeness.geometry import fit_long_edge, working_dimensions
from exo.likeness.models import (
    IdentityReference,
    PhotoUuid,
    SelectedPhoto,
    SelectionSettings,
    TaskManifest,
)
from exo.likeness.recipes import IDENTITY_LOCK, RECIPES, EditStep, Recipe
from exo.likeness.training import (
    build_training_config,
    caption_for,
    export_latest_adapter,
    suggested_epochs,
    write_dataset,
)
from exo.likeness.workspace import TaskWorkspace, human_size, list_workspaces

NOW = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)


def manifest_with(
    *uuids: str, pulled: dict[PhotoUuid, str] | None = None
) -> TaskManifest:
    return TaskManifest(
        task_name="me",
        created_at=NOW,
        settings=SelectionSettings(
            person_name="Me", earliest_capture=NOW, target_count=5
        ),
        selected=[
            SelectedPhoto(
                uuid=PhotoUuid(uuid),
                captured_at=NOW,
                framing="close_up",
                head_angle="frontal",
                ranking_score=0.7,
                is_favorite=False,
            )
            for uuid in uuids
        ],
        pulled_files=pulled or {},
    )


def test_task_names_cannot_escape_the_home(tmp_path: Path):
    with pytest.raises(ValueError):
        TaskWorkspace(tmp_path, "../elsewhere")
    with pytest.raises(ValueError):
        TaskWorkspace(tmp_path, "Me")


def test_release_frees_photos_but_keeps_results(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path, "me")
    workspace.save_manifest(manifest_with("a", pulled={PhotoUuid("a"): "a.jpeg"}))
    workspace.save_identity(
        IdentityReference(
            embedder_name="test",
            embedding=[1.0],
            source_face_count=1,
            discarded_face_count=0,
            created_at=NOW,
        )
    )
    for folder, name in (
        ("originals", "a.jpeg"),
        ("training", "x.zip"),
        ("lora", "me.safetensors"),
        ("outputs", "final.png"),
    ):
        (workspace.directory / folder).mkdir(parents=True)
        (workspace.directory / folder / name).write_bytes(b"0" * 100)

    assert workspace.release() == 200
    assert not workspace.originals_directory.exists()
    assert not workspace.training_directory.exists()
    assert (workspace.lora_directory / "me.safetensors").exists()
    assert (workspace.outputs_directory / "final.png").exists()
    assert workspace.load_manifest().pulled_files == {}
    assert workspace.load_identity().embedding == [1.0]
    assert [found.task_name for found in list_workspaces(tmp_path)] == ["me"]


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(1536) == "1.5 KB"
    assert human_size(3 * 1024**3) == "3.0 GB"


def test_geometry():
    width, height = working_dimensions(4032, 3024, 1.0)
    assert width % 16 == 0 and height % 16 == 0
    assert 0.9e6 < width * height < 1.1e6
    assert abs(width / height - 4 / 3) < 0.02
    assert fit_long_edge(800, 600, 1536) == (800, 600)
    assert fit_long_edge(4000, 3000, 1000) == (1000, 750)


def test_recipes_lock_identity_only_for_realistic_looks():
    for recipe in RECIPES.values():
        edits = [step for step in recipe.steps if isinstance(step, EditStep)]
        if recipe.style == "realistic":
            assert recipe.identity.enforce
            assert all(IDENTITY_LOCK in step.instruction for step in edits)
        else:
            assert not recipe.identity.enforce
        assert Recipe.model_validate_json(recipe.model_dump_json()) == recipe


def test_training_config_follows_the_mflux_example(tmp_path: Path):
    config = build_training_config(
        data_directory=tmp_path / "data",
        output_directory=tmp_path / "run",
        image_count=50,
        quantize=8,
    )
    assert config["model"] == "z-image-turbo"
    assert config["data"] == str(tmp_path / "data")
    assert config["training_loop"] == {
        "num_epochs": 40,
        "batch_size": 1,
        "timestep_low": 4,
        "timestep_high": 9,
    }
    targets = config["lora_layers"]
    assert isinstance(targets, dict)
    assert suggested_epochs(1) == 100
    assert suggested_epochs(1000) == 8


class CopyWriter:
    def write_training_copy(
        self, *, source: Path, destination: Path, long_edge: int
    ) -> None:
        destination.write_bytes(source.read_bytes())


def test_write_dataset_captions_pulled_photos(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path, "me")
    manifest = manifest_with("a", "b", pulled={PhotoUuid("a"): "a.jpeg"})
    workspace.originals_directory.mkdir(parents=True)
    (workspace.originals_directory / "a.jpeg").write_bytes(b"jpeg")

    config_path, count = write_dataset(
        manifest=manifest,
        workspace=workspace,
        trigger_word="ohwx",
        subject_class="man",
        writer=CopyWriter(),
        quantize=None,
    )
    data = workspace.training_directory / "data"
    assert count == 1
    assert (data / "001.txt").read_text() == caption_for(
        manifest.selected[0], "ohwx", "man"
    )
    assert (
        data / "001.txt"
    ).read_text() == "a close-up portrait photo of ohwx man, facing the camera"
    assert json.loads(config_path.read_text())["data"] == str(data)


def test_export_takes_the_newest_checkpoint(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path, "me")
    run = workspace.training_directory / "run" / "checkpoints"
    run.mkdir(parents=True)
    for iteration in ("0000250", "0000500"):
        with zipfile.ZipFile(run / f"{iteration}_checkpoint.zip", "w") as archive:
            archive.writestr(f"{iteration}_adapter.safetensors", iteration)
            archive.writestr(f"{iteration}_optimizer.safetensors", "big")

    adapter = export_latest_adapter(workspace)
    assert adapter.name == "me-0000500.safetensors"
    assert adapter.read_text() == "0000500"
