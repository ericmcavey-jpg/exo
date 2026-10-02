"""Turn a task's photos into an mflux LoRA training run, then keep only the adapter.

mflux trains LoRAs for Z-Image and FLUX.2 (not FLUX.1 or Qwen-Image). The config
below follows mflux's own Z-Image Turbo example; the trained adapter is used with
`mflux-generate-z-image-turbo --lora-paths <adapter>`.
"""

import json
import zipfile
from pathlib import Path
from typing import Final, Protocol

from exo.likeness.models import Framing, HeadAngle, SelectedPhoto, TaskManifest
from exo.likeness.workspace import TaskWorkspace

TRAINING_MODEL: Final = "z-image-turbo"
TARGET_TRAINING_STEPS: Final = 2000
DATASET_LONG_EDGE: Final = 1536

FRAMING_PHRASES: Final[dict[Framing, str]] = {
    "close_up": "a close-up portrait photo",
    "medium": "a waist-up photo",
    "wide": "a full-body photo",
}
ANGLE_PHRASES: Final[dict[HeadAngle, str]] = {
    "frontal": "facing the camera",
    "three_quarter_left": "head turned slightly to the side",
    "three_quarter_right": "head turned slightly to the side",
    "profile": "seen in profile",
}


class TrainingImageWriter(Protocol):
    def write_training_copy(
        self, *, source: Path, destination: Path, long_edge: int
    ) -> None: ...


def caption_for(photo: SelectedPhoto, trigger_word: str, subject_class: str) -> str:
    return (
        f"{FRAMING_PHRASES[photo.framing]} of {trigger_word} {subject_class}, "
        f"{ANGLE_PHRASES[photo.head_angle]}"
    )


def suggested_epochs(
    image_count: int, target_steps: int = TARGET_TRAINING_STEPS
) -> int:
    return min(100, max(8, round(target_steps / max(1, image_count))))


def lora_targets(rank: int) -> list[dict[str, object]]:
    """The layer set from mflux's Z-Image Turbo training example."""
    block_paths = [
        "layers.{block}.attention.to_q",
        "layers.{block}.attention.to_k",
        "layers.{block}.attention.to_v",
        "layers.{block}.attention.to_out.0",
        "layers.{block}.feed_forward.w1",
        "layers.{block}.feed_forward.w2",
        "layers.{block}.feed_forward.w3",
    ]
    targets: list[dict[str, object]] = [
        {"module_path": path, "blocks": {"start": 0, "end": 30}, "rank": rank}
        for path in block_paths
    ]
    targets.append({"module_path": "cap_embedder.1", "rank": rank})
    targets.append({"module_path": "all_final_layer.2-1.linear", "rank": rank})
    return targets


def build_training_config(
    *,
    data_directory: Path,
    output_directory: Path,
    image_count: int,
    quantize: int | None,
    rank: int = 16,
    seed: int = 4,
) -> dict[str, object]:
    checkpoint_every = 250
    return {
        "model": TRAINING_MODEL,
        "data": str(data_directory),
        "seed": seed,
        "steps": 9,
        "guidance": 0.0,
        "quantize": quantize,
        "max_resolution": 1024,
        "low_ram": False,
        "training_loop": {
            "num_epochs": suggested_epochs(image_count),
            "batch_size": 1,
            "timestep_low": 4,
            "timestep_high": 9,
        },
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {
            "save_frequency": checkpoint_every,
            "output_path": str(output_directory),
        },
        "monitoring": {
            "preview_width": 896,
            "preview_height": 1152,
            "plot_frequency": 1,
            "generate_image_frequency": checkpoint_every,
        },
        "lora_layers": {"targets": lora_targets(rank)},
    }


def write_dataset(
    *,
    manifest: TaskManifest,
    workspace: TaskWorkspace,
    trigger_word: str,
    subject_class: str,
    writer: TrainingImageWriter,
    quantize: int | None,
) -> tuple[Path, int]:
    """Write captioned images and train.json. Returns the config path and image count."""
    data_directory = workspace.training_directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    image_count = 0
    for photo in manifest.selected:
        filename = manifest.pulled_files.get(photo.uuid)
        if filename is None:
            continue
        source = workspace.originals_directory / filename
        if not source.exists():
            continue
        image_count += 1
        stem = f"{image_count:03d}"
        writer.write_training_copy(
            source=source,
            destination=data_directory / f"{stem}.jpg",
            long_edge=DATASET_LONG_EDGE,
        )
        (data_directory / f"{stem}.txt").write_text(
            caption_for(photo, trigger_word, subject_class)
        )
    if image_count == 0:
        raise FileNotFoundError(
            "No pulled photos to train on. Run `exo-likeness pull` for this task first."
        )
    (data_directory / "preview_1.txt").write_text(
        f"a portrait photo of {trigger_word} {subject_class} smiling outdoors in soft daylight"
    )
    config_path = workspace.training_directory / "train.json"
    config = build_training_config(
        data_directory=data_directory,
        output_directory=workspace.training_directory / "run",
        image_count=image_count,
        quantize=quantize,
    )
    config_path.write_text(json.dumps(config, indent=2))
    return config_path, image_count


def export_latest_adapter(workspace: TaskWorkspace) -> Path:
    """Copy the newest checkpoint's LoRA adapter into lora/, which survives `release`."""
    checkpoints = sorted(
        workspace.training_directory.rglob("*_checkpoint.zip"),
        key=lambda path: (path.name, path.stat().st_mtime),
    )
    if not checkpoints:
        raise FileNotFoundError(
            "No training checkpoints found. Run the mflux-train command first."
        )
    latest = checkpoints[-1]
    iteration = latest.name.split("_", 1)[0]
    with zipfile.ZipFile(latest) as archive:
        adapter_names = [
            name for name in archive.namelist() if name.endswith("_adapter.safetensors")
        ]
        if not adapter_names:
            raise FileNotFoundError(f"{latest} contains no LoRA adapter.")
        workspace.lora_directory.mkdir(parents=True, exist_ok=True)
        destination = (
            workspace.lora_directory / f"{workspace.task_name}-{iteration}.safetensors"
        )
        destination.write_bytes(archive.read(adapter_names[0]))
    return destination
