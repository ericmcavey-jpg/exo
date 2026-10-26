"""Start or extend a task from a folder of photo files instead of the Photos library.

For someone whose photos are not in your library: they send the files (AirDrop, iCloud
Drive), and `import` copies them into the task as if `select` and `pull` had picked
them. Each file gets a stable UUID derived from its content, so importing the same
folder twice, or two identical copies of a photo, adds it once.
"""

import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from exo.likeness.geometry import FaceBox
from exo.likeness.models import (
    PhotoUuid,
    SelectedPhoto,
    SelectionSettings,
    TaskManifest,
)
from exo.likeness.selection import classify_framing, classify_head_angle
from exo.likeness.workspace import TaskWorkspace
from exo.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from exo.likeness.identity import InsightFaceEmbedder

IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".heic"})
# Fixed namespace so a file's UUID is the same on every Mac and every run.
IMPORT_NAMESPACE: Final = uuid.UUID("6f1c2d0e-5b7a-4e8f-9c3d-2a1b0e9f8d7c")
EXIF_IFD: Final = 0x8769
DATE_TIME_ORIGINAL: Final = 0x9003


class FoundFace(FrozenModel):
    box: FaceBox
    yaw_radians: float | None
    image_width: int
    image_height: int

    @property
    def relative_size(self) -> float:
        """Face size as a share of the image's short side, like Photos' face size."""
        left, top, right, bottom = self.box
        short_side = min(self.image_width, self.image_height)
        if short_side <= 0:
            return 0.0
        return max(right - left, bottom - top) / short_side


class PhotoInspector(Protocol):
    def find_largest_face(self, image_path: Path) -> FoundFace | None: ...

    def capture_time(self, image_path: Path) -> datetime | None: ...


class ImportResult(FrozenModel):
    imported: list[PhotoUuid]
    already_in_task: list[str]
    excluded: list[str]
    without_face: list[str]


def photo_uuid_for(path: Path) -> PhotoUuid:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PhotoUuid(str(uuid.uuid5(IMPORT_NAMESPACE, digest)).upper())


def import_folder(
    *,
    workspace: TaskWorkspace,
    folder: Path,
    person_name: str,
    inspector: PhotoInspector,
) -> ImportResult:
    """Copy the folder's photos into the task and select every one with a face."""
    previous = workspace.load_manifest() if workspace.manifest_path.exists() else None
    selected = [] if previous is None else list(previous.selected)
    pulled = {} if previous is None else dict(previous.pulled_files)
    excluded_uuids = {} if previous is None else previous.excluded
    known = {photo.uuid for photo in selected}
    imported: list[PhotoUuid] = []
    already_in_task: list[str] = []
    excluded: list[str] = []
    without_face: list[str] = []
    workspace.originals_directory.mkdir(parents=True, exist_ok=True)
    for source in sorted(folder.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        photo_uuid = photo_uuid_for(source)
        if photo_uuid in excluded_uuids:
            excluded.append(source.name)
            continue
        if photo_uuid in known:
            already_in_task.append(source.name)
            continue
        face = inspector.find_largest_face(source)
        if face is None:
            without_face.append(source.name)
            continue
        filename = f"{photo_uuid}{source.suffix.lower()}"
        shutil.copy2(source, workspace.originals_directory / filename)
        captured_at = (
            inspector.capture_time(source)
            or datetime.fromtimestamp(source.stat().st_mtime).astimezone()
        )
        selected.append(
            SelectedPhoto(
                uuid=photo_uuid,
                captured_at=captured_at,
                framing=classify_framing(face.relative_size),
                head_angle=classify_head_angle(face.yaw_radians),
                ranking_score=0.0,
                is_favorite=False,
            )
        )
        pulled[photo_uuid] = filename
        known.add(photo_uuid)
        imported.append(photo_uuid)
    settings = (
        previous.settings
        if previous is not None
        else SelectionSettings(
            person_name=person_name,
            earliest_capture=None,
            target_count=max(1, len(selected)),
        )
    )
    workspace.save_manifest(
        TaskManifest(
            task_name=workspace.task_name,
            created_at=datetime.now().astimezone()
            if previous is None
            else previous.created_at,
            settings=settings,
            selected=selected,
            pulled_files=pulled,
            excluded=excluded_uuids,
        )
    )
    return ImportResult(
        imported=imported,
        already_in_task=already_in_task,
        excluded=excluded,
        without_face=without_face,
    )


class InsightFacePhotoInspector:
    """Finds faces with InsightFace and reads capture times from EXIF."""

    def __init__(self, embedder: "InsightFaceEmbedder") -> None:
        self._embedder = embedder

    def find_largest_face(self, image_path: Path) -> FoundFace | None:
        from PIL import Image, ImageOps

        found = self._embedder.largest_face_box_and_yaw(image_path)
        if found is None:
            return None
        box, yaw = found
        with Image.open(image_path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        return FoundFace(
            box=box, yaw_radians=yaw, image_width=width, image_height=height
        )

    def capture_time(self, image_path: Path) -> datetime | None:
        from PIL import Image

        with Image.open(image_path) as opened:
            exif = opened.getexif().get_ifd(EXIF_IFD)
        value = exif.get(DATE_TIME_ORIGINAL)
        if not isinstance(value, str):
            return None
        try:
            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").astimezone()
        except ValueError:
            return None
