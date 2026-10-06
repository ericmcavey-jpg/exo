"""Typed records shared across the likeness toolkit."""

from datetime import datetime
from typing import Literal, NewType

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

PhotoUuid = NewType("PhotoUuid", str)

Framing = Literal["close_up", "medium", "wide"]
HeadAngle = Literal[
    "frontal", "three_quarter_left", "three_quarter_right", "profile", "unknown"
]


class FaceMetrics(FrozenModel):
    """What Photos' face analysis recorded about the target person's face in one photo."""

    quality: float | None
    relative_size: float | None
    yaw_radians: float | None
    eyes_closed: bool
    smiling: bool | None


class PhotoCandidate(FrozenModel):
    """Metadata for one library photo, read from the local Photos database (no download)."""

    uuid: PhotoUuid
    captured_at: datetime
    width: int
    height: int
    person_count: int
    is_favorite: bool
    is_screenshot: bool
    is_hidden: bool
    is_original_local: bool
    overall_score: float | None
    sharpness_score: float | None
    lighting_score: float | None
    composition_score: float | None
    face: FaceMetrics | None


class SelectionSettings(FrozenModel):
    person_name: str
    # None means the library Photos is using; otherwise a .photoslibrary path.
    library_path: str | None = None
    earliest_capture: datetime | None
    target_count: int = Field(ge=1)
    minimum_short_side_pixels: int = 1024
    minimum_face_size: float = 0.03
    minimum_seconds_between_picks: float = 90.0
    maximum_picks_per_day: int = 4
    close_up_share: float = 0.40
    medium_share: float = 0.35
    wide_share: float = 0.25


class SelectedPhoto(FrozenModel):
    uuid: PhotoUuid
    captured_at: datetime
    framing: Framing
    head_angle: HeadAngle
    ranking_score: float
    is_favorite: bool


class TaskManifest(FrozenModel):
    """Everything needed to re-create a task's photo set without keeping the photos."""

    task_name: str
    created_at: datetime
    settings: SelectionSettings
    selected: list[SelectedPhoto]
    pulled_files: dict[PhotoUuid, str] = {}


class IdentityReference(FrozenModel):
    """Averaged face embedding of the person, kept after the source photos are released."""

    embedder_name: str
    embedding: list[float]
    source_face_count: int
    discarded_face_count: int
    created_at: datetime
