"""Choose a flattering, varied set of photos of one person from library metadata alone.

Everything here is pure: it works on `PhotoCandidate` records that the Photos
adapter reads from the local database, so ranking never downloads a photo.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, final

from exo.likeness.models import (
    Framing,
    HeadAngle,
    PhotoCandidate,
    SelectedPhoto,
    SelectionSettings,
)

CLOSE_UP_MINIMUM_FACE_SIZE: Final = 0.25
MEDIUM_MINIMUM_FACE_SIZE: Final = 0.10
FRONTAL_MAXIMUM_YAW_RADIANS: Final = math.radians(15)
THREE_QUARTER_MAXIMUM_YAW_RADIANS: Final = math.radians(50)
NEUTRAL_SCORE: Final = 0.5
HEAD_ANGLES: Final[tuple[HeadAngle, ...]] = (
    "frontal",
    "three_quarter_left",
    "three_quarter_right",
    "profile",
)


@final
@dataclass(frozen=True)
class SelectionResult:
    selected: list[SelectedPhoto]
    eligible_count: int
    rejection_counts: dict[str, int]


def classify_framing(relative_face_size: float | None) -> Framing:
    if relative_face_size is None:
        return "medium"
    if relative_face_size >= CLOSE_UP_MINIMUM_FACE_SIZE:
        return "close_up"
    if relative_face_size >= MEDIUM_MINIMUM_FACE_SIZE:
        return "medium"
    return "wide"


def classify_head_angle(yaw_radians: float | None) -> HeadAngle:
    if yaw_radians is None or abs(yaw_radians) <= FRONTAL_MAXIMUM_YAW_RADIANS:
        return "frontal"
    if abs(yaw_radians) > THREE_QUARTER_MAXIMUM_YAW_RADIANS:
        return "profile"
    return "three_quarter_left" if yaw_radians < 0 else "three_quarter_right"


def rejection_reason(
    candidate: PhotoCandidate, settings: SelectionSettings
) -> str | None:
    """Why a photo cannot be used for a likeness set, or None if it can."""
    if candidate.is_screenshot:
        return "screenshot"
    if candidate.is_hidden:
        return "hidden in Photos"
    if candidate.person_count != 1:
        return "other people in the photo"
    if (
        settings.earliest_capture is not None
        and candidate.captured_at < settings.earliest_capture
    ):
        return "older than the date range"
    if min(candidate.width, candidate.height) < settings.minimum_short_side_pixels:
        return "low resolution"
    if candidate.face is None:
        return "no face data for this person"
    if candidate.face.eyes_closed:
        return "eyes closed"
    if (
        candidate.face.relative_size is not None
        and candidate.face.relative_size < settings.minimum_face_size
    ):
        return "face too small"
    return None


def _unit_interval(value: float | None) -> float:
    if value is None:
        return NEUTRAL_SCORE
    return min(1.0, max(0.0, value))


def ranking_score(candidate: PhotoCandidate) -> float:
    """Blend Photos' own aesthetic scores with face quality; favorites get a strong boost."""
    face_quality = None if candidate.face is None else candidate.face.quality
    score = (
        0.30 * _unit_interval(candidate.overall_score)
        + 0.20 * _unit_interval(candidate.sharpness_score)
        + 0.15 * _unit_interval(candidate.lighting_score)
        + 0.10 * _unit_interval(candidate.composition_score)
        + 0.25 * _unit_interval(face_quality)
    )
    if candidate.is_favorite:
        score += 0.15
    if candidate.face is not None and candidate.face.smiling:
        score += 0.05
    return round(score, 4)


def _to_selected(candidate: PhotoCandidate) -> SelectedPhoto:
    face = candidate.face
    return SelectedPhoto(
        uuid=candidate.uuid,
        captured_at=candidate.captured_at,
        framing=classify_framing(None if face is None else face.relative_size),
        head_angle=classify_head_angle(None if face is None else face.yaw_radians),
        ranking_score=ranking_score(candidate),
        is_favorite=candidate.is_favorite,
    )


@final
class _SpacingGuard:
    """Rejects near-duplicates (bursts) and too many picks from the same day."""

    def __init__(self, settings: SelectionSettings) -> None:
        self._minimum_seconds = settings.minimum_seconds_between_picks
        self._maximum_per_day = settings.maximum_picks_per_day
        self._picked: list[SelectedPhoto] = []
        self._per_day: Counter[date] = Counter()

    def allows(self, photo: SelectedPhoto) -> bool:
        if self._per_day[photo.captured_at.date()] >= self._maximum_per_day:
            return False
        return all(
            abs((photo.captured_at - picked.captured_at).total_seconds())
            >= self._minimum_seconds
            for picked in self._picked
        )

    def record(self, photo: SelectedPhoto) -> None:
        self._picked.append(photo)
        self._per_day[photo.captured_at.date()] += 1


def _framing_quotas(settings: SelectionSettings) -> dict[Framing, int]:
    shares: dict[Framing, float] = {
        "close_up": settings.close_up_share,
        "medium": settings.medium_share,
        "wide": settings.wide_share,
    }
    total_share = sum(shares.values()) or 1.0
    return {
        framing: round(settings.target_count * share / total_share)
        for framing, share in shares.items()
    }


def _pick_round_robin_by_angle(
    pool: list[SelectedPhoto],
    quota: int,
    guard: _SpacingGuard,
    taken: set[str],
) -> list[SelectedPhoto]:
    """Take the best photos while cycling through head angles so no angle dominates."""
    queues: dict[HeadAngle, list[SelectedPhoto]] = {
        angle: [photo for photo in pool if photo.head_angle == angle]
        for angle in HEAD_ANGLES
    }
    picked: list[SelectedPhoto] = []
    while len(picked) < quota and any(queues.values()):
        active_angles: list[HeadAngle] = [
            angle for angle, queue in queues.items() if queue
        ]
        active_angles.sort(key=lambda angle: -queues[angle][0].ranking_score)
        for angle in active_angles:
            queue = queues[angle]
            while queue:
                photo = queue.pop(0)
                if photo.uuid in taken or not guard.allows(photo):
                    continue
                guard.record(photo)
                taken.add(photo.uuid)
                picked.append(photo)
                break
            if len(picked) >= quota:
                break
    return picked


def select_photos(
    candidates: Sequence[PhotoCandidate], settings: SelectionSettings
) -> SelectionResult:
    rejection_counts: Counter[str] = Counter()
    eligible: list[SelectedPhoto] = []
    for candidate in candidates:
        reason = rejection_reason(candidate, settings)
        if reason is None:
            eligible.append(_to_selected(candidate))
        else:
            rejection_counts[reason] += 1
    eligible.sort(key=lambda photo: photo.ranking_score, reverse=True)

    guard = _SpacingGuard(settings)
    taken: set[str] = set()
    selected: list[SelectedPhoto] = []
    for framing, quota in _framing_quotas(settings).items():
        pool = [photo for photo in eligible if photo.framing == framing]
        selected.extend(_pick_round_robin_by_angle(pool, quota, guard, taken))

    # Fill any shortfall (e.g. few full-body shots) with the best remaining photos.
    for photo in eligible:
        if len(selected) >= settings.target_count:
            break
        if photo.uuid not in taken and guard.allows(photo):
            guard.record(photo)
            taken.add(photo.uuid)
            selected.append(photo)

    selected.sort(key=lambda photo: photo.ranking_score, reverse=True)
    return SelectionResult(
        selected=selected[: settings.target_count],
        eligible_count=len(eligible),
        rejection_counts=dict(rejection_counts),
    )
