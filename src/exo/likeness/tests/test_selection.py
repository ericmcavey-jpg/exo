import math
from datetime import datetime, timedelta, timezone

from exo.likeness.models import (
    FaceMetrics,
    PhotoCandidate,
    PhotoUuid,
    SelectionSettings,
)
from exo.likeness.selection import (
    classify_framing,
    classify_head_angle,
    ranking_score,
    rejection_reason,
    select_photos,
)

START = datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc)


def make_candidate(
    uuid: str,
    *,
    captured_at: datetime = START,
    face_size: float | None = 0.30,
    yaw_degrees: float = 0.0,
    overall: float | None = 0.5,
    favorite: bool = False,
    person_count: int = 1,
    largest_other_face_size: float | None = None,
    width: int = 3024,
    height: int = 4032,
    eyes_closed: bool = False,
    screenshot: bool = False,
    has_face: bool = True,
) -> PhotoCandidate:
    return PhotoCandidate(
        uuid=PhotoUuid(uuid),
        captured_at=captured_at,
        width=width,
        height=height,
        person_count=person_count,
        largest_other_face_size=largest_other_face_size,
        is_favorite=favorite,
        is_screenshot=screenshot,
        is_hidden=False,
        is_original_local=False,
        overall_score=overall,
        sharpness_score=0.5,
        lighting_score=0.5,
        composition_score=0.5,
        face=FaceMetrics(
            quality=0.5,
            relative_size=face_size,
            yaw_radians=math.radians(yaw_degrees),
            eyes_closed=eyes_closed,
            smiling=None,
        )
        if has_face
        else None,
    )


def settings(target: int = 10, **overrides: float) -> SelectionSettings:
    return SelectionSettings.model_validate(
        {
            "person_name": "Me",
            "earliest_capture": None,
            "target_count": target,
            **overrides,
        }
    )


def test_framing_and_angle_buckets():
    assert classify_framing(0.40) == "close_up"
    assert classify_framing(0.15) == "medium"
    assert classify_framing(0.05) == "wide"
    assert classify_framing(None) == "medium"
    assert classify_head_angle(math.radians(5)) == "frontal"
    assert classify_head_angle(math.radians(-30)) == "three_quarter_left"
    assert classify_head_angle(math.radians(30)) == "three_quarter_right"
    assert classify_head_angle(math.radians(70)) == "profile"
    assert classify_head_angle(None) == "unknown"


def test_rejection_reasons():
    rules = settings()
    assert rejection_reason(make_candidate("a"), rules) is None
    assert rejection_reason(make_candidate("a", screenshot=True), rules) == "screenshot"
    assert (
        rejection_reason(make_candidate("a", person_count=2), rules)
        == "other people in the photo"
    )
    assert (
        rejection_reason(make_candidate("a", width=800, height=600), rules)
        == "low resolution"
    )
    assert (
        rejection_reason(make_candidate("a", eyes_closed=True), rules) == "eyes closed"
    )
    assert (
        rejection_reason(make_candidate("a", face_size=0.01), rules) == "face too small"
    )
    assert (
        rejection_reason(make_candidate("a", has_face=False), rules)
        == "no face data for this person"
    )
    dated = SelectionSettings(
        person_name="Me", earliest_capture=START + timedelta(days=1), target_count=5
    )
    assert rejection_reason(make_candidate("a"), dated) == "older than the date range"


def test_favorites_rank_higher():
    assert ranking_score(make_candidate("a", favorite=True)) > ranking_score(
        make_candidate("b")
    )


def test_burst_shots_and_daily_cap_are_thinned():
    burst = [
        make_candidate(f"burst{index}", captured_at=START + timedelta(seconds=index))
        for index in range(5)
    ]
    same_day = [
        make_candidate(f"day{index}", captured_at=START + timedelta(hours=index + 1))
        for index in range(8)
    ]
    result = select_photos(burst + same_day, settings(target=20))
    picked = {photo.uuid for photo in result.selected}
    assert len(picked & {f"burst{index}" for index in range(5)}) == 1
    assert len(result.selected) == 4  # the default cap of four photos per day


def test_angles_stay_varied_within_a_framing():
    frontal = [
        make_candidate(
            f"front{index}", captured_at=START + timedelta(days=index), overall=0.9
        )
        for index in range(6)
    ]
    profile = make_candidate(
        "side", captured_at=START + timedelta(days=30), yaw_degrees=70, overall=0.1
    )
    result = select_photos(
        [*frontal, profile],
        settings(target=3, close_up_share=1.0, medium_share=0.0, wide_share=0.0),
    )
    assert "side" in {photo.uuid for photo in result.selected}


def test_shortfall_in_one_framing_is_filled_from_others():
    close_ups = [
        make_candidate(f"close{index}", captured_at=START + timedelta(days=index))
        for index in range(10)
    ]
    result = select_photos(close_ups, settings(target=8))
    assert len(result.selected) == 8
    assert result.eligible_count == 10
    assert result.rejection_counts == {}


def test_background_people_are_allowed_but_companions_are_not():
    rules = settings()
    background = make_candidate("a", person_count=3, largest_other_face_size=0.05)
    companion = make_candidate("b", person_count=2, largest_other_face_size=0.20)
    no_face_sizes = make_candidate("c", person_count=2)
    assert rejection_reason(background, rules) is None
    assert rejection_reason(companion, rules) == "other people in the photo"
    assert rejection_reason(no_face_sizes, rules) == "other people in the photo"
