from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from exo.likeness.models import PhotoUuid
from exo.likeness.photo_library import (
    build_pull_command,
    candidate_from_photo,
    match_pulled_files,
)


def fake_photo(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "uuid": "ABC-123",
        "date": datetime(2025, 5, 4, 12, 30),
        "width": 4032,
        "height": 3024,
        "persons": ["Me"],
        "favorite": True,
        "screenshot": False,
        "hidden": False,
        "ismissing": True,
        "score": SimpleNamespace(
            overall=0.8,
            sharply_focused_subject=0.7,
            pleasant_lighting=0.6,
            well_framed_subject=0.5,
        ),
        "face_info": [
            SimpleNamespace(name="Someone Else", size=0.5, quality=0.9),
            SimpleNamespace(
                name="Me",
                size=0.2,
                quality=0.4,
                yaw=0.1,
                left_eye_closed=False,
                right_eye_closed=False,
                has_smile=True,
            ),
            SimpleNamespace(name="Me", size=0.3, quality=-1.0, yaw=0.2),
        ],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_maps_osxphotos_fields_and_picks_the_persons_largest_face():
    candidate = candidate_from_photo(fake_photo(), "Me")
    assert candidate is not None
    assert candidate.uuid == "ABC-123"
    assert candidate.captured_at.tzinfo is not None
    assert candidate.is_favorite
    assert not candidate.is_original_local
    assert candidate.overall_score == 0.8
    assert candidate.face is not None
    assert candidate.face.relative_size == 0.3
    assert candidate.face.quality is None  # Photos' "not scored yet" marker
    assert candidate.face.smiling is None


def test_missing_fields_degrade_instead_of_crashing():
    candidate = candidate_from_photo(
        fake_photo(score=None, face_info=None, width="wide", persons=None), "Me"
    )
    assert candidate is not None
    assert candidate.overall_score is None
    assert candidate.face is None
    assert candidate.width == 0
    assert candidate.person_count == 0
    assert candidate_from_photo(fake_photo(uuid=None), "Me") is None


def test_prefers_your_edit_over_the_original():
    uuids = [PhotoUuid("one"), PhotoUuid("two"), PhotoUuid("three")]
    exported = [Path("one.jpeg"), Path("one_edited.jpeg"), Path("two.jpeg")]
    assert match_pulled_files(uuids, exported) == {
        "one": Path("one_edited.jpeg"),
        "two": Path("two.jpeg"),
    }


def test_pull_command_downloads_only_listed_photos():
    command = build_pull_command(
        Path("/w/uuids.txt"), Path("/w/originals"), Path("/lib.photoslibrary")
    )
    assert command[command.index("--uuid-from-file") + 1] == "/w/uuids.txt"
    assert "--download-missing" in command
    assert command[-2:] == ["--library", "/lib.photoslibrary"]
