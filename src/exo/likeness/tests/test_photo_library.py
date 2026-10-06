from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from exo.likeness.models import PhotoUuid
from exo.likeness.photo_library import (
    build_pull_command,
    candidate_from_photo,
    find_photo_libraries,
    match_pulled_files,
    open_library_failure_message,
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


def test_reads_raw_eye_flags_integer_smiles_and_missing_head_angle():
    face = SimpleNamespace(
        name="Me",
        size=0.3,
        quality=0.6,
        yaw=0,
        has_smile=1,
        _info={"left_eye_closed": 0, "right_eye_closed": 1},
    )
    candidate = candidate_from_photo(fake_photo(face_info=[face]), "Me")
    assert candidate is not None and candidate.face is not None
    assert candidate.face.eyes_closed
    assert candidate.face.smiling is True
    assert candidate.face.yaw_radians is None


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
    assert command[-2:] == ["--library", "/lib.photoslibrary"]
    # iCloud downloads only work for the library Photos itself is using.
    assert "--download-missing" not in command
    system_library = build_pull_command(
        Path("/w/uuids.txt"), Path("/w/originals"), None
    )
    assert "--download-missing" in system_library and "--library" not in system_library


def test_finds_libraries_up_to_three_folders_deep(tmp_path: Path):
    drive = tmp_path / "Drive"
    expected = [
        drive / "Photos Library.photoslibrary",
        drive / "Backups" / "2024" / "Old.photoslibrary",
    ]
    for library in expected:
        library.mkdir(parents=True)
    (drive / "a" / "b" / "c" / "Too Deep.photoslibrary").mkdir(parents=True)
    (drive / "not-a-library.photoslibrary").write_text("")
    assert find_photo_libraries([drive, tmp_path / "missing"]) == sorted(expected)


def test_failure_message_points_ssh_users_at_remote_full_disk_access():
    message = open_library_failure_message(
        library_path=None,
        underlying_error="Could not get path to photo library database",
        over_ssh=True,
        libraries_found=[Path("/Volumes/Drive/Photos Library.photoslibrary")],
    )
    assert '--library "/Volumes/Drive/Photos Library.photoslibrary"' in message
    assert "Allow full disk access for remote users" in message
    local = open_library_failure_message(
        library_path=Path("/x.photoslibrary"),
        underlying_error="boom",
        over_ssh=False,
        libraries_found=[],
    )
    assert "No --library was given" not in local
    assert "terminal app has Full Disk Access" in local
