"""Read the Apple Photos library through osxphotos and pull originals on demand.

osxphotos is not an exo dependency; scripts/likeness adds it for the commands that need it.
Reading needs Full Disk Access for the terminal app. Queries use only the local
Photos database, which holds faces, People names and Photos' aesthetic scores even
when "Optimize Mac Storage" keeps the originals in iCloud. Originals are fetched
only by `pull_originals`, for the photos a task selected.

Every attribute read goes through the defensive helpers below so that an osxphotos
release that renames a field degrades to "unknown" instead of crashing.
"""

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, cast

from exo.likeness.models import FaceMetrics, PhotoCandidate, PhotoUuid


class PhotosLibraryUnavailableError(RuntimeError):
    pass


class _PhotosDatabase(Protocol):
    def photos(
        self, *, persons: list[str], images: bool, movies: bool
    ) -> list[object]: ...


class _PhotosDatabaseFactory(Protocol):
    def __call__(self, dbfile: str | None = None) -> _PhotosDatabase: ...


def _attribute(source: object, name: str) -> object:
    value: object = getattr(source, name, None)
    return value


def _optional_float(source: object, name: str) -> float | None:
    value = _attribute(source, name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _flag(source: object, name: str) -> bool:
    return _attribute(source, name) is True


def _optional_flag(source: object, name: str) -> bool | None:
    """Photos stores some flags as 0/1 integers rather than booleans."""
    value = _attribute(source, name)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


def _whole_number(source: object, name: str) -> int:
    value = _attribute(source, name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _text_items(source: object, name: str) -> list[str]:
    value = _attribute(source, name)
    if not isinstance(value, list | tuple):
        return []
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, str)]


def _object_items(source: object, name: str) -> list[object]:
    value = _attribute(source, name)
    if not isinstance(value, list | tuple):
        return []
    return list(cast(Sequence[object], value))


def _eyes_closed(face: object) -> bool:
    """Closed-eye flags; osxphotos keeps them only in the raw face record.

    Libraries from macOS Ventura onward leave these empty, so this is a no-op there.
    """
    raw_record = _attribute(face, "_info")
    if not isinstance(raw_record, dict):
        return False
    record = cast(dict[object, object], raw_record)
    return any(
        record.get(key) in (True, 1) for key in ("left_eye_closed", "right_eye_closed")
    )


def _target_face(photo: object, person_name: str) -> FaceMetrics | None:
    """The person's face in this photo; the largest one if Photos tagged several."""
    faces = [
        face
        for face in _object_items(photo, "face_info")
        if _attribute(face, "name") == person_name
    ]
    if not faces:
        return None
    face = max(faces, key=lambda item: _optional_float(item, "size") or 0.0)
    quality = _optional_float(face, "quality")
    yaw = _optional_float(face, "yaw")
    return FaceMetrics(
        # Photos stores -1 when it has not scored a face yet.
        quality=None if quality is None or quality < 0 else quality,
        relative_size=_optional_float(face, "size"),
        # osxphotos reports 0 when Photos has no head angle, which is always the
        # case for libraries from macOS Ventura onward.
        yaw_radians=None if yaw is None or yaw == 0 else yaw,
        eyes_closed=_eyes_closed(face),
        smiling=_optional_flag(face, "has_smile"),
    )


def candidate_from_photo(photo: object, person_name: str) -> PhotoCandidate | None:
    """Map one osxphotos PhotoInfo to a PhotoCandidate, or None if it lacks a uuid/date."""
    uuid = _attribute(photo, "uuid")
    captured_at = _attribute(photo, "date")
    if not isinstance(uuid, str) or not isinstance(captured_at, datetime):
        return None
    score = _attribute(photo, "score")
    return PhotoCandidate(
        uuid=PhotoUuid(uuid),
        # Normalize to an aware datetime so date-range comparisons never mix naive/aware.
        captured_at=captured_at.astimezone(),
        width=_whole_number(photo, "width"),
        height=_whole_number(photo, "height"),
        person_count=len(_text_items(photo, "persons")),
        is_favorite=_flag(photo, "favorite"),
        is_screenshot=_flag(photo, "screenshot"),
        is_hidden=_flag(photo, "hidden"),
        is_original_local=not _flag(photo, "ismissing"),
        overall_score=_optional_float(score, "overall"),
        sharpness_score=_optional_float(score, "sharply_focused_subject"),
        lighting_score=_optional_float(score, "pleasant_lighting"),
        composition_score=_optional_float(score, "well_framed_subject"),
        face=_target_face(photo, person_name),
    )


def _load_osxphotos() -> ModuleType:
    if sys.platform != "darwin":
        raise PhotosLibraryUnavailableError(
            "The Photos library can only be read on macOS."
        )
    try:
        return importlib.import_module("osxphotos")
    except ImportError as error:
        raise PhotosLibraryUnavailableError(
            "osxphotos is not installed. Run the toolkit through scripts/likeness, which adds it."
        ) from error


LIBRARY_SEARCH_PATTERNS: Final = (
    "*.photoslibrary",
    "*/*.photoslibrary",
    "*/*/*.photoslibrary",
)


def find_photo_libraries(search_roots: Sequence[Path]) -> list[Path]:
    """Photos libraries up to three folders deep under each root (e.g. each drive)."""
    found: set[Path] = set()
    for root in search_roots:
        for pattern in LIBRARY_SEARCH_PATTERNS:
            try:
                found.update(path for path in root.glob(pattern) if path.is_dir())
            except OSError:
                continue
    return sorted(found)


def default_library_search_roots() -> list[Path]:
    volumes = Path("/Volumes")
    drives = (
        sorted(path for path in volumes.iterdir() if path.is_dir())
        if volumes.is_dir()
        else []
    )
    return [Path.home() / "Pictures", *drives]


def open_library_failure_message(
    *,
    library_path: Path | None,
    underlying_error: str,
    over_ssh: bool,
    libraries_found: Sequence[Path],
) -> str:
    lines = [f"Could not open the Photos library ({underlying_error})."]
    if library_path is None:
        lines.append(
            "No --library was given, and osxphotos could not tell which library Photos "
            "uses on this Mac (Photos may never have been opened here)."
        )
        if libraries_found:
            lines.append("Photos libraries found; pass one with --library:")
            lines.extend(f'  --library "{path}"' for path in libraries_found)
        else:
            lines.append(
                "No Photos libraries were found in ~/Pictures or on mounted drives. "
                "If yours is on an external drive, check it is mounted: ls /Volumes"
            )
    if over_ssh:
        lines.append(
            "You are connected over SSH, where the Terminal app's Full Disk Access does "
            "not apply. On the Mac itself, open System Settings > General > Sharing, "
            "click (i) next to Remote Login, and turn on "
            '"Allow full disk access for remote users".'
        )
    else:
        lines.append(
            "Also check that your terminal app has Full Disk Access "
            "(System Settings > Privacy & Security > Full Disk Access)."
        )
    return "\n".join(lines)


def open_library(library_path: Path | None) -> _PhotosDatabase:
    """Open the Photos database (the system library unless a path is given)."""
    module = _load_osxphotos()
    factory = cast(_PhotosDatabaseFactory, module.PhotosDB)
    if library_path is not None:
        library_path = library_path.expanduser()
    if library_path is not None and not library_path.exists():
        raise PhotosLibraryUnavailableError(
            f"{library_path} does not exist. If it is on an external drive, check the "
            "drive is mounted (ls /Volumes)."
        )
    try:
        return factory(None if library_path is None else str(library_path))
    except Exception as error:
        raise PhotosLibraryUnavailableError(
            open_library_failure_message(
                library_path=library_path,
                underlying_error=str(error),
                over_ssh="SSH_CONNECTION" in os.environ,
                libraries_found=find_photo_libraries(default_library_search_roots()),
            )
        ) from error


def list_people(database: _PhotosDatabase) -> list[tuple[str, int]]:
    """Named people in Photos with their photo counts, most photographed first."""
    counts = _attribute(database, "persons_as_dict")
    if not isinstance(counts, dict):
        return []
    people = [
        (name, count)
        for name, count in cast(dict[object, object], counts).items()
        if isinstance(name, str) and isinstance(count, int) and name != "_UNKNOWN_"
    ]
    return sorted(people, key=lambda person: person[1], reverse=True)


def find_candidates(
    database: _PhotosDatabase, person_name: str
) -> list[PhotoCandidate]:
    photos = database.photos(persons=[person_name], images=True, movies=False)
    candidates = (candidate_from_photo(photo, person_name) for photo in photos)
    return [candidate for candidate in candidates if candidate is not None]


def _osxphotos_command() -> list[str]:
    executable = shutil.which("osxphotos")
    return [executable] if executable else [sys.executable, "-m", "osxphotos"]


def build_pull_command(
    uuid_list_file: Path, destination: Path, library_path: Path | None
) -> list[str]:
    """iCloud downloads go through Photos itself, so they only work for the library
    Photos is using; a library on another drive is copied from as-is."""
    command = [
        *_osxphotos_command(),
        "export",
        str(destination),
        "--uuid-from-file",
        str(uuid_list_file),
    ]
    if library_path is None:
        command.extend(["--download-missing", "--use-photokit"])
    command.extend(
        [
            "--convert-to-jpeg",
            "--jpeg-quality",
            "0.95",
            "--skip-live",
            "--skip-raw",
            "--filename",
            "{uuid}",
        ]
    )
    if library_path is not None:
        command.extend(["--library", str(library_path)])
    return command


def match_pulled_files(
    uuids: Sequence[PhotoUuid], exported: Sequence[Path]
) -> dict[PhotoUuid, Path]:
    """Pick one exported file per photo, preferring your own edit in Photos over the original."""
    by_stem = {path.stem: path for path in exported}
    matched: dict[PhotoUuid, Path] = {}
    for uuid in uuids:
        edited = by_stem.get(f"{uuid}_edited")
        original = by_stem.get(uuid)
        chosen = edited or original
        if chosen is not None:
            matched[uuid] = chosen
    return matched


def pull_originals(
    uuids: Sequence[PhotoUuid], destination: Path, library_path: Path | None
) -> dict[PhotoUuid, Path]:
    """Export full-resolution copies, downloading from iCloud only the ones not on disk."""
    _load_osxphotos()
    destination.mkdir(parents=True, exist_ok=True)
    uuid_list_file = destination / ".pull-uuids.txt"
    uuid_list_file.write_text("\n".join(uuids) + "\n")
    try:
        subprocess.run(
            build_pull_command(uuid_list_file, destination, library_path), check=True
        )
    finally:
        uuid_list_file.unlink(missing_ok=True)

    exported = [path for path in destination.iterdir() if path.is_file()]
    matched = match_pulled_files(uuids, exported)
    kept = set(matched.values())
    for path in exported:
        if path not in kept and path.stem.removesuffix("_edited") in uuids:
            path.unlink()  # the unedited twin of a photo you edited in Photos
    return matched
