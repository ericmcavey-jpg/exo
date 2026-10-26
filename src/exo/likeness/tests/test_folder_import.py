from datetime import datetime, timezone
from pathlib import Path

from exo.likeness.folder_import import FoundFace, import_folder, photo_uuid_for
from exo.likeness.models import PhotoUuid
from exo.likeness.workspace import TaskWorkspace

CAPTURED = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


class FakeInspector:
    def __init__(self, faces: dict[str, FoundFace | None]) -> None:
        self._faces = faces

    def find_largest_face(self, image_path: Path) -> FoundFace | None:
        return self._faces.get(image_path.name)

    def capture_time(self, image_path: Path) -> datetime | None:
        return CAPTURED


def face(size: float, yaw: float | None = 0.0) -> FoundFace:
    return FoundFace(
        box=(0.0, 0.0, size, size), yaw_radians=yaw, image_width=1000, image_height=2000
    )


def write_folder(folder: Path, files: dict[str, bytes]) -> None:
    folder.mkdir()
    for name, content in files.items():
        (folder / name).write_bytes(content)


def test_import_selects_photos_with_faces_and_copies_them(tmp_path: Path) -> None:
    folder = tmp_path / "sent"
    write_folder(
        folder,
        {"a.JPG": b"one", "b.jpg": b"two", "c.jpg": b"three", "notes.txt": b"x"},
    )
    workspace = TaskWorkspace(tmp_path / "home", "lisa")
    inspector = FakeInspector(
        {"a.JPG": face(400), "b.jpg": face(150, yaw=0.6), "c.jpg": None}
    )

    result = import_folder(
        workspace=workspace, folder=folder, person_name="Lisa", inspector=inspector
    )

    assert result.without_face == ["c.jpg"]
    manifest = workspace.load_manifest()
    assert [photo.framing for photo in manifest.selected] == ["close_up", "medium"]
    assert manifest.selected[1].head_angle == "three_quarter_right"
    assert manifest.selected[0].captured_at == CAPTURED
    assert manifest.settings.person_name == "Lisa"
    copied = workspace.pulled_paths()
    assert sorted(path.read_bytes() for path in copied) == [b"one", b"two"]
    assert all(path.suffix == ".jpg" for path in copied)


def test_identical_files_and_reimports_are_added_once(tmp_path: Path) -> None:
    folder = tmp_path / "sent"
    write_folder(folder, {"a.jpg": b"same", "a 2.jpg": b"same"})
    workspace = TaskWorkspace(tmp_path / "home", "lisa")
    inspector = FakeInspector({"a.jpg": face(400), "a 2.jpg": face(400)})

    first = import_folder(
        workspace=workspace, folder=folder, person_name="Lisa", inspector=inspector
    )
    second = import_folder(
        workspace=workspace, folder=folder, person_name="Lisa", inspector=inspector
    )

    assert len(first.imported) == 1
    assert first.already_in_task == ["a.jpg"]
    assert second.imported == []
    assert len(workspace.load_manifest().selected) == 1


def test_excluded_photos_are_not_imported_again(tmp_path: Path) -> None:
    folder = tmp_path / "sent"
    write_folder(folder, {"a.jpg": b"keep", "b.jpg": b"drop"})
    workspace = TaskWorkspace(tmp_path / "home", "lisa")
    inspector = FakeInspector({"a.jpg": face(400), "b.jpg": face(400)})
    import_folder(
        workspace=workspace, folder=folder, person_name="Lisa", inspector=inspector
    )
    workspace.exclude([str(photo_uuid_for(folder / "b.jpg"))], "blurry")

    result = import_folder(
        workspace=workspace, folder=folder, person_name="Lisa", inspector=inspector
    )

    assert result.excluded == ["b.jpg"]
    assert [photo.uuid for photo in workspace.load_manifest().selected] == [
        photo_uuid_for(folder / "a.jpg")
    ]


def test_photo_uuid_is_stable_and_uppercase(tmp_path: Path) -> None:
    path = tmp_path / "x.jpg"
    path.write_bytes(b"content")
    assert photo_uuid_for(path) == photo_uuid_for(path)
    assert photo_uuid_for(path) == PhotoUuid(photo_uuid_for(path).upper())
