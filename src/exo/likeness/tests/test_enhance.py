import shutil
from pathlib import Path
from typing import final

from exo.likeness.enhance import (
    EnhancementEffects,
    EnhancementReport,
    RestorationOptions,
    enhance_image,
)
from exo.likeness.geometry import FaceBox
from exo.likeness.identity import IdentityPolicy
from exo.likeness.recipes import RECIPES, EditStep, Recipe, RestoreStep
from exo.likeness.texture import Sharpness


@final
class FakeImages:
    """Photos measure soft unless `acutance` says otherwise; blending copies the image."""

    def __init__(self, short_edge: int = 768, acutance: float = 0.05) -> None:
        self._short_edge = short_edge
        self._acutance = acutance
        self.blends: list[float] = []
        self.measured_faces: list[FaceBox | None] = []

    def prepare_working_copy(
        self, *, source: Path, destination: Path, target_megapixels: float
    ) -> Path:
        shutil.copyfile(source, destination)
        return destination

    def short_edge(self, path: Path) -> int:
        return self._short_edge

    def measure_sharpness(self, path: Path, face: FaceBox | None) -> Sharpness:
        self.measured_faces.append(face)
        return Sharpness(acutance=self._acutance, noise=0.5, measured_on="face")

    def blend_texture(
        self, *, restored: Path, reference: Path, destination: Path, strength: float
    ) -> Path:
        self.blends.append(strength)
        shutil.copyfile(restored, destination)
        return destination


@final
class FakeRestorer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def restore(
        self,
        *,
        source: Path,
        destination: Path,
        short_edge: int,
        softness: float,
        seed: int,
    ) -> Path:
        self.calls.append(short_edge)
        destination.write_text(f"restored({source.read_text()})")
        return destination


@final
class FakeEditor:
    """Each attempt writes 'edit<attempt>'; the identity fake scores by that label."""

    def edit(
        self, *, source: Path, step: EditStep, seed: int, destination: Path
    ) -> Path:
        attempt = destination.stem.rsplit("attempt", 1)[1]
        destination.write_text(f"edit{attempt}")
        return destination


@final
class FakeIdentity:
    def __init__(self, scores: dict[str, float | None]) -> None:
        self._scores = scores

    def similarity(self, image_path: Path) -> float | None:
        return self._scores.get(image_path.read_text(), 0.80)

    def face_box(self, image_path: Path) -> FaceBox | None:
        return (10.0, 20.0, 110.0, 140.0)


def realistic_recipe(attempts: int = 2) -> Recipe:
    return Recipe(
        name="test",
        summary="",
        style="realistic",
        steps=[RestoreStep(), EditStep(instruction="retouch", attempts=attempts)],
        identity=IdentityPolicy(enforce=True),
        final_short_edge=2048,
    )


def run(
    tmp_path: Path,
    recipe: Recipe,
    identity: FakeIdentity | None,
    restorer: FakeRestorer | None = None,
    final_short_edge: int | None = None,
    keep_intermediates: bool = False,
    images: FakeImages | None = None,
    restoration: RestorationOptions | None = None,
) -> EnhancementReport:
    source = tmp_path / "source.jpg"
    source.write_text("source")
    return enhance_image(
        source=source,
        recipe=recipe,
        output_directory=tmp_path / "out",
        effects=EnhancementEffects(
            editor=FakeEditor(),
            restorer=restorer or FakeRestorer(),
            images=images or FakeImages(),
            identity=identity,
        ),
        final_short_edge=final_short_edge,
        keep_intermediates=keep_intermediates,
        restoration=restoration,
    )


def test_drifting_attempt_is_rejected_and_retried(tmp_path: Path):
    report = run(
        tmp_path, realistic_recipe(), FakeIdentity({"edit1": 0.30, "edit2": 0.78})
    )
    edits = [step for step in report.steps if step.kind == "edit"]
    assert [step.accepted for step in edits] == [False, True]
    assert (tmp_path / "out" / "final.png").read_text() == "restored(edit2)"
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "final.png",
        "report.json",
    ]


def test_edit_is_skipped_when_every_attempt_drifts(tmp_path: Path):
    report = run(
        tmp_path, realistic_recipe(), FakeIdentity({"edit1": 0.2, "edit2": 0.1})
    )
    assert report.steps[-2].note.startswith("every attempt drifted")
    assert (tmp_path / "out" / "final.png").read_text() == "restored(restored(source))"


def test_stylized_recipes_only_report_similarity(tmp_path: Path):
    report = run(tmp_path, RECIPES["anime"], FakeIdentity({"edit1": None}))
    assert all(step.accepted for step in report.steps)


def test_upscale_can_be_skipped_and_intermediates_kept(tmp_path: Path):
    restorer = FakeRestorer()
    run(
        tmp_path,
        realistic_recipe(),
        None,
        restorer,
        final_short_edge=0,
        keep_intermediates=True,
    )
    assert restorer.calls == [768]  # only the restore step, no final upscale
    assert (tmp_path / "out" / "final.png").read_text() == "edit1"
    assert (tmp_path / "out" / "00-working.png").exists()


def clarity_like_recipe() -> Recipe:
    return Recipe(
        name="clarity-test",
        summary="",
        style="realistic",
        steps=[RestoreStep()],
        identity=IdentityPolicy(enforce=True),
        final_short_edge=2048,
    )


def test_restore_that_ends_a_recipe_upscales_in_one_pass(tmp_path: Path):
    restorer = FakeRestorer()
    run(tmp_path, clarity_like_recipe(), FakeIdentity({}), restorer)
    assert restorer.calls == [2048]
    assert (tmp_path / "out" / "final.png").read_text() == "restored(source)"


def test_restoration_that_drifts_from_your_face_is_dropped(tmp_path: Path):
    # Source scores 0.80; restoring drops it to 0.72, beyond the 0.05 tolerance.
    report = run(
        tmp_path, clarity_like_recipe(), FakeIdentity({"restored(source)": 0.72})
    )
    assert not report.steps[0].accepted
    assert "kept the image from before this step" in report.steps[0].note
    assert (tmp_path / "out" / "final.png").read_text() == "source"


def test_upscale_is_judged_against_the_edit_not_the_original(tmp_path: Path):
    # The edit costs 0.08 (allowed for edits); a faithful upscale then costs 0.01.
    report = run(
        tmp_path,
        realistic_recipe(),
        FakeIdentity({"edit1": 0.72, "restored(edit1)": 0.71}),
    )
    assert all(step.accepted for step in report.steps)
    assert (tmp_path / "out" / "final.png").read_text() == "restored(edit1)"


def native_clarity_recipe() -> Recipe:
    return RECIPES["clarity"]


def test_clarity_restores_at_native_size_without_enlarging(tmp_path: Path):
    restorer = FakeRestorer()
    run(tmp_path, native_clarity_recipe(), FakeIdentity({}), restorer)
    assert restorer.calls == [768]


def test_sharp_photo_is_not_restored(tmp_path: Path):
    restorer = FakeRestorer()
    images = FakeImages(acutance=0.30)
    report = run(
        tmp_path, native_clarity_recipe(), FakeIdentity({}), restorer, images=images
    )
    assert restorer.calls == []
    assert report.steps[0].note.startswith("skipped SeedVR2: the face is already sharp")
    assert report.steps[0].sharpness is not None
    assert images.measured_faces == [(10.0, 20.0, 110.0, 140.0)]
    assert (tmp_path / "out" / "final.png").read_text() == "source"


def test_sharp_photo_is_still_restored_when_skipping_is_off(tmp_path: Path):
    restorer = FakeRestorer()
    run(
        tmp_path,
        native_clarity_recipe(),
        FakeIdentity({}),
        restorer,
        images=FakeImages(acutance=0.30),
        restoration=RestorationOptions(minimum_acutance=None),
    )
    assert restorer.calls == [768]


def test_enlarging_is_never_skipped_for_sharpness(tmp_path: Path):
    restorer = FakeRestorer()
    run(
        tmp_path,
        clarity_like_recipe(),
        FakeIdentity({}),
        restorer,
        images=FakeImages(acutance=0.30),
    )
    assert restorer.calls == [2048]


def test_restored_output_gets_the_original_texture_blended_back(tmp_path: Path):
    images = FakeImages()
    report = run(
        tmp_path,
        native_clarity_recipe(),
        FakeIdentity({}),
        images=images,
        restoration=RestorationOptions(texture_strength=0.4),
    )
    assert images.blends == [0.4]
    assert "40% of the original's fine texture" in report.steps[0].note


def test_texture_blend_can_be_turned_off(tmp_path: Path):
    images = FakeImages()
    run(
        tmp_path,
        native_clarity_recipe(),
        FakeIdentity({}),
        images=images,
        restoration=RestorationOptions(texture_strength=0),
    )
    assert images.blends == []
