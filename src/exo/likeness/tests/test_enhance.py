import shutil
from pathlib import Path
from typing import final

from exo.likeness.enhance import EnhancementEffects, EnhancementReport, enhance_image
from exo.likeness.identity import IdentityPolicy
from exo.likeness.recipes import RECIPES, EditStep, Recipe, RestoreStep


@final
class FakeImages:
    def __init__(self, short_edge: int = 768) -> None:
        self._short_edge = short_edge

    def prepare_working_copy(
        self, *, source: Path, destination: Path, target_megapixels: float
    ) -> Path:
        shutil.copyfile(source, destination)
        return destination

    def short_edge(self, path: Path) -> int:
        return self._short_edge


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
            images=FakeImages(),
            identity=identity,
        ),
        final_short_edge=final_short_edge,
        keep_intermediates=keep_intermediates,
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
