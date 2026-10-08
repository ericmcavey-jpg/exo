"""Run a recipe on one photo: restore, edit with identity checks, then upscale.

Restoration skips photos that are already sharp, and blends the input's own fine
texture and colors back into SeedVR2's output so faces do not turn waxy.

The pipeline is pure orchestration over injected effects (editor, restorer, image
operations, identity scorer), so it can be tested without models or Pillow.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, final

from exo.likeness.geometry import FaceBox
from exo.likeness.identity import IdentityPolicy, IdentityVerdict, judge_identity
from exo.likeness.recipes import EditStep, Recipe, RestoreStep
from exo.likeness.texture import (
    DEFAULT_MAXIMUM_NOISE,
    DEFAULT_MINIMUM_ACUTANCE,
    Sharpness,
)
from exo.utils.pydantic_ext import FrozenModel


class ImageEditor(Protocol):
    def edit(
        self, *, source: Path, step: EditStep, seed: int, destination: Path
    ) -> Path: ...


class ImageRestorer(Protocol):
    def restore(
        self,
        *,
        source: Path,
        destination: Path,
        short_edge: int,
        softness: float,
        seed: int,
    ) -> Path: ...


class ImageOperations(Protocol):
    def prepare_working_copy(
        self, *, source: Path, destination: Path, target_megapixels: float
    ) -> Path: ...

    def short_edge(self, path: Path) -> int: ...

    def measure_sharpness(self, path: Path, face: FaceBox | None) -> Sharpness: ...

    def blend_texture(
        self, *, restored: Path, reference: Path, destination: Path, strength: float
    ) -> Path: ...


class IdentityScorer(Protocol):
    def similarity(self, image_path: Path) -> float | None: ...

    def face_box(self, image_path: Path) -> FaceBox | None: ...


class RestorationOptions(FrozenModel):
    """How SeedVR2 passes are applied.

    `texture_strength` is how much of SeedVR2's finest detail is replaced with the
    input's own (0 keeps SeedVR2's, 1 keeps only the input's). A restore that does
    not enlarge is skipped when the face already measures at least
    `minimum_acutance` sharp with at most `maximum_noise` grain; None always restores.
    """

    texture_strength: float = 0.5
    minimum_acutance: float | None = DEFAULT_MINIMUM_ACUTANCE
    maximum_noise: float = DEFAULT_MAXIMUM_NOISE


@final
@dataclass(frozen=True)
class EnhancementEffects:
    editor: ImageEditor
    restorer: ImageRestorer
    images: ImageOperations
    identity: IdentityScorer | None


class StepRecord(FrozenModel):
    step_number: int
    kind: Literal["restore", "edit", "final"]
    attempt: int
    seed: int
    accepted: bool
    verdict: IdentityVerdict | None
    note: str
    sharpness: Sharpness | None = None


class EnhancementReport(FrozenModel):
    recipe: str
    source: str
    final_output: str
    source_similarity: float | None
    final_similarity: float | None
    steps: list[StepRecord]


def _score(effects: EnhancementEffects, path: Path) -> float | None:
    return None if effects.identity is None else effects.identity.similarity(path)


def _run_edit_step(
    *,
    step: EditStep,
    step_number: int,
    current: Path,
    output_directory: Path,
    base_seed: int,
    effects: EnhancementEffects,
    policy: IdentityPolicy,
    source_similarity: float | None,
    records: list[StepRecord],
) -> Path:
    for attempt in range(1, step.attempts + 1):
        seed = base_seed + step_number * 100 + attempt
        candidate = effects.editor.edit(
            source=current,
            step=step,
            seed=seed,
            destination=output_directory
            / f"{step_number:02d}-edit-attempt{attempt}.png",
        )
        verdict = (
            None
            if effects.identity is None
            else judge_identity(policy, source_similarity, _score(effects, candidate))
        )
        accepted = verdict is None or verdict.accepted
        records.append(
            StepRecord(
                step_number=step_number,
                kind="edit",
                attempt=attempt,
                seed=seed,
                accepted=accepted,
                verdict=verdict,
                note="no identity reference, so the edit was not checked"
                if verdict is None
                else verdict.reason,
            )
        )
        if accepted:
            return candidate
    records.append(
        StepRecord(
            step_number=step_number,
            kind="edit",
            attempt=step.attempts,
            seed=base_seed + step_number * 100 + step.attempts,
            accepted=False,
            verdict=None,
            note="every attempt drifted from your face; kept the image from before this step",
        )
    )
    return current


def _run_restore(
    *,
    kind: Literal["restore", "final"],
    step_number: int,
    current: Path,
    destination: Path,
    short_edge: int,
    softness: float,
    base_seed: int,
    effects: EnhancementEffects,
    policy: IdentityPolicy,
    options: RestorationOptions,
    records: list[StepRecord],
) -> Path:
    """Restore/upscale with SeedVR2, keeping the previous image if your face drifted.

    Judged against the image going in, so an edit's accepted change is not counted
    again here: this pass only has to leave the face as it found it.
    """
    sharpness: Sharpness | None = None
    enlarging = short_edge > effects.images.short_edge(current)
    if not enlarging and options.minimum_acutance is not None:
        face = None if effects.identity is None else effects.identity.face_box(current)
        sharpness = effects.images.measure_sharpness(current, face)
        if sharpness.is_already_sharp(options.minimum_acutance, options.maximum_noise):
            records.append(
                StepRecord(
                    step_number=step_number,
                    kind=kind,
                    attempt=1,
                    seed=base_seed,
                    accepted=True,
                    verdict=None,
                    note=(
                        f"skipped SeedVR2: the {sharpness.measured_on} is already sharp "
                        f"(acutance {sharpness.acutance:.3f} >= {options.minimum_acutance:.3f}, "
                        f"noise {sharpness.noise:.1f} <= {options.maximum_noise:.1f})"
                    ),
                    sharpness=sharpness,
                )
            )
            return current

    similarity_before = _score(effects, current)
    candidate = effects.restorer.restore(
        source=current,
        destination=destination,
        short_edge=short_edge,
        softness=softness,
        seed=base_seed,
    )
    action = f"restored with SeedVR2 at a {short_edge}px short edge"
    if options.texture_strength > 0:
        candidate = effects.images.blend_texture(
            restored=candidate,
            reference=current,
            destination=destination.with_name(f"{destination.stem}-texture.png"),
            strength=options.texture_strength,
        )
        action += (
            f", with {options.texture_strength:.0%} of the original's fine texture"
        )
    restoration_policy = policy.model_copy(
        update={"maximum_drop": policy.maximum_restoration_drop}
    )
    verdict = (
        None
        if effects.identity is None
        else judge_identity(
            restoration_policy, similarity_before, _score(effects, candidate)
        )
    )
    if verdict is None or verdict.accepted:
        records.append(
            StepRecord(
                step_number=step_number,
                kind=kind,
                attempt=1,
                seed=base_seed,
                accepted=True,
                verdict=verdict,
                note=action,
                sharpness=sharpness,
            )
        )
        return candidate
    records.append(
        StepRecord(
            step_number=step_number,
            kind=kind,
            attempt=1,
            seed=base_seed,
            accepted=False,
            verdict=verdict,
            note=f"{action}, but {verdict.reason}; kept the image from before this step",
            sharpness=sharpness,
        )
    )
    return current


def enhance_image(
    *,
    source: Path,
    recipe: Recipe,
    output_directory: Path,
    effects: EnhancementEffects,
    base_seed: int = 7,
    target_megapixels: float = 1.0,
    final_short_edge: int | None = None,
    keep_intermediates: bool = False,
    restoration: RestorationOptions | None = None,
) -> EnhancementReport:
    """Enhance `source` into `output_directory/final.png` and write `report.json` beside it."""
    output_directory.mkdir(parents=True, exist_ok=True)
    policy = recipe.identity
    options = restoration or RestorationOptions()
    working = effects.images.prepare_working_copy(
        source=source,
        destination=output_directory / "00-working.png",
        target_megapixels=target_megapixels,
    )
    source_similarity = _score(effects, working)
    records: list[StepRecord] = []
    current = working
    upscale_to = (
        recipe.final_short_edge if final_short_edge is None else final_short_edge
    )
    upscale_done = False

    for step_number, step in enumerate(recipe.steps, start=1):
        if isinstance(step, RestoreStep):
            short_edge = step.short_edge or effects.images.short_edge(current)
            # Each SeedVR2 pass changes the face a little, so a restore that ends the
            # recipe also does the final upscale in the same pass.
            if (
                step_number == len(recipe.steps)
                and upscale_to
                and upscale_to > short_edge
            ):
                short_edge = upscale_to
                upscale_done = True
            current = _run_restore(
                kind="restore",
                step_number=step_number,
                current=current,
                destination=output_directory / f"{step_number:02d}-restore.png",
                short_edge=short_edge,
                softness=step.softness,
                base_seed=base_seed,
                effects=effects,
                policy=policy,
                options=options,
                records=records,
            )
        else:
            current = _run_edit_step(
                step=step,
                step_number=step_number,
                current=current,
                output_directory=output_directory,
                base_seed=base_seed,
                effects=effects,
                policy=policy,
                source_similarity=source_similarity,
                records=records,
            )

    if (
        not upscale_done
        and upscale_to
        and upscale_to > effects.images.short_edge(current)
    ):
        current = _run_restore(
            kind="final",
            step_number=len(recipe.steps) + 1,
            current=current,
            destination=output_directory / "99-upscale.png",
            short_edge=upscale_to,
            softness=0.0,
            base_seed=base_seed,
            effects=effects,
            policy=policy,
            options=options,
            records=records,
        )
    final_path = output_directory / "final.png"
    shutil.copyfile(current, final_path)

    report = EnhancementReport(
        recipe=recipe.name,
        source=str(source),
        final_output=str(final_path),
        source_similarity=source_similarity,
        final_similarity=_score(effects, final_path),
        steps=records,
    )
    (output_directory / "report.json").write_text(report.model_dump_json(indent=2))

    if not keep_intermediates:
        for path in output_directory.iterdir():
            if path.is_file() and path.name not in {"final.png", "report.json"}:
                path.unlink()
    return report
