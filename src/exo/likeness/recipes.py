"""Ready-made enhancement recipes: realistic retouching and stylized looks.

Realistic recipes run every edit through the identity gate (see identity.py) and
retry or skip edits that drift from your face. Stylized recipes are expected to
change your face, so the gate only reports similarity there.
"""

from typing import Annotated, Final, Literal

from pydantic import Field

from exo.likeness.identity import IdentityPolicy
from exo.utils.pydantic_ext import FrozenModel

IDENTITY_LOCK: Final = (
    " Keep exactly the same person: identical face shape, facial features, eye shape and "
    "color, nose, mouth, skin tone, facial hair, hairline, age and body shape. "
    "Photorealistic, natural skin texture, no plastic or airbrushed look."
)

STYLE_RECOGNIZABILITY: Final = (
    " Keep the person clearly recognizable: same face shape, eye shape, hairstyle and hair "
    "color, facial hair, glasses, skin tone, expression and outfit colors."
)


class RestoreStep(FrozenModel):
    """SeedVR2 restoration: sharpens and de-noises without inventing a new face."""

    kind: Literal["restore"] = "restore"
    short_edge: int | None = None
    softness: float = 0.0


class EditStep(FrozenModel):
    """One instruction sent to the exo-hosted image-editing model."""

    kind: Literal["edit"] = "edit"
    instruction: str
    input_fidelity: Literal["low", "high"] = "high"
    quality: Literal["low", "medium", "high"] = "high"
    guidance: float | None = None
    attempts: int = Field(default=2, ge=1, le=5)


RecipeStep = Annotated[RestoreStep | EditStep, Field(discriminator="kind")]


class Recipe(FrozenModel):
    name: str
    summary: str
    style: Literal["realistic", "stylized"]
    steps: list[RecipeStep]
    identity: IdentityPolicy
    final_short_edge: int | None = 2048


REALISTIC_IDENTITY: Final = IdentityPolicy(enforce=True)
STYLIZED_IDENTITY: Final = IdentityPolicy(enforce=False, require_face=False)


def _realistic(name: str, summary: str, *steps: RestoreStep | EditStep) -> Recipe:
    return Recipe(
        name=name,
        summary=summary,
        style="realistic",
        steps=list(steps),
        identity=REALISTIC_IDENTITY,
    )


def _stylized(name: str, summary: str, instruction: str) -> Recipe:
    return Recipe(
        name=name,
        summary=summary,
        style="stylized",
        steps=[
            EditStep(
                instruction=instruction + STYLE_RECOGNIZABILITY,
                input_fidelity="low",
                quality="medium",
                attempts=1,
            )
        ],
        identity=STYLIZED_IDENTITY,
    )


RECIPES: Final[dict[str, Recipe]] = {
    recipe.name: recipe
    for recipe in (
        _realistic(
            "clarity",
            "Sharpen, de-noise and upscale only. No retouching, so no identity risk.",
            RestoreStep(),
        ),
        _realistic(
            "polish",
            "Clarity plus a professional retouch: flattering light, true skin tones, light cleanup.",
            RestoreStep(),
            EditStep(
                instruction=(
                    "Retouch this photo like a professional portrait retoucher. Even out the "
                    "lighting so the face is softly and flatteringly lit, correct the white "
                    "balance to natural skin tones, gently reduce under-eye shadows and "
                    "temporary blemishes, and tidy stray hairs. Keep everything else in the "
                    "photo unchanged." + IDENTITY_LOCK
                )
            ),
        ),
        _realistic(
            "groomed",
            "Well-rested and well-groomed: clearer eyes, neat hair and facial hair, even complexion.",
            RestoreStep(),
            EditStep(
                instruction=(
                    "Make the person look well rested and well groomed: brighter, clearer "
                    "eyes, subtly whiter teeth if they are visible, neatly styled hair, tidy "
                    "facial hair edges, an even complexion and upright posture. Keep the "
                    "clothing, background and framing unchanged." + IDENTITY_LOCK
                )
            ),
        ),
        _realistic(
            "headshot",
            "Studio headshot: soft key light, clean grey backdrop, sharp eyes.",
            RestoreStep(),
            EditStep(
                instruction=(
                    "Turn this into a professional studio headshot: soft key light from the "
                    "front left with gentle fill, a clean, softly lit neutral grey backdrop, "
                    "sharp focus on the eyes and a subtle shallow depth of field. Keep the "
                    "same clothing, neatly pressed." + IDENTITY_LOCK
                )
            ),
        ),
        _realistic(
            "golden-hour",
            "Relight as warm golden-hour sun with a soft rim light; background softly blurred.",
            RestoreStep(),
            EditStep(
                instruction=(
                    "Relight the scene with warm golden-hour sunlight from the side and a soft "
                    "rim light on the hair and shoulders. Use a warm, natural color grade and "
                    "keep the same location, gently out of focus." + IDENTITY_LOCK
                )
            ),
        ),
        _stylized(
            "anime",
            "Modern anime illustration with clean line art and cel shading.",
            "Redraw this person as a high-quality modern anime character illustration: clean "
            "line art, cel shading, vivid but soft colors, detailed expressive eyes and a "
            "simple painted background.",
        ),
        _stylized(
            "3d-animated",
            "Character from a modern 3D animated feature film.",
            "Transform this person into a character from a modern 3D animated feature film: "
            "slightly stylized proportions, large expressive eyes, soft subsurface-scattered "
            "skin, smooth stylized hair, cinematic soft lighting and a gently blurred, "
            "colorful background.",
        ),
        _stylized(
            "comic",
            "Bold comic-book illustration with ink outlines and halftone shading.",
            "Redraw this person as a bold comic-book illustration: confident ink outlines, "
            "halftone shading, flat saturated colors and dramatic lighting.",
        ),
        _stylized(
            "watercolor",
            "Loose, elegant watercolor portrait on textured paper.",
            "Paint this person as a loose, elegant watercolor portrait on textured paper with "
            "soft washes, gentle color bleeds and visible brush strokes.",
        ),
    )
}
