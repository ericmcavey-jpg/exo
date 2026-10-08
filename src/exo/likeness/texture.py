"""Measure how sharp a photo already is, and put its real texture back after SeedVR2.

SeedVR2 sharpens convincingly but repaints fine detail: skin turns waxy, beard hair
becomes drawn strokes, and it can shift colors such as eye color. Blending puts the
input's own fine texture and colors back on top of SeedVR2's cleaner structure.

Pure numpy on float arrays, so it can be tested without Pillow or models.
"""

import math
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from exo.utils.pydantic_ext import FrozenModel

FloatArray = npt.NDArray[np.float32]

# Calibrated on 41 iPhone photos at the 6 MP clarity working size: faces whose
# eyelashes and hairline were crisp at 100% scored 0.19 or more, visibly soft ones
# 0.09 to 0.14.
DEFAULT_MINIMUM_ACUTANCE: Final = 0.18
# Grain in 8-bit levels. Phone photos measured 0.2 to 1.4; visible grain is about 3.
DEFAULT_MAXIMUM_NOISE: Final = 3.0


class Sharpness(FrozenModel):
    acutance: float
    noise: float
    measured_on: str

    def is_already_sharp(self, minimum_acutance: float, maximum_noise: float) -> bool:
        return self.acutance >= minimum_acutance and self.noise <= maximum_noise


def gaussian_blur(values: FloatArray, sigma: float) -> FloatArray:
    """Separable Gaussian blur of a 2-D array, reflecting at the edges."""
    blurred: FloatArray = values.astype(np.float32)
    if sigma <= 0:
        return blurred
    for axis in (0, 1):
        length = cast(int, blurred.shape[axis])
        radius = min(max(1, int(3 * sigma + 0.5)), length - 1)
        if radius < 1:
            continue
        kernel = [
            math.exp(-(offset**2) / (2 * sigma * sigma))
            for offset in range(-radius, radius + 1)
        ]
        total = sum(kernel)
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded: FloatArray = np.pad(blurred, padding, mode="reflect")
        result: FloatArray = np.zeros_like(blurred)
        for index, weight in enumerate(kernel):
            window = (
                padded[index : index + length]
                if axis == 0
                else padded[:, index : index + length]
            )
            result += np.float32(weight / total) * window
        blurred = result
    return blurred


def blend_fine_texture(
    restored: FloatArray, reference: FloatArray, *, strength: float, sigma: float
) -> FloatArray:
    """Swap SeedVR2's finest detail for the reference's, and keep the reference's colors.

    Both inputs are YCbCr arrays of the same shape (height, width, 3). Luminance is
    split at `sigma`: SeedVR2's coarser structure is kept, and `strength` (0 to 1) of
    its fine band is replaced by the reference's. Chroma comes from the reference, as
    restoration only needs to change detail, never colors.
    """
    if restored.shape != reference.shape:
        raise ValueError(
            f"Cannot blend {restored.shape} with a reference of {reference.shape}"
        )
    strength = min(1.0, max(0.0, strength))
    restored_luma = restored[..., 0]
    reference_luma = reference[..., 0]
    restored_coarse = gaussian_blur(restored_luma, sigma)
    reference_fine = reference_luma - gaussian_blur(reference_luma, sigma)
    luma = (
        restored_coarse
        + (1 - strength) * (restored_luma - restored_coarse)
        + strength * reference_fine
    )
    return np.stack([luma, reference[..., 1], reference[..., 2]], axis=-1).astype(
        np.float32
    )


def acutance(luma: FloatArray) -> float:
    """Steepness of the strongest edges relative to the image's contrast.

    Blur widens edges, so their steepest slope drops. Dividing by the tonal range
    makes the score independent of exposure. Measure it on the face: a sharp
    background behind a soft face should not count.
    """
    if min(luma.shape) < 8:
        return 0.0
    horizontal = luma[1:-1, 2:] - luma[1:-1, :-2]
    vertical = luma[2:, 1:-1] - luma[:-2, 1:-1]
    slope = np.hypot(horizontal, vertical) / 2
    tonal_range = float(np.percentile(luma, 99) - np.percentile(luma, 1))
    if tonal_range <= 0:
        return 0.0
    return float(np.percentile(slope, 99.5)) / tonal_range


def noise_level(luma: FloatArray) -> float:
    """Immerkær's noise estimate (standard deviation), on the flattest half of the image."""
    if min(luma.shape) < 8:
        return 0.0
    laplacian = (
        luma[:-2, :-2]
        - 2 * luma[:-2, 1:-1]
        + luma[:-2, 2:]
        - 2 * luma[1:-1, :-2]
        + 4 * luma[1:-1, 1:-1]
        - 2 * luma[1:-1, 2:]
        + luma[2:, :-2]
        - 2 * luma[2:, 1:-1]
        + luma[2:, 2:]
    )
    gradient = np.abs(luma[1:-1, 2:] - luma[1:-1, :-2]) + np.abs(
        luma[2:, 1:-1] - luma[:-2, 1:-1]
    )
    flat = cast(npt.NDArray[np.bool_], gradient <= np.percentile(gradient, 50))
    flat_laplacian = cast(FloatArray, laplacian[flat])
    mean_deviation = float(np.mean(np.abs(flat_laplacian)))
    return math.sqrt(math.pi / 2) * mean_deviation / 6
