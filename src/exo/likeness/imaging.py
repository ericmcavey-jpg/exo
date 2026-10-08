"""Pillow-backed image operations (macOS exo environments ship Pillow with mflux)."""

from pathlib import Path
from typing import final

import numpy as np
from PIL import Image, ImageOps

from exo.likeness.geometry import FaceBox, fit_long_edge, working_dimensions
from exo.likeness.texture import (
    Sharpness,
    acutance,
    blend_fine_texture,
    noise_level,
)

# Fine texture is split off below this blur radius (pixels at the input's scale).
TEXTURE_SIGMA = 2.0


def _open_upright_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


@final
class PillowImageOperations:
    def prepare_working_copy(
        self, *, source: Path, destination: Path, target_megapixels: float
    ) -> Path:
        image = _open_upright_rgb(source)
        size = working_dimensions(image.width, image.height, target_megapixels)
        image.resize(size, Image.Resampling.LANCZOS).save(destination)
        return destination

    def short_edge(self, path: Path) -> int:
        with Image.open(path) as opened:
            return min(opened.size)

    def measure_sharpness(self, path: Path, face: FaceBox | None) -> Sharpness:
        image = _open_upright_rgb(path).convert("L")
        measured_on = "whole image"
        if face is not None:
            left, top, right, bottom = face
            box = (
                max(0, round(left)),
                max(0, round(top)),
                min(image.width, round(right)),
                min(image.height, round(bottom)),
            )
            if box[2] - box[0] >= 32 and box[3] - box[1] >= 32:
                image = image.crop(box)
                measured_on = "face"
        luma = np.asarray(image, dtype=np.float32)
        return Sharpness(
            acutance=round(acutance(luma), 4),
            noise=round(noise_level(luma), 2),
            measured_on=measured_on,
        )

    def blend_texture(
        self, *, restored: Path, reference: Path, destination: Path, strength: float
    ) -> Path:
        restored_image = _open_upright_rgb(restored)
        reference_image = _open_upright_rgb(reference)
        scale = restored_image.width / reference_image.width
        if reference_image.size != restored_image.size:
            reference_image = reference_image.resize(
                restored_image.size, Image.Resampling.LANCZOS
            )
        blended = blend_fine_texture(
            np.asarray(restored_image.convert("YCbCr"), dtype=np.float32),
            np.asarray(reference_image.convert("YCbCr"), dtype=np.float32),
            strength=strength,
            sigma=TEXTURE_SIGMA * max(1.0, scale),
        )
        pixels = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        Image.fromarray(pixels, mode="YCbCr").convert("RGB").save(destination)
        return destination

    def write_training_copy(
        self, *, source: Path, destination: Path, long_edge: int
    ) -> None:
        image = _open_upright_rgb(source)
        size = fit_long_edge(image.width, image.height, long_edge)
        image.resize(size, Image.Resampling.LANCZOS).save(destination, quality=95)
