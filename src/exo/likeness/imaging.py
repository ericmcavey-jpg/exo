"""Pillow-backed image operations (macOS exo environments ship Pillow with mflux)."""

from pathlib import Path
from typing import final

from PIL import Image, ImageOps

from exo.likeness.geometry import fit_long_edge, working_dimensions


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

    def write_training_copy(
        self, *, source: Path, destination: Path, long_edge: int
    ) -> None:
        image = _open_upright_rgb(source)
        size = fit_long_edge(image.width, image.height, long_edge)
        image.resize(size, Image.Resampling.LANCZOS).save(destination, quality=95)
