import math


def working_dimensions(
    width: int, height: int, target_megapixels: float, multiple: int = 16
) -> tuple[int, int]:
    """Shrink to about `target_megapixels`, keeping aspect ratio, snapped to `multiple`.

    Editing models run at the input's size, so a 48 MP phone photo is brought down to
    roughly 1 MP before editing; SeedVR2 restores the resolution at the end. Smaller
    photos are never enlarged here: SeedVR2 does the enlarging, which keeps detail.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    scale = min(1.0, math.sqrt(target_megapixels * 1_000_000 / (width * height)))

    def snap(value: float) -> int:
        return max(multiple * 16, round(value / multiple) * multiple)

    return snap(width * scale), snap(height * scale)


def fit_long_edge(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Shrink (never enlarge) so the longer side is at most `long_edge`."""
    scale = min(1.0, long_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))
