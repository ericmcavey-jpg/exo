import numpy as np
import numpy.typing as npt

from exo.likeness.geometry import working_dimensions
from exo.likeness.texture import (
    acutance,
    blend_fine_texture,
    gaussian_blur,
    noise_level,
)

FloatArray = npt.NDArray[np.float32]


def random_image(seed: int, shape: tuple[int, int, int] = (64, 48, 3)) -> FloatArray:
    return (np.random.default_rng(seed).random(shape) * 255).astype(np.float32)


def edge(width: float) -> FloatArray:
    """A vertical dark-to-light edge, `width` pixels wide."""
    columns = np.clip((np.arange(64, dtype=np.float32) - 32) / width + 0.5, 0, 1)
    return np.tile(columns * 200 + 20, (64, 1)).astype(np.float32)


def test_blur_keeps_flat_areas_and_softens_detail():
    flat = np.full((20, 30), 100.0, dtype=np.float32)
    assert np.allclose(gaussian_blur(flat, 2.0), 100.0)
    noisy = random_image(1)[..., 0]
    assert gaussian_blur(noisy, 2.0).std() < noisy.std() / 2


def test_blend_keeps_reference_colors_and_mixes_only_fine_detail():
    restored, reference = random_image(2), random_image(3)
    unchanged = blend_fine_texture(restored, reference, strength=0.0, sigma=2.0)
    assert np.allclose(unchanged[..., 0], restored[..., 0], atol=1e-3)
    assert np.array_equal(unchanged[..., 1:], reference[..., 1:])

    swapped = blend_fine_texture(restored, reference, strength=1.0, sigma=2.0)
    expected_luma = gaussian_blur(restored[..., 0], 2.0) + (
        reference[..., 0] - gaussian_blur(reference[..., 0], 2.0)
    )
    assert np.allclose(swapped[..., 0], expected_luma, atol=1e-3)


def test_acutance_drops_as_an_edge_softens():
    assert acutance(edge(1)) > 2 * acutance(edge(6))
    # Independent of contrast.
    assert abs(acutance(edge(3)) - acutance(edge(3) * 0.5)) < 1e-6


def test_noise_level_tracks_grain():
    rng = np.random.default_rng(4)
    flat = np.full((128, 128), 120.0, dtype=np.float32)
    assert noise_level(flat) < 0.01
    assert 4 < noise_level(flat + rng.normal(0, 5, flat.shape).astype(np.float32)) < 6


def test_zero_megapixels_keeps_the_native_size():
    assert working_dimensions(3024, 4032, 0) == (3024, 4032)
    assert working_dimensions(3024, 4032, 6.0) == (2128, 2832)
