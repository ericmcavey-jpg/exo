# pyright: reportPrivateUsage=false

import pytest

from exo.worker.engines.mlx.generator.generate import _pipeline_prefill_chunk_sizes


def _qk_products(sizes: list[int], prefix_hit_length: int) -> list[int]:
    processed = 0
    products: list[int] = []
    for size in sizes:
        products.append(size * (prefix_hit_length + processed + size))
        processed += size
    return products


def test_glm_5315_cached_suffix_respects_qk_budget() -> None:
    """A 240,835-token hit must constrain its 15,081-token suffix immediately."""
    prefix_hit_length = 240_835
    maximum_qk = 64_000_000

    sizes = _pipeline_prefill_chunk_sizes(
        total=15_081,
        max_step=2_048,
        min_step=128,
        max_qk=maximum_qk,
        prefix_hit_length=prefix_hit_length,
        step_tiers=(2_048, 1_024, 512, 256, 128),
    )

    # The final token is handled by pipeline_parallel_prefill's post-loop.
    assert sum(sizes) == 15_080
    assert sizes[0] == 256
    assert max(sizes) <= 256
    assert 128 in sizes
    assert max(_qk_products(sizes, prefix_hit_length)) <= maximum_qk


def test_no_prefix_preserves_adaptive_schedule_behavior() -> None:
    expected = ([1_024] * 14) + [744]

    assert _pipeline_prefill_chunk_sizes(
        total=15_081,
        max_step=1_024,
        min_step=128,
        max_qk=64_000_000,
    ) == expected


def test_configured_tiers_choose_2048_for_safe_empty_context() -> None:
    sizes = _pipeline_prefill_chunk_sizes(
        total=15_081,
        max_step=2_048,
        min_step=128,
        max_qk=64_000_000,
        step_tiers=(2_048, 1_024, 512, 256, 128),
    )

    assert sizes[0] == 2_048
    assert max(_qk_products(sizes, prefix_hit_length=0)) <= 64_000_000


def test_disabled_qk_budget_preserves_fixed_chunks_with_cached_prefix() -> None:
    assert _pipeline_prefill_chunk_sizes(
        total=2_501,
        max_step=1_024,
        min_step=128,
        max_qk=0,
        prefix_hit_length=240_835,
    ) == [1_024, 1_024, 452]


def test_configured_tiers_fail_closed_when_128_exceeds_budget() -> None:
    with pytest.raises(ValueError, match="no configured adaptive prefill tier"):
        _pipeline_prefill_chunk_sizes(
            total=1_000,
            max_step=2_048,
            min_step=128,
            max_qk=64_000_000,
            prefix_hit_length=500_000,
            step_tiers=(2_048, 1_024, 512, 256, 128),
        )
