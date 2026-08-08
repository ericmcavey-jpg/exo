from exo.worker.engines.mlx.patches.eval_every_n_layers import (
    apply_eval_every_n_layers_patch,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    # EXO_MLX_CACHE_LIMIT_MB caps (or with 0, forbids) MLX's internal buffer cache.
    # MLX keeps freed Metal buffers around for reuse; on a fleet running at a ~3% margin
    # that is memory we cannot spare. 0 returns every buffer to the system immediately.
    # Trade-off: more allocator traffic, so only set it when memory is the binding
    # constraint. Unset = stock MLX behaviour.
    import os as _os

    _cl = _os.environ.get("EXO_MLX_CACHE_LIMIT_MB")
    if _cl is not None:
        try:
            import mlx.core as _mx

            _bytes = max(0, int(_cl)) * 1024 * 1024
            _mx.set_cache_limit(_bytes)
            print(f"[exo] MLX buffer cache limit set to {_cl} MB")
        except Exception as _e:  # never let a tuning knob break startup
            print(f"[exo] could not set MLX cache limit: {_e}")
    patch_yarn_rope()
    apply_batch_gen_patch()
    apply_eval_every_n_layers_patch()
    from exo.worker.engines.mlx.patches.eval_every_n_layers import apply_dsv32_eval_patch
    apply_dsv32_eval_patch()
    from exo.worker.engines.mlx.patches.eval_every_n_layers import (
        apply_glm_moe_dsa_eval_patch,
    )
    apply_glm_moe_dsa_eval_patch()
    from exo.worker.engines.mlx.patches.eval_every_n_layers import (
        apply_dsv4_eval_patch,
    )
    apply_dsv4_eval_patch()
