"""Patch LlamaModel.__call__ to force mx.eval(h) every N layers.

Diagnostic/mitigation for the M4Max "GPU Timeout Error" (SIGABRT,
kIOGPUCommandBufferCallbackErrorTimeout) seen loading dense Llama-405B-class
models (e.g. Hermes-4-405B) in pipeline-parallel mode. Hypothesis (per
external review 2026-07-06): MLX's lazy evaluation lets many layers'
worth of ops accumulate into one giant Metal command buffer before a
sync point forces evaluation; if that buffer takes longer than Metal's
hard, unconfigurable command-buffer watchdog to execute, the OS aborts
the process. Forcing periodic eval breaks one large graph into many
smaller command buffers, each with a much shorter runtime.

Controlled by EXO_EVAL_EVERY_N_LAYERS (int, default 0 = disabled/no
change from stock behavior). Set to 1, 2, or 4 to test.
"""

import os
import time

import mlx.core as mx
from mlx_lm.models.llama import LlamaModel
from mlx_lm.models.llama import create_attention_mask

from exo.worker.runner.bootstrap import logger

_EVAL_EVERY_N_LAYERS = int(os.environ.get("EXO_EVAL_EVERY_N_LAYERS", "0"))
# DECODE cadence, separate and OFF by default (2026-08-08). The eval exists to
# break a LONG Metal command buffer — a prefill chunk over many layers, or a
# cold fault-in — so the 5s watchdog cannot fire. Decode is one token through
# the shard: the buffer is tiny and the watchdog was never the risk. Applying
# the prefill cadence to decode cost ~30 forced GPU syncs PLUS ~30 fsync'd log
# writes PER TOKEN PER NODE, which is the dominant term in DeepSeek-V3.2's
# 0.15 tok/s. Set EXO_EVAL_EVERY_N_LAYERS_DECODE only if a decode-time GPU
# timeout is ever actually observed.
_EVAL_EVERY_N_LAYERS_DECODE = int(
    os.environ.get("EXO_EVAL_EVERY_N_LAYERS_DECODE", "0"))
_TRACKER_PATH = os.environ.get("EXO_EVAL_TRACKER_PATH", "/tmp/exo_eval_tracker.log")
_call_counter = 0


def _write_tracker(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(_TRACKER_PATH, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def _patched_llama_call(self, inputs, cache=None, input_embeddings=None):
    global _call_counter
    _call_counter += 1
    call_id = _call_counter
    n_layers = len(self.layers)
    _write_tracker(f"call#{call_id} START n_layers={n_layers}")

    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = self.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(self.layers)

    fa_mask = create_attention_mask(h, cache[self.fa_idx])
    swa_mask = None
    if self.swa_idx is not None:
        swa_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.sliding_window
        )

    is_prefill = h.shape[1] > 1
    for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
        mask = swa_mask if layer.use_sliding else fa_mask
        h = layer(h, mask, cache=layer_cache)
        _eval_n = _EVAL_EVERY_N_LAYERS if is_prefill else _EVAL_EVERY_N_LAYERS_DECODE
        if _eval_n and (i + 1) % _eval_n == 0:
            mx.eval(h)
            if is_prefill:
                _write_tracker(f"call#{call_id} layer {i + 1}/{n_layers} eval OK")

    out = self.norm(h)
    _write_tracker(f"call#{call_id} DONE (all {n_layers} layers)")
    return out


def apply_eval_every_n_layers_patch() -> None:
    if _EVAL_EVERY_N_LAYERS <= 0:
        return
    logger.info(f"Patching LlamaModel.__call__: mx.eval every {_EVAL_EVERY_N_LAYERS} layer(s)")
    LlamaModel.__call__ = _patched_llama_call


# ---- 2026-07-12: DeepseekV32Model support (GLM-5.2 / DeepSeek-V3.2 arch) ----
# Faithful copy of mlx_lm.models.deepseek_v32.DeepseekV32Model.__call__ with
# mx.eval(h) every N layers inside the layer loop. The recv happens before the
# loop and sends after it, so periodic eval cannot cross a pending distributed
# op; the first eval simply forces the pending recv, which is safe.
try:
    from mlx_lm.models import deepseek_v32 as _dsv32_mod
    _DSV32_AVAILABLE = True
except ImportError:
    _DSV32_AVAILABLE = False


def _patched_dsv32_call(self, x, cache=None):
    global _call_counter
    _call_counter += 1
    call_id = _call_counter
    h = self.embed_tokens(x)

    pipeline_rank = self.pipeline_rank
    pipeline_size = self.pipeline_size

    if cache is None:
        cache = [None] * self.num_layers
    mask = _dsv32_mod.create_attention_mask(
        h, cache[0][0] if cache[0] else None, return_array=True
    )

    if pipeline_rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (pipeline_rank + 1))

    n_layers = self.num_layers
    _write_tracker(f"dsv32 call#{call_id} START n_layers={n_layers}")
    is_prefill = h.shape[1] > 1
    for i in range(self.num_layers):
        h = self.layers[self.start_idx + i](h, mask, cache[i])
        _eval_n = _EVAL_EVERY_N_LAYERS if is_prefill else _EVAL_EVERY_N_LAYERS_DECODE
        if _eval_n and (i + 1) % _eval_n == 0:
            mx.eval(h)
            if is_prefill:
                _write_tracker(f"dsv32 call#{call_id} layer {i + 1}/{n_layers} eval OK")

    if pipeline_rank != 0:
        h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
        if cache[-1] is not None:
            cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

    if pipeline_size > 1:
        h = mx.distributed.all_gather(h)[: h.shape[0]]

    _write_tracker(f"dsv32 call#{call_id} DONE (all {n_layers} layers)")
    return self.norm(h)


def apply_dsv32_eval_patch() -> None:
    if _EVAL_EVERY_N_LAYERS <= 0 or not _DSV32_AVAILABLE:
        return
    logger.info(
        f"Patching DeepseekV32Model.__call__: mx.eval every {_EVAL_EVERY_N_LAYERS} layer(s)"
    )
    _dsv32_mod.DeepseekV32Model.__call__ = _patched_dsv32_call


# --- GLM-5.2 (GlmMoeDsaModel) ------------------------------------------------
# GlmMoeDsaModel SUBCLASSES DeepseekV32Model but defines its own __call__, so
# assigning DeepseekV32Model.__call__ above is shadowed and never runs for GLM.
# Measured 2026-08-07: /tmp/exo_eval_tracker.log had no GLM entries across six
# 6-bit load attempts, i.e. all 78 layers accumulated into one command buffer
# with no eval to break it up -- the kIOGPUCommandBufferCallbackErrorTimeout we
# kept hitting. This is a faithful copy of the subclass __call__ plus the same
# periodic eval, and an optional allocator purge on the same cadence.
try:
    from mlx_lm.models import glm_moe_dsa as _glm_mod

    _GLM_AVAILABLE = True
except ImportError:
    _GLM_AVAILABLE = False

# Purge the MLX buffer cache every N layers. Prefill only: shapes grow every
# chunk so cached buffers are never reused, and freeing them keeps the residency
# footprint from ratcheting up mid-pass. Decode reuses shapes every step, so
# purging there would just churn the allocator for nothing.
_CLEAR_CACHE_EVERY_N_LAYERS = int(
    os.environ.get("EXO_CLEAR_CACHE_EVERY_N_LAYERS", "0")
)


def _patched_glm_moe_dsa_call(self, x, cache=None):
    global _call_counter
    _call_counter += 1
    call_id = _call_counter
    h = self.embed_tokens(x)

    pipeline_rank = self.pipeline_rank
    pipeline_size = self.pipeline_size

    if cache is None:
        cache = [None] * self.num_layers
    mask = _glm_mod.create_attention_mask(
        h, cache[0][0] if cache[0] else None, return_array=True
    )

    # Receive from the previous process in the pipeline
    if pipeline_rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (pipeline_rank + 1), stream=mx.cpu)

    n_layers = self.num_layers
    is_prefill = h.shape[1] > 1
    _write_tracker(
        f"glm call#{call_id} START n_layers={n_layers} rank={pipeline_rank} "
        f"seq={h.shape[1]} prefill={is_prefill}"
    )
    prev_topk_indices = None
    for i in range(n_layers):
        h, prev_topk_indices = self.layers[self.start_idx + i](
            h, mask, cache[i], prev_topk_indices
        )
        _eval_n = _EVAL_EVERY_N_LAYERS if is_prefill else _EVAL_EVERY_N_LAYERS_DECODE
        if _eval_n and (i + 1) % _eval_n == 0:
            mx.eval(h)
            if is_prefill:
                _write_tracker(f"glm call#{call_id} layer {i + 1}/{n_layers} eval OK")
        # Purge on its own cadence. It can only free buffers the eval above has
        # already released, so keep it a multiple of _EVAL_EVERY_N_LAYERS.
        if (
            is_prefill
            and _CLEAR_CACHE_EVERY_N_LAYERS
            and (i + 1) % _CLEAR_CACHE_EVERY_N_LAYERS == 0
        ):
            mx.clear_cache()

    # Send to the next process in the pipeline
    if pipeline_rank != 0:
        h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size, stream=mx.cpu)
        if cache[-1] is not None:
            cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

    if pipeline_size > 1:
        h = mx.distributed.all_gather(h, stream=mx.cpu)[: h.shape[0]]

    _write_tracker(f"glm call#{call_id} DONE (all {n_layers} layers)")
    return self.norm(h)


# --- DeepSeek-V4 (DeepseekV4Model) -------------------------------------------
# 2026-08-07: V4 has its OWN __call__ and is covered by NEITHER the Llama nor the
# DeepseekV32 patch above -- the same subclass/standalone blind spot that made the GLM
# fix inert. DeepSeek-V4-Flash survived unpatched, so this was left alone; V4-Pro does
# not. Measured: V4-Pro (791 GiB, 61 layers, 6 nodes) loaded clean and reached
# RunnerRunning on all six, then the FIRST generation died with
# kIOGPUCommandBufferCallbackErrorTimeout on m3a/m3b/mbp/m3d simultaneously -- with
# 74-244 GiB free on every node, so not memory. 61 layers' worth of ops, each faulting
# lazily-mmap'd NFS-backed weights in on first touch, is far more wall-clock inside one
# command buffer than Flash ever built.
try:
    from mlx_lm.models import deepseek_v4 as _dsv4_mod

    _DSV4_AVAILABLE = True
except ImportError:
    _DSV4_AVAILABLE = False


def _patched_dsv4_call(self, inputs, cache=None):
    global _call_counter
    _call_counter += 1
    call_id = _call_counter

    B, S = inputs.shape
    h = self.embed_tokens(inputs)
    h = mx.broadcast_to(
        h[:, :, None, :],
        (B, S, self.args.hc_mult, h.shape[-1]),
    )
    h = mx.contiguous(h)

    if cache is None:
        cache = [None] * len(self.layers)

    n_layers = len(self.layers)
    is_prefill = S > 1
    _write_tracker(f"dsv4 call#{call_id} START n_layers={n_layers} seq={S}")
    for i, layer in enumerate(self.layers):
        h = layer(h, cache[i], inputs)
        _eval_n = _EVAL_EVERY_N_LAYERS if is_prefill else _EVAL_EVERY_N_LAYERS_DECODE
        if _eval_n and (i + 1) % _eval_n == 0:
            mx.eval(h)
            if is_prefill:
                _write_tracker(f"dsv4 call#{call_id} layer {i + 1}/{n_layers} eval OK")
        if (
            is_prefill
            and _CLEAR_CACHE_EVERY_N_LAYERS
            and (i + 1) % _CLEAR_CACHE_EVERY_N_LAYERS == 0
        ):
            mx.clear_cache()

    h = self.hc_head(h)
    _write_tracker(f"dsv4 call#{call_id} DONE (all {n_layers} layers)")
    return self.norm(h)


def apply_dsv4_eval_patch() -> None:
    if _EVAL_EVERY_N_LAYERS <= 0 or not _DSV4_AVAILABLE:
        return
    logger.info(
        f"Patching DeepseekV4Model.__call__: mx.eval every {_EVAL_EVERY_N_LAYERS} layer(s)"
    )
    _dsv4_mod.DeepseekV4Model.__call__ = _patched_dsv4_call


def apply_glm_moe_dsa_eval_patch() -> None:
    if _EVAL_EVERY_N_LAYERS <= 0 or not _GLM_AVAILABLE:
        return
    logger.info(
        f"Patching GlmMoeDsaModel.__call__: mx.eval every {_EVAL_EVERY_N_LAYERS} "
        f"layer(s), clear_cache every {_CLEAR_CACHE_EVERY_N_LAYERS or 'never'}"
    )
    _glm_mod.GlmMoeDsaModel.__call__ = _patched_glm_moe_dsa_call
