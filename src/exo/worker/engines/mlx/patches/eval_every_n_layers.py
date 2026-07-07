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

    for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
        mask = swa_mask if layer.use_sliding else fa_mask
        h = layer(h, mask, cache=layer_cache)
        if _EVAL_EVERY_N_LAYERS and (i + 1) % _EVAL_EVERY_N_LAYERS == 0:
            mx.eval(h)
            _write_tracker(f"call#{call_id} layer {i + 1}/{n_layers} eval OK")

    out = self.norm(h)
    _write_tracker(f"call#{call_id} DONE (all {n_layers} layers)")
    return out


def apply_eval_every_n_layers_patch() -> None:
    if _EVAL_EVERY_N_LAYERS <= 0:
        return
    logger.info(f"Patching LlamaModel.__call__: mx.eval every {_EVAL_EVERY_N_LAYERS} layer(s)")
    LlamaModel.__call__ = _patched_llama_call
