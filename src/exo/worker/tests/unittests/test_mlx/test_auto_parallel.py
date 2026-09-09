import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    patch_pipeline_model,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return x * 2

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_pipeline_model(inner_model, group)

        x = mx.ones((1, 4))
        result = model(x)
        mx.eval(result)
        success = result.shape == x.shape
        result_queue.put((rank, success, result))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


def test_single_wrapper_delegates_attributes() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    assert wrapped.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert wrapped.use_sliding is True  # type: ignore[attr-defined]


def test_composed_wrappers_delegate_attributes() -> None:
    mock = MockLayer()
    group = mx.distributed.init()

    first = PipelineFirstLayer(mock, r=0, group=group)
    composed = PipelineLastLayer(first, r=0, s=1, group=group)

    assert composed.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert composed.use_sliding is True  # type: ignore[attr-defined]


def test_missing_attribute_raises() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    with pytest.raises(AttributeError):
        _ = wrapped.nonexistent_attr  # type: ignore[attr-defined]


class _FullIndexerLayer(mlx_nn.Module):
    def __init__(self, index_topk: int) -> None:
        super().__init__()
        self.index_topk = index_topk

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        del mask, cache, prev_topk_indices
        if x.shape[1] <= self.index_topk:
            return x + 1, None
        topk = mx.broadcast_to(
            mx.arange(self.index_topk, dtype=mx.uint32)[None, None, None, :],
            (x.shape[0], 1, x.shape[1], self.index_topk),
        )
        return x + 1, topk


class _SharedIndexerLayer(mlx_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_topk: mx.array | None = None

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        del mask, cache
        self.received_topk = prev_topk_indices
        return x + 1, prev_topk_indices


def _patch_distributed_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[mx.array], list[tuple[int, ...]]]:
    mailbox: list[mx.array] = []
    send_shapes: list[tuple[int, ...]] = []

    def fake_send(value: mx.array, *_args: object, **_kwargs: object) -> mx.array:
        mx.eval(value)
        mailbox.append(value)
        send_shapes.append(tuple(value.shape))
        return value

    def fake_recv_like(
        _template: mx.array, *_args: object, **_kwargs: object
    ) -> mx.array:
        return mailbox.pop(0)

    monkeypatch.setattr(mx.distributed, "send", fake_send)
    monkeypatch.setattr(mx.distributed, "recv_like", fake_recv_like)
    def fake_async_eval(*_values: mx.array) -> None:
        return None

    monkeypatch.setattr(mx, "async_eval", fake_async_eval)
    return mailbox, send_shapes


def _run_shared_topk_pipeline_device(
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        x = mx.zeros((1, 5, 3))
        if rank == 0:
            sender = PipelineLastLayer(
                _FullIndexerLayer(index_topk=4),  # pyright: ignore[reportArgumentType]
                r=0,
                s=2,
                group=group,
                send_shared_topk=True,
            )
            sender.is_prefill = True
            sender.queue_sends = True
            sender(x)
            flush_prefill_sends()
            result_queue.put((rank, True, None))  # pyright: ignore[reportAny]
        else:
            receiver_layer = _SharedIndexerLayer()
            receiver = PipelineFirstLayer(
                receiver_layer,  # pyright: ignore[reportArgumentType]
                r=1,
                group=group,
                shared_topk_width=4,
            )
            receiver.is_prefill = True
            receiver(x)
            assert receiver_layer.received_topk is not None
            mx.eval(receiver_layer.received_topk)
            result_queue.put(  # pyright: ignore[reportAny]
                (
                    rank,
                    True,
                    (
                        str(receiver_layer.received_topk.dtype),
                        tuple(receiver_layer.received_topk.shape),
                    ),
                )
            )
    except Exception as error:
        result_queue.put((rank, False, str(error)))  # pyright: ignore[reportAny]


def test_shared_indexer_boundary_short_context_sends_activation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, send_shapes = _patch_distributed_mailbox(monkeypatch)
    group = object()  # The mailbox replacements do not inspect the group.
    sender = PipelineLastLayer(
        _FullIndexerLayer(index_topk=4),  # pyright: ignore[reportArgumentType]
        r=0,
        s=2,
        group=group,  # type: ignore[arg-type]
        send_shared_topk=True,
    )
    receiver_layer = _SharedIndexerLayer()
    receiver = PipelineFirstLayer(
        receiver_layer,  # pyright: ignore[reportArgumentType]
        r=1,
        group=group,  # type: ignore[arg-type]
        shared_topk_width=4,
    )
    sender.is_prefill = receiver.is_prefill = True

    x = mx.zeros((1, 4, 3))
    sender(x)
    receiver(mx.zeros_like(x))

    assert send_shapes == [(1, 4, 3)]
    assert receiver_layer.received_topk is None
    assert not mailbox


def test_shared_indexer_boundary_queued_prefill_orders_activation_before_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_prefill_sends()
    mailbox, send_shapes = _patch_distributed_mailbox(monkeypatch)
    group = object()  # The mailbox replacements do not inspect the group.
    sender = PipelineLastLayer(
        _FullIndexerLayer(index_topk=4),  # pyright: ignore[reportArgumentType]
        r=0,
        s=2,
        group=group,  # type: ignore[arg-type]
        send_shared_topk=True,
    )
    receiver_layer = _SharedIndexerLayer()
    receiver = PipelineFirstLayer(
        receiver_layer,  # pyright: ignore[reportArgumentType]
        r=1,
        group=group,  # type: ignore[arg-type]
        shared_topk_width=4,
    )
    sender.is_prefill = receiver.is_prefill = True
    sender.queue_sends = True

    x = mx.zeros((1, 5, 3))
    sender(x)
    assert not mailbox
    flush_prefill_sends()
    receiver(mx.zeros_like(x))

    assert send_shapes == [(1, 5, 3), (1, 1, 5, 4)]
    assert receiver_layer.received_topk is not None
    assert tuple(receiver_layer.received_topk.shape) == (1, 1, 5, 4)
    assert not mailbox


def test_shared_indexer_boundary_transfers_uint32_over_local_ring() -> None:
    ctx = mp.get_context("spawn")
    hosts = ["127.0.0.1:29600", "127.0.0.1:29601"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(hosts, file)
        hostfile_path = file.name

    try:
        result_queue: Any = ctx.Queue()
        processes = [
            ctx.Process(
                target=_run_shared_topk_pipeline_device,
                args=(rank, hostfile_path, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)

        results: dict[int, tuple[bool, object]] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            results[rank] = (success, value)

        assert results == {
            0: (True, None),
            1: (True, ("mlx.core.uint32", (1, 1, 5, 4))),
        }
    finally:
        os.unlink(hostfile_path)


def test_composed_call_works() -> None:
    ctx = mp.get_context("spawn")

    world_size = 2
    base_port = 29500

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    try:
        result_queue: Any = ctx.Queue()

        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_pipeline_device,
                args=(rank, world_size, hostfile_path, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=10)  # pyright: ignore[reportAny]

        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            if success:
                results[rank] = value
            else:
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}"
        )

        for rank in range(world_size):
            assert rank in results, (
                f"Device {rank} failed: {errors.get(rank, 'unknown')}"
            )
            result_array = results[rank]
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)
