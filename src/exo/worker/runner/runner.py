import queue
import threading
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

from anyio import ClosedResourceError, EndOfStream

from exo.shared.constants import ENABLE_DISAGGREGATION
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    GenerationTask,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.server import (
    PrefillRequest,
    PrefillServer,
)
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger

PREFILL_PICKUP_TIMEOUT_SECONDS = 3
PREFILL_FINISH_TIMEOUT_SECONDS = 300


@dataclass
class PrefillTask:
    request: PrefillRequest
    wfile: BinaryIO
    started: threading.Event
    done: threading.Event


class _TaskStreamClosed:
    pass


WorkItem = Task | PrefillTask | _TaskStreamClosed


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        builder: Builder,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | Engine = builder

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            GenerationTask,
        ] = {}

        self._prefill_server: PrefillServer | None = None
        self._prefill_server_port: int | None = None
        self._work_queue: queue.Queue[WorkItem] = queue.Queue()
        self._task_reader_thread: threading.Thread | None = None

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _start_prefill_server(self) -> int | None:
        if not ENABLE_DISAGGREGATION:
            return None
        if self.device_rank != 0:
            return None
        if self._prefill_server_port is not None:
            return self._prefill_server_port

        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:
            req = PrefillTask(
                request=request,
                wfile=wfile,
                started=threading.Event(),
                done=threading.Event(),
            )
            self._work_queue.put(req)
            if not req.started.wait(timeout=PREFILL_PICKUP_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} not picked up within "
                    f"{PREFILL_PICKUP_TIMEOUT_SECONDS}s — runner busy"
                )
                return False
            if not req.done.wait(timeout=PREFILL_FINISH_TIMEOUT_SECONDS):
                logger.warning(
                    f"Prefill request {request.request_id} did not finish within "
                    f"{PREFILL_FINISH_TIMEOUT_SECONDS}s"
                )
            return True

        port = random_ephemeral_port()
        self._prefill_server = PrefillServer(resolve=resolve, host="0.0.0.0", port=port)
        self._prefill_server_port = port
        return self._prefill_server_port

    def _start_task_reader(self) -> None:
        if self._task_reader_thread is not None:
            return

        def loop() -> None:
            try:
                with self.task_receiver:
                    for task in self.task_receiver:
                        self._work_queue.put(task)
            except (EndOfStream, ClosedResourceError):
                pass
            finally:
                self._work_queue.put(_TaskStreamClosed())

        self._task_reader_thread = threading.Thread(target=loop, name="task-reader")
        self._task_reader_thread.start()

    def _serve_prefill(self, req: PrefillTask) -> None:
        req.started.set()
        try:
            assert isinstance(self.generator, Engine)
            self.generator.serve_prefill(req.request, req.wfile)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to serve prefill request {req.request.request_id}"
            )
        finally:
            req.done.set()

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        self._start_task_reader()
        try:
            while True:
                item = self._work_queue.get()
                if isinstance(item, _TaskStreamClosed):
                    break
                if isinstance(item, PrefillTask):
                    self._serve_prefill(item)
                    continue
                if item.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(item.task_id)
                self.handle_first_task(item)
                if isinstance(self.current_status, RunnerShutdown):
                    break
        finally:
            if self._prefill_server is not None:
                self._prefill_server.stop()
                self._prefill_server = None
            self.task_receiver.close()
            if self._task_reader_thread is not None:
                self._task_reader_thread.join(timeout=5)
                self._task_reader_thread = None

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.connect(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                isinstance(self.current_status, (RunnerConnected, RunnerIdle))
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                for load_progress in self.generator.load(self.bound_instance):
                    self.update_status(
                        RunnerLoading(
                            layers_loaded=load_progress.layers_loaded,
                            total_layers=load_progress.total,
                        )
                    )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, Engine)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                # --- cpu-stream residency warmup (gradual wiring) ---
                # 2026-08-07: this pass MUST honor EXO_SKIP_WARMUP. It sat ABOVE the
                # EXO_SKIP_WARMUP check below, so the flag only ever guarded
                # generator.warmup() while THIS loop -- the expensive half -- ran
                # unconditionally. It touches every parameter, converting lazily-mmap'd
                # NFS-backed weights into hard residency. On DeepSeek-V4-Pro (791 GiB,
                # 18 layers = 233.5 GiB on m3c) that walk drove a 256 GiB node past its
                # limit and jetsam SIGKILLed the runner ~2 min into StartWarmup, taking
                # the whole 6-node ring down with it -- after all six had already reached
                # RunnerReady. The try/except made it look safe; a SIGKILL is not
                # catchable, so "non-fatal" was never true.
                # Lazy paging on the first real request is the proven path for models
                # this size (GLM-5.2 604 GiB: 9s and 23s first probes, both clean).
                # EXO_CPU_RESIDENCY=1 forces the residency pass back ON while leaving
                # EXO_SKIP_WARMUP=1 in place, so generator.warmup() -- which is separately
                # fatal on NFS-backed models this size -- stays skipped. These are two
                # different things and conflating them under one flag is what made the
                # pass run unconditionally in the first place.
                # WHY TRY IT AGAIN (2026-08-07): it is the only mechanism that removes
                # weight fault-in from the Metal command buffer ENTIRELY (CPU stream, no
                # watchdog) rather than subdividing it. Chunking hit its asymptote at
                # chunk=16 (m3c tracker lines 34 -> 122 -> 183, i.e. 3.6x then 1.5x).
                # It OOM'd before under conditions since fixed: 18 layers + lm_head on the
                # NFS-serving node, no purge loops, unstripped baseline. 791.3 GiB against
                # 819.4 GiB effective says it fits -- barely.
                _force_residency = os.environ.get("EXO_CPU_RESIDENCY") == "1"
                _skip_warmup = (
                    os.environ.get("EXO_SKIP_WARMUP") == "1" and not _force_residency
                )
                if _force_residency:
                    logger.info("EXO_CPU_RESIDENCY=1: running cpu residency pass")
                if _skip_warmup:
                    logger.info(
                        "skipping cpu residency pass because EXO_SKIP_WARMUP=1 "
                        "(weights page in on first request)"
                    )
                try:
                    import time as _t
                    import mlx.core as _mx
                    from mlx.utils import tree_flatten as _tf
                    _rm = None if _skip_warmup else getattr(self.generator, "model", None)
                    if _rm is not None:
                        _t0 = _t.time()
                        _params = _tf(_rm.parameters())
                        _n = len(_params)
                        _done = 0
                        for _, _p in _params:
                            # CPU-stream touch: faults pages in with NO Metal watchdog.
                            _mx.eval(_mx.sum(_p.view(_mx.uint8), stream=_mx.cpu))
                            _done += 1
                            if _done % 50 == 0:
                                logger.info(
                                    f"cpu residency: {_done}/{_n} params touched "
                                    f"({_t.time() - _t0:.0f}s)"
                                )
                        logger.info(
                            f"cpu residency complete: {_n} params resident in "
                            f"{_t.time() - _t0:.0f}s"
                        )
                except Exception as _re:
                    logger.warning(f"cpu residency warmup failed (non-fatal): {_re}")

                if os.environ.get("EXO_SKIP_WARMUP") == "1":
                    logger.info("skipping runner warmup because EXO_SKIP_WARMUP=1")
                else:
                    self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self._start_prefill_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(
                    RunnerReady(prefill_server_port=self._prefill_server_port)
                )
                logger.info("runner ready")

            case TextGeneration() | ImageEdits() | ImageGeneration() if isinstance(
                self.current_status, RunnerReady
            ):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        self.generator.close()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_generation(self, task: GenerationTask):
        assert isinstance(self.generator, Engine)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def handle_generation_tasks(self, starting_task: GenerationTask):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, Engine)

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_generation(starting_task)

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case CancelledResponse():
                        finished.append(task_id)
                    case FinishedResponse():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case other:
                        self.send_chunk(other, self.active_tasks[task_id].command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                continue
            if isinstance(item, _TaskStreamClosed):
                return ExitCode.Shutdown
            if isinstance(item, PrefillTask):
                self._serve_prefill(item)
                continue
            if item.task_id in self.seen:
                logger.warning("repeat task - potential error")
                continue
            self.seen.add(item.task_id)
            match item:
                case TextGeneration() | ImageGeneration() | ImageEdits():
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                case Shutdown():
                    self.shutdown(item)
                    return ExitCode.Shutdown
                case _:
                    raise ValueError(
                        f"Received {item.__class__.__name__} outside of state machine in {self.current_status=}"
                    )

        # Release MLX's GPU buffer pool from the just-finished generation(s) so
        # the next request's prefill starts with maximum free memory. Without
        # this, pooled buffers accumulate across requests on a loaded instance
        # and a memory-tight node OOMs on the 2nd/3rd task of a multi-task run
        # ([METAL] command-buffer Insufficient Memory -> SIGABRT). clear_cache
        # only frees UNUSED pooled buffers; model weights and any live KV cache
        # are untouched. (2026-07-09)
        import mlx.core as mx

        mx.clear_cache()
        self.update_status(RunnerReady(prefill_server_port=self._prefill_server_port))
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_chunk(
        self,
        chunk: Chunk,
        command_id: CommandId,
    ):
        assert isinstance(self.generator, Engine)
        self.event_sender.send(ChunkGenerated(command_id=command_id, chunk=chunk))
