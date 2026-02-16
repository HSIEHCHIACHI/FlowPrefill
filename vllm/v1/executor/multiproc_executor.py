# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread, Lock
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

import itertools

logger = init_logger(__name__)


class MultiprocExecutor(Executor):
    """
    A multiprocessing executor that supports:
    1. Pipelined submission (Executor doesn't block on send).
    2. Sequence-ID based response matching (Handles out-of-order responses).
    3. Parallel execution inside Workers (via ThreadPool for virtual_runners).
    """

    supports_pp: bool = True

    # Robust single-writer RPC sender
    _RPC_SEND_SENTINEL = object()

    def _start_rpc_sender(self) -> None:
        self._rpc_send_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._rpc_sender_shutdown = threading.Event()
        self._rpc_sender_thread = Thread(
            target=self._rpc_sender_loop,
            daemon=True,
            name="RpcBroadcastSender",
        )
        self._rpc_sender_thread.start()

    def _stop_rpc_sender(self) -> None:
        if hasattr(self, "_rpc_sender_shutdown"):
            self._rpc_sender_shutdown.set()
        if hasattr(self, "_rpc_send_queue"):
            try:
                self._rpc_send_queue.put_nowait(self._RPC_SEND_SENTINEL)
            except Exception:
                pass
        if hasattr(self, "_rpc_sender_thread") and self._rpc_sender_thread.is_alive():
            self._rpc_sender_thread.join(timeout=1.0)

    def _fail_all_pending(self, exc: Exception) -> None:
        with self.pending_requests_lock:
            pending_items = list(self.pending_requests.items())
            self.pending_requests.clear()
        for _, futs_by_rank in pending_items:
            for fut in futs_by_rank.values():
                if not fut.done():
                    fut.set_exception(exc)

    def _rpc_sender_loop(self) -> None:
        while not self.shutdown_event.is_set() and not self._rpc_sender_shutdown.is_set():
            try:
                item = self._rpc_send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is self._RPC_SEND_SENTINEL:
                break

            try:
                # single-writer
                mq = getattr(self, "rpc_broadcast_mq", None)
                if mq is None:
                    raise RuntimeError("rpc_broadcast_mq is None while sending")
                mq.enqueue(item)
            except Exception as e:
                logger.exception("RPC sender failed while enqueueing to shm MQ.")
                self.is_failed = True
                self._fail_all_pending(RuntimeError(f"RPC send failed: {e}"))
                try:
                    self.shutdown()
                except Exception:
                    pass
                break

    def _try_cleanup_seq(self, seq_id: int) -> None:
        with self.pending_requests_lock:
            futs = self.pending_requests.get(seq_id)
            if not futs:
                return
            if all(f.done() for f in futs.values()):
                self.pending_requests.pop(seq_id, None)

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: FailureCallback | None = None

        # Sequence counter to tag every RPC request uniquely
        self.seq_counter = itertools.count()
        self.seq_lock = threading.Lock()
        # Map: seq_id -> {worker_rank: Future}
        self.pending_requests: dict[int, dict[int, Future]] = {}
        self.pending_requests_lock = threading.Lock()

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)

        # Start single-writer sender thread BEFORE any potential multi-threaded RPC
        self._start_rpc_sender()

        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))

            # Workers must be created before wait_for_ready to avoid deadlock
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()

            # Start one background thread per worker to listen for responses
            # This enables full-duplex communication.
            self.response_threads = []
            for w in self.workers:
                t = Thread(target=self._worker_response_listener,
                           args=(w,),
                           daemon=True,
                           name=f"ResponseListener-{w.rank}")
                t.start()
                self.response_threads.append(t)

            success = True
        finally:
            if not success:
                # stop sender to avoid leaking thread / writing to half-initialized MQ
                try:
                    self._stop_rpc_sender()
                except Exception:
                    pass

                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None

    def _worker_response_listener(self, worker_handle: "WorkerProcHandle"):
        """Background thread: Reads from Worker -> Resolves Future"""
        while not self.shutdown_event.is_set():
            try:
                if self.is_failed:
                    break

                # Short timeout allows checking shutdown_event periodically
                try:
                    # Expect: (status, result, seq_id)
                    status, result, seq_id = worker_handle.worker_response_mq.dequeue(timeout=0.2)
                except TimeoutError:
                    continue
                except Exception as e:
                    if self.shutdown_event.is_set():
                        break
                    logger.error(f"Error reading from worker {worker_handle.rank}: {e}")
                    self.is_failed = True
                    # fail all pending to avoid deadlocks
                    self._fail_all_pending(RuntimeError(f"Worker response MQ read failed: {e}"))
                    break

                if seq_id is None:
                    logger.error(f"Protocol Error: Missing seq_id from worker {worker_handle.rank}")
                    continue

                # Find the future associated with this seq_id and rank
                with self.pending_requests_lock:
                    worker_futures = self.pending_requests.get(seq_id)
                    if not worker_futures:
                        # Request might have been timed out or cancelled
                        continue
                    future = worker_futures.get(worker_handle.rank)

                if future and not future.done():
                    if status == WorkerProc.ResponseStatus.SUCCESS:
                        future.set_result(result)
                    else:
                        future.set_exception(RuntimeError(result))

                # cleanup if all futures done (important for non_block mode)
                self._try_cleanup_seq(seq_id)

            except Exception:
                if not self.shutdown_event.is_set():
                    logger.exception("Response listener thread crashed")
                    self.is_failed = True
                    self._fail_all_pending(RuntimeError("Response listener crashed."))

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            # fail pending so callers don't hang
            _self._fail_all_pending(RuntimeError(f"Worker died unexpectedly: {proc_name}"))
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
        virtual_runner: int = None, 
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:

        if not self.has_connector:
            # Get output from output_rank only
            futures_or_results = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, virtual_runner), 
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

            # If non_block=True, collective_rpc returns List[Future]
            return futures_or_results[0]

        # Get output from all workers (TP/PP aggregation case)
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, virtual_runner), 
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch",
                            unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        outputs = self.collective_rpc("take_draft_token_ids",
                                      unique_reply_rank=self.output_rank)
        return outputs[0]

    def collective_rpc(self,
                       method: str | Callable,
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict | None = None,
                       non_block: bool = False,
                       unique_reply_rank: int | None = None) -> list[Any]:
        """
        Sends RPC request to workers.
        Returns List[Future] if non_block=True, else List[Any].

        Robustness:
        - Multi-thread safe enqueue: all sends go through a single sender thread.
        - seq_id generation protected for free-threading.
        """
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        kwargs = kwargs or {}

        # Generate unique Sequence ID (thread-safe)
        with self.seq_lock:
            seq_id = next(self.seq_counter)

        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)

            target_workers_indices = (
                [unique_reply_rank] if unique_reply_rank is not None
                else range(len(self.workers))
            )

            # Prepare Futures
            request_futures: dict[int, Future] = {}
            response_futures_list: list[Future] = []

            for rank in target_workers_indices:
                fut = Future()
                request_futures[rank] = fut
                response_futures_list.append(fut)

            # Register futures BEFORE sending
            with self.pending_requests_lock:
                self.pending_requests[seq_id] = request_futures

            # Send Request via single-writer sender thread:
            # Protocol: (method, args, kwargs, output_rank, seq_id)
            payload = (send_method, args, kwargs, unique_reply_rank, seq_id)
            try:
                self._rpc_send_queue.put(payload, timeout=5.0)
            except Exception as e:
                with self.pending_requests_lock:
                    self.pending_requests.pop(seq_id, None)
                raise RuntimeError(f"Failed to enqueue RPC payload to sender thread: {e}")

            if non_block:
                return response_futures_list

            # Wait for results (Block)
            results: list[Any] = []
            deadline = None if timeout is None else time.monotonic() + timeout

            for fut in response_futures_list:
                wait_time = None
                if deadline is not None:
                    wait_time = deadline - time.monotonic()
                    if wait_time < 0:
                        raise TimeoutError(f"RPC call to {method} timed out.")
                results.append(fut.result(timeout=wait_time))

            # Cleanup finished seq_id (may already be cleaned by listener)
            with self.pending_requests_lock:
                self.pending_requests.pop(seq_id, None)

            return results

        except Exception as e:
            with self.pending_requests_lock:
                self.pending_requests.pop(seq_id, None)
            raise e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""
        def wait_for_termination(procs, timeout):
            if not time:
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            # Stop sender FIRST to avoid writing shm after teardown
            try:
                self._stop_rpc_sender()
            except Exception:
                pass

            # Fail pending requests so callers don't hang
            try:
                self._fail_all_pending(RuntimeError("Executor is shutting down."))
            except Exception:
                pass

            if workers := getattr(self, 'workers', None):
                for w in workers:
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

            # Join listener threads
            if hasattr(self, 'response_threads'):
                for t in self.response_threads:
                    if t.is_alive():
                        t.join(timeout=0.1)

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        # Allow higher concurrency as we support thread-pooled workers
        return 32

    def _get_output_rank(self) -> int:
        return self.world_size - self.parallel_config.tensor_parallel_size


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process with internal parallelism."""

    READY_STR = "READY"

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq: MessageQueue = MessageQueue(1, 1)
        else:
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            self.worker_response_mq, _ = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)

        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        # Lock to protect concurrent writes to shared memory response queue
        self.response_lock = Lock()

        # Internal thread pool to allow executing multiple virtual_runners simultaneously
        self.execution_thread_pool = ThreadPoolExecutor(
            max_workers=256, thread_name_prefix="WorkerExec"
        )

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy")
            self.async_output_copy_thread.start()

        # Initialize device
        self.worker.init_device()

        # Set process title and log prefix
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel)

        # Load model
        self._init_message_queues(input_shm_handle, vllm_config)
        self.worker.load_model()

        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        writer.close()
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:
        e = Exception("WorkerProc initialization failed.")
        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None
                finally:
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.worker.shutdown()
        if self.execution_thread_pool:
            self.execution_thread_pool.shutdown(wait=False)
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()

        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    death_pipe.recv()
                except EOFError:
                    logger.info("Parent process exited, terminating worker")
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(
                target=monitor_parent_death, daemon=True, name="WorkerDeathMonitor"
            )
            death_monitor.start()

        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)

            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                }
            )

            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any, seq_id: int):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq using a lock for thread safety.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output), seq_id)
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output, seq_id)

        if (response_mq := self.worker_response_mq) is not None:
            with self.response_lock:
                response_mq.enqueue(result)

    def handle_output(self, output: Any, seq_id: int):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put((output, seq_id))
        else:
            self.enqueue_output(output, seq_id)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            output, seq_id = self.async_output_queue.get()
            self.enqueue_output(output, seq_id)

    def _execute_task_concurrently(self, method_name, pickled_method, args, kwargs, output_rank, seq_id):
        """Helper to execute the task in a thread and handle the response"""
        try:
            if method_name is not None:
                func = getattr(self.worker, method_name)
            else:
                func = partial(cloudpickle.loads(pickled_method), self.worker)

            output = func(*args, **kwargs)

        except Exception as e:
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception(f"WorkerProc hit an exception in method: {method_name}")
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e, seq_id)
            return

        if output_rank is None or self.rank == output_rank:
            self.handle_output(output, seq_id)

    def worker_busy_loop(self, cancel: threading.Event | None = None):
        """
        Main busy loop for Multiprocessing Workers.
        Now acts as a dispatcher to the thread pool for supported methods.
        """
        while True:
            method_data = self.rpc_broadcast_mq.dequeue(cancel=cancel, indefinite=True)
            method, args, kwargs, output_rank, seq_id = method_data

            method_name = None
            pickled_method = None

            if isinstance(method, str):
                method_name = method
            elif isinstance(method, bytes):
                pickled_method = method

            should_parallelize = (method_name == "execute_model")
            if should_parallelize:
                self.execution_thread_pool.submit(
                    self._execute_task_concurrently,
                    method_name, pickled_method, args, kwargs, output_rank, seq_id
                )
            else:
                self._execute_task_concurrently(
                    method_name, pickled_method, args, kwargs, output_rank, seq_id
                )

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    default_omp_num_threads = 1
    if (
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
