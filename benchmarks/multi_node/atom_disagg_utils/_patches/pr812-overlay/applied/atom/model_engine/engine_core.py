# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import logging
import pickle
import queue
import threading
import time
from contextlib import ExitStack
from typing import List

import torch
import zmq
from atom.config import Config, ParallelConfig
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence
from atom.utils import init_exit_handler, make_zmq_socket
from atom.utils.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)

from atom.kv_transfer.disaggregation import KVOutputAggregator

logger = logging.getLogger("atom")


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used within EngineCore.
    SHUTDOWN = b"\x05"
    # Stream output for callbacks
    STREAM = b"\x06"
    # Signal that EngineCore is fully initialized and ready
    READY = b"\x07"


class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[List[Sequence]]()
        self.stream_output_queue = (
            queue.Queue()
        )  # Queue for streaming intermediate outputs
        self.input_address = input_address
        self.output_address = output_address
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()

        # Start input thread BEFORE _init_data_parallel so that CoreManager
        # can receive the input socket connection and proceed to start the
        # remaining DP ranks.  Without this, _init_data_parallel blocks on
        # rendezvous waiting for all DP ranks, but they haven't been spawned
        # yet because CoreManager is still waiting for *this* rank's socket.
        # The READY signal (sent at the end of __init__) gates actual request
        # processing, so starting the input thread early is safe.
        self.input_thread = threading.Thread(
            target=self.process_input_sockets, args=(self.input_address,), daemon=True
        )
        self.input_thread.start()

        self.profile_enbaled = config.torch_profiler_dir is not None
        self.mark_trace = getattr(config, "mark_trace", False)
        init_exit_handler(self)
        self._init_data_parallel(config)

        # Initialize model runner processes
        try:
            good = False
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tensor_parallel_size,
                "atom.model_engine.model_runner.ModelRunner",
                config,
            )
            block_info = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            num_blocks = block_info["num_kvcache_blocks"]
            config.per_req_cache_equiv_blocks = block_info.get(
                "per_req_cache_equiv_blocks", 0
            )
            config.num_per_req_cache_groups = block_info.get(
                "num_per_req_cache_groups", 0
            )
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            config.num_kvcache_blocks = num_blocks
            if not config.enforce_eager:
                # Start profiler before cudagraph capture only if mark-trace is enabled.
                if self.profile_enbaled and self.mark_trace:
                    self.runner_mgr.call_func(
                        "start_profiler", "capture_graph", wait_out=True
                    )
                cap_cost, bs = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{bs} cost: {cap_cost:.2f} seconds"
                )
                if self.profile_enbaled and self.mark_trace:
                    # Persist a dedicated capture-graph trace immediately.
                    self.runner_mgr.call_func("stop_profiler", wait_out=True)
            good = True
        finally:
            logger.info(
                f"{self.label}: load model runner {'success' if good else 'failed'}"
            )
            if not good:
                self._finalizer()

        self.scheduler = Scheduler(config)

        self.kv_transfer_enabled = bool(config.kv_transfer_config)
        if self.kv_transfer_enabled:
            self.kv_aggregator = KVOutputAggregator(
                world_size=config.tensor_parallel_size
            )

        self._send_ready_signal()
        logger.info(f"{self.label}: EngineCore fully initialized and ready")

    def _send_ready_signal(self):
        self.output_queue.put_nowait(("READY", None))

    def _init_data_parallel(self, config: Config):
        pass

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self.runner_mgr.keep_monitoring = False
        try:
            self.runner_mgr.call_func("exit")
        except Exception:
            pass  # shared memory may already be freed
        for proc in self.runner_mgr.procs:
            try:
                alive = proc.is_alive()
            except ValueError:
                continue  # process object already closed by CoreManager
            if alive:
                proc.join(timeout=5)
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait([get_exit_sequence()])
        self.output_thread.join(timeout=0.5)

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine: EngineCore = None
        try:
            if config.parallel_config.data_parallel_size > 1:
                engine = DPEngineCoreProc(config, input_address, output_address)
            else:
                engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception as e:
            logger.error(f"run_engine: exception: {e}", exc_info=True)
            raise e
        finally:
            if engine is not None:
                engine.exit()

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self.pull_and_process_input_queue()
            if shutdown:
                break
            if not self.scheduler.is_finished():
                self._process_engine_step()

    def _process_engine_step(self):
        result = self.scheduler.schedule()
        if result is None:
            if self.kv_transfer_enabled:
                kvoutput = self.runner_mgr.call_func_with_aggregation(
                    "async_proc_aggregation"
                )
                self.scheduler._update_from_kv_xfer_finished(kvoutput)
            return False
        scheduled_batch, seqs = result

        if scheduled_batch is None:
            logger.debug("%s: No sequences to schedule, skipping forward", self.label)
            if self.kv_transfer_enabled:
                kvoutput = self.runner_mgr.call_func_with_aggregation(
                    "async_proc_aggregation"
                )
                self.scheduler._update_from_kv_xfer_finished(kvoutput)
            return False

        # Dispatch KV connector metadata to workers (triggers async KV load)
        if (
            self.kv_transfer_enabled
            and scheduled_batch.connector_meta_output is not None
        ):
            self.runner_mgr.call_func(
                "process_kvconnector_output", scheduled_batch.connector_meta_output
            )

        # Run the model forward pass if there are actual sequences
        has_seqs = len(scheduled_batch.req_ids) > 0
        if has_seqs:
            fwd_out = self.runner_mgr.call_func(
                "forward", scheduled_batch, wait_out=True
            )

        # Aggregate KV transfer status from all workers (only when PD disaggregation is active)
        if self.kv_transfer_enabled:
            kvoutput = self.runner_mgr.call_func_with_aggregation(
                "async_proc_aggregation"
            )
            self.scheduler._update_from_kv_xfer_finished(kvoutput)

        if not has_seqs:
            logger.debug("%s: Empty scheduled batch, skipping postprocess", self.label)
            return False

        seqs = seqs.values()
        # Pass stream_output_queue to postprocess for streaming callbacks
        finished_seqs = self.scheduler.postprocess(
            seqs, fwd_out, stream_output_queue=self.stream_output_queue
        )

        # Send stream outputs to main process via output_queue
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                # Send stream outputs as intermediate results
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass

        if finished_seqs:
            self.output_queue.put_nowait(finished_seqs)
        return True

    def pull_and_process_input_queue(self):
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    logger.debug(f"{self.label}: input_queue get exit engine")
                    return True
                recv_reqs.append(seq)
        if len(recv_reqs) > 0:
            logger.debug(f"{self.label}: put {len(recv_reqs)} reqs to scheduler")
            self.scheduler.extend(recv_reqs)
        return False

    def process_input_sockets(self, input_address: str):
        """Input socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")
            alive = True

            while alive:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    obj = input_socket.recv(copy=False)
                    request_type, reqs = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.ADD:
                        req_ids = [req.id for req in reqs]
                        logger.debug(
                            f"{self.label}: input get {request_type} {req_ids}"
                        )
                        self.input_queue.put_nowait(reqs)
                    elif request_type == EngineCoreRequestType.UTILITY:
                        # Handle utility commands like start_profile/stop_profile
                        cmd = reqs.get("cmd") if isinstance(reqs, dict) else None
                        logger.debug(f"{self.label}: input get UTILITY command: {cmd}")
                        if cmd == "start_profile":
                            self.start_profiler()
                        elif cmd == "stop_profile":
                            self.stop_profiler()
                        elif cmd == "get_mtp_stats":
                            self.print_mtp_statistics()
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(f"{self.label}: input get {request_type}")
                        self.input_queue.put_nowait([get_exit_sequence()])
                        alive = False
                        reason = request_type
            logger.debug(f"{self.label}: input thread exit due to {reason}")

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                item = self.output_queue.get()
                if isinstance(item, tuple) and item[0] == "STREAM":
                    # Send stream outputs
                    stream_outputs = item[1]
                    obj = pickle.dumps((EngineCoreRequestType.STREAM, stream_outputs))
                    socket.send(obj)
                    continue

                if isinstance(item, tuple) and item[0] == "READY":
                    # Send READY signal to indicate EngineCore is fully initialized
                    obj = pickle.dumps((EngineCoreRequestType.READY, None))
                    socket.send(obj)
                    logger.debug(f"{self.label}: sent READY signal")
                    continue

                # Regular finished sequences
                seqs = item
                valid_seqs = [
                    seq for seq in seqs if seq.status != SequenceStatus.EXIT_ENGINE
                ]
                num_valid = len(valid_seqs)
                if num_valid > 0:
                    obj = pickle.dumps((EngineCoreRequestType.ADD, valid_seqs))
                    socket.send(obj)
                    logger.info(f"{self.label}: output send {num_valid} reqs")
                if len(valid_seqs) != len(seqs):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.debug(
                        f"{self.label}: output send {EngineCoreRequestType.SHUTDOWN}"
                    )
                    break

    def start_profiler(self):
        if self.profile_enbaled:
            self.runner_mgr.call_func("start_profiler", wait_out=True)

    def stop_profiler(self):
        if self.profile_enbaled:
            logger.info("Profiler stopping...")
            t0 = time.monotonic()
            self.runner_mgr.call_func("stop_profiler", wait_out=True)
            logger.info("Profiler stopped in %.1fs", time.monotonic() - t0)

    def print_mtp_statistics(self):
        if self.scheduler.spec_stats is not None:
            self.scheduler.spec_stats._log()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available (MTP not enabled or no tokens processed)\n"
            )


class DPEngineCoreProc(EngineCore):
    def __init__(self, config: Config, input_address: str, output_address: str):
        # self.dp_group = config.parallel_config.dp_group
        self.dp_rank = config.parallel_config.data_parallel_rank
        # self.dp_group = config.parallel_config.stateless_init_dp_group()
        super().__init__(config, input_address, output_address)
        # Initialize to True so first iteration reaches all_reduce
        self.engines_running = True
        self._shutting_down = False

    def _init_data_parallel(self, config: Config):
        dp_rank = config.parallel_config.data_parallel_rank
        dp_size = config.parallel_config.data_parallel_size
        local_dp_rank = config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_group = config.parallel_config.stateless_init_dp_group()

    def exit(self):
        super().exit()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self.pull_and_process_input_queue()

            local_is_prefill, local_num_tokens, local_num_reqs = (
                self.scheduler.get_next_batch_info()
            )
            local_unfinished = not self.scheduler.is_finished()

            (
                global_has_prefill,
                global_max_tokens,
                global_max_reqs,
                global_has_unfinished,
                global_shutdown,
            ) = self._sync_dp_state(
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_unfinished,
                shutdown,
            )

            if global_shutdown and not global_has_unfinished:
                logger.info(
                    f"{self.label}: All DP ranks agreed to shutdown, exiting busy_loop"
                )
                break

            if not global_has_unfinished and not self.engines_running:
                self.engines_running = False
                continue

            if global_has_prefill and not local_is_prefill:
                # We must do dummy prefill to sync here
                # Since we want to split mori output in moe, we need to make dp all run prefill or all run decode
                dummy_reqs = min(
                    global_max_reqs, 2
                )  # dummy reqs at 2: just enough for TBO agreement, avoid wasting compute.
                logger.info(
                    f"{self.label}: Running dummy prefill ({global_max_tokens} tokens, {dummy_reqs} reqs) "
                    f"to sync with other DP ranks doing prefill"
                )
                self._execute_dummy_prefill(global_max_tokens, dummy_reqs)
            else:
                executed = self._process_engine_step()
                if not executed:
                    if global_has_prefill:
                        # get_next_batch_info predicted prefill but schedule()
                        # skipped it (e.g. WAITING_FOR_REMOTE_KVS).  Other DP
                        # ranks already committed to dummy_prefill, so we must
                        # match to keep the all-reduce in sync.
                        logger.info(
                            f"{self.label}: Predicted prefill was not scheduled, "
                            f"falling back to dummy prefill ({global_max_tokens} "
                            f"tokens) to stay in sync with other DP ranks"
                        )
                        self._execute_dummy_prefill(global_max_tokens)
                    else:
                        self._execute_dummy_batch()

            self.engines_running = global_has_unfinished

    def _execute_dummy_batch(self):
        return self.runner_mgr.call_func("dummy_execution", wait_out=True)

    def _execute_dummy_prefill(self, num_tokens: int, num_reqs: int = 1):
        return self.runner_mgr.call_func(
            "dummy_prefill_execution", num_tokens, num_reqs, wait_out=True
        )

    def _sync_dp_state(
        self,
        local_is_prefill: bool,
        local_num_tokens: int,
        local_num_reqs: int,
        local_has_unfinished: bool,
        local_shutdown: bool = False,
    ) -> tuple[bool, int, int, bool, bool]:
        if self._shutting_down:
            return (
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_has_unfinished,
                True,
            )

        try:
            # Pack all state: [is_prefill, num_tokens, num_reqs, has_unfinished, shutdown]
            state_tensor = torch.tensor(
                [
                    1 if local_is_prefill else 0,
                    local_num_tokens,
                    local_num_reqs,
                    1 if local_has_unfinished else 0,
                    1 if local_shutdown else 0,
                ],
                dtype=torch.int64,
                device="cpu",
            )
            torch.distributed.all_reduce(
                state_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_has_prefill = state_tensor[0].item() == 1
            global_max_tokens = state_tensor[1].item()
            global_max_reqs = state_tensor[2].item()
            global_has_unfinished = state_tensor[3].item() == 1
            global_shutdown = state_tensor[4].item() == 1
            return (
                global_has_prefill,
                global_max_tokens,
                global_max_reqs,
                global_has_unfinished,
                global_shutdown,
            )
        except RuntimeError as e:
            logger.warning(f"{self.label}: _sync_dp_state failed: {e}")
            # If sync fails, assume shutdown to prevent hang
            self._shutting_down = True
            return (
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_has_unfinished,
                True,
            )

    def _sync_shutdown_state(self, local_should_shutdown: bool) -> bool:
        try:
            tensor = torch.tensor(
                [local_should_shutdown], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_should_shutdown = bool(tensor.item())
            return global_should_shutdown
        except RuntimeError as e:
            # If all_reduce fails, it means other ranks are shutting down
            logger.warning(
                f"{self.label}: Shutdown sync failed, assuming shutdown: {e}"
            )
            return True

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        if self._shutting_down:
            logger.info(f"{self.label}: Skipping DP sync during shutdown")
            return local_unfinished
        try:
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        except RuntimeError as e:
            # Handle case where other ranks have already shut down
            logger.warning(f"{self.label}: DP sync failed during shutdown: {e}")
            return local_unfinished
