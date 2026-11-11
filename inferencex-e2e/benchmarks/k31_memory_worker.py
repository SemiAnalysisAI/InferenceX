import json
import os
import time

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger("vllm.k31_memory_worker")


class MemoryDiagnosticWorker(Worker):
    """Opt-in through --worker-cls; delegate all serving operations unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_observer_enabled = True
        self._memory_next_sample = 0.0
        self._memory_step = 0

    def _memory_record(self, phase, scheduler_output=None):
        if not self._memory_observer_enabled:
            return
        try:
            stats = torch.cuda.memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            record = {
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "phase": phase,
                "step": self._memory_step,
                "driver_free_bytes": free,
                "driver_total_bytes": total,
            }
            for key in (
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "active_bytes.all.current",
                "inactive_split_bytes.all.current",
                "allocated_bytes.all.peak",
                "reserved_bytes.all.peak",
                "num_alloc_retries",
                "num_ooms",
            ):
                record[key] = stats.get(key)
            if scheduler_output is not None:
                record["scheduled_tokens"] = scheduler_output.total_num_scheduled_tokens
                record["scheduled_requests"] = len(scheduler_output.num_scheduled_tokens)
            logger.info("UMBP_MEMORY %s", json.dumps(record, sort_keys=True))
        except Exception:
            # A failing diagnostic must not replace the model's original error.
            self._memory_observer_enabled = False
            logger.warning("UMBP_MEMORY observer disabled after sampling failure", exc_info=True)

    def init_device(self):
        result = super().init_device()
        self._memory_record("init_device_after")
        return result

    def load_model(self, *, load_dummy_weights=False):
        self._memory_record("load_model_before")
        result = super().load_model(load_dummy_weights=load_dummy_weights)
        self._memory_record("load_model_after")
        return result

    def determine_available_memory(self):
        self._memory_record("profile_before")
        result = super().determine_available_memory()
        self._memory_record("profile_after")
        return result

    def initialize_from_config(self, kv_cache_config):
        self._memory_record("kv_init_before")
        result = super().initialize_from_config(kv_cache_config)
        self._memory_record("kv_init_after")
        return result

    def compile_or_warm_up_model(self):
        self._memory_record("capture_before")
        result = super().compile_or_warm_up_model()
        self._memory_record("capture_after")
        return result

    def execute_model(self, scheduler_output):
        self._memory_step += 1
        now = time.monotonic()
        sample = self._memory_observer_enabled and now >= self._memory_next_sample
        if sample:
            # At most one pair per minute and rank; no background thread or barrier.
            self._memory_next_sample = now + 60.0
            self._memory_record("execute_before", scheduler_output)
        try:
            return super().execute_model(scheduler_output)
        finally:
            if sample:
                self._memory_record("execute_after", scheduler_output)
