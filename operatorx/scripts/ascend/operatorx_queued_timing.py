import hashlib
import time
from pathlib import Path

FIRST = True
PROTOCOL = "queued_10_events_5_warmups_25ms_ramp_4_write_shield_v1"
SOURCE_URL = "https://github.com/SemiAnalysisAI/InferenceX/blob/24e1caa715c0cd57e141969562ad06403023ca00/operatorx/runners/common/timing.py"


def time_op(npu, fn, cache, config):
    global FIRST
    if (config["samples"], config["warmups"], config["warmup_ms"]) != (10, 5, 25):
        raise ValueError("Queued protocol requires 10 samples, 5 warmups, 25 ms ramp")
    begun = time.perf_counter_ns()
    for _ in range(5):
        output = fn()
    npu.synchronize()
    floor_ms = 1000 if FIRST else 25
    FIRST = False
    ramp_start = time.perf_counter_ns()
    ramp_calls = 0
    while (time.perf_counter_ns() - ramp_start) / 1e6 < floor_ms:
        if ramp_calls >= 1000000:
            raise RuntimeError("Queued warmup iteration bound exceeded")
        output = fn()
        npu.synchronize()
        ramp_calls += 1
    warmup_ms = (time.perf_counter_ns() - begun) / 1e6
    starts = [npu.Event(enable_timing=True) for _ in range(10)]
    ends = [npu.Event(enable_timing=True) for _ in range(10)]
    wall_start = time.perf_counter_ns()
    for _ in range(4):
        cache.zero_()
    for start, end in zip(starts, ends):
        cache.zero_()
        start.record()
        output = fn()
        end.record()
    npu.synchronize()
    batch_wall_us = (time.perf_counter_ns() - wall_start) / 1000
    event_us = [float(s.elapsed_time(e)) * 1000 for s, e in zip(starts, ends)]
    latency = sorted(event_us)[5]
    cooldown = min(latency * 1e-6 * 15 * 4, 1.0)
    time.sleep(cooldown)
    timing = {
        "protocol": PROTOCOL,
        "device_event_samples_us": event_us,
        "aggregation": "upper median: sorted(samples)[5] of 10",
        "requested_samples": 10,
        "warmups": 5,
        "actual_warmups": 5 + ramp_calls,
        "minimum_warmup_ms": 25,
        "effective_ramp_floor_ms": floor_ms,
        "actual_warmup_ms": warmup_ms,
        "mode": "eager",
        "npu_graph": False,
        "cache_write_bytes_per_sample": config["cache_write_bytes"],
        "cache_write_operation": "uint8 zero_ queued before each start event; no per-sample synchronization",
        "cache_write_in_timed_interval": False,
        "cold_cache_proven": False,
        "launch_shield": "four queued cache zero_ writes; not CUDA _sleep",
        "shield_writes": 4,
        "synchronizations_inside_sample_loop": 0,
        "queued_batch_wall_us": batch_wall_us,
        "queued_wall_us_per_call": batch_wall_us / 10,
        "wall_interval": "shield writes + all ten cache writes and event-bracketed calls + final synchronization",
        "cooldown_seconds": cooldown,
        "cooldown_inside_timer": False,
        "timing_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operatorx_reference_url": SOURCE_URL,
        "gpu_throttle_retry_equivalence_claimed": False,
        "gpu_graph_equivalence_claimed": False,
    }
    return output, timing
