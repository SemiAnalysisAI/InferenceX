#!/usr/bin/env python3
"""CollectiveX — shared EP (expert-parallel) dispatch/combine benchmark harness.

Backend-agnostic core. The per-backend adapters (`ep_deepep.py`, `ep_mori.py`)
implement a small duck-typed protocol; this module owns the source-tokens-per-rank
sweep, the timing, the correctness gate, and the provenance-tagged JSON doc.

Fair-comparison contract (see docs/methodology.md):
  * **Deterministic shared routing trace** (`routing.py`): the per-token expert IDs +
    gate weights are generated once from a fixed seed over the *global* batch and are
    identical on every SKU; each rank materializes its slice. So every platform runs
    the *same* problem (no per-rank/per-platform RNG in the adapters).
  * **Explicit measurement contract**: layout-and-dispatch-v1 includes routing-layout
    generation in dispatch timing. Combine excludes staging.
    Isolated sum is derived independently at each percentile and is not a measured chained op.
  * **Correct collective percentile**: each iteration's latency is reduced MAX across
    ranks first (a collective finishes with its slowest rank), THEN percentiled —
    `median_i(max_r)`, not `max_r(median_i)`.
  * **One line = one fixed config**; only T varies. Both `tokens_per_rank` and
    `global_tokens = T * ep_size` are recorded as explicit chart coordinates.

stdlib-only at module top (torch is passed in by the entrypoint; `routing` is imported
lazily inside run_sweep) so this file `py_compile`s without torch.

Backend protocol:
    name, mode, combine_needs_redispatch, backend_provenance(dict)
    buffer_cap(args) -> int|None
    make_problem(T, idx, weights, x) -> problem   # materialize this rank's trace slice
    dispatch(problem) -> handle                   # pure dispatch comm (timed)
    stage(problem, handle)                        # expert-output placement
    stage_device_work                             # true only when stage launches device work
    combine(problem, handle) -> tensor            # pure combine comm (timed)
    inspect_dispatch(problem, handle) -> view     # normalized payload/expert/weight metadata
    combine_transformed(problem, handle, tensor) -> tensor
    recv_tokens(handle) -> int                    # realized tokens received this rank
    finalize(rc) -> int|NoReturn
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import types

import contracts
import identity
import workload as workload_contract

# Raw v1 result emitted by one benchmark case. Publication uses a separate contract.
SCHEMA_VERSION = 1

# Every comparison-grade EP point uses the same literal timing profile on every SKU/backend.
# Eight timed iterations keep each MoRI burst well below its sustained-iteration wedge, 64 trials
# provide 512 observations per operation, and 32 warmups meet Blackwell's measured clock-ramp floor.
SAMPLING_CONTRACT = identity.V1_CASE_PROFILE["sampling_contract"]
TIMED_SAMPLES_PER_POINT = 512
TIMED_ITERS_PER_TRIAL = 8
TRIALS_PER_POINT = 64
WARMUP_ITERS_PER_TRIAL = 32
WARMUP_SEMANTICS = "full-roundtrip-before-each-component-trial-point-v1"
QUALIFICATION_RUNS = 3
ROUTING_SEED = 67
ROUTING_GENERATOR = workload_contract.GENERATOR_VERSION
ACTIVATION_PROFILE = "canonical-counter-source-v4"
ACTIVATION_GENERATOR = workload_contract.ACTIVATION_GENERATOR
PLACEMENT = "packed"
COMPONENT_ORDER_CONTRACT = "qualification-hash-rotated-components-v1"
LOW_LATENCY_MODE = "low-latency"
LOW_LATENCY_MAX_TOKENS_PER_RANK = 128
LOW_LATENCY_MEASUREMENT_CONTRACT = "expert-packed-weighted-combine-v1"
LOW_LATENCY_COMPONENT_ORDER_CONTRACT = "qualification-hash-rotated-components-v1"
LOW_LATENCY_ORACLE_CONTRACT = "expert-assignment-transform-v1"
LOW_LATENCY_CORRECTNESS_SCOPE = "expert-assignment-and-weighted-combine"

# Phase-default sweeps — token-size regimes, NOT distinct kernels (both run normal
# mode; "decode"/"prefill" name the small/large-token regime). Powers of two for a
# clean log x-axis; clamped to the backend buffer ceiling (MoRI's registerable heap).
DECODE_LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_LADDER = [128, 256, 512, 1024, 2048, 4096]
CONDITIONING_LADDERS = {
    phase: list(ladder) for phase, ladder in contracts.V1_CONDITIONING_LADDERS.items()
}
CONDITIONING_ROUNDS_PER_SHAPE = contracts.V1_CONDITIONING_ROUNDS_PER_SHAPE
CONDITIONING_CONTRACT = identity.V1_CASE_PROFILE["conditioning_contract"]
ORACLE_CONTRACT = identity.V1_CASE_PROFILE["oracle_contract"]
ORACLE_RTOL = 5e-2
ORACLE_ATOL = 2e-2


def _oracle_tolerances(backend) -> tuple[float, float]:
    tolerance = identity.combine_oracle_tolerances(backend.communication_precision)
    return tolerance["atol"], tolerance["rtol"]

EPLB_REDUNDANT_EXPERTS = 32
EPLB_REFERENCE_TOKENS_PER_RANK = 2048
EPLB_PLANNER = "greedy-rank-major-v1"
V1_PROFILE = {
    "dispatch_dtype": "bf16",
    "combine_dtype": "bf16",
    "combine_quant_mode": "none",
    "mode": "normal",
    "measurement_contract": "layout-and-dispatch-v1",
    "resource_mode": "fixed-profile",
    "placement": PLACEMENT,
    "activation_profile": ACTIVATION_PROFILE,
    "activation_generator": ACTIVATION_GENERATOR,
    "routing_generator": ROUTING_GENERATOR,
    "component_order_contract": COMPONENT_ORDER_CONTRACT,
    "conditioning_contract": CONDITIONING_CONTRACT,
    "eplb_reference_tokens_per_rank": EPLB_REFERENCE_TOKENS_PER_RANK,
    "eplb_redundant_experts": EPLB_REDUNDANT_EXPERTS,
    "eplb_planner": EPLB_PLANNER,
    # DeepEP/UCCL use this only as the fallback when their tuned default is not exported.
    "num_sms": 24,
}


def precision_byte_provenance(
    axis: dict, logical_copies: int, hidden: int
) -> dict[str, int | str]:
    """Return comparable logical activation and required scale bytes for one direction."""
    if logical_copies < 0 or hidden < 0:
        raise ValueError("logical precision byte dimensions must be non-negative")
    bits_per_value = {
        "bf16": 16,
        "fp8-e4m3fn": 8,
        "fp8-e4m3fnuz": 8,
        "logfmt10": 10,
    }.get(axis["communication_format"])
    if bits_per_value is None:
        raise ValueError(f"unknown communication format {axis['communication_format']!r}")
    activation_data_bytes = logical_copies * math.ceil(hidden * bits_per_value / 8)
    scale_bytes_per_value = {None: 0, "f32": 4, "implicit-logfmt10": 0}.get(
        axis["scale_dtype"]
    )
    if scale_bytes_per_value is None:
        raise ValueError(f"unknown communication scale dtype {axis['scale_dtype']!r}")
    group_size = axis["scale_group_size"]
    scale_groups = math.ceil(hidden / group_size) if group_size is not None else 0
    scale_bytes = logical_copies * scale_groups * scale_bytes_per_value
    return {
        "accounting_contract": "activation-data-plus-scales-v1",
        "activation_data_bytes": activation_data_bytes,
        "scale_bytes": scale_bytes,
        "total_logical_bytes": activation_data_bytes + scale_bytes,
    }

def format_collective_version(raw) -> str:
    """Normalize PyTorch's tuple or packed NCCL/RCCL version representation."""
    if isinstance(raw, int):
        if raw < 10_000:
            return f"{raw // 1000}.{raw // 100 % 10}.{raw % 100}"
        return f"{raw // 10_000}.{raw // 100 % 100}.{raw % 100}"
    if isinstance(raw, (tuple, list)):
        return ".".join(map(str, raw))
    return str(raw) if raw not in (None, "") else "unknown"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Add the varying v1 inputs; fixed profile values are not CLI axes."""
    ap.set_defaults(**V1_PROFILE)
    ap.add_argument("--mode", default="normal", choices=["normal", LOW_LATENCY_MODE])
    ap.add_argument(
        "--precision-profile",
        default="",
        choices=("", *identity.V1_PRECISION_PROFILES),
        help="exact native dispatch/combine communication profile; blank selects BF16 control",
    )
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"],
                    help="token-size regime: decode (small T) / prefill (large T) — picks the default ladder")
    ap.add_argument("--tokens-ladder", default="",
                    help="space/comma-separated source-tokens-per-rank sweep; blank = phase default")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256, help="TOTAL experts (fixed across EP degrees)")
    ap.add_argument("--routing", default="uniform", choices=["uniform"])
    # EPLB (Expert-Parallel Load Balancer): replicate hot experts onto redundant physical
    # slots + balanced-place so per-rank load equalizes. A pure routing-trace transform
    # (tests/eplb.py); experts becomes num_logical+redundant. The remedy for `zipf` skew.
    ap.add_argument("--eplb", action="store_true",
                    help="apply EPLB expert replication/placement to the routing trace")
    # Canonical workloads consume pre-generated trace bytes instead of the
    # seeded runtime generator, so a result is provably the SAME workload as another machine's
    # (checksum match). Points at a dir of <workload_id>.npz/.manifest.json (make_workloads.py).
    ap.add_argument("--workload-dir", default="",
                    help="dir of canonical workload traces; empty = seeded runtime generation (dev)")
    ap.add_argument("--case-id", default="")
    ap.add_argument("--suite", default="")
    ap.add_argument("--workload-name", default="")
    ap.add_argument("--required-publication", default="")
    ap.add_argument("--seed", type=int, default=ROUTING_SEED)
    ap.add_argument(
        "--qualification-index",
        type=int,
        choices=range(1, QUALIFICATION_RUNS + 1),
        default=os.environ.get("CX_QUALIFICATION_INDEX", "1"),
        help="one-based qualification repeat used for deterministic measurement ordering",
    )
    # 32: B300/Blackwell needs ~30 untimed iters to reach steady-state GPU clocks +
    # establish NVLink/NVSHMEM connections — at warmup=8 its dispatch read ~1787us
    # (cold), at warmup>=30 it settles to ~85us (faster than H100, reproducible within
    # ~2.5%). H100/MI355X reach steady state much sooner; the extra iters are harmless.
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS_PER_TRIAL,
                    help=f"untimed full roundtrips before each trial/point; fixed by "
                         f"{SAMPLING_CONTRACT} to {WARMUP_ITERS_PER_TRIAL}")
    ap.add_argument("--iters", type=int, default=TIMED_ITERS_PER_TRIAL,
                    help=f"timed iterations per trial; fixed by {SAMPLING_CONTRACT} to "
                         f"{TIMED_ITERS_PER_TRIAL}")
    ap.add_argument("--trials", type=int, default=TRIALS_PER_POINT,
                    help=f"timed trials; fixed by {SAMPLING_CONTRACT} to {TRIALS_PER_POINT}")
    # provenance / output
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--scope", required=True, choices=["scale-up", "scale-out"])
    ap.add_argument("--scale-up-transport", required=True)
    ap.add_argument("--scale-out-transport", default="")
    # gpus-per-node=0 means one node containing the whole EP group.
    ap.add_argument("--gpus-per-node", type=int, default=0)
    ap.add_argument("--scale-up-domain", type=int, default=0, help="0 = gpus_per_node*ep (one domain)")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)


def token_ladder(spec: str, phase: str, cap: int | None) -> tuple[list[int], list[int]]:
    """Return (ladder, dropped): explicit spec else the phase default; positive ints;
    clamped to `cap` with dropped points reported (never silently truncated)."""
    if spec and spec.strip():
        want = [int(t) for t in spec.replace(",", " ").split() if t]
    else:
        want = DECODE_LADDER if phase == "decode" else PREFILL_LADDER
    want = sorted({t for t in want if t > 0})
    if cap is not None:
        return [t for t in want if t <= cap], [t for t in want if t > cap]
    return want, []


def sampling_contract_error(iters: int, trials: int, warmup: int) -> str | None:
    """Return a user-facing error unless the exact cross-SKU timing profile is used."""
    expected = (TIMED_ITERS_PER_TRIAL, TRIALS_PER_POINT, WARMUP_ITERS_PER_TRIAL)
    observed = (iters, trials, warmup)
    if observed != expected:
        return (f"{SAMPLING_CONTRACT} requires exactly iters:trials:warmup="
                f"{expected[0]}:{expected[1]}:{expected[2]} on every SKU/backend; got "
                f"{observed[0]}:{observed[1]}:{observed[2]} "
                f"({iters * trials if iters > 0 and trials > 0 else 'invalid'} timed samples)")
    return None


def qualification_order(
    values: list, qualification_index: int, trial_index: int, *, identity_key: str = ""
) -> list:
    """Return a deterministic, position-balanced order for one qualification trial.

    Official runs bind the base permutation to the case identity. The cyclic schedule then gives
    every value every position equally often over 64 trials while qualification repeats start at
    different offsets. Keeping the empty-key behavior stable preserves local diagnostic fixtures.
    """
    if not values or len(values) != len(set(values)):
        raise ValueError("qualification order requires non-empty unique values")
    if qualification_index not in range(1, QUALIFICATION_RUNS + 1):
        raise ValueError(f"qualification_index must be in 1..{QUALIFICATION_RUNS}")
    if type(trial_index) is not int or trial_index < 0:
        raise ValueError("trial_index must be a non-negative integer")
    if not isinstance(identity_key, str):
        raise ValueError("qualification identity_key must be a string")
    base_values = list(values)
    if identity_key:
        base_values.sort(
            key=lambda value: hashlib.sha256(
                f"{identity_key}\0{qualification_index}\0{value}".encode("utf-8")
            ).digest()
        )
    position = trial_index + qualification_index - 1
    cycle, offset = divmod(position, len(values))
    base = base_values if cycle % 2 == 0 else list(reversed(base_values))
    return base[offset:] + base[:offset]


def sampled_component_evidence(trials: list[list[float]]) -> dict:
    """Validate and copy private 64x8 trial blocks without flattening their boundaries."""
    if not trials:
        return {"availability": "unavailable", "sample_count": 0, "trials": None}
    if len(trials) != TRIALS_PER_POINT:
        raise ValueError(
            f"measured component needs {TRIALS_PER_POINT} trial blocks; got {len(trials)}"
        )
    normalized: list[list[float]] = []
    for trial in trials:
        if len(trial) != TIMED_ITERS_PER_TRIAL:
            raise ValueError(
                f"measured trial needs {TIMED_ITERS_PER_TRIAL} samples; got {len(trial)}"
            )
        block = []
        for sample in trial:
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise ValueError("measured samples must be numeric")
            value = float(sample)
            if not math.isfinite(value) or value < 0:
                raise ValueError("measured samples must be finite and non-negative")
            block.append(value)
        normalized.append(block)
    count = sum(map(len, normalized))
    if count != TIMED_SAMPLES_PER_POINT:
        raise ValueError(
            f"measured component needs {TIMED_SAMPLES_PER_POINT} samples; got {count}"
        )
    return {"availability": "measured", "sample_count": count, "trials": normalized}


def _stats_vec(xs: list[int]) -> dict:
    """min/mean/max/CV (+ empty count) of a per-rank count vector — self-describing source-token
    or load summary without dumping the full vector."""
    n = len(xs) or 1
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    cv = (var ** 0.5 / mean) if mean > 0 else 0.0
    return {"min": min(xs) if xs else 0, "mean": round(mean, 3),
            "max": max(xs) if xs else 0, "cv": round(cv, 4),
            "empty_ranks": sum(1 for x in xs if x == 0), "total": sum(xs), "ranks": n}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[i]


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _series_provenance(provenance: dict) -> dict:
    """Retain stable semantic build identity while keeping raw binaries diagnostic."""
    return contracts.series_provenance(provenance)


def _write_bytes_atomic(path: str, payload: bytes) -> tuple[str, int]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _write_json_atomic(path: str, value) -> tuple[str, int]:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    return _write_bytes_atomic(path, payload)


def time_us(torch, fn, warmup: int, iters: int, pre=None, post=None) -> list[float]:
    """Per-iteration CUDA-event latencies (µs) for THIS rank.

    Without `pre`: times `fn()`. With `pre`: runs `pre()` UNTIMED each iteration (sync
    before the start event so its GPU work can't bleed in), then times `fn(pre_result)`.
    `post(result)` runs after the end event and synchronization, so stateful backends can
    consume/reset a timed operation without charging that cleanup to its latency. Returns
    the raw per-iteration series; the caller reduces across ranks per iteration before
    percentiling.
    """
    def sample():
        arg = pre() if pre is not None else None
        if pre is not None:
            torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        result = fn(arg) if pre is not None else fn()
        e.record()
        torch.cuda.synchronize()
        elapsed = s.elapsed_time(e) * 1000.0  # ms -> us
        if post is not None:
            post(result)
            torch.cuda.synchronize()
        return elapsed

    for _ in range(max(0, warmup)):
        if pre is not None:
            a = pre()
            torch.cuda.synchronize()
            fn(a)
        else:
            fn()
        # sync EACH warmup iteration, not just once after the loop: the measured-roundtrip fn
        # interleaves dispatch+combine on a backend's persistent comm buffer, so back-to-back
        # un-synced warmup iterations let iter N+1's dispatch race iter N's combine (CUDA abort
        # on a rank -> NCCL-watchdog SIGABRT). Cheap (warmup is small); timed samples already sync.
        torch.cuda.synchronize()
    return [sample() for _ in range(iters)]


def kernel_generation(backend) -> str:
    """Return the adapter's explicit kernel family when one exists."""
    declared = getattr(backend, "kernel_generation", None)
    if declared:
        return declared
    return {
        "deepep": "v1",
        "deepep-v2": "v2-elastic-buffer",
        "deepep-hybrid": "hybrid",
    }.get(backend.name, "n-a")


def _reduce_vec(torch, dist, device, vals, op):
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return [float(x) for x in t.tolist()]


def _reduce_int(torch, dist, device, v: int, op) -> int:
    t = torch.tensor([int(v)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=op)
    return int(t.item())


def _same_hash_across_ranks(torch, dist, device, digest: str) -> bool:
    parts = [int(digest[offset:offset + 8], 16) for offset in range(0, 64, 8)]
    low = torch.tensor(parts, device=device, dtype=torch.int64)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return bool(torch.equal(low, high))


def _tensor_sha256(*tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _normalized_expert_metadata(torch, expert_ids, weights):
    """Sort each row by global expert ID while keeping -1 sentinels last."""
    valid = expert_ids >= 0
    keys = torch.where(valid, expert_ids.to(torch.int64), torch.full_like(expert_ids, 1 << 30))
    order = torch.argsort(keys, dim=1, stable=True)
    sorted_ids = torch.gather(expert_ids.to(torch.int64), 1, order)
    sorted_weights = torch.gather(weights.to(torch.float32), 1, order)
    sorted_valid = sorted_ids >= 0
    return (
        torch.where(sorted_valid, sorted_ids, torch.full_like(sorted_ids, -1)),
        sorted_weights.masked_fill(~sorted_valid, 0),
    )


def expert_packed_slot_map(
    counts,
    src_info,
    layout_range,
    *,
    tokens_per_rank: int,
    experts_per_rank: int,
    world_size: int,
) -> list[tuple[int, int, int]]:
    """Decode and validate DeepEP's expert-packed receive metadata.

    ``src_info`` stores a source-local token index. The source rank is carried by
    the corresponding packed ``layout_range`` interval, so neither field is
    independently sufficient to identify a source token.
    """
    if tokens_per_rank <= 0 or experts_per_rank <= 0 or world_size <= 0:
        raise ValueError("expert-packed dimensions must be positive")
    if len(counts) != experts_per_rank:
        raise ValueError("expert-packed count shape differs from local experts")
    if len(src_info) != experts_per_rank or len(layout_range) != experts_per_rank:
        raise ValueError("expert-packed metadata shape differs from local experts")

    mask = (1 << 32) - 1
    slots: list[tuple[int, int, int]] = []
    pairs: set[tuple[int, int]] = set()
    for local_expert in range(experts_per_rank):
        count = counts[local_expert]
        if type(count) is not int or count < 0:
            raise ValueError("expert-packed receive count is invalid")
        if len(layout_range[local_expert]) != world_size:
            raise ValueError("expert-packed layout rank dimension is invalid")
        if len(src_info[local_expert]) < count:
            raise ValueError("expert-packed source metadata is truncated")

        covered = [False] * count
        for source_rank, encoded in enumerate(layout_range[local_expert]):
            if type(encoded) is not int or encoded < 0:
                raise ValueError("expert-packed layout range is invalid")
            begin, span = encoded >> 32, encoded & mask
            if begin > count or begin + span > count:
                raise ValueError("expert-packed layout range exceeds valid slots")
            for packed_position in range(begin, begin + span):
                if covered[packed_position]:
                    raise ValueError("expert-packed layout ranges overlap")
                covered[packed_position] = True
                local_source = src_info[local_expert][packed_position]
                if (
                    type(local_source) is not int
                    or local_source < 0
                    or local_source >= tokens_per_rank
                ):
                    raise ValueError("expert-packed source token index is invalid")
                source_id = source_rank * tokens_per_rank + local_source
                pair = (source_id, local_expert)
                if pair in pairs:
                    raise ValueError("expert-packed source/expert assignment is duplicated")
                pairs.add(pair)
                slots.append((local_expert, packed_position, source_id))
        if not all(covered):
            raise ValueError("expert-packed layout ranges omit valid receive slots")
    return slots


def expert_packed_dispatch_view(
    torch,
    packed_payload,
    packed_counts,
    packed_src_info,
    packed_layout_range,
    *,
    rank: int,
    tokens_per_rank: int,
    experts_per_rank: int,
    world_size: int,
):
    """Return the valid expert-packed rows with exact global source identities."""
    if packed_payload.ndim != 3:
        raise ValueError("expert-packed payload must have shape [experts, slots, hidden]")
    if packed_payload.shape[0] != experts_per_rank:
        raise ValueError("expert-packed payload expert dimension is invalid")
    if tuple(packed_counts.shape) != (experts_per_rank,):
        raise ValueError("expert-packed count tensor shape is invalid")
    if tuple(packed_src_info.shape[:1]) != (experts_per_rank,):
        raise ValueError("expert-packed source tensor shape is invalid")
    if tuple(packed_layout_range.shape) != (experts_per_rank, world_size):
        raise ValueError("expert-packed layout tensor shape is invalid")
    if packed_src_info.ndim != 2 or packed_src_info.shape[1] < packed_payload.shape[1]:
        raise ValueError("expert-packed source tensor capacity is invalid")

    counts = [int(value) for value in packed_counts.detach().cpu().tolist()]
    if any(count > packed_payload.shape[1] for count in counts):
        raise ValueError("expert-packed receive count exceeds payload capacity")
    slots = expert_packed_slot_map(
        counts,
        packed_src_info.detach().cpu().tolist(),
        packed_layout_range.detach().cpu().tolist(),
        tokens_per_rank=tokens_per_rank,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )
    device = packed_payload.device
    local_expert_slots = torch.tensor(
        [slot[0] for slot in slots], device=device, dtype=torch.int64
    )
    packed_positions = torch.tensor(
        [slot[1] for slot in slots], device=device, dtype=torch.int64
    )
    source_ids = torch.tensor(
        [slot[2] for slot in slots], device=device, dtype=torch.int64
    )
    expert_ids = local_expert_slots + rank * experts_per_rank
    payload = packed_payload[local_expert_slots, packed_positions]
    return types.SimpleNamespace(
        payload=payload,
        source_ids=source_ids,
        expert_ids=expert_ids,
        local_expert_counts=packed_counts.to(torch.int64),
        local_expert_slots=local_expert_slots,
        packed_positions=packed_positions,
        ordering_contract="expert-major/layout-addressed-packed-slot-v1",
    )


def _expert_transform(torch, payload, expert_ids, weights, combine_weight_semantics):
    """Build one local expert aggregate for the v1 unweighted combine contract."""
    if combine_weight_semantics != "unweighted-rank-sum":
        raise ValueError("v1 requires unweighted rank-sum combine")
    valid = expert_ids >= 0
    expert = expert_ids.clamp(min=0).to(torch.int64)
    gate = weights.to(torch.float32).masked_fill(~valid, 0)
    scale = ((expert * 17 + 5) % 31 + 1).to(torch.float32) / 32
    offset_a = (((expert * 29 + 7) % 37) - 18).to(torch.float32) / 64
    offset_b = (((expert * 43 + 11) % 41) - 20).to(torch.float32) / 128
    scale_sum = (gate * scale).sum(dim=1, keepdim=True)
    offset_a_sum = (gate * offset_a).sum(dim=1, keepdim=True)
    offset_b_sum = (gate * offset_b).sum(dim=1, keepdim=True)
    columns = torch.arange(payload.shape[1], device=payload.device, dtype=torch.int64)
    pattern = (((columns * 13) % 17) - 8).to(torch.float32) / 8
    transformed = (
        payload.float() * scale_sum + offset_a_sum + offset_b_sum * pattern.unsqueeze(0)
    )
    return transformed.to(payload.dtype)


def _expert_transform_expanded(torch, payload, expert_ids):
    """Apply the oracle transform to one row per token/expert assignment."""
    expert = expert_ids.to(torch.int64)
    scale = (((expert * 17 + 5) % 31 + 1).to(torch.float32) / 32).unsqueeze(1)
    offset_a = ((((expert * 29 + 7) % 37) - 18).to(torch.float32) / 64).unsqueeze(1)
    offset_b = ((((expert * 43 + 11) % 41) - 20).to(torch.float32) / 128).unsqueeze(1)
    columns = torch.arange(payload.shape[1], device=payload.device, dtype=torch.int64)
    pattern = (((columns * 13) % 17) - 8).to(torch.float32) / 8
    transformed = payload.float() * scale + offset_a + offset_b * pattern.unsqueeze(0)
    return transformed.to(payload.dtype)


def _expected_transformed_combine(torch, problem):
    """Independently derive sum_i gate_i * expert_i(x) for each source token."""
    semantic_x = getattr(problem, "oracle_x", problem.x)
    expected = torch.zeros_like(semantic_x, dtype=torch.float32)
    expert_ids = problem.topk_idx.to(torch.int64)
    weights = problem.topk_weights.to(torch.float32)
    columns = torch.arange(semantic_x.shape[1], device=semantic_x.device, dtype=torch.int64)
    pattern = (((columns * 13) % 17) - 8).to(torch.float32) / 8
    for slot in range(expert_ids.shape[1]):
        expert = expert_ids[:, slot]
        gate = weights[:, slot].unsqueeze(1)
        scale = (((expert * 17 + 5) % 31 + 1).to(torch.float32) / 32).unsqueeze(1)
        offset_a = ((((expert * 29 + 7) % 37) - 18).to(torch.float32) / 64).unsqueeze(1)
        offset_b = ((((expert * 43 + 11) % 41) - 20).to(torch.float32) / 128).unsqueeze(1)
        expert_output = semantic_x.float() * scale + offset_a + offset_b * pattern.unsqueeze(0)
        expected.add_(gate * expert_output)
    return expected


def _baseline_precision_axis() -> dict:
    return {
        "encoded_payload_valid": True,
        "scales_finite": None,
        "scales_positive": None,
        "dequantized_semantics": True,
        "saturation_count": 0,
        "saturation_rate": 0.0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "passed": True,
    }


def _precision_evidence(backend, problem, view, combined, expected_combined) -> dict:
    method = getattr(backend, "precision_evidence", None)
    if method is not None:
        evidence = method(problem, view)
        combine_axis = backend.communication_precision["combine"]
        if combine_axis["communication_format"] != "bf16":
            oracle_atol, oracle_rtol = _oracle_tolerances(backend)
            if combined.shape == expected_combined.shape and combined.numel():
                absolute = (combined.float() - expected_combined.float()).abs()
                max_abs_error = float(absolute.max().item())
                max_rel_error = max_abs_error / (
                    float(expected_combined.float().abs().max().item()) + 1e-6
                )
                tolerance = oracle_atol + oracle_rtol * expected_combined.float().abs()
                semantics = bool((absolute <= tolerance).all().item())
            elif combined.shape == expected_combined.shape:
                max_abs_error = max_rel_error = 0.0
                semantics = True
            else:
                max_abs_error = max_rel_error = 1e30
                semantics = False
            direction = evidence["combine"]
            direction.update({
                "dequantized_semantics": semantics,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
            })
            scale_ok = (
                direction["scales_finite"] is not False
                and direction["scales_positive"] is not False
            )
            direction["passed"] = bool(
                direction["encoded_payload_valid"]
                and semantics
                and scale_ok
                and direction["saturation_count"] == 0
            )
            evidence["passed"] = bool(
                evidence["dispatch"]["passed"] and direction["passed"]
            )
        return evidence
    profile_id = getattr(backend, "precision_profile_id", None)
    if profile_id != identity.V1_CONTROL_PRECISION_PROFILE:
        failed = _baseline_precision_axis()
        failed.update({"encoded_payload_valid": False, "dequantized_semantics": False,
                       "passed": False})
        return {"profile_id": profile_id, "dispatch": failed, "combine": dict(failed),
                "passed": False}
    return {
        "profile_id": profile_id,
        "dispatch": _baseline_precision_axis(),
        "combine": _baseline_precision_axis(),
        "passed": True,
    }


def _failed_precision_evidence(backend) -> dict:
    failed = _baseline_precision_axis()
    failed.update({"encoded_payload_valid": False, "dequantized_semantics": False,
                   "passed": False})
    return {
        "profile_id": getattr(backend, "precision_profile_id", None),
        "dispatch": failed,
        "combine": dict(failed),
        "passed": False,
    }


def aggregate_precision_evidence(evidence_by_rank: list[dict]) -> dict:
    """Collapse pre/post rank evidence without hiding any direction's worst observation."""
    records = [record[phase] for record in evidence_by_rank for phase in ("pre", "post")]
    profile_ids = {record["profile_id"] for record in records}
    if len(profile_ids) != 1:
        raise ValueError("precision evidence profiles differ across ranks or oracle passes")
    result = {"profile_id": profile_ids.pop()}
    for direction in ("dispatch", "combine"):
        axes = [record[direction] for record in records]
        rank_counts = []
        for rank_index in range(len(evidence_by_rank)):
            rank_counts.append(max(
                evidence_by_rank[rank_index][phase][direction]["saturation_count"]
                for phase in ("pre", "post")
            ))
        scale_finite = [axis["scales_finite"] for axis in axes]
        scale_positive = [axis["scales_positive"] for axis in axes]
        result[direction] = {
            "encoded_payload_valid": all(axis["encoded_payload_valid"] for axis in axes),
            "scales_finite": (
                None if all(value is None for value in scale_finite)
                else all(value is True for value in scale_finite)
            ),
            "scales_positive": (
                None if all(value is None for value in scale_positive)
                else all(value is True for value in scale_positive)
            ),
            "dequantized_semantics": all(axis["dequantized_semantics"] for axis in axes),
            "saturation_count": sum(rank_counts),
            "saturation_rate": max(axis["saturation_rate"] for axis in axes),
            "max_abs_error": max(axis["max_abs_error"] for axis in axes),
            "max_rel_error": max(axis["max_rel_error"] for axis in axes),
            "passed": all(axis["passed"] for axis in axes),
        }
    result["passed"] = all(result[direction]["passed"] for direction in ("dispatch", "combine"))
    return result


def _run_expert_packed_oracle(
    torch,
    routing,
    backend,
    problem,
    global_idx,
    global_weights,
    rank: int,
    experts_per_rank: int,
    seed: int,
):
    """Verify an expert-packed dispatch and native gate-weighted combine."""
    contract = LOW_LATENCY_ORACLE_CONTRACT
    oracle_atol, oracle_rtol = _oracle_tolerances(backend)
    handle = backend.dispatch(problem)
    torch.cuda.synchronize()
    try:
        packed = backend.inspect_expert_dispatch(problem, handle)
        view = expert_packed_dispatch_view(
            torch,
            packed.payload,
            packed.local_expert_counts,
            packed.source_info,
            packed.layout_range,
            rank=rank,
            tokens_per_rank=problem.T,
            experts_per_rank=experts_per_rank,
            world_size=backend.world_size,
        )
        decoded_source_ids = routing.decode_source_ids(view.payload, seed)
    except Exception as inspection_error:
        try:
            problem.recv_tokens = backend.recv_tokens(handle)
            backend.stage(problem, handle)
            backend.combine(problem, handle)
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            raise inspection_error from cleanup_error
        return {
            "_precision": _failed_precision_evidence(backend),
            "contract": contract,
            "passed": False,
            "ordering_contract": "adapter-inspection-failed",
            "order_sha256": None,
            "dispatch_sha256": None,
            "combine_weight_semantics": getattr(
                backend, "combine_weight_semantics", "undeclared"
            ),
            "receive_count": 0,
            "atol": oracle_atol,
            "max_absolute_error": None,
            "max_elementwise_relative_error": None,
            "max_relative_error": None,
            "max_weight_error": None,
            "rtol": oracle_rtol,
            "checks": {
                "combine_values": False,
                "counts": False,
                "metadata": False,
                "multiplicity": False,
                "payload": False,
                "source_set": False,
                "weights": False,
            },
        }

    device = problem.x.device
    world_size = backend.world_size
    total_experts = experts_per_rank * world_size
    global_idx_device = global_idx.to(device=device, dtype=torch.int64)
    global_weights_device = global_weights.to(device=device, dtype=torch.float32)
    source_grid = torch.arange(
        global_idx_device.shape[0], device=device, dtype=torch.int64
    ).unsqueeze(1).expand_as(global_idx_device)
    local_mask = (global_idx_device // experts_per_rank) == rank
    expected_sources = source_grid[local_mask]
    expected_experts = global_idx_device[local_mask]
    expected_pair_weights = global_weights_device[local_mask]

    receive_count = int(view.payload.shape[0])
    shape_ok = (
        view.payload.ndim == 2
        and view.source_ids.shape == (receive_count,)
        and view.expert_ids.shape == (receive_count,)
        and view.local_expert_counts.shape == (experts_per_rank,)
    )
    source_range = bool(
        receive_count == 0
        or (
            (view.source_ids >= 0)
            & (view.source_ids < global_idx_device.shape[0])
        ).all().item()
    )
    expected_payload = (
        routing.activations_for_source_ids(
            view.source_ids, problem.x.shape[1], seed, problem.x.dtype
        )
        if source_range
        else torch.empty_like(view.payload)
    )
    normalize_payload = getattr(backend, "oracle_dispatch_payload", None)
    if source_range and normalize_payload is not None:
        expected_payload = normalize_payload(expected_payload)
    payload_ok = bool(
        source_range
        and torch.equal(decoded_source_ids.to(torch.int64), view.source_ids)
        and torch.equal(view.payload, expected_payload)
    )

    actual_keys = view.source_ids * total_experts + view.expert_ids
    expected_keys = expected_sources * total_experts + expected_experts
    actual_order = torch.argsort(actual_keys, stable=True)
    expected_order = torch.argsort(expected_keys, stable=True)
    canonical_sources = view.source_ids.index_select(0, actual_order)
    canonical_experts = view.expert_ids.index_select(0, actual_order)
    canonical_expected_weights = expected_pair_weights.index_select(0, expected_order)
    expected_local_idx = global_idx_device[
        rank * problem.T:(rank + 1) * problem.T
    ]
    metadata_ok = bool(
        shape_ok
        and torch.equal(problem.topk_idx.to(torch.int64), expected_local_idx)
        and torch.equal(
            actual_keys.index_select(0, actual_order),
            expected_keys.index_select(0, expected_order),
        )
    )
    expected_counts = torch.bincount(
        expected_experts - rank * experts_per_rank, minlength=experts_per_rank
    )
    counts_ok = torch.equal(
        view.local_expert_counts.to(torch.int64), expected_counts.to(torch.int64)
    )
    actual_multiplicity = torch.bincount(
        view.source_ids, minlength=global_idx_device.shape[0]
    )
    expected_multiplicity = torch.bincount(
        expected_sources, minlength=global_idx_device.shape[0]
    )
    multiplicity_ok = torch.equal(actual_multiplicity, expected_multiplicity)
    source_set_ok = torch.equal(
        torch.sort(torch.unique(view.source_ids)).values,
        torch.sort(torch.unique(expected_sources)).values,
    )

    expected_local_weights = global_weights_device[
        rank * problem.T:(rank + 1) * problem.T
    ]
    if problem.topk_weights.shape == expected_local_weights.shape:
        max_weight_error = (
            float((problem.topk_weights.float() - expected_local_weights).abs().max().item())
            if expected_local_weights.numel()
            else 0.0
        )
    else:
        max_weight_error = None
    weights_ok = max_weight_error == 0.0
    ordering_contract = f"canonical-source-expert-v1/{view.ordering_contract}"
    order_sha256 = _tensor_sha256(canonical_sources, canonical_experts)
    dispatch_sha256 = _tensor_sha256(
        canonical_sources, canonical_experts, canonical_expected_weights
    )

    handle.oracle_local_expert_slots = view.local_expert_slots
    handle.oracle_packed_positions = view.packed_positions
    problem.recv_tokens = receive_count
    transformed = _expert_transform_expanded(torch, view.payload, view.expert_ids)
    view.combine_input = transformed
    combined = backend.combine_transformed(problem, handle, transformed)
    torch.cuda.synchronize()
    expected_combined = _expected_transformed_combine(torch, problem)
    if combined.shape == expected_combined.shape and combined.numel():
        absolute_error = (combined.float() - expected_combined).abs()
        max_absolute_error = float(absolute_error.max().item())
        max_relative_error = max_absolute_error / (
            float(expected_combined.abs().max().item()) + 1e-6
        )
        max_elementwise_relative_error = float(
            (absolute_error / expected_combined.abs().clamp_min(oracle_atol)).max().item()
        )
        combine_values_ok = bool(torch.allclose(
            combined.float(), expected_combined, rtol=oracle_rtol, atol=oracle_atol
        ))
    elif combined.shape == expected_combined.shape:
        max_absolute_error = 0.0
        max_elementwise_relative_error = 0.0
        max_relative_error = 0.0
        combine_values_ok = True
    else:
        max_absolute_error = None
        max_elementwise_relative_error = None
        max_relative_error = None
        combine_values_ok = False
    tolerance = oracle_rtol
    precision = _precision_evidence(backend, problem, view, combined, expected_combined)
    checks = {
        "combine_values": combine_values_ok,
        "counts": counts_ok,
        "metadata": metadata_ok,
        "multiplicity": multiplicity_ok,
        "payload": payload_ok,
        "source_set": source_set_ok,
        "weights": weights_ok,
    }
    return {
        "_precision": precision,
        "contract": contract,
        "passed": bool(
            all(checks.values())
            and precision["passed"]
            and ordering_contract
            and max_relative_error is not None
            and max_relative_error < tolerance
        ),
        "atol": oracle_atol,
        "combine_weight_semantics": backend.combine_weight_semantics,
        "ordering_contract": ordering_contract,
        "order_sha256": order_sha256,
        "dispatch_sha256": dispatch_sha256,
        "receive_count": receive_count,
        "max_absolute_error": max_absolute_error,
        "max_elementwise_relative_error": max_elementwise_relative_error,
        "max_relative_error": max_relative_error,
        "max_weight_error": max_weight_error,
        "rtol": oracle_rtol,
        "checks": checks,
    }


def _run_expert_oracle(
    torch,
    routing,
    backend,
    problem,
    global_idx,
    global_weights,
    rank: int,
    experts_per_rank: int,
    seed: int,
):
    """Verify one real dispatch/transform/combine without entering a timed region."""
    if getattr(backend, "oracle_layout", "token-rank") == "expert-packed":
        return _run_expert_packed_oracle(
            torch,
            routing,
            backend,
            problem,
            global_idx,
            global_weights,
            rank,
            experts_per_rank,
            seed,
        )
    oracle_atol, oracle_rtol = _oracle_tolerances(backend)
    handle = backend.dispatch(problem)
    torch.cuda.synchronize()
    try:
        view = backend.inspect_dispatch(problem, handle)
        source_ids = routing.decode_source_ids(view.payload, seed)
    except Exception as inspection_error:
        try:
            problem.recv_tokens = backend.recv_tokens(handle)
            backend.stage(problem, handle)
            backend.combine(problem, handle)
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            raise inspection_error from cleanup_error
        return {
            "_precision": _failed_precision_evidence(backend),
            "contract": ORACLE_CONTRACT,
            "passed": False,
            "ordering_contract": "adapter-inspection-failed",
            "order_sha256": None,
            "dispatch_sha256": None,
            "combine_weight_semantics": getattr(
                backend, "combine_weight_semantics", "undeclared"
            ),
            "receive_count": 0,
            "atol": oracle_atol,
            "max_absolute_error": None,
            "max_elementwise_relative_error": None,
            "max_relative_error": None,
            "max_weight_error": None,
            "rtol": oracle_rtol,
            "checks": {
                "combine_values": False,
                "counts": False,
                "metadata": False,
                "multiplicity": False,
                "payload": False,
                "source_set": False,
                "weights": False,
            },
        }

    receive_count = int(view.payload.shape[0])
    shape_ok = (
        view.payload.ndim == 2
        and view.expert_ids.shape == (receive_count, problem.topk_idx.shape[1])
        and view.weights.shape == view.expert_ids.shape
    )
    source_range = bool(
        receive_count == 0
        or ((source_ids >= 0) & (source_ids < global_idx.shape[0])).all().item()
    )
    if source_range:
        expected_idx = global_idx.to(problem.x.device).index_select(0, source_ids)
        expected_weights = global_weights.to(problem.x.device).index_select(0, source_ids)
        local = (expected_idx // experts_per_rank) == rank
        expected_ids = torch.where(local, expected_idx, torch.full_like(expected_idx, -1))
        expected_weights = expected_weights.masked_fill(~local, 0)
        expected_payload = routing.activations_for_source_ids(
            source_ids, problem.x.shape[1], seed, problem.x.dtype
        )
        normalize_payload = getattr(backend, "oracle_dispatch_payload", None)
        if normalize_payload is not None:
            expected_payload = normalize_payload(expected_payload)
    else:
        expected_ids = torch.full_like(view.expert_ids, -1)
        expected_weights = torch.zeros_like(view.weights)
        expected_payload = torch.empty_like(view.payload)
    actual_ids, actual_weights = _normalized_expert_metadata(
        torch, view.expert_ids, view.weights
    )
    expected_ids, expected_weights = _normalized_expert_metadata(
        torch, expected_ids, expected_weights
    )
    expected_sources = (
        ((global_idx // experts_per_rank) == rank).any(dim=1).nonzero(as_tuple=True)[0]
    ).to(problem.x.device)
    source_set_ok = (
        source_range
        and source_ids.numel() == torch.unique(source_ids).numel()
        and torch.equal(torch.sort(source_ids).values, expected_sources)
    )
    payload_ok = source_range and torch.equal(view.payload, expected_payload)
    metadata_ok = shape_ok and torch.equal(actual_ids, expected_ids)
    max_weight_error = (
        float((actual_weights - expected_weights).abs().max().item())
        if actual_weights.numel()
        else 0.0
    )
    weights_ok = max_weight_error == 0.0
    valid_expected = expected_ids >= 0
    expected_local = expected_ids[valid_expected] - rank * experts_per_rank
    expected_counts = torch.bincount(expected_local, minlength=experts_per_rank)
    counts_ok = torch.equal(
        view.local_expert_counts.to(torch.int64), expected_counts.to(torch.int64)
    )
    multiplicity_ok = torch.equal(
        (actual_ids >= 0).sum(dim=1), (expected_ids >= 0).sum(dim=1)
    )
    # Receive-slot assignment may use atomics and is not a semantic EP guarantee. Compare
    # pre/post dispatch evidence in canonical source-token order without changing the native path.
    canonical_order = torch.argsort(source_ids.to(torch.int64), stable=True)
    canonical_sources = source_ids.to(torch.int64).index_select(0, canonical_order)
    canonical_ids = actual_ids.to(torch.int64).index_select(0, canonical_order)
    canonical_weights = actual_weights.index_select(0, canonical_order)
    ordering_contract = f"canonical-source-id-v1/{view.ordering_contract}"
    order_sha256 = _tensor_sha256(canonical_sources)
    dispatch_sha256 = _tensor_sha256(
        canonical_sources, canonical_ids, canonical_weights
    )

    problem.recv_tokens = receive_count
    combine_weight_semantics = backend.combine_weight_semantics
    transformed = _expert_transform(
        torch, view.payload, actual_ids, actual_weights, combine_weight_semantics
    )
    view.combine_input = transformed
    combined = backend.combine_transformed(problem, handle, transformed)
    torch.cuda.synchronize()
    expected_combined = _expected_transformed_combine(torch, problem)
    if combined.shape == expected_combined.shape and combined.numel():
        absolute_error = (combined.float() - expected_combined).abs()
        max_absolute_error = float(absolute_error.max().item())
        max_relative_error = max_absolute_error / (
            float(expected_combined.abs().max().item()) + 1e-6
        )
        max_elementwise_relative_error = float(
            (absolute_error / expected_combined.abs().clamp_min(oracle_atol)).max().item()
        )
        combine_values_ok = bool(torch.allclose(
            combined.float(), expected_combined, rtol=oracle_rtol, atol=oracle_atol
        ))
    elif combined.shape == expected_combined.shape:
        max_absolute_error = 0.0
        max_elementwise_relative_error = 0.0
        max_relative_error = 0.0
        combine_values_ok = True
    else:
        max_absolute_error = None
        max_elementwise_relative_error = None
        max_relative_error = None
        combine_values_ok = False
    tolerance = oracle_rtol
    precision = _precision_evidence(backend, problem, view, combined, expected_combined)
    checks = {
        "combine_values": combine_values_ok,
        "counts": counts_ok,
        "metadata": metadata_ok,
        "multiplicity": multiplicity_ok,
        "payload": payload_ok,
        "source_set": source_set_ok,
        "weights": weights_ok,
    }
    return {
        "_precision": precision,
        "contract": ORACLE_CONTRACT,
        "passed": bool(
            all(checks.values())
            and precision["passed"]
            and ordering_contract
            and max_relative_error is not None
            and max_relative_error < tolerance
        ),
        "atol": oracle_atol,
        "combine_weight_semantics": combine_weight_semantics,
        "ordering_contract": ordering_contract,
        "order_sha256": order_sha256,
        "dispatch_sha256": dispatch_sha256,
        "receive_count": receive_count,
        "max_absolute_error": max_absolute_error,
        "max_elementwise_relative_error": max_elementwise_relative_error,
        "max_relative_error": max_relative_error,
        "max_weight_error": max_weight_error,
        "rtol": oracle_rtol,
        "checks": checks,
    }


def _histogram(xs: list[float], nbins: int = 40) -> dict:
    """Compact equal-width summary of the exact private cross-rank-max samples."""
    if not xs:
        return {"n": 0}
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {"n": len(xs), "min": lo, "max": hi, "bins": nbins, "counts": [len(xs)]}
    counts = [0] * nbins
    span = hi - lo
    for x in xs:
        b = min(nbins - 1, int((x - lo) / span * nbins))
        counts[b] += 1
    return {"n": len(xs), "min": round(lo, 3), "max": round(hi, 3), "bins": nbins, "counts": counts}


def _derive_publication_status(v: dict) -> str:
    """Classify raw attempts; only the isolated coverage publisher may promote evidence."""
    if v["execution_status"] != "complete":
        return "failed"
    if v["semantic_correctness"] != "pass" or v["measurement_conformance"] != "conformant" \
       or v["workload_identity"] == "inconsistent":
        return "invalid"
    # Per-case producers cannot prove exact matrix coverage, repeat stability, or controlled
    # cohorts. Keep even sound attempts diagnostic until the isolated publisher validates them.
    return "diagnostic"


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    mode = getattr(args, "mode", "normal")
    requested_precision = getattr(args, "precision_profile", "") or None
    resolved_precision_id = requested_precision or identity.V1_CONTROL_PRECISION_PROFILE
    try:
        profile_case = {"mode": mode}
        if requested_precision is not None:
            profile_case["precision_profile"] = requested_precision
        case_profile = identity.profile_for_case(profile_case)
        communication_precision = identity.precision_profile(resolved_precision_id)
    except identity.IdentityError as exc:
        if rank == 0:
            print(f"ERROR: {exc}")
        return 2
    sampling_error = sampling_contract_error(args.iters, args.trials, args.warmup)
    if sampling_error:
        if rank == 0:
            print(f"ERROR: {sampling_error}")
        return 2
    import routing  # torch-based; imported lazily so the module byte-compiles without torch
    import eplb     # stdlib planner + torch remap (the EPLB transform)

    ep_size = world_size
    # EPLB (if on): run_ep.py already bumped args.experts to the PHYSICAL count and stashed the
    # logical count, so experts_per_rank below is physical. The trace is built over LOGICAL
    # experts then remapped to physical (build_trace), so the whole sweep runs over the
    # balanced physical placement with no adapter change.
    eplb_on = getattr(args, "eplb", False)
    num_logical = getattr(args, "num_logical_experts", args.experts)
    if args.experts % ep_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide ep_size ({ep_size})")
        return 2
    experts_per_rank = args.experts // ep_size
    if getattr(backend, "mode", None) != mode:
        if rank == 0:
            print(f"ERROR: backend mode {getattr(backend, 'mode', None)!r} != {mode!r}")
        return 2
    if (
        getattr(backend, "precision_profile_id", None) != resolved_precision_id
        or getattr(backend, "communication_precision", None) != communication_precision
    ):
        if rank == 0:
            print("ERROR: backend did not realize the requested communication precision")
        return 2
    expected_weight_semantics = (
        "gate-weighted-sum"
        if case_profile["combine_semantics"] == "gate-weighted"
        else "unweighted-rank-sum"
    )
    if getattr(backend, "combine_weight_semantics", None) != expected_weight_semantics:
        if rank == 0:
            print(
                f"ERROR: {mode} requires combine semantics {expected_weight_semantics}"
            )
        return 2
    if mode == LOW_LATENCY_MODE and (
        args.phase != "decode"
        or getattr(backend, "oracle_layout", None) != "expert-packed"
        or getattr(backend, "payload_unit", None) != "token-expert"
    ):
        if rank == 0:
            print("ERROR: low-latency requires decode expert-packed token-expert execution")
        return 2

    cap = backend.buffer_cap(args)
    conditioning_ladder = CONDITIONING_LADDERS[args.phase]
    if cap is not None and cap < conditioning_ladder[-1]:
        if rank == 0:
            print(f"ERROR: {backend.name} buffer cap {cap} cannot run the v1 conditioning ladder")
        return 2
    ladder, dropped = token_ladder(args.tokens_ladder, args.phase, cap)
    if rank == 0 and dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed {backend.name} buffer cap {cap} "
              f"(hidden={args.hidden}); not silently truncated.")
    if not ladder:
        if rank == 0:
            print(f"ERROR: empty token ladder (phase={args.phase}, cap={cap})")
        return 2
    MAX, MIN, SUM = dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.SUM

    # EPLB plan (once): estimate logical load from the global logical trace at the largest
    # ladder T (most samples), then replicate+place. Held fixed across all T (as real EPLB
    # plans from an observed load estimate). build_trace builds the LOGICAL trace and remaps
    # to physical when the plan is present; otherwise it's the identity (logical == physical).
    eplb_plan = None
    eplb_calibration = None
    if eplb_on:
        calibration_id, calibration_checksums, calibration_rows, _ = (
            workload_contract.canonical_eplb_calibration_member(
                args.routing,
                args.hidden,
                args.topk,
                num_logical,
                ep_size,
                EPLB_REFERENCE_TOKENS_PER_RANK,
                args.seed,
            )
        )
        ref_idx = torch.tensor(
            calibration_rows,
            dtype=torch.int64,
        )
        eplb_calibration = {
            "token_offset": workload_contract.EPLB_CALIBRATION_TOKEN_OFFSET,
            "trace_sha256": calibration_checksums["trace"],
            "window": workload_contract.EPLB_CALIBRATION_WINDOW,
            "workload_id": calibration_id,
        }
        if ref_idx.shape != (
            EPLB_REFERENCE_TOKENS_PER_RANK * ep_size,
            args.topk,
        ):
            raise RuntimeError("EPLB calibration trace dimensions differ from the contract")
        load = torch.bincount(ref_idx.reshape(-1), minlength=num_logical).float().tolist()
        eplb_plan = eplb.build_plan(load, args.experts, ep_size)
        if rank == 0:
            print(f"NOTE: EPLB {num_logical}->{args.experts} experts ({ep_size}x{experts_per_rank}); "
                  f"per-rank load imbalance {eplb_plan['imbalance_before']:.2f}x -> "
                  f"{eplb_plan['imbalance_after']:.2f}x; {eplb_plan['replicated_experts']} experts "
                  f"replicated (hottest {eplb_plan['max_replicas']}x)")

    canonical = bool(getattr(args, "workload_dir", ""))
    loaded_workload_ids, loaded_checksums = [], {}
    if canonical:
        import workload as _wl

    def build_trace(gt):
        # canonical: load pre-serialized trace bytes (verified by checksum) so this run is
        # provably the SAME workload as any other consuming the same files. else: seeded gen.
        if canonical:
            wid = _wl.compute_workload_id(
                args.routing, args.hidden, args.topk, num_logical, ep_size, gt, args.seed
            )
            idx_np, w_np, man = _wl.load_workload(os.path.join(args.workload_dir, f"{wid}.npz"), verify=True)
            idx_l = torch.from_numpy(idx_np).to(torch.int64)
            w = torch.from_numpy(w_np).to(torch.float32)
            if wid not in loaded_workload_ids:
                loaded_workload_ids.append(wid)
                loaded_checksums[wid] = man.get("checksums")
        else:
            idx_l, w = routing.build_global_routing(
                gt, num_logical, args.topk, args.routing, args.seed
            )
        return (eplb.remap_idx(idx_l, eplb_plan) if eplb_plan is not None else idx_l), w

    # Fabric/clock warm-up BEFORE any timed point (review: H200 had an anomalous cold
    # first point and a 40% decode-vs-prefill mismatch at the shared T=128). Gradually
    # ramp through the small ladder shapes untimed — warms clocks/fabric for everyone
    # and is also cold-jump-safe for MoRI.
    def warm_roundtrips(problem, count):
        for _ in range(count):
            handle = backend.dispatch(problem)
            if not hasattr(problem, "recv_tokens"):
                # Dynamic receive cardinality is stable for this fixed routing trace. Cache it
                # during untimed conditioning so adapters never read a device scalar in timing.
                problem.recv_tokens = backend.recv_tokens(handle)
            backend.stage(problem, handle)
            backend.combine(problem, handle)
            torch.cuda.synchronize()

    for wt in conditioning_ladder:
        # Warm-only shapes need not have canonical manifests: they are never measured or emitted.
        wi, ww = routing.build_global_routing(
            wt * ep_size, num_logical, args.topk, args.routing, args.seed,
        )
        if eplb_plan is not None:
            wi = eplb.remap_idx(wi, eplb_plan)
        wsi, wsw = routing.rank_slice(wi, ww, rank, wt)
        wx = routing.rank_activations(wt, args.hidden, args.seed, rank, device, torch.bfloat16)
        wp = backend.make_problem(wt, wsi.to(device), wsw.to(device), wx)
        warm_roundtrips(wp, CONDITIONING_ROUNDS_PER_SHAPE)
    torch.cuda.synchronize()
    dist.barrier()
    # Setup may materialize deferred provenance such as DeepEP V2 JIT CUBINs.
    # Resolve it after conditioning but before correctness or timed measurements.
    capture_deferred_provenance = getattr(backend, "capture_deferred_provenance", None)
    if capture_deferred_provenance is not None:
        capture_deferred_provenance()
    provenance_issues = contracts.backend_provenance_issues(
        backend.name, backend.backend_provenance
    )
    if provenance_issues:
        if rank == 0:
            print(
                f"ERROR: unpinned provenance {provenance_issues} "
                f"in {backend.backend_provenance}"
            )
        return 4
    # ---- Pass 1: build each deterministic problem and run the expert oracle. ----
    problems, gate, gts, global_traces, input_snapshots = {}, {}, {}, {}, {}
    routing_hashes = set()
    for T in ladder:
        counts = [T] * ep_size
        gt = T * ep_size
        gts[T] = gt
        idx_g, w_g = build_trace(gt)
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank, weights=w_g)
        gpn = args.gpus_per_node or ep_size
        rstats["locality"] = routing.routing_locality(idx_g, experts_per_rank, ep_size, max(1, T),
                                                      gpn, args.scale_up_domain or None)
        rstats["source_token_stats"] = _stats_vec(counts)
        routing_hashes.add(rstats["routing_hash"])
        my_off, my_cnt = rank * T, T
        idx_s = idx_g[my_off:my_off + my_cnt].contiguous()
        w_s = w_g[my_off:my_off + my_cnt].contiguous()
        x = routing.rank_activations(my_cnt, args.hidden, args.seed, rank, device, torch.bfloat16)
        problem = backend.make_problem(my_cnt, idx_s.to(device), w_s.to(device), x)
        input_snapshots[T] = (
            problem.x.clone(), problem.topk_idx.clone(), problem.topk_weights.clone()
        )
        oracle = _run_expert_oracle(
            torch, routing, backend, problem, idx_g, w_g, rank, experts_per_rank,
            args.seed,
        )
        precision_pre = oracle.pop("_precision")
        before_x, before_idx, before_weights = input_snapshots[T]
        pre_input_unchanged = (
            torch.equal(problem.x, before_x)
            and torch.equal(problem.topk_idx, before_idx)
            and torch.equal(problem.topk_weights, before_weights)
        )
        problems[T] = problem
        global_traces[T] = (idx_g, w_g)
        gate[T] = {
            "rstats": rstats,
            "recv_local": oracle["receive_count"],
            "max_rel": oracle["max_relative_error"] or 0.0,
            "local_ok": int(oracle["passed"]),
            "oracle_pre": oracle,
            "precision_pre": precision_pre,
            "pre_input_unchanged": pre_input_unchanged,
        }

    # ---- Pass 2: every backend uses the same ascending point order and conditioning ramp.
    # Per-iteration cross-rank MAX samples are pooled across trials. ----
    disp_pool = {T: [] for T in ladder}     # pooled per-iteration cross-rank MAX (dispatch)
    stage_pool = {T: [] for T in ladder}    # measured only when stage launches device work
    comb_pool = {T: [] for T in ladder}     # ... combine
    rt_pool = {T: [] for T in ladder}       # independently measured round trip
    disp_trials = {T: [] for T in ladder}
    stage_trials = {T: [] for T in ladder}
    comb_trials = {T: [] for T in ladder}
    rt_trials = {T: [] for T in ladder}
    stage_device_work = getattr(backend, "stage_device_work", False)
    if type(stage_device_work) is not bool:
        raise ValueError("backend.stage_device_work must be a boolean")
    dispatch_needs_cleanup = getattr(backend, "dispatch_needs_combine_cleanup", False)
    if type(dispatch_needs_cleanup) is not bool:
        raise ValueError("backend.dispatch_needs_combine_cleanup must be a boolean")
    order_identity = args.case_id or _sha256_json({
        "backend": backend.name,
        "ep_size": ep_size,
        "mode": mode,
        "phase": args.phase,
        "precision_profile": resolved_precision_id,
        "runner": args.runner,
        "suite": args.suite,
    })
    observed_component_orders = []
    for trial_index in range(args.trials):
        order = qualification_order(
            list(ladder), args.qualification_index, trial_index,
            identity_key=f"{order_identity}:tokens",
        )
        for T in order:
            problem = problems[T]
            # Stateful paired APIs may expose only a measured round trip.
            # Do not synthesize component latency from that measurement.
            roundtrip_only = getattr(backend, "roundtrip_only", False)

            def rt_once(p=problem):
                hh = backend.dispatch(p)
                backend.stage(p, hh)
                return backend.combine(p, hh)

            available_components = ["roundtrip"]
            if not roundtrip_only:
                available_components.extend(["dispatch", "combine"])
                if stage_device_work:
                    available_components.append("stage")
            component_order = qualification_order(
                available_components,
                args.qualification_index,
                trial_index,
                identity_key=f"{order_identity}:components:{T}",
            )
            observed_component_orders.append({
                "components": component_order,
                "tokens_per_rank": T,
                "trial_index": trial_index,
            })
            measured = {name: [] for name in ("dispatch", "stage", "combine", "roundtrip")}

            def prep_stage(p=problem):
                return backend.dispatch(p)

            def prep_combine(p=problem):
                hh = backend.dispatch(p)
                backend.stage(p, hh)
                return hh

            def finish_dispatch(hh, p=problem):
                backend.stage(p, hh)
                backend.combine(p, hh)

            for component_name in component_order:
                # Every measured component receives the same 32 synchronized full-roundtrip
                # warmups immediately before its timed trial.
                warm_roundtrips(problem, args.warmup)
                if component_name == "roundtrip":
                    measured[component_name] = time_us(
                        torch, lambda p=problem: rt_once(p), 0, args.iters
                    )
                elif component_name == "dispatch":
                    measured[component_name] = time_us(
                        torch, lambda p=problem: backend.dispatch(p), 0, args.iters,
                        post=finish_dispatch if dispatch_needs_cleanup else None,
                    )
                elif component_name == "stage":
                    measured[component_name] = time_us(
                        torch,
                        lambda hh, p=problem: backend.stage(p, hh),
                        0,
                        args.iters,
                        pre=prep_stage,
                    )
                elif component_name == "combine":
                    if backend.combine_needs_redispatch:
                        measured[component_name] = time_us(
                            torch,
                            lambda hh, p=problem: backend.combine(p, hh),
                            0,
                            args.iters,
                            pre=prep_combine,
                        )
                    else:
                        hh = prep_combine()
                        torch.cuda.synchronize()
                        measured[component_name] = time_us(
                            torch,
                            lambda p=problem, hx=hh: backend.combine(p, hx),
                            0,
                            args.iters,
                        )
                else:  # pragma: no cover - generated from the fixed list above
                    raise RuntimeError(f"unknown timed component {component_name!r}")
            disp_iters = measured["dispatch"]
            stage_iters = measured["stage"]
            comb_iters = measured["combine"]
            rt_iters = measured["roundtrip"]
            # per-iteration cross-rank MAX (the distributed-op latency per iter), pooled.
            if disp_iters:
                reduced_dispatch = _reduce_vec(torch, dist, device, disp_iters, MAX)
                reduced_combine = _reduce_vec(torch, dist, device, comb_iters, MAX)
                disp_trials[T].append(reduced_dispatch)
                comb_trials[T].append(reduced_combine)
                disp_pool[T] += reduced_dispatch
                comb_pool[T] += reduced_combine
            if stage_iters:
                reduced_stage = _reduce_vec(torch, dist, device, stage_iters, MAX)
                stage_trials[T].append(reduced_stage)
                stage_pool[T] += reduced_stage
            reduced_roundtrip = _reduce_vec(torch, dist, device, rt_iters, MAX)
            rt_trials[T].append(reduced_roundtrip)
            rt_pool[T] += reduced_roundtrip

    # ---- Pass 3: prove timed inputs were immutable and repeat the full oracle. ----
    for T in ladder:
        problem = problems[T]
        before_x, before_idx, before_weights = input_snapshots[T]
        input_unchanged = gate[T]["pre_input_unchanged"] and (
            torch.equal(problem.x, before_x)
            and torch.equal(problem.topk_idx, before_idx)
            and torch.equal(problem.topk_weights, before_weights)
        )
        idx_g, w_g = global_traces[T]
        post = _run_expert_oracle(
            torch, routing, backend, problem, idx_g, w_g, rank, experts_per_rank,
            args.seed,
        )
        precision_post = post.pop("_precision")
        pre = gate[T]["oracle_pre"]
        order_stable = (
            pre["ordering_contract"] == post["ordering_contract"]
            and pre["order_sha256"] == post["order_sha256"]
            and pre["dispatch_sha256"] == post["dispatch_sha256"]
        )
        gate[T].update({
            "input_unchanged": input_unchanged,
            "local_ok": int(pre["passed"] and post["passed"] and input_unchanged and order_stable),
            "max_rel": max(pre["max_relative_error"] or 0.0, post["max_relative_error"] or 0.0),
            "oracle_post": post,
            "precision_post": precision_post,
            "order_stable": order_stable,
        })

    # ---- Pass 4: percentiles (p50/p90/p95/p99, nearest-rank) from pooled samples + bytes + row ----
    def pcts(xs):
        return ({"p50": percentile(xs, 50), "p90": percentile(xs, 90),
                 "p95": percentile(xs, 95), "p99": percentile(xs, 99)} if xs else None)

    def component(percentiles, count, *, derived=False):
        if percentiles is None:
            return {"availability": "unavailable", "origin": None,
                    "percentiles_us": None, "sample_count": 0}
        return {
            "availability": "derived" if derived else "measured",
            "origin": "derived-percentile-sum" if derived else "measured",
            "percentiles_us": percentiles,
            "sample_count": 0 if derived else count,
        }
    rows = []
    all_anomalies = []
    thr_rt = 3.0
    for T in ladder:
        gt = gts[T]
        g = gate[T]
        rstats = g["rstats"]
        d, s, c, rt = disp_pool[T], stage_pool[T], comb_pool[T], rt_pool[T]
        dp, sp, cp, rtp = pcts(d), pcts(s), pcts(c), pcts(rt)
        # isolated_sum = SUM of the isolated dispatch+stage+combine percentiles. Stage contributes
        # zero when it is explicitly not applicable. This is NOT a measured chained operation
        # (can't reveal shared sync / launch amortization / overlap) — do NOT use for throughput
        # or SLO capacity. The MEASURED round trip (rtp) is the real chained latency.
        isum = (
            {key: dp[key] + (sp[key] if sp is not None else 0.0) + cp[key] for key in dp}
            if dp and cp else None
        )
        recv_total = _reduce_int(torch, dist, device, g["recv_local"], SUM)
        recv_max = _reduce_int(torch, dist, device, g["recv_local"], MAX)
        recv_min = _reduce_int(torch, dist, device, g["recv_local"], MIN)
        global_ok = _reduce_int(torch, dist, device, g["local_ok"], MIN)
        max_rel = _reduce_vec(torch, dist, device, [g["max_rel"]], MAX)[0]
        point_ok = bool(global_ok) and recv_total > 0
        rank_evidence = [None] * world_size
        dist.all_gather_object(
            rank_evidence,
            {
                "input_unchanged": g["input_unchanged"],
                "order_stable": g["order_stable"],
                "post_timing": g["oracle_post"],
                "pre_timing": g["oracle_pre"],
                "rank": rank,
            },
        )
        precision_rank_evidence = [None] * world_size
        dist.all_gather_object(
            precision_rank_evidence,
            {"pre": g["precision_pre"], "post": g["precision_post"]},
        )
        precision_evidence = aggregate_precision_evidence(precision_rank_evidence)
        # Canonical LOGICAL payload byte contracts (from the routing trace, NOT backend recv
        # tensors): token-rank = one copy per unique (token,dest-rank); token-expert = one copy
        # per routed (token,expert). routed_copies = token-rank copies; gt*topk = token-expert.
        token_rank_copies = rstats["routed_copies"]
        logical_copies = (
            sum(rstats["expert_assignments_per_rank"])
            if case_profile["payload_unit"] == "token-expert"
            else token_rank_copies
        )
        H = args.hidden
        throughput = {
            percentile_name: gt / (latency_us * 1e-6)
            for percentile_name, latency_us in rtp.items()
        }
        dispatch_bytes = precision_byte_provenance(
            communication_precision["dispatch"], logical_copies, H
        )
        combine_bytes = precision_byte_provenance(
            communication_precision["combine"], logical_copies, H
        )
        stage_bytes = {
            "accounting_contract": "activation-data-plus-scales-v1",
            "activation_data_bytes": 0,
            "scale_bytes": 0,
            "total_logical_bytes": 0,
        }
        roundtrip_bytes = {
            "accounting_contract": "activation-data-plus-scales-v1",
            **{
                field: dispatch_bytes[field] + combine_bytes[field]
                for field in (
                    "activation_data_bytes", "scale_bytes", "total_logical_bytes"
                )
            },
        }
        # Contract-level anomalies are attached to the row and rolled into validity.
        #   roundtrip_gt_isolated_sum: measured RT p99 >> Σ(isolated dispatch+combine) p99.
        #   roundtrip_lt_component_floor: measured RT p50 < max(dispatch,combine) p50 — a chained
        #     op can't finish faster than its slowest required component (sync semantics violated).
        row_anoms = []
        if isum and isum["p99"] > 0 and rtp["p99"] > thr_rt * isum["p99"]:
            row_anoms.append({"type": "roundtrip_gt_isolated_sum", "T": T,
                              "roundtrip_p99": round(rtp["p99"], 2), "isolated_sum_p99": round(isum["p99"], 2),
                              "ratio": round(rtp["p99"] / isum["p99"], 2), "threshold": thr_rt})
        floor = (
            max(dp["p50"], cp["p50"], sp["p50"] if sp is not None else 0.0)
            if dp and cp else None
        )
        if floor and rtp["p50"] > 0 and rtp["p50"] < 0.95 * floor:
            row_anoms.append({"type": "roundtrip_lt_component_floor", "T": T,
                              "roundtrip_p50": round(rtp["p50"], 2), "component_floor_p50": round(floor, 2)})
        all_anomalies.extend(row_anoms)
        rows.append({
            "anomalies": row_anoms,
            "components": {
                "combine": component(cp, len(c)),
                "dispatch": component(dp, len(d)),
                "isolated_sum": component(isum, 0, derived=True),
                "roundtrip": component(rtp, len(rt)),
                "stage": component(sp, len(s)),
            },
            "correctness": {
                "contract": case_profile["oracle_contract"],
                "max_relative_error": max_rel,
                "passed": point_ok,
                "precision": precision_evidence,
                "rank_evidence": rank_evidence,
                "scope": case_profile["correctness_scope"],
            },
            "global_tokens": gt,
            "byte_provenance": {
                "combine": combine_bytes,
                "dispatch": dispatch_bytes,
                "roundtrip": roundtrip_bytes,
                "stage": stage_bytes,
            },
            "receive": {
                "max": recv_max,
                "mean": recv_total / world_size,
                "min": recv_min,
                "total": recv_total,
            },
            "routing": {
                "empty_expert_count": rstats["empty_expert_count"],
                "empty_rank_count": rstats["empty_rank_count"],
                "expert_assignment_rank_cv": rstats["expert_assignment_rank_cv"],
                "expert_assignments_per_rank": rstats["expert_assignments_per_rank"],
                "expert_load_cv": rstats["expert_load_cv"],
                "expert_load_max": rstats["expert_load_max"],
                "expert_load_mean": rstats["expert_load_mean"],
                "expert_load_min": rstats["expert_load_min"],
                "fanout_histogram": rstats["fanout_hist"],
                "fanout_max": rstats["fanout_max"],
                "fanout_mean": rstats["fanout_mean"],
                "fanout_min": rstats["fanout_min"],
                "hash": rstats["routing_hash"],
                "hotspot_ratio": rstats["hotspot_ratio"],
                "locality": rstats.get("locality"),
                "payload_copies_per_rank": rstats["payload_copies_per_rank"],
                "payload_rank_cv": rstats["payload_rank_cv"],
                "routed_copies": rstats["routed_copies"],
                "source_token_stats": rstats.get("source_token_stats"),
            },
            "sample_histograms": {
                "dispatch": _histogram(d) if d else None,
                "stage": _histogram(s) if s else None,
                "combine": _histogram(c) if c else None,
                "roundtrip": _histogram(rt),
            },
            "token_rate_at_latency_percentile": throughput,
            "tokens_per_rank": T,
        })
        if rank == 0:
            component_log = (f"disp p50/p99={dp['p50']:7.1f}/{dp['p99']:7.1f} "
                             f"comb {cp['p50']:6.1f}/{cp['p99']:6.1f} " if dp and cp
                             else "components=unavailable ")
            print(f"  T={T:<5} {component_log}"
                  f"RT p50/p99={rtp['p50']:7.1f}/{rtp['p99']:7.1f}us n={len(rt)} fanout={rstats['fanout_mean']:.2f} "
                  f"recv[min/mean/max]={recv_min}/{recv_total // world_size}/{recv_max} "
                  f"correct={point_ok}")

    # Cross-rank workload-identity proof: every rank must have built the SAME global routing
    # (one hash per T here); confirm all ranks agree by hashing the per-T hash set and
    # MIN/MAX-reducing it — a mismatch means NVIDIA and AMD did NOT run identical routing.
    trace_sig = hashlib.sha256("|".join(sorted(routing_hashes)).encode()).hexdigest()
    routing_consistent = _same_hash_across_ranks(torch, dist, device, trace_sig)

    # Capture again after correctness and timing so no lazily generated kernel can escape
    # the implementation identity recorded in the artifact.
    if capture_deferred_provenance is not None:
        capture_deferred_provenance()

    # status=valid requires correctness AND a proven-identical routing trace across ranks.
    all_ok = bool(rows) and all(r["correctness"]["passed"] for r in rows) and routing_consistent

    # Adapters never self-label official; status is derived from these gates.
    prov = backend.backend_provenance
    allocation_stratum_sha256 = getattr(args, "allocation_stratum_sha256", None)
    provenance_complete = contracts.provenance_complete(
        prov,
        backend.name,
        getattr(args, "git_run", None),
        allocation_stratum_sha256=allocation_stratum_sha256,
        image_digest=getattr(args, "image_digest", None),
        image_verified=getattr(args, "image_digest_verified", False),
        squash_sha256=getattr(args, "squash_sha256", None),
    )
    resource_profile = contracts.project_resource_profile(prov)
    resource_conformance = resource_profile["conformance_class"]
    # record the canonical workload identity consumed (one trace per T -> set of ids/checksums).
    if canonical and loaded_workload_ids:
        args.workload_id = identity.workload_id(
            {
                "members": [
                    {"checksums": loaded_checksums[member], "workload_id": member}
                    for member in sorted(loaded_workload_ids)
                ]
            }
        )
        args.workload_members = sorted(loaded_workload_ids)
        args.workload_checksums = loaded_checksums
    canonical_workload = bool(getattr(args, "workload_id", None))
    activation_identity = workload_contract.compute_activation_identity(args.seed, args.hidden)
    # EPLB identity covers replica placement, not only counts.
    eplb_mapping_hash = None
    if eplb_plan is not None:
        eplb_mapping_hash = eplb.mapping_hash(eplb_plan)
    anomaly_free = len(all_anomalies) == 0
    validity = {
        "execution_status": "complete" if rows else "failed",
        "semantic_correctness": (
            "pass" if rows and all(r["correctness"]["passed"] for r in rows) else "fail"
        ),
        "workload_identity": "consistent-across-ranks" if routing_consistent else "inconsistent",
        "workload_source": "canonical-serialized" if canonical_workload else "seeded-runtime",
        "measurement_conformance": "conformant",   # run_ep gate rejects nonconformant pre-run
        "sampling_conformance": "conformant",      # fixed-512-v1 gate rejects any other profile
        "resource_conformance": resource_conformance,
        "provenance_complete": provenance_complete,
        # anomaly-free unless a contract-level timing anomaly fired (then diagnostic, see above).
        "anomaly_free": anomaly_free,
    }
    publication_status = _derive_publication_status(validity)

    shape = {  # FIXED line identity (no T, no per-backend resource knobs)
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank,
        "precision_profile": resolved_precision_id,
        "dispatch_precision": communication_precision["dispatch"],
        "combine_precision": communication_precision["combine"],
        "routing": args.routing, "eplb": bool(eplb_plan), "num_logical_experts": num_logical,
        # V2 is reserved for the PR #605 ElasticBuffer adapter; package versions never imply it.
        "kernel_gen": kernel_generation(backend),
        "activation_profile": ACTIVATION_PROFILE,
    }
    generated_at = args.timestamp or _dt.datetime.now().astimezone().isoformat()
    realized_placement = getattr(args, "realized_placement", None)
    nodes = (
        realized_placement["nodes"]
        if realized_placement is not None
        else int(os.environ.get("SLURM_NNODES", "1"))
    )
    scheduled_case = {
            "backend": backend.name,
            "canonical": canonical,
            "eplb": bool(eplb_plan),
            "ep": ep_size,
            "experts": num_logical,
            "gpus_per_node": args.gpus_per_node or ep_size,
            "hidden": args.hidden,
            "ladder": " ".join(map(str, ladder)),
            "mode": mode,
            "nodes": nodes,
            "phase": args.phase,
            "required_publication": args.required_publication or "diagnostic",
            "routing": args.routing,
            "samples_per_point": TIMED_SAMPLES_PER_POINT,
            "scale_up_domain": args.scale_up_domain or (args.gpus_per_node or ep_size),
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "suite": args.suite or "manual",
            "timing": f"{args.iters}:{args.trials}:{args.warmup}",
            "topk": args.topk,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "warmup_semantics": WARMUP_SEMANTICS,
            "workload": args.workload_name or "manual",
    }
    if requested_precision is not None:
        scheduled_case["precision_profile"] = requested_precision
    case_factors = {
        "case": scheduled_case,
        "profile": case_profile,
        "sku": args.runner,
    }
    computed_case_id = identity.digest("case", case_factors)
    if args.case_id and args.case_id != computed_case_id:
        raise ValueError(
            f"scheduled case ID does not match realized factors: {args.case_id} != {computed_case_id}"
        )
    case_identifier = args.case_id or computed_case_id
    git_run = getattr(args, "git_run", None) or {}
    allocation_factors = {
        "artifact": git_run.get("artifact"),
        "execution_id": getattr(args, "allocation_execution_id", None),
        "job": git_run.get("job"),
        "qualification_index": args.qualification_index,
        "repo": git_run.get("repo"),
        "run_attempt": git_run.get("run_attempt"),
        "run_id": git_run.get("run_id"),
        "runner": args.runner,
        "source_sha": git_run.get("source_sha"),
    }
    allocation_identifier = identity.allocation_id(allocation_factors)
    try:
        attempt_ordinal = int(os.environ.get("CX_ATTEMPT_ID", "1"))
    except ValueError:
        attempt_ordinal = 0
    if attempt_ordinal <= 0:
        raise ValueError("CX_ATTEMPT_ID must be a positive integer")
    attempt_identifier = identity.attempt_id(
        allocation=allocation_identifier, case=case_identifier, ordinal=attempt_ordinal
    )
    runtime_fingerprint = getattr(args, "runtime_fingerprint", None) or {}
    implementation_contract = {
        "kernel_generation": kernel_generation(backend),
        "name": backend.name,
        "provenance": _series_provenance(backend.backend_provenance),
        "resource_profile": resource_profile,
    }
    public_config = contracts.public_series_config(
        kernel_generation=implementation_contract["kernel_generation"],
        provenance=backend.backend_provenance,
        resource_profile=resource_profile,
        resource_mode=args.resource_mode,
        device_product=getattr(args, "runtime_device_product", None),
    )
    series_factors = {
        "backend": backend.name,
        "implementation_contract_sha256": _sha256_json(implementation_contract),
        "public_config_sha256": contracts.public_series_config_sha256(public_config),
        "routing_control_sha256": contracts.routing_implementation_control_sha256(
            implementation_contract
        ),
        "case_id": case_identifier,
        "image_digest": getattr(args, "image_digest", None),
        "runtime_fingerprint_sha256": _sha256_json(runtime_fingerprint),
        "source_sha": git_run.get("source_sha"),
        "squash_sha256": getattr(args, "squash_sha256", None),
        "workload_id": getattr(args, "workload_id", None) or trace_sig,
    }
    series_identifier = identity.series_id(series_factors)

    sample_points = []
    for row in rows:
        token_count = row["tokens_per_rank"]

        sample_point = {
            "components": {
                "combine": sampled_component_evidence(comb_trials[token_count]),
                "dispatch": sampled_component_evidence(disp_trials[token_count]),
                "roundtrip": sampled_component_evidence(rt_trials[token_count]),
                "stage": sampled_component_evidence(stage_trials[token_count]),
            },
            "tokens_per_rank": token_count,
        }
        sample_sha256 = _sha256_json(sample_point)
        point_identifier = identity.point_id(
            series=series_identifier, tokens_per_rank=token_count
        )
        evidence_identifier = identity.evidence_id(
            point=point_identifier,
            allocation=allocation_identifier,
            attempt=attempt_identifier,
            sample_sha256=sample_sha256,
        )
        sample_point.update(
            {
                "evidence_id": evidence_identifier,
                "point_id": point_identifier,
                "sample_sha256": sample_sha256,
            }
        )
        sample_points.append(sample_point)
        row.update({
            "evidence_id": evidence_identifier,
            "point_id": point_identifier,
            "sample_sha256": sample_sha256,
        })

    samples_path = args.out[:-5] + ".samples.json" if args.out.endswith(".json") else args.out + ".samples.json"
    samples_document = {
        "allocation_id": allocation_identifier,
        "attempt_id": attempt_identifier,
        "case_id": case_identifier,
        "format": "collectivex.samples.v1",
        "points": sample_points,
        "qualification_index": args.qualification_index,
        "sampling": {
            "iterations_per_trial": args.iters,
            "reduction": case_profile["rank_reduction"],
            "trials": args.trials,
        },
        "schema_version": 1,
        "series_id": series_identifier,
    }
    samples_payload = contracts.canonical_json_bytes(samples_document)
    samples_sha256 = hashlib.sha256(samples_payload).hexdigest()
    samples_bytes = len(samples_payload)
    sample_artifact = {
        "bytes": samples_bytes,
        "format": "collectivex.samples.v1",
        "path": os.path.basename(samples_path),
        "sha256": samples_sha256,
    }
    headline = next((r for r in rows if r["tokens_per_rank"] == 64), rows[len(rows) // 2])
    eplb_record = (
        {
            "calibration_token_offset": eplb_calibration["token_offset"],
            "calibration_trace_sha256": eplb_calibration["trace_sha256"],
            "calibration_window": eplb_calibration["window"],
            "calibration_workload_id": eplb_calibration["workload_id"],
            "enabled": True,
            "imbalance_after": eplb_plan["imbalance_after"],
            "imbalance_before": eplb_plan["imbalance_before"],
            "mapping_hash": eplb_mapping_hash,
            "max_replicas": eplb_plan["max_replicas"],
            "num_logical_experts": num_logical,
            "num_physical_experts": args.experts,
            "num_redundant": args.experts - num_logical,
            "planner": EPLB_PLANNER,
            "reference_tokens_per_rank": EPLB_REFERENCE_TOKENS_PER_RANK,
            "replicated_experts": eplb_plan["replicated_experts"],
        }
        if eplb_plan
        else {
            "calibration_token_offset": None,
            "calibration_trace_sha256": None,
            "calibration_window": None,
            "calibration_workload_id": None,
            "enabled": False,
            "imbalance_after": None,
            "imbalance_before": None,
            "mapping_hash": None,
            "max_replicas": None,
            "num_logical_experts": num_logical,
            "num_physical_experts": args.experts,
            "num_redundant": 0,
            "planner": None,
            "reference_tokens_per_rank": None,
            "replicated_experts": 0,
        }
    )
    doc = {
        "format": "collectivex.ep.v1",
        "schema_version": SCHEMA_VERSION,
        "record_type": "case-attempt",
        "generated_at": generated_at,
        "identity": {
            "allocation_factors": allocation_factors,
            "allocation_id": allocation_identifier,
            "attempt_id": attempt_identifier,
            "attempt_ordinal": attempt_ordinal,
            "case_factors": case_factors,
            "case_id": case_identifier,
            "series_factors": series_factors,
            "series_id": series_identifier,
        },
        "case": {
            "attempt_ordinal": attempt_ordinal,
            "backend": backend.name,
            "eplb": eplb_record,
            "ep_size": ep_size,
            "mode": mode,
            "phase": args.phase,
            "required_publication": args.required_publication or "diagnostic",
            "resource_mode": "fixed-profile",
            "runner": args.runner,
            "shape": shape,
            "suite": args.suite or "manual",
            "workload_name": args.workload_name or "manual",
        },
        "workload": {
            "activation_generator": ACTIVATION_GENERATOR,
            "activation_identity": activation_identity,
            "activation_profile": ACTIVATION_PROFILE,
            "cross_rank_consistent": routing_consistent,
            "manifest_checksums": getattr(args, "workload_checksums", None),
            "members": getattr(args, "workload_members", None),
            "routing_generator": ROUTING_GENERATOR,
            "source": validity["workload_source"],
            "trace_hashes": sorted(routing_hashes),
            "trace_signature": trace_sig,
            "workload_id": getattr(args, "workload_id", None),
        },
        "measurement": {
            "component_order_contract": case_profile["component_order_contract"],
            "conditioning": {
                "contract": CONDITIONING_CONTRACT,
                "ladder": conditioning_ladder,
                "roundtrips_per_shape": CONDITIONING_ROUNDS_PER_SHAPE,
            },
            "contract": case_profile["contract"],
            "execution_order_sha256": _sha256_json(observed_component_orders),
            "qualification_index": args.qualification_index,
            "rows": rows,
            "sampling": {
                "contract": SAMPLING_CONTRACT,
                "iterations_per_trial": args.iters,
                "percentile_method": case_profile["percentile_method"],
                "reduction": case_profile["rank_reduction"],
                "samples_per_component": TIMED_SAMPLES_PER_POINT,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
                "warmup_semantics": WARMUP_SEMANTICS,
            },
            "source_allocation": "even",
        },
        "implementation": {
            "kernel_generation": kernel_generation(backend),
            "name": backend.name,
            "provenance": backend.backend_provenance,
            "resource_profile": resource_profile,
        },
        "topology": {
            "device_count": getattr(args, "runtime_device_count", None),
            "device_product": getattr(args, "runtime_device_product", None),
            "gpus_per_node": args.gpus_per_node or ep_size,
            "nodes": nodes,
            "placement": "packed",
            "realized_placement": realized_placement,
            "scale_up_domain": args.scale_up_domain or (args.gpus_per_node or ep_size),
            "scale_up_transport": args.scale_up_transport,
            "scale_out_transport": args.scale_out_transport or None,
            "scope": args.scope,
            "topology_class": args.topology_class,
            "transport": args.transport,
            "world_size": world_size,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "provenance": {
            "allocation_stratum_sha256": allocation_stratum_sha256,
            "command": getattr(args, "reproduction_command", ""),
            "distributed_launcher": getattr(args, "distributed_launcher", None),
            "git_run": getattr(args, "git_run", None),
            "image": {
                "arch": getattr(args, "image_arch", None),
                "digest": getattr(args, "image_digest", "") or None,
                "digest_verified": getattr(args, "image_digest_verified", False),
                "reference": getattr(args, "image", "") or None,
                "squash_sha256": getattr(args, "squash_sha256", None),
            },
            "redaction": "sanitized-v1",
        },
        "sample_artifact": sample_artifact,
        "outcome": {
            "publication_status": publication_status,
            "reasons": [] if all_ok else ["semantic correctness or routing identity failed"],
            "status": "success" if all_ok else "invalid",
            "validity": validity,
        },
    }
    contracts.validate_raw_document(doc, samples_document)
    if rank == 0:
        _write_bytes_atomic(samples_path, samples_payload)
        _write_json_atomic(args.out, doc)
        dispatch_percentiles = headline["components"]["dispatch"]["percentiles_us"]
        dispatch_p99 = dispatch_percentiles["p99"] if dispatch_percentiles else None
        component_summary = (f"disp_p99={dispatch_p99:.1f}us "
                             if dispatch_p99 is not None
                             else "components=unavailable ")
        print(f"{backend.name} ep-dispatch-combine [{args.phase}/{mode}/{case_profile['contract']}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts, routing_consistent={routing_consistent}, "
              f"headline T={headline['tokens_per_rank']} {component_summary}"
              f"-> {args.out}")
    # A complete invalid document is still a successfully captured terminal outcome. Launchers
    # inspect its status to fail the case without conflating it with an execution failure.
    return 0
