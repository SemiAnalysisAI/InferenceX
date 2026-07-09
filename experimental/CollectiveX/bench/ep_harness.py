#!/usr/bin/env python3
"""CollectiveX — shared EP (expert-parallel) dispatch/combine benchmark harness.

Backend-agnostic core. The per-backend adapters (`ep_deepep_v2.py`, `ep_mori.py`)
implement a small duck-typed protocol; this module owns the source-tokens-per-rank
sweep, the timing, the correctness gate, and the emitted JSON doc.

Fair-comparison rules (see docs/methodology.md):
  * **Deterministic shared routing trace** (`routing.py`): the per-token expert IDs +
    gate weights are generated once from a fixed seed over the *global* batch and are
    identical on every SKU; each rank materializes its slice. So every platform runs
    the *same* problem (no per-rank/per-platform RNG in the adapters).
  * Dispatch timing includes routing-layout generation. Combine excludes staging.
    Isolated sum is derived independently at each percentile and is not a measured chained op.
  * **Correct collective percentile**: each iteration's latency is reduced MAX across
    ranks first (a collective finishes with its slowest rank), THEN percentiled —
    `median_i(max_r)`, not `max_r(median_i)`.
  * **One line = one fixed config**; only T varies. Both `tokens_per_rank` and
    `global_tokens = T * ep_size` are recorded as explicit chart coordinates.

stdlib-only at module top (torch is passed in by the entrypoint; `routing` is imported
lazily inside run_sweep) so this file `py_compile`s without torch.

Backend protocol:
    name, mode, combine_needs_redispatch
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
import json
import math
import os
import re
import types


_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def is_case_id(value) -> bool:
    return bool(isinstance(value, str) and _CASE_ID.fullmatch(value))


def case_id(sku: str, case: dict) -> str:
    parts = (
        sku,
        case["backend"],
        case.get("workload") or "manual",
        case["mode"],
        case["phase"],
        f"ep{int(case['ep'])}",
        case["routing"],
    )
    values = [_NON_SLUG.sub("-", str(part).lower()).strip("-") for part in parts]
    if not all(values):
        raise ValueError("case ID contains an empty factor")
    return "-".join(values)


# Every comparison-grade EP point uses the same literal timing profile on every SKU/backend.
# Eight timed iterations keep each MoRI burst well below its sustained-iteration wedge, 64 trials
# provide 512 observations per operation, and 32 warmups meet Blackwell's measured clock-ramp floor.
TIMED_SAMPLES_PER_POINT = 512
TIMED_ITERS_PER_TRIAL = 8
TRIALS_PER_POINT = 64
WARMUP_ITERS_PER_TRIAL = 32
WARMUP_SEMANTICS = "full-roundtrip-before-each-component-trial-point-v1"
ROUTING_SEED = 67
PLACEMENT = "packed"

# Phase-default sweeps — token-size regimes, NOT distinct kernels (both run normal
# mode; "decode"/"prefill" name the small/large-token regime). Powers of two for a
# clean log x-axis; clamped to the backend buffer ceiling (MoRI's registerable heap).
DECODE_LADDER = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_LADDER = [128, 256, 512, 1024, 2048, 4096]
# Conditioning replays a fixed phase ramp before each measured shape to settle
# clocks and routing state; these rounds are never timed or emitted. The ladders
# and round count are fixed for every benchmark case.
CONDITIONING_LADDERS = {
    "decode": [1, 2, 4, 8, 16, 32, 64, 128],
    "prefill": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}
CONDITIONING_ROUNDS_PER_SHAPE = 8
# Dispatch and combine are fixed BF16, so the combine oracle uses one frozen gate.
ORACLE_RTOL = 5e-2
ORACLE_ATOL = 2e-2

def logical_byte_provenance(logical_copies: int, hidden: int) -> dict[str, int]:
    """Return comparable logical BF16 activation bytes for one direction.

    Dispatch and combine both move BF16 (2 bytes/value) with no separate scale
    payload, so ``scale_bytes`` is always zero.
    """
    if logical_copies < 0 or hidden < 0:
        raise ValueError("logical byte dimensions must be non-negative")
    activation_data_bytes = logical_copies * hidden * 2
    return {
        "activation_data_bytes": activation_data_bytes,
        "scale_bytes": 0,
        "total_logical_bytes": activation_data_bytes,
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
    # DeepEP uses this only when its installed Buffer does not expose a tuned default.
    ap.set_defaults(num_sms=24)
    ap.add_argument("--mode", default="normal", choices=["normal"])
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"],
                    help="token-size regime: decode (small T) / prefill (large T) — picks the default ladder")
    ap.add_argument("--tokens-ladder", default="",
                    help="space/comma-separated source-tokens-per-rank sweep; blank = phase default")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256, help="TOTAL experts (fixed across EP degrees)")
    ap.add_argument("--routing", default="uniform", choices=["uniform"])
    ap.add_argument("--case-id", default="")
    ap.add_argument("--suite", default="")
    ap.add_argument("--workload-name", default="")
    ap.add_argument("--seed", type=int, default=ROUTING_SEED)
    ap.add_argument(
        "--version",
        type=int,
        default=os.environ.get("CX_VERSION", "1"),
        help="iterable benchmark version copied verbatim into the emitted result",
    )
    # 32: B300/Blackwell needs ~30 untimed iters to reach steady-state GPU clocks +
    # establish NVLink/NVSHMEM connections — at warmup=8 its dispatch read ~1787us
    # (cold), at warmup>=30 it settles to ~85us (faster than H100, reproducible within
    # ~2.5%). H100/MI355X reach steady state much sooner; the extra iters are harmless.
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS_PER_TRIAL,
                    help=f"untimed full roundtrips before each trial/point; fixed to {WARMUP_ITERS_PER_TRIAL}")
    ap.add_argument("--iters", type=int, default=TIMED_ITERS_PER_TRIAL,
                    help=f"timed iterations per trial; fixed to {TIMED_ITERS_PER_TRIAL}")
    ap.add_argument("--trials", type=int, default=TRIALS_PER_POINT,
                    help=f"timed trials; fixed to {TRIALS_PER_POINT}")
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


def sampling_error(iters: int, trials: int, warmup: int) -> str | None:
    """Return a user-facing error unless the exact cross-SKU timing profile is used."""
    expected = (TIMED_ITERS_PER_TRIAL, TRIALS_PER_POINT, WARMUP_ITERS_PER_TRIAL)
    observed = (iters, trials, warmup)
    if observed != expected:
        return ("CollectiveX requires exactly iters:trials:warmup="
                f"{expected[0]}:{expected[1]}:{expected[2]} on every SKU/backend; got "
                f"{observed[0]}:{observed[1]}:{observed[2]} "
                f"({iters * trials if iters > 0 and trials > 0 else 'invalid'} timed samples)")
    return None


def trial_order(values: list, trial_index: int) -> list:
    """Rotate and reverse values so each occupies every timing position."""
    if not values or len(values) != len(set(values)):
        raise ValueError("trial order requires non-empty unique values")
    if type(trial_index) is not int or trial_index < 0:
        raise ValueError("trial_index must be a non-negative integer")
    cycle, offset = divmod(trial_index, len(values))
    base = list(values) if cycle % 2 == 0 else list(reversed(values))
    return base[offset:] + base[:offset]


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


def _write_bytes_atomic(path: str, payload: bytes) -> None:
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


def _write_json_atomic(path: str, value) -> None:
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


def _same_tensors_across_ranks(torch, dist, device, *tensors) -> bool:
    matches = True
    for tensor in tensors:
        observed = tensor.to(device=device, non_blocking=False)
        reference = observed.clone() if dist.get_rank() == 0 else torch.empty_like(observed)
        dist.broadcast(reference, src=0)
        matches = matches and bool(torch.equal(observed, reference))
    result = torch.tensor([int(matches)], device=device, dtype=torch.int64)
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return bool(result.item())


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
    oracle_atol, oracle_rtol = ORACLE_ATOL, ORACLE_RTOL
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
            "passed": False,
            "ordering": "adapter-inspection-failed",
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
    ordering = view.ordering_contract

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
        "passed": bool(
            all(checks.values())
            and ordering
            and max_relative_error is not None
            and max_relative_error < tolerance
        ),
        "atol": oracle_atol,
        "combine_weight_semantics": combine_weight_semantics,
        "ordering": ordering,
        "receive_count": receive_count,
        "max_absolute_error": max_absolute_error,
        "max_elementwise_relative_error": max_elementwise_relative_error,
        "max_relative_error": max_relative_error,
        "max_weight_error": max_weight_error,
        "rtol": oracle_rtol,
        "checks": checks,
    }


def run_sweep(args, backend, torch, dist, device, rank: int, world_size: int) -> int:
    """Drive the source-tokens-per-rank sweep for one fully-specified line."""
    mode = getattr(args, "mode", "normal")
    if mode != "normal":
        if rank == 0:
            print(f"ERROR: unknown CollectiveX case mode {mode!r}")
        return 2
    timing_error = sampling_error(args.iters, args.trials, args.warmup)
    if timing_error:
        if rank == 0:
            print(f"ERROR: {timing_error}")
        return 2
    import routing  # torch-based; imported lazily so the module byte-compiles without torch

    ep_size = world_size
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
    expected_weight_semantics = "unweighted-rank-sum"
    if getattr(backend, "combine_weight_semantics", None) != expected_weight_semantics:
        if rank == 0:
            print(
                f"ERROR: {mode} requires combine semantics {expected_weight_semantics}"
            )
        return 2

    spec = backend.make_inputs(args)
    if not spec.ok:
        if rank == 0:
            print(f"ERROR: {spec.message}")
        return spec.rc
    cap = spec.cap
    conditioning_ladder = spec.conditioning_ladder
    ladder, dropped = spec.ladder, spec.dropped
    if rank == 0 and dropped:
        print(f"NOTE: dropped tokens/rank {dropped} — exceed {backend.name} buffer cap {cap} "
              f"(hidden={args.hidden}); not silently truncated.")
    MAX, MIN, SUM = dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.SUM

    # Size the communicator from the resolved numeric shape. Buffer sizing needs the
    # ladder numbers (not the input tensors), so it runs after make_inputs rather than
    # in __init__ — resolving the historical build-buffers-before-inputs inversion.
    # make_inputs already materialized deterministic per-rank routing and activations.
    backend.create_buffer(spec)

    for wt in conditioning_ladder:
        # Fabric/clock warm-up BEFORE any timed point (review: H200 had an anomalous
        # cold first point and a 40% decode-vs-prefill mismatch at the shared T=128).
        # make_inputs already materialized these warm-only per-rank inputs; ramping
        # through the small ladder shapes untimed warms clocks/fabric for everyone and
        # is cold-jump-safe for MoRI. Warm-only shapes carry no canonical manifest:
        # they are never measured or emitted.
        cwp = spec.conditioning_points[wt]
        wp = backend.make_problem(
            wt, cwp.topk_idx.to(device), cwp.topk_weights.to(device), cwp.activations
        )
        backend.warm(wp, CONDITIONING_ROUNDS_PER_SHAPE)
    torch.cuda.synchronize()
    dist.barrier()
    # ---- Pass 1: build each deterministic problem and run the expert oracle. ----
    problems, gate, gts, global_traces, input_snapshots = {}, {}, {}, {}, {}
    routing_consistent = True
    for T in ladder:
        counts = [T] * ep_size
        gt = T * ep_size
        gts[T] = gt
        point = spec.points[T]
        idx_g, w_g = point.global_idx, point.global_weights
        rstats = routing.routing_stats(idx_g, args.experts, experts_per_rank, weights=w_g)
        gpn = args.gpus_per_node or ep_size
        rstats["locality"] = routing.routing_locality(idx_g, experts_per_rank, ep_size, max(1, T),
                                                      gpn, args.scale_up_domain or None)
        rstats["source_token_stats"] = _stats_vec(counts)
        point_routing_consistent = _same_tensors_across_ranks(
            torch, dist, device, idx_g, w_g
        )
        routing_consistent = routing_consistent and point_routing_consistent
        my_cnt = T
        problem = backend.make_problem(
            my_cnt, point.topk_idx.to(device), point.topk_weights.to(device), point.activations
        )
        input_snapshots[T] = (
            problem.x.clone(), problem.topk_idx.clone(), problem.topk_weights.clone()
        )
        oracle = _run_expert_oracle(
            torch, routing, backend, problem, idx_g, w_g, rank, experts_per_rank,
            args.seed,
        )
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
    for trial_index in range(args.trials):
        order = trial_order(list(ladder), trial_index)
        for T in order:
            problem = problems[T]
            # timed_components() encodes the roundtrip-only vs full-component contract
            # (and whether stage launches device work) once, in the base class.
            component_order = trial_order(backend.timed_components(), trial_index)
            measured = {name: [] for name in ("dispatch", "stage", "combine", "roundtrip")}
            for component_name in component_order:
                # The base template gives every component the same synchronized
                # full-roundtrip warm-up before its timed trial and encodes the two
                # branch rules (dispatch cleanup, combine re-dispatch) internally.
                measured[component_name] = backend.benchmark_component(
                    component_name, problem, args.warmup, args.iters
                )
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
        pre = gate[T]["oracle_pre"]
        order_stable = pre["ordering"] == post["ordering"]
        gate[T].update({
            "input_unchanged": input_unchanged,
            "local_ok": int(pre["passed"] and post["passed"] and input_unchanged and order_stable),
            "max_rel": max(pre["max_relative_error"] or 0.0, post["max_relative_error"] or 0.0),
            "oracle_post": post,
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
        # Canonical LOGICAL payload byte contracts (from the routing trace, NOT backend recv
        # tensors): token-rank = one copy per unique (token,dest-rank); token-expert = one copy
        # per routed (token,expert). routed_copies = token-rank copies; gt*topk = token-expert.
        token_rank_copies = rstats["routed_copies"]
        logical_copies = token_rank_copies
        H = args.hidden
        throughput = {
            percentile_name: gt / (latency_us * 1e-6)
            for percentile_name, latency_us in rtp.items()
        }
        dispatch_bytes = logical_byte_provenance(logical_copies, H)
        combine_bytes = logical_byte_provenance(logical_copies, H)
        stage_bytes = {
            "activation_data_bytes": 0,
            "scale_bytes": 0,
            "total_logical_bytes": 0,
        }
        roundtrip_bytes = {
            **{
                field: dispatch_bytes[field] + combine_bytes[field]
                for field in (
                    "activation_data_bytes", "scale_bytes", "total_logical_bytes"
                )
            },
        }
        rows.append({
            "components": {
                "combine": component(cp, len(c)),
                "dispatch": component(dp, len(d)),
                "isolated_sum": component(isum, 0, derived=True),
                "roundtrip": component(rtp, len(rt)),
                "stage": component(sp, len(s)),
            },
            "correctness": {
                "max_relative_error": max_rel,
                "passed": point_ok,
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
                "hotspot_ratio": rstats["hotspot_ratio"],
                "locality": rstats.get("locality"),
                "payload_copies_per_rank": rstats["payload_copies_per_rank"],
                "payload_rank_cv": rstats["payload_rank_cv"],
                "routed_copies": rstats["routed_copies"],
                "source_token_stats": rstats.get("source_token_stats"),
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

    # status=valid requires correctness AND a proven-identical routing trace across ranks.
    all_ok = bool(rows) and all(r["correctness"]["passed"] for r in rows) and routing_consistent

    validity = {
        "execution_status": "complete" if rows else "failed",
        "semantic_correctness": (
            "pass" if rows and all(r["correctness"]["passed"] for r in rows) else "fail"
        ),
        "workload_identity": "consistent-across-ranks" if routing_consistent else "inconsistent",
        "measurement_conformance": "conformant",   # run_ep gate rejects nonconformant pre-run
        "sampling_conformance": "conformant",      # fixed-512-v1 gate rejects any other profile
    }

    shape = {  # FIXED line identity (no T, no per-backend resource knobs)
        "hidden": args.hidden, "topk": args.topk, "experts": args.experts,
        "experts_per_rank": experts_per_rank,
        "routing": args.routing, "num_logical_experts": num_logical,
        # V2 is reserved for the PR #605 ElasticBuffer adapter; package versions never imply it.
        "kernel_gen": kernel_generation(backend),
    }
    generated_at = args.timestamp or _dt.datetime.now().astimezone().isoformat()
    realized_placement = getattr(args, "realized_placement", None)
    nodes = (
        realized_placement["nodes"]
        if realized_placement is not None
        else int(os.environ.get("SLURM_NNODES", "1"))
    )
    # A scheduled sweep case always carries a matrix-issued --case-id; ad-hoc manual runs do
    # not. The old canonical-workload machinery (serialized traces) is gone — every workload is
    # now seeded-runtime — so "canonical" means "matrix-scheduled case", matching sweep_matrix's
    # canonical:True on scheduled cases and CX_CANONICAL in the container env.
    canonical = bool(args.case_id)
    scheduled_case = {
            "backend": backend.name,
            "canonical": canonical,
            "ep": ep_size,
            "experts": num_logical,
            "gpus_per_node": args.gpus_per_node or ep_size,
            "hidden": args.hidden,
            "ladder": " ".join(map(str, ladder)),
            "mode": mode,
            "nodes": nodes,
            "phase": args.phase,
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
    case_factors = {"case": scheduled_case, "sku": args.runner}
    computed_case_id = case_id(args.runner, scheduled_case)
    if args.case_id and args.case_id != computed_case_id:
        raise ValueError(
            f"scheduled case ID does not match realized factors: {args.case_id} != {computed_case_id}"
        )
    case_identifier = args.case_id or computed_case_id
    git_run = getattr(args, "git_run", None) or {}
    allocation_factors = {
        "run_attempt": git_run.get("run_attempt"),
        "run_id": git_run.get("run_id"),
        "source_sha": git_run.get("source_sha"),
    }
    try:
        attempt_ordinal = int(os.environ.get("CX_ATTEMPT_ID", "1"))
    except ValueError:
        attempt_ordinal = 0
    if attempt_ordinal <= 0:
        raise ValueError("CX_ATTEMPT_ID must be a positive integer")
    headline = next((r for r in rows if r["tokens_per_rank"] == 64), rows[len(rows) // 2])
    doc = {
        "version": args.version,
        "record_type": "case-attempt",
        "generated_at": generated_at,
        "identity": {
            "allocation_factors": allocation_factors,
            "attempt_ordinal": attempt_ordinal,
            "case_factors": case_factors,
            "case_id": case_identifier,
        },
        "case": {
            "attempt_ordinal": attempt_ordinal,
            "backend": backend.name,
            "ep_size": ep_size,
            "mode": mode,
            "phase": args.phase,
            "resource_mode": "fixed-profile",
            "runner": args.runner,
            "shape": shape,
            "suite": args.suite or "manual",
            "workload_name": args.workload_name or "manual",
        },
        "workload": {
            "cross_rank_consistent": routing_consistent,
        },
        "measurement": {
            "combine_dtype": "bf16",
            "combine_semantics": "activation-only",
            "dispatch_dtype": "bf16",
            "payload_unit": "token-rank",
            "rows": rows,
            "sampling": {
                "iterations_per_trial": args.iters,
                "samples_per_component": TIMED_SAMPLES_PER_POINT,
                "trials": args.trials,
                "warmup_iterations": args.warmup,
            },
        },
        "implementation": {
            "kernel_generation": kernel_generation(backend),
            "name": backend.name,
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
        "runtime": getattr(args, "runtime", {}),
        "provenance": {
            "image": getattr(args, "image", "") or None,
            "source_sha": git_run.get("source_sha"),
        },
        "outcome": {
            "reasons": [] if all_ok else ["semantic correctness or routing identity failed"],
            "status": "success" if all_ok else "invalid",
            "validity": validity,
        },
    }
    if rank == 0:
        _write_json_atomic(args.out, doc)
        dispatch_percentiles = headline["components"]["dispatch"]["percentiles_us"]
        dispatch_p99 = dispatch_percentiles["p99"] if dispatch_percentiles else None
        component_summary = (f"disp_p99={dispatch_p99:.1f}us "
                             if dispatch_p99 is not None
                             else "components=unavailable ")
        print(f"{backend.name} ep-dispatch-combine [{args.phase}/{mode}]: "
              f"status={doc['outcome']['status']} {len(rows)} pts, routing_consistent={routing_consistent}, "
              f"headline T={headline['tokens_per_rank']} {component_summary}"
              f"-> {args.out}")
    # CI honesty: run_sweep's return code is the only success signal cx_run_shard (and thus CI)
    # reads — the doc is uploaded regardless, via the launcher's always() stage step. A captured
    # `invalid` outcome (semantic correctness or cross-rank routing identity failed) must therefore
    # fail the leg, not ride as a green success; otherwise a persistent oracle failure is invisible
    # in CI and could autopublish an invalid doc. Agree the verdict across ranks (MIN) so every
    # rank exits identically and the distributed case fails as one.
    outcome_ok = bool(_reduce_int(torch, dist, device, int(all_ok), dist.ReduceOp.MIN))
    return 0 if outcome_ok else 3
