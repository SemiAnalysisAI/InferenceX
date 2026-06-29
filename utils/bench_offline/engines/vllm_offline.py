"""vLLM offline plugin: mirrors `vllm serve` flags from the b300 vllm_mtp
launch script, but uses the in-process `vllm.LLM` API and runs
`LLM.generate(prompts)` instead of HTTP requests.

Step semantics: the benchmark unit is *decode rounds* (main-model forward
passes). The warmup batch calibrates
tokens-per-round; the timed batch's token budget is scaled so each request
runs ~`--decode-steps` forward passes. Per-request rounds come from request
metrics where available; for spec runs without per-request fields we fall
back to the engine's aggregate spec-decoding counters (drafts == rounds)
read via `llm.get_metrics()` deltas around each batch.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from latency_utils import (
    append_latency_sample,
    decode_window_s,
    request_rounds,
    timed_max_tokens,
    tokens_per_round_estimate,
)


def _apply_routing_sim(args: argparse.Namespace) -> str:
    """Force simulated MoE routing for idealized expert balance.

    The router factory swaps every FusedMoE router (DSV4's experts included) for
    the RoutingSimulator when
    VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set. Expert load becomes
    (statistically) balanced and model outputs invalid — performance
    measurement only. Must be set before engine construction; spawned
    EngineCore/DP workers inherit it via os.environ."""
    strategy = args.routing_sim_strategy
    if strategy == "none":
        return "real"
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = strategy
    print(f"[vllm_offline] MoE routing simulation ON: {strategy} "
          "(idealized expert balance; model outputs are invalid, MTP acceptance "
          "is reported as observed under simulated routing)")
    return f"simulated_{strategy}"


def _routing_label(args: argparse.Namespace) -> str:
    if args.routing_sim_strategy == "none":
        return "real"
    return f"simulated_{args.routing_sim_strategy}"


def _build_llm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tokenizer_mode="deepseek_v4",
        kv_cache_dtype="fp8",
        block_size=256,
        enable_prefix_caching=False,
        max_model_len=args.engine_max_len,
        # Sized for one in-flight prefill at ISL + the timed-run token budget
        # upper bound — same retune as the HTTP path's
        # MAX_NUM_BATCHED_TOKENS=ISL+OSL change.
        max_num_batched_tokens=(
            args.infinitebench_input_len
            + (args.mtp + 1) * args.decode_steps),
        max_num_seqs=args.batch_size,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE",
                            "custom_ops": ["all"]},
    )
    if args.mtp > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.mtp,
        }
    if getattr(args, "nnodes", 1) > 1:
        llm_kwargs["nnodes"] = args.nnodes
        llm_kwargs["node_rank"] = args.node_rank
        llm_kwargs["master_addr"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        llm_kwargs["master_port"] = int(os.environ.get("MASTER_PORT", "29501"))
    return llm_kwargs


def _spec_draft_counter(llm, mtp: int) -> Optional[float]:
    """Aggregate decode-round count from vLLM's spec-decode engine metrics,
    if the installed version exposes them. Prefers num_drafts (one draft
    per decode round); falls back to num_draft_tokens / mtp (each round
    drafts `mtp` tokens)."""
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None
    draft_tokens = None
    for m in metrics:
        name = getattr(m, "name", "")
        if name in ("vllm:spec_decode_num_drafts",
                    "vllm:spec_decode_num_draft_tokens"):
            value = getattr(m, "value", None)
            if value is None:
                values = getattr(m, "values", None)
                if values:
                    value = sum(values)
            if value is not None and name == "vllm:spec_decode_num_drafts":
                return float(value)
            if value is not None and draft_tokens is None:
                draft_tokens = float(value)
    if draft_tokens is not None and mtp > 0:
        return draft_tokens / mtp
    return None


def _extract_batch(outputs: List[Any], mtp: int) -> Dict[str, Any]:
    total_output_tokens = sum(
        len(o.outputs[0].token_ids) if o.outputs else 0 for o in outputs)

    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []
    tpot_steps: List[float] = []
    decode_tokens_per_req: List[int] = []
    decode_rounds_per_req: List[int] = []
    decode_windows: List[Optional[float]] = []
    n_decode_tokens = 0
    n_rounds = 0
    for o in outputs:
        n_out = len(o.outputs[0].token_ids) if o.outputs else 0
        decode_tokens = max(n_out - 1, 0)
        n_decode_tokens += decode_tokens
        decode_tokens_per_req.append(decode_tokens)
        m = getattr(o, "metrics", None)
        append_latency_sample(m, n_out, ttfts, e2els, tpots)
        window = decode_window_s(m)
        decode_windows.append(window)

        rounds: Optional[int] = None
        if mtp <= 0:
            rounds = request_rounds(decode_tokens, None, mtp)
        elif m is not None:
            # Per-request spec accounting when the engine surfaces it:
            # emitted = rounds + accepted  =>  rounds = emitted - accepted.
            n_emitted = (getattr(m, "num_emitted_tokens", None)
                         or getattr(o, "num_emitted_tokens", None))
            n_accepted = (getattr(m, "num_accepted_tokens", None)
                          or getattr(o, "num_accepted_tokens", None))
            if n_emitted is not None and int(n_emitted) > 0:
                iters = int(n_emitted) - int(n_accepted or 0)
                rounds = request_rounds(decode_tokens, iters, mtp)
        if rounds is not None:
            decode_rounds_per_req.append(rounds)
            n_rounds += rounds
            if window is not None and rounds > 0:
                tpot_steps.append(window / rounds)

    return {
        "total_output_tokens": total_output_tokens,
        "ttfts_s": ttfts,
        "tpots_s": tpots,
        "tpot_steps_s": tpot_steps,
        "e2els_s": e2els,
        "decode_tokens_per_req": decode_tokens_per_req,
        "decode_rounds_per_req": decode_rounds_per_req,
        "decode_windows_s": decode_windows,
        "decode_tokens_total": n_decode_tokens,
        "decode_rounds_total": n_rounds,
    }


def _backfill_rounds_from_counters(stats: Dict[str, Any], mtp: int,
                                   drafts_delta: Optional[float],
                                   tag: str) -> None:
    """When per-request spec telemetry is missing, distribute the engine's
    aggregate draft count (one draft per decode round per request) uniformly
    across the batch — the batch runs in lockstep, so uniform is faithful."""
    if mtp <= 0 or stats["decode_rounds_per_req"]:
        return
    n_req = len(stats["decode_tokens_per_req"])
    if not drafts_delta or drafts_delta <= 0 or n_req == 0:
        return
    rounds_per_req = max(int(round(drafts_delta / n_req)), 1)
    stats["decode_rounds_per_req"] = [rounds_per_req] * n_req
    stats["decode_rounds_total"] = rounds_per_req * n_req
    windows = stats.get("decode_windows_s") or []
    tpot_steps = [w / rounds_per_req for w in windows
                  if w is not None and w > 0]
    stats["tpot_steps_s"] = tpot_steps
    print(f"[vllm_offline] {tag}: per-request spec telemetry unavailable; "
          f"using aggregate spec counters ({drafts_delta:.0f} drafts / "
          f"{n_req} requests = {rounds_per_req} rounds/request).")


def _generate_with_metrics(llm, prompt_strs: List[str], sampling_dict: Dict[str, Any],
                           args: argparse.Namespace) -> Dict[str, Any]:
    """Warmup (calibration) → sleep → step-targeted timed batch; extract
    per-request metrics. Returns the dict shape `run_offline.py` expects."""
    from vllm import SamplingParams

    decode_steps = args.decode_steps
    warmup_sampling = SamplingParams(
        **{**sampling_dict, "max_tokens": decode_steps})

    print(f"[vllm_offline] Warmup batch: {len(prompt_strs)} prompts, "
          f"max_tokens={decode_steps}")
    drafts_before = _spec_draft_counter(llm, args.mtp) if args.mtp > 0 else None
    t0 = time.perf_counter()
    warmup_outputs = llm.generate(prompt_strs, warmup_sampling, use_tqdm=False)
    warmup_s = time.perf_counter() - t0
    print(f"[vllm_offline] Warmup done in {warmup_s:.3f}s")
    drafts_after_warmup = _spec_draft_counter(llm, args.mtp) if args.mtp > 0 else None

    warmup_stats = _extract_batch(warmup_outputs, args.mtp)
    if args.mtp > 0 and not warmup_stats["decode_rounds_per_req"]:
        delta = None
        if drafts_before is not None and drafts_after_warmup is not None:
            delta = drafts_after_warmup - drafts_before
        _backfill_rounds_from_counters(warmup_stats, args.mtp, delta, "warmup")
    tokens_per_round = tokens_per_round_estimate(
        warmup_stats["decode_tokens_total"],
        warmup_stats["decode_rounds_total"])
    if args.mtp > 0 and not warmup_stats["decode_rounds_per_req"]:
        print("[vllm_offline] WARN: spec run but neither per-request spec "
              "metrics nor aggregate spec counters available; cannot "
              "calibrate decode-step budget. Timed run keeps "
              "max_tokens=decode_steps and step metrics will fall back to "
              "per-token accounting.")
    timed_budget = timed_max_tokens(decode_steps, tokens_per_round)
    time.sleep(2.0)

    timed_sampling = SamplingParams(
        **{**sampling_dict, "max_tokens": timed_budget})
    print(f"[vllm_offline] Timed batch: {len(prompt_strs)} prompts, "
          f"warmup tokens/round={tokens_per_round:.3f}, "
          f"max_tokens={timed_budget} (targeting {decode_steps} "
          f"forward passes/request)")
    drafts_before_timed = _spec_draft_counter(llm, args.mtp) if args.mtp > 0 else None
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_strs, timed_sampling, use_tqdm=False)
    timed_s = time.perf_counter() - t0
    print(f"[vllm_offline] Timed done in {timed_s:.3f}s")
    drafts_after_timed = _spec_draft_counter(llm, args.mtp) if args.mtp > 0 else None

    stats = _extract_batch(outputs, args.mtp)
    if args.mtp > 0 and not stats["decode_rounds_per_req"]:
        delta = None
        if drafts_before_timed is not None and drafts_after_timed is not None:
            delta = drafts_after_timed - drafts_before_timed
        _backfill_rounds_from_counters(stats, args.mtp, delta, "timed")

    stats.pop("decode_windows_s", None)
    stats.update({
        "warmup_seconds": warmup_s,
        "timed_seconds": timed_s,
        "warmup_tokens_per_round": tokens_per_round,
        "timed_max_tokens": timed_budget,
    })
    return stats


def _dp_worker(local_rank: int, global_rank: int, dp_size: int,
               master_ip: str, master_port: int,
               llm_kwargs: Dict[str, Any], sampling_dict: Dict[str, Any],
               prompt_strs_slice: List[str], args: argparse.Namespace,
               result_q: "mp.Queue") -> None:
    """Per-rank worker: sets DP env vars, builds LLM, runs warmup+timed,
    pushes metrics back to parent via queue."""
    os.environ["VLLM_DP_RANK"] = str(global_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    _apply_routing_sim(args)

    try:
        from vllm import LLM
        if global_rank == 0:
            print(f"[vllm_offline DP rank {global_rank}] LLM kwargs: {llm_kwargs}")
        t_load = time.perf_counter()
        llm = LLM(**llm_kwargs)
        print(f"[vllm_offline DP rank {global_rank}] Engine init: "
              f"{time.perf_counter() - t_load:.2f}s, "
              f"prompts={len(prompt_strs_slice)}")
        rank_metrics = _generate_with_metrics(
            llm, prompt_strs_slice, sampling_dict, args)
        result_q.put({"rank": global_rank, "ok": True, "metrics": rank_metrics})
    except Exception as exc:
        import traceback
        result_q.put({
            "rank": global_rank, "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
            "tb": traceback.format_exc(),
        })


def _build_sampling_dict(args: argparse.Namespace) -> Dict[str, Any]:
    # max_tokens is set per-phase (warmup calibration vs step-targeted timed
    # run) inside _generate_with_metrics.
    return dict(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.decode_steps,
        ignore_eos=args.ignore_eos,
    )


def _run_dp(args: argparse.Namespace,
            prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    import json as _json
    from vllm.utils.network_utils import get_open_port

    nnodes = getattr(args, "nnodes", 1)
    node_rank = getattr(args, "node_rank", 0)
    total_dp_size = args.tp
    local_dp_size = total_dp_size // nnodes

    llm_kwargs = _build_llm_kwargs(args)
    # DP-attn: each worker runs LLM(tensor_parallel_size=1, EP=True)
    # WITHOUT data_parallel_size as an LLM kwarg. ParallelConfig falls
    # back to VLLM_DP_SIZE/VLLM_DP_RANK env vars ("offline SPMD case",
    # parallel.py:774-786). Strip multi-node TP kwargs — workers are
    # single-GPU; cross-node coordination uses VLLM_DP_* env vars.
    for k in ("nnodes", "node_rank", "master_addr", "master_port"):
        llm_kwargs.pop(k, None)
    llm_kwargs["tensor_parallel_size"] = 1
    llm_kwargs["enable_expert_parallel"] = True

    sampling_dict = _build_sampling_dict(args)
    prompt_strs = [p for (p, _, _) in prompts]

    # Even-split prompts across ALL DP ranks.
    floor = len(prompt_strs) // total_dp_size
    rem = len(prompt_strs) % total_dp_size

    def slice_for(rank: int) -> List[str]:
        start = rank * floor + min(rank, rem)
        end = (rank + 1) * floor + min(rank + 1, rem)
        s = prompt_strs[start:end]
        return s if s else ["Placeholder"]

    if nnodes > 1:
        master_ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("VLLM_DP_PORT",
                                         os.environ.get("MASTER_PORT", "29501")))
    else:
        master_ip = "127.0.0.1"
        master_port = get_open_port()

    rank_offset = node_rank * local_dp_size
    print(f"[vllm_offline] DP-attn: node {node_rank}/{nnodes}, "
          f"spawning {local_dp_size} local workers (global ranks "
          f"{rank_offset}..{rank_offset + local_dp_size - 1}, "
          f"total_dp_size={total_dp_size}), master={master_ip}:{master_port}")

    ctx = mp.get_context("spawn")
    result_q: "mp.Queue" = ctx.Queue()
    procs: List[mp.Process] = []
    for local_rank in range(local_dp_size):
        global_rank = rank_offset + local_rank
        p = ctx.Process(
            target=_dp_worker,
            args=(local_rank, global_rank, total_dp_size,
                  master_ip, master_port,
                  llm_kwargs, sampling_dict, slice_for(global_rank),
                  args, result_q),
        )
        p.start()
        procs.append(p)

    rank_results: List[Dict[str, Any]] = []
    deadline = time.time() + 1800
    while len(rank_results) < local_dp_size and time.time() < deadline:
        try:
            rank_results.append(result_q.get(timeout=60))
        except Exception:
            alive = [p.pid for p in procs if p.is_alive()]
            print(f"[vllm_offline] waiting on workers (alive PIDs: {alive})")

    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            print(f"[vllm_offline] killing stuck worker PID={p.pid}")
            p.kill()

    failed = [r for r in rank_results if not r.get("ok")]
    if failed:
        for r in failed:
            print(f"[vllm_offline] rank {r['rank']} FAILED: {r.get('err')}\n"
                  f"{r.get('tb','')}")
        raise RuntimeError(
            f"{len(failed)}/{local_dp_size} DP workers failed on node {node_rank}")
    if len(rank_results) < local_dp_size:
        raise RuntimeError(
            f"Only {len(rank_results)}/{local_dp_size} workers reported on node {node_rank}")

    # Multi-node: follower writes partial results to shared FS and exits.
    if nnodes > 1 and node_rank > 0:
        partial_path = Path(args.result_dir) / f".dp_partial_node{node_rank}.json"
        partial_path.write_text(_json.dumps(rank_results, default=str))
        print(f"[vllm_offline DP] node {node_rank}: wrote {len(rank_results)} "
              f"rank results to {partial_path}")
        return {}

    # Multi-node leader: read follower partial results.
    if nnodes > 1:
        for nid in range(1, nnodes):
            partial_path = Path(args.result_dir) / f".dp_partial_node{nid}.json"
            print(f"[vllm_offline DP] leader: waiting for {partial_path}")
            while not partial_path.exists():
                time.sleep(2)
            remote_results = _json.loads(partial_path.read_text())
            rank_results.extend(remote_results)
            print(f"[vllm_offline DP] leader: read {len(remote_results)} "
                  f"rank results from node {nid}")

    rank_metrics = [r["metrics"] for r in
                    sorted(rank_results, key=lambda r: r["rank"])]

    total_input_tokens = sum(plen for (_, plen, _) in prompts)
    agg: Dict[str, Any] = {
        "warmup_seconds": max(m["warmup_seconds"] for m in rank_metrics),
        "timed_seconds": max(m["timed_seconds"] for m in rank_metrics),
        "total_output_tokens": sum(m["total_output_tokens"] for m in rank_metrics),
        "total_input_tokens": total_input_tokens,
        "ttfts_s": [v for m in rank_metrics for v in m["ttfts_s"]],
        "tpots_s": [v for m in rank_metrics for v in m["tpots_s"]],
        "tpot_steps_s": [v for m in rank_metrics for v in m["tpot_steps_s"]],
        "e2els_s": [v for m in rank_metrics for v in m["e2els_s"]],
        "decode_tokens_per_req": [v for m in rank_metrics for v in m["decode_tokens_per_req"]],
        "decode_rounds_per_req": [v for m in rank_metrics for v in m["decode_rounds_per_req"]],
        "decode_tokens_total": sum(m["decode_tokens_total"] for m in rank_metrics),
        "decode_rounds_total": sum(m["decode_rounds_total"] for m in rank_metrics),
        "warmup_tokens_per_round": (
            sum(m["warmup_tokens_per_round"] for m in rank_metrics)
            / len(rank_metrics)),
        "timed_max_tokens": max(m["timed_max_tokens"] for m in rank_metrics),
        "latency_metrics_source": "vllm_request_metrics_dp",
        "moe_routing": _routing_label(args),
    }
    print(f"[vllm_offline DP] aggregated across {total_dp_size} ranks: "
          f"timed={agg['timed_seconds']:.2f}s, "
          f"out_toks={agg['total_output_tokens']}, "
          f"decode_rounds={agg['decode_rounds_total']}")
    return agg


def _run_single(args: argparse.Namespace,
                prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    from vllm import LLM

    llm_kwargs = _build_llm_kwargs(args)
    llm_kwargs["tensor_parallel_size"] = args.tp
    if args.ep > 1:
        llm_kwargs["enable_expert_parallel"] = True

    print(f"[vllm_offline] LLM kwargs: {llm_kwargs}")

    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[vllm_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]
    rank_metrics = _generate_with_metrics(
        llm, prompt_strs, _build_sampling_dict(args), args)
    rank_metrics["total_input_tokens"] = sum(plen for (_, plen, _) in prompts)
    rank_metrics["latency_metrics_source"] = "vllm_request_metrics"
    rank_metrics["moe_routing"] = _routing_label(args)
    return rank_metrics


def run(args: argparse.Namespace,
        prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    _apply_routing_sim(args)
    if args.dp_attn:
        return _run_dp(args, prompts)
    return _run_single(args, prompts)


def run_follower(args: argparse.Namespace) -> None:
    """Follower node: construct LLM engine (starts NCCL workers that
    participate in the leader's all-reduce), then block until the leader
    writes a sentinel file after benchmark completion."""
    from vllm import LLM

    _apply_routing_sim(args)
    llm_kwargs = _build_llm_kwargs(args)
    llm_kwargs["tensor_parallel_size"] = args.tp
    if args.ep > 1:
        llm_kwargs["enable_expert_parallel"] = True

    print(f"[vllm_offline follower] node_rank={args.node_rank} "
          f"LLM kwargs: {llm_kwargs}")

    t_load = time.perf_counter()
    _llm = LLM(**llm_kwargs)
    print(f"[vllm_offline follower] Engine init: "
          f"{time.perf_counter() - t_load:.2f}s — waiting for leader")

    sentinel = Path(args.result_dir) / ".offline_follower_exit"
    while not sentinel.exists():
        time.sleep(2)
    print("[vllm_offline follower] Leader signalled done, exiting.")
