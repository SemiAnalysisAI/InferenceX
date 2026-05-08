"""vLLM offline plugin: mirrors `vllm serve` flags from
benchmarks/single_node/dsv4_fp4_b300_vllm_mtp.sh, but uses the in-process
`vllm.LLM` API and runs `LLM.generate(prompts)` instead of HTTP requests.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Tuple

from latency_utils import append_latency_sample


def _build_llm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tokenizer_mode="deepseek_v4",
        kv_cache_dtype="fp8",
        block_size=256,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        # Sized for one in-flight prefill at ISL+OSL — same retune as the
        # HTTP path's MAX_NUM_BATCHED_TOKENS=ISL+OSL change.
        max_num_batched_tokens=args.infinitebench_input_len + args.infinitebench_output_len,
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
    return llm_kwargs


def _build_sampling_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.infinitebench_output_len,
        ignore_eos=args.ignore_eos,
    )


def _generate_with_metrics(llm, prompt_strs: List[str], sampling,
                           output_len: int) -> Dict[str, Any]:
    """Warmup → sleep → timed; extract per-request metrics. Returns the dict
    shape that `run_offline.py` expects."""
    print(f"[vllm_offline] Warmup batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    _ = llm.generate(prompt_strs, sampling, use_tqdm=False)
    warmup_s = time.perf_counter() - t0
    print(f"[vllm_offline] Warmup done in {warmup_s:.3f}s")
    time.sleep(2.0)

    print(f"[vllm_offline] Timed batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_strs, sampling, use_tqdm=False)
    timed_s = time.perf_counter() - t0
    print(f"[vllm_offline] Timed done in {timed_s:.3f}s")

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []
    decode_tokens_per_req: List[int] = []
    decode_iters_per_req: List[int] = []
    n_decode_tokens = 0
    for o in outputs:
        n_out = len(o.outputs[0].token_ids) if o.outputs else 0
        decode_tokens = max(n_out - 1, 0)
        n_decode_tokens += decode_tokens
        decode_tokens_per_req.append(decode_tokens)
        m = getattr(o, "metrics", None)
        append_latency_sample(m, n_out, ttfts, e2els, tpots)
        if m is not None:
            n_emitted = (getattr(m, "num_emitted_tokens", None)
                         or getattr(o, "num_emitted_tokens", None))
            n_accepted = (getattr(m, "num_accepted_tokens", None)
                          or getattr(o, "num_accepted_tokens", None))
            if n_emitted is not None and int(n_emitted) > 0:
                iters = int(n_emitted) - int(n_accepted or 0)
                if iters > 0:
                    decode_iters_per_req.append(iters)

    return {
        "warmup_seconds": warmup_s,
        "timed_seconds": timed_s,
        "total_output_tokens": total_output_tokens,
        "ttfts_s": ttfts,
        "tpots_s": tpots,
        "e2els_s": e2els,
        "decode_tokens_per_req": decode_tokens_per_req,
        "decode_iters_per_req": decode_iters_per_req,
        "decode_tokens_total": n_decode_tokens,
    }


def _dp_worker(local_rank: int, global_rank: int, dp_size: int,
               master_ip: str, master_port: int,
               llm_kwargs: Dict[str, Any], sampling_dict: Dict[str, Any],
               prompt_strs_slice: List[str], output_len: int,
               result_q: "mp.Queue") -> None:
    """Per-rank worker: sets DP env vars, builds LLM, runs warmup+timed,
    pushes metrics back to parent via queue."""
    os.environ["VLLM_DP_RANK"] = str(global_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    try:
        from vllm import LLM, SamplingParams
        sampling = SamplingParams(**sampling_dict)
        if global_rank == 0:
            print(f"[vllm_offline DP rank {global_rank}] LLM kwargs: {llm_kwargs}")
            print(f"[vllm_offline DP rank {global_rank}] SamplingParams: {sampling}")
        t_load = time.perf_counter()
        llm = LLM(**llm_kwargs)
        print(f"[vllm_offline DP rank {global_rank}] Engine init: "
              f"{time.perf_counter() - t_load:.2f}s, "
              f"prompts={len(prompt_strs_slice)}")
        rank_metrics = _generate_with_metrics(
            llm, prompt_strs_slice, sampling, output_len)
        result_q.put({"rank": global_rank, "ok": True, "metrics": rank_metrics})
    except Exception as exc:
        import traceback
        result_q.put({
            "rank": global_rank, "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
            "tb": traceback.format_exc(),
        })


def _run_dp(args: argparse.Namespace,
            prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    from vllm.utils.network_utils import get_open_port

    dp_size = args.tp
    llm_kwargs = _build_llm_kwargs(args)
    # DP-attn: each of `tp` workers runs LLM(tensor_parallel_size=1, EP=True)
    # WITHOUT data_parallel_size as an LLM kwarg. The single-process check in
    # vllm/entrypoints/llm.py rejects LLM(data_parallel_size>1) per process,
    # but ParallelConfig falls back to VLLM_DP_SIZE/VLLM_DP_RANK env vars
    # ("offline SPMD case", parallel.py:774-786). So each worker only sees
    # tp=1, ep on, and the env vars give it its DP rank/topology — same shape
    # as examples/offline_inference/data_parallel.py, which pops
    # data_parallel_size from kwargs before LLM(...).
    llm_kwargs["tensor_parallel_size"] = 1
    llm_kwargs["enable_expert_parallel"] = True

    sampling_dict = _build_sampling_dict(args)
    prompt_strs = [p for (p, _, _) in prompts]

    # Even-split prompts across DP ranks (data_parallel.py example pattern).
    floor = len(prompt_strs) // dp_size
    rem = len(prompt_strs) % dp_size

    def slice_for(rank: int) -> List[str]:
        start = rank * floor + min(rank, rem)
        end = (rank + 1) * floor + min(rank + 1, rem)
        s = prompt_strs[start:end]
        return s if s else ["Placeholder"]

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    print(f"[vllm_offline] DP-attn: spawning {dp_size} workers "
          f"(data_parallel_size={dp_size}, tensor_parallel_size=1, "
          f"enable_expert_parallel=True), master={master_ip}:{master_port}")

    ctx = mp.get_context("spawn")
    result_q: "mp.Queue" = ctx.Queue()
    procs: List[mp.Process] = []
    for rank in range(dp_size):
        p = ctx.Process(
            target=_dp_worker,
            args=(rank, rank, dp_size, master_ip, master_port,
                  llm_kwargs, sampling_dict, slice_for(rank),
                  args.infinitebench_output_len, result_q),
        )
        p.start()
        procs.append(p)

    rank_results: List[Dict[str, Any]] = []
    deadline = time.time() + 1800  # 30 min cap for DP run
    while len(rank_results) < dp_size and time.time() < deadline:
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
        raise RuntimeError(f"{len(failed)}/{dp_size} DP workers failed")
    if len(rank_results) < dp_size:
        raise RuntimeError(
            f"Only {len(rank_results)}/{dp_size} workers reported back")

    rank_metrics = [r["metrics"] for r in
                    sorted(rank_results, key=lambda r: r["rank"])]

    # Aggregate. timed/warmup wall = max across ranks (parallel). Token totals
    # sum. Per-request samples concatenate. Input tokens come from the full
    # prompts list at the parent (we know per-prompt token counts already).
    total_input_tokens = sum(plen for (_, plen, _) in prompts)
    agg: Dict[str, Any] = {
        "warmup_seconds": max(m["warmup_seconds"] for m in rank_metrics),
        "timed_seconds": max(m["timed_seconds"] for m in rank_metrics),
        "total_output_tokens": sum(m["total_output_tokens"] for m in rank_metrics),
        "total_input_tokens": total_input_tokens,
        "ttfts_s": [v for m in rank_metrics for v in m["ttfts_s"]],
        "tpots_s": [v for m in rank_metrics for v in m["tpots_s"]],
        "e2els_s": [v for m in rank_metrics for v in m["e2els_s"]],
        "decode_tokens_per_req": [v for m in rank_metrics for v in m["decode_tokens_per_req"]],
        "decode_iters_per_req": [v for m in rank_metrics for v in m["decode_iters_per_req"]],
        "decode_tokens_total": sum(m["decode_tokens_total"] for m in rank_metrics),
        "latency_metrics_source": "vllm_request_metrics_dp",
    }
    print(f"[vllm_offline DP] aggregated across {dp_size} ranks: "
          f"timed={agg['timed_seconds']:.2f}s, "
          f"out_toks={agg['total_output_tokens']}, "
          f"decode_toks={agg['decode_tokens_total']}")
    return agg


def _run_single(args: argparse.Namespace,
                prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm_kwargs = _build_llm_kwargs(args)
    llm_kwargs["tensor_parallel_size"] = args.tp
    if args.ep > 1:
        llm_kwargs["enable_expert_parallel"] = True

    sampling = SamplingParams(**_build_sampling_dict(args))
    print(f"[vllm_offline] LLM kwargs: {llm_kwargs}")
    print(f"[vllm_offline] SamplingParams: {sampling}")

    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[vllm_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]
    rank_metrics = _generate_with_metrics(
        llm, prompt_strs, sampling, args.infinitebench_output_len)
    rank_metrics["total_input_tokens"] = sum(plen for (_, plen, _) in prompts)
    rank_metrics["latency_metrics_source"] = "vllm_request_metrics"
    return rank_metrics


def run(args: argparse.Namespace,
        prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    if args.dp_attn:
        return _run_dp(args, prompts)
    return _run_single(args, prompts)
