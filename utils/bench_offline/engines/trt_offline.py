"""TRT-LLM offline plugin: mirrors `trtllm-serve` flags from the b300
trt_mtp launch script, but uses the in-process `tensorrt_llm.LLM` API and
runs `LLM.generate(prompts)`.

Step semantics: the benchmark unit is *decode rounds* (engine iterations
after the prefill-emit iteration). TRT's RequestPerfMetrics exposes firstIter/lastIter, so per-request rounds
are exact for both spec and non-spec runs: rounds = lastIter - firstIter.
The warmup batch calibrates tokens-per-round and the timed batch's token
budget targets ~`--decode-steps` forward passes per request.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

from latency_utils import (
    append_latency_sample,
    decode_window_s,
    request_rounds,
    timed_max_tokens,
    tokens_per_round_estimate,
)


def _request_iters(output: Any) -> Optional[int]:
    """Decode iterations from TRT's RequestPerfMetrics. firstIter is the
    iteration that emitted the prefill token; lastIter the final decode
    iteration — the difference counts decode/verify rounds only."""
    perf = getattr(output, "request_perf_metrics", None)
    if perf is None:
        return None
    first_iter = (getattr(perf, "firstIter", None)
                  if getattr(perf, "firstIter", None) is not None
                  else getattr(perf, "first_iter", None))
    last_iter = (getattr(perf, "lastIter", None)
                 if getattr(perf, "lastIter", None) is not None
                 else getattr(perf, "last_iter", None))
    if first_iter is not None and last_iter is not None and last_iter >= first_iter:
        return max(int(last_iter) - int(first_iter), 0)
    return None


def _extract_batch(outputs: List[Any], mtp: int,
                   fallback_output_len: int) -> Dict[str, Any]:
    total_output_tokens = 0
    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []
    tpot_steps: List[float] = []
    decode_tokens_per_req: List[int] = []
    decode_rounds_per_req: List[int] = []
    n_decode_tokens = 0
    n_rounds = 0
    for o in outputs:
        toks = getattr(o, "outputs", None)
        n_out = (len(toks[0].token_ids)
                 if toks and hasattr(toks[0], "token_ids")
                 else fallback_output_len)
        total_output_tokens += n_out
        decode_tokens = max(n_out - 1, 0)
        n_decode_tokens += decode_tokens
        decode_tokens_per_req.append(decode_tokens)

        metric_candidates = [getattr(o, "metrics_dict", None)]
        for output in toks or []:
            perf_metrics = getattr(output, "request_perf_metrics", None)
            timing_metrics = getattr(perf_metrics, "timing_metrics", None)
            metric_candidates.extend([perf_metrics, timing_metrics])
        metric_candidates.append(getattr(o, "metrics", None))

        window: Optional[float] = None
        for candidate in metric_candidates:
            if append_latency_sample(candidate, n_out, ttfts, e2els, tpots):
                window = decode_window_s(candidate)
                break

        iters: Optional[int] = None
        for output in toks or []:
            iters = _request_iters(output)
            if iters is not None:
                break
        rounds = request_rounds(decode_tokens, iters, mtp)
        if rounds is not None:
            decode_rounds_per_req.append(rounds)
            n_rounds += rounds
            if window is not None and window > 0 and rounds > 0:
                tpot_steps.append(window / rounds)

    return {
        "total_output_tokens": total_output_tokens,
        "ttfts_s": ttfts,
        "tpots_s": tpots,
        "tpot_steps_s": tpot_steps,
        "e2els_s": e2els,
        "decode_tokens_per_req": decode_tokens_per_req,
        "decode_rounds_per_req": decode_rounds_per_req,
        "decode_tokens_total": n_decode_tokens,
        "decode_rounds_total": n_rounds,
    }


def run(args: argparse.Namespace,
        prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    if args.routing_sim_strategy != "none":
        print("[trt_offline] WARN: --routing-sim-strategy="
              f"{args.routing_sim_strategy} requested, but TRT-LLM exposes "
              "no MoE routing simulation. Running REAL routing; result is "
              "labeled moe_routing=real and should not be compared against "
              "idealized expert-balance rows.")
    from tensorrt_llm import LLM, SamplingParams

    decode_steps = args.decode_steps

    def sampling(max_tokens: int) -> "SamplingParams":
        return SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            ignore_eos=args.ignore_eos,
            return_perf_metrics=True,
        )

    # Build the inline pytorch-backend config dict — equivalent to the YAML
    # the trt_mtp.sh script writes out (kv_cache_config, moe_config, mtp).
    extra_config: Dict[str, Any] = {
        "cuda_graph_config": {
            "enable_padding": True,
            "max_batch_size": args.batch_size,
        },
        "enable_attention_dp": args.dp_attn,
        "kv_cache_config": {
            "tokens_per_block": 128,
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.65,
            "enable_block_reuse": False,
        },
        "stream_interval": 10,
        "moe_config": {"backend": "TRTLLM"},
    }
    if args.dp_attn:
        extra_config["attention_dp_config"] = {
            "batching_wait_iters": 0,
            "enable_balance": True,
            "timeout_iters": 60,
        }
    if args.mtp > 0:
        extra_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp,
        }

    # Worst-case token budget per request is (mtp+1) tokens per round.
    output_budget = (args.mtp + 1) * decode_steps
    max_num_tokens = max(
        args.infinitebench_input_len + output_budget
        + (args.mtp + 1) * args.batch_size + 256,
        8192,
    )

    llm_kwargs: Dict[str, Any] = dict(
        model=args.model,
        backend="pytorch",
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tp,
        moe_expert_parallel_size=args.ep,
        max_batch_size=args.batch_size,
        max_seq_len=args.engine_max_len,
        max_num_tokens=max_num_tokens,
        custom_tokenizer="deepseek_v4",
        return_perf_metrics=True,
        **extra_config,
    )

    print(f"[trt_offline] LLM kwargs: {llm_kwargs}")

    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[trt_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]

    warmup_sampling = sampling(decode_steps)
    print(f"[trt_offline] Warmup batch: {len(prompt_strs)} prompts, "
          f"max_tokens={decode_steps}")
    t0 = time.perf_counter()
    warmup_outputs = llm.generate(prompt_strs, warmup_sampling)
    warmup_s = time.perf_counter() - t0
    print(f"[trt_offline] Warmup done in {warmup_s:.3f}s")
    warmup_stats = _extract_batch(warmup_outputs, args.mtp, decode_steps)
    tokens_per_round = tokens_per_round_estimate(
        warmup_stats["decode_tokens_total"],
        warmup_stats["decode_rounds_total"])
    if args.mtp > 0 and not warmup_stats["decode_rounds_per_req"]:
        print("[trt_offline] WARN: spec run but no firstIter/lastIter perf "
              "metrics; cannot calibrate decode-step budget. Timed run "
              "keeps max_tokens=decode_steps and step metrics will fall "
              "back to per-token accounting.")
    timed_budget = timed_max_tokens(decode_steps, tokens_per_round)
    time.sleep(2.0)

    timed_sampling = sampling(timed_budget)
    print(f"[trt_offline] Timed batch: {len(prompt_strs)} prompts, "
          f"warmup tokens/round={tokens_per_round:.3f}, "
          f"max_tokens={timed_budget} (targeting {decode_steps} "
          f"forward passes/request)")
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_strs, timed_sampling)
    timed_s = time.perf_counter() - t0
    print(f"[trt_offline] Timed done in {timed_s:.3f}s")

    stats = _extract_batch(outputs, args.mtp, timed_budget)
    total_input_tokens = sum(plen for (_, plen, _) in prompts)
    stats.update({
        "warmup_seconds": warmup_s,
        "timed_seconds": timed_s,
        "total_input_tokens": total_input_tokens,
        "warmup_tokens_per_round": tokens_per_round,
        "timed_max_tokens": timed_budget,
        "latency_metrics_source": "trt_perf_metrics",
        "moe_routing": "real",
    })
    return stats
