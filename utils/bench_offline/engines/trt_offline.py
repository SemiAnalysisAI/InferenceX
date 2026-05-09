"""TRT-LLM offline plugin: mirrors `trtllm-serve` flags from
benchmarks/single_node/dsv4_fp4_b300_trt_mtp.sh, but uses the in-process
`tensorrt_llm.LLM` API and runs `LLM.generate(prompts)`.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

from latency_utils import append_latency_sample


def run(args: argparse.Namespace,
        prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    from tensorrt_llm import LLM, SamplingParams

    output_len = args.infinitebench_output_len
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=output_len,
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

    max_num_tokens = max(
        args.infinitebench_input_len + args.infinitebench_output_len
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
        max_seq_len=args.max_model_len,
        max_num_tokens=max_num_tokens,
        custom_tokenizer="deepseek_v4",
        return_perf_metrics=True,
        **extra_config,
    )

    print(f"[trt_offline] LLM kwargs: {llm_kwargs}")
    print(f"[trt_offline] SamplingParams: {sampling}")

    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[trt_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]

    print(f"[trt_offline] Warmup batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    _ = llm.generate(prompt_strs, sampling)
    warmup_s = time.perf_counter() - t0
    print(f"[trt_offline] Warmup done in {warmup_s:.3f}s")
    time.sleep(2.0)

    print(f"[trt_offline] Timed batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_strs, sampling)
    timed_s = time.perf_counter() - t0
    print(f"[trt_offline] Timed done in {timed_s:.3f}s")

    total_output_tokens = 0
    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []
    decode_tokens_per_req: List[int] = []
    decode_iters_per_req: List[int] = []
    n_decode_tokens = 0
    for o in outputs:
        toks = getattr(o, "outputs", None)
        n_out = len(toks[0].token_ids) if toks and hasattr(toks[0], "token_ids") else output_len
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

        for candidate in metric_candidates:
            if append_latency_sample(candidate, n_out, ttfts, e2els, tpots):
                break

        # Decode iters for CANN-style spec acceptance. TRT's RequestPerfMetrics
        # exposes firstIter / lastIter (engine iter range where this request
        # had a token generated). decode_iters = lastIter - firstIter excludes
        # the prefill-emit iter (which fires once) and counts the decode/verify
        # iters that emit subsequent tokens.
        for output in toks or []:
            perf = getattr(output, "request_perf_metrics", None)
            if perf is None:
                continue
            first_iter = (getattr(perf, "firstIter", None)
                          or getattr(perf, "first_iter", None))
            last_iter = (getattr(perf, "lastIter", None)
                         or getattr(perf, "last_iter", None))
            if first_iter is not None and last_iter is not None and last_iter >= first_iter:
                decode_iters_per_req.append(max(int(last_iter) - int(first_iter), 0))
                break

    total_input_tokens = sum(plen for (_, plen, _) in prompts)
    return {
        "warmup_seconds": warmup_s,
        "timed_seconds": timed_s,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "ttfts_s": ttfts,
        "tpots_s": tpots,
        "e2els_s": e2els,
        "decode_tokens_per_req": decode_tokens_per_req,
        "decode_iters_per_req": decode_iters_per_req,
        "decode_tokens_total": n_decode_tokens,
        "latency_metrics_source": "trt_perf_metrics",
    }
