"""Offline lockstep benchmark for DeepSeek-V4 on Blackwell.

Replicates cann-recipes-infer/models/deepseek-v4/infer.sh + infer.py shape:
  1. Load engine once (vLLM / SGLang / TRT-LLM offline mode).
  2. Build a single batch of `batch_size` InfiniteBench 8K/256 prompts.
  3. Run one warmup `engine.generate(prompts)` call.
  4. Run one timed `engine.generate(prompts)` call.
  5. Record total wall-clock, total output tokens, derived per-chip metrics.
  6. Emit a result JSON consumable by utils/process_result.py.

This is the engine-side analog of the HTTP serving benchmark
utils/bench_serving/benchmark_serving.py — identical prompt construction,
but no continuous batching, no request-rate, no HTTP front-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Use sibling imports rather than `utils.bench_offline.*` — utils/ has no
# __init__.py, and the namespace-package import that works in the parent
# breaks in vLLM/TRT spawn workers (multiprocessing re-runs this file via
# runpy and the child interpreter can't resolve `utils.bench_offline`).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_ENGINES_DIR = _THIS_DIR / "engines"
if str(_ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINES_DIR))
# Also expose utils/bench_serving so encoding_dsv4 and friends are importable.
_UTILS_DIR = _THIS_DIR.parent
_BENCH_SERVING = _UTILS_DIR / "bench_serving"
if str(_BENCH_SERVING) not in sys.path:
    sys.path.insert(0, str(_BENCH_SERVING))

from prompts import DEFAULT_INFINITEBENCH_TASK, build_infinitebench_prompts


def _load_tokenizer(model: str, tokenizer_mode: str = "auto",
                    trust_remote_code: bool = True):
    """Use vllm's get_tokenizer if available — DSV4-Pro's tokenizer requires
    vllm's `tokenizer_mode='deepseek_v4'` since plain transformers AutoTokenizer
    rejects the `model_type: deepseek_v4` config. Fall back to AutoTokenizer
    for non-DSV4 models."""
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer as _vllm_get
    except ImportError:
        _vllm_get = None
    if _vllm_get is not None:
        try:
            return _vllm_get(
                model,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            pass
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code)


def _engine_run(args: argparse.Namespace,
                prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    if args.engine == "vllm":
        from vllm_offline import run as engine_run
    elif args.engine == "sglang":
        from sglang_offline import run as engine_run
    elif args.engine == "trt":
        from trt_offline import run as engine_run
    else:
        raise ValueError(f"Unknown engine: {args.engine}")
    return engine_run(args, prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang", "trt"], required=True)
    parser.add_argument("--model", required=True,
                        help="Model path or HF id (also used as tokenizer source).")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--dp-attn", action="store_true")
    parser.add_argument("--num-chips", type=int, default=8,
                        help="Total chips (==tp here for single-node).")
    parser.add_argument("--max-model-len", type=int, default=9472)
    parser.add_argument("--mtp", type=int, default=2,
                        help="Number of MTP / EAGLE speculative tokens (0 disables).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    # Workload
    parser.add_argument("--infinitebench-task", default=DEFAULT_INFINITEBENCH_TASK)
    parser.add_argument("--infinitebench-input-len", type=int, default=8192)
    parser.add_argument("--infinitebench-output-len", type=int, default=256)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of prompts in the single warmup+timed batch "
                        "(== CANN's `data_config.batch_size`).")
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--dsv4", action="store_true", default=True)
    parser.add_argument("--dsv4-thinking-mode", default="chat",
                        choices=["chat", "thinking"])
    parser.add_argument("--moe-runner-backend", default="flashinfer_mxfp4",
                        help="SGLang MoE runner backend (flashinfer_mxfp4 for FP4/Blackwell, marlin for FP8/Hopper)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--tokenizer-mode", default="deepseek_v4",
                        help="Passed to vllm.get_tokenizer; use 'deepseek_v4' "
                        "for DSV4-Pro, 'auto' for HF-recognized models.")
    # Output
    parser.add_argument("--result-dir", default=".")
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--metadata", nargs="*", default=[],
                        help='Extra "key=value" entries to embed in the result JSON.')
    args = parser.parse_args()

    print(f"[run_offline] engine={args.engine} bs={args.batch_size} "
          f"isl={args.infinitebench_input_len} osl={args.infinitebench_output_len} "
          f"mtp={args.mtp} tp={args.tp} ep={args.ep} dp_attn={args.dp_attn}")

    tokenizer = _load_tokenizer(
        args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    prompts = build_infinitebench_prompts(
        dataset_path=args.dataset_path,
        task=args.infinitebench_task,
        input_len=args.infinitebench_input_len,
        output_len=args.infinitebench_output_len,
        num_prompts=args.batch_size,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
        dsv4=args.dsv4,
        dsv4_thinking_mode=args.dsv4_thinking_mode,
    )

    metrics = _engine_run(args, prompts)

    # Compute distributions from per-request engine timing (apples-to-apples
    # with the online HTTP path, which derives mean_tpot_ms from engine-side
    # decode-step timing too — i.e. excludes prefill from per-token rate).
    # Falls back to the simple throughput/concurrency formula when the engine
    # didn't surface per-request metrics.
    timed_s = metrics["timed_seconds"]
    total_output_tokens = metrics["total_output_tokens"]

    # Wall-clock aggregate (kept for reference). The dashboard's
    # `output_tput_per_gpu` will be derived from the decode-only number
    # below — see the `output_throughput` assignment in `result`.
    wall_clock_output_throughput = (total_output_tokens / timed_s
                                    if timed_s > 0 else 0.0)
    wall_clock_output_throughput_per_chip = (
        wall_clock_output_throughput / args.num_chips)

    def _stats_ms(samples: List[float], iqr_filter: bool = False
                  ) -> Dict[str, float]:
        """Compute mean/median/p* in milliseconds. When iqr_filter=True,
        drop samples above q3 + 1.5×IQR before computing mean (matches
        cann-recipes-infer's executor/utils/common_utils.py::process_infer_time
        outlier-trim on decode-step times). Percentiles are still computed on
        the full sorted set so the tail isn't hidden."""
        if not samples:
            return {}
        full_sorted = sorted(samples)
        n_full = len(full_sorted)

        def _pct_of(arr, p):
            n = len(arr)
            if n == 0:
                return 0.0
            k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return arr[k]

        # Mean: optionally filter upper-tail outliers via IQR 1.5x rule.
        mean_arr = full_sorted
        n_filtered = n_full
        if iqr_filter and n_full >= 4:
            q1 = _pct_of(full_sorted, 25)
            q3 = _pct_of(full_sorted, 75)
            upper = q3 + 1.5 * (q3 - q1)
            mean_arr = [x for x in full_sorted if x <= upper]
            n_filtered = len(mean_arr) or n_full
            if n_filtered != n_full:
                print(f"[run_offline] IQR-filtered {n_full - n_filtered}/"
                      f"{n_full} TPOT outliers above q3+1.5*IQR={upper*1000:.2f}ms.")
        mean_v = sum(mean_arr) / n_filtered

        return {
            "mean": mean_v * 1000.0,
            "median": _pct_of(full_sorted, 50) * 1000.0,
            "p90": _pct_of(full_sorted, 90) * 1000.0,
            "p99": _pct_of(full_sorted, 99) * 1000.0,
            "p99.9": _pct_of(full_sorted, 99.9) * 1000.0,
            "std": (((sum((x - mean_v) ** 2 for x in mean_arr)
                      / n_filtered) ** 0.5) * 1000.0),
            "_n_full": n_full,
            "_n_iqr_filtered": n_filtered,
        }

    ttft_samples = metrics.get("ttfts_s") or []
    tpot_samples = metrics.get("tpots_s") or []
    e2el_samples = metrics.get("e2els_s") or []
    ttft_stats = _stats_ms(ttft_samples)
    # TPOT mean uses the CANN process_infer_time IQR outlier filter
    # (q3 + 1.5*IQR upper bound) so jittery per-request decode steps don't
    # inflate the headline number. Percentiles still reflect the full tail.
    tpot_stats = _stats_ms(tpot_samples, iqr_filter=True)
    e2el_stats = _stats_ms(e2el_samples)

    used_tpot_fallback = not tpot_stats
    used_ttft_fallback = not ttft_stats
    used_e2el_fallback = not e2el_stats
    tpot_fallback_ms = None
    output_len = args.infinitebench_output_len
    if used_tpot_fallback:
        tpot_fallback_ms = (timed_s * 1000.0 / max(output_len, 1))
        print("[run_offline] WARN: no engine TPOT samples; using full "
              f"wall-clock TPOT fallback ({tpot_fallback_ms:.2f} ms).")
        tpot_stats = {
            "mean": tpot_fallback_ms, "median": tpot_fallback_ms,
            "p90": tpot_fallback_ms, "p99": tpot_fallback_ms,
            "p99.9": tpot_fallback_ms, "std": 0.0,
        }
    if used_ttft_fallback:
        ttft_stats = {"mean": 0.0, "median": 0.0, "p90": 0.0,
                      "p99": 0.0, "p99.9": 0.0, "std": 0.0}
    if used_e2el_fallback:
        e2el_stats = {"mean": timed_s * 1000.0, "median": timed_s * 1000.0,
                      "p90": timed_s * 1000.0, "p99": timed_s * 1000.0,
                      "p99.9": timed_s * 1000.0, "std": 0.0}

    engine_latency_source = metrics.get("latency_metrics_source") or "engine"
    if not (ttft_samples or tpot_samples or e2el_samples):
        latency_metrics_source = "fallback_wall_clock"
    elif used_tpot_fallback or used_ttft_fallback:
        latency_metrics_source = f"{engine_latency_source}_partial"
    else:
        latency_metrics_source = engine_latency_source

    # Headline TPOT = mean across per-request TPOT samples (with IQR filter
    # already applied by _stats_ms). For uniform-osl workloads this matches
    # CANN's token-weighted aggregate; for varying-osl it's a simple mean
    # of ratios (close enough; not our use case).
    mean_tpot_ms = tpot_stats["mean"]
    decode_per_user = (1000.0 / mean_tpot_ms) if mean_tpot_ms > 0 else 0.0
    decode_throughput = decode_per_user * args.batch_size
    decode_throughput_per_chip = decode_throughput / args.num_chips

    # Per-decode-step TPOT for spec-decoding AR display. Real engine
    # telemetry only — leave unreported when the engine doesn't surface
    # per-request iter counts.
    decode_tokens_per_req = metrics.get("decode_tokens_per_req") or []
    decode_iters_per_req = metrics.get("decode_iters_per_req") or []
    sum_decode_tokens = sum(decode_tokens_per_req)
    sum_decode_iters = sum(decode_iters_per_req)
    spec_tokens_per_step = None
    mean_tpot_per_step_ms = None
    median_tpot_per_step_ms = None
    if sum_decode_iters > 0 and sum_decode_tokens > 0:
        spec_tokens_per_step = sum_decode_tokens / sum_decode_iters
        mean_tpot_per_step_ms = mean_tpot_ms * spec_tokens_per_step
        median_tpot_per_step_ms = tpot_stats["median"] * spec_tokens_per_step

    # Use decode-only as the canonical "output throughput" so the dashboard's
    # `output_tput_per_gpu` (= output_throughput / tp_size) shows decode tok/s
    # per chip — matching CANN's reported metric. Wall-clock values are kept
    # in `wall_clock_*` fields for reference.
    output_throughput = decode_throughput
    output_throughput_per_chip = decode_throughput_per_chip
    total_input_tokens = metrics.get("total_input_tokens") or 0
    input_throughput = (total_input_tokens / timed_s
                        if timed_s > 0 else 0.0)
    total_token_throughput = output_throughput + input_throughput

    result: Dict[str, Any] = {
        "engine": args.engine,
        "engine_mode": "offline",
        "model_id": args.model,
        "served_model_name": args.served_model_name or args.model,
        "tp": args.tp,
        "ep": args.ep,
        "dp_attn": args.dp_attn,
        "num_chips": args.num_chips,
        "mtp": args.mtp,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "dataset_name": "infinitebench",
        "infinitebench_task": args.infinitebench_task,
        "infinitebench_input_len": args.infinitebench_input_len,
        "infinitebench_output_len": args.infinitebench_output_len,
        "dsv4_thinking_mode": args.dsv4_thinking_mode,
        "batch_size": args.batch_size,
        "max_concurrency": args.batch_size,
        "num_prompts": args.batch_size,
        "warmup_seconds": metrics.get("warmup_seconds"),
        "timed_seconds": timed_s,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": metrics.get("total_input_tokens"),
        "latency_metrics_source": latency_metrics_source,
        "ttft_sample_count": len(ttft_samples),
        "tpot_sample_count": len(tpot_samples),
        "e2el_sample_count": len(e2el_samples),
        "used_ttft_fallback": used_ttft_fallback,
        "used_tpot_fallback": used_tpot_fallback,
        "used_e2el_fallback": used_e2el_fallback,
        # decode_throughput: derived from per-request decode-step time
        # (mean_tpot_ms × batch_size⁻¹) — apples-to-apples with the HTTP
        # serving path, which also reports decode-only TPOT.
        "decode_throughput": decode_throughput,
        "decode_throughput_from_mean_tpot": decode_throughput,
        "decode_throughput_per_chip_from_mean_tpot": decode_throughput_per_chip,
        # output_throughput: decode-only — what process_result divides by
        # tp_size for output_tput_per_gpu. We bind this to the engine's
        # decode-step rate so the dashboard column shows decode tok/s/chip
        # (apples-to-apples with CANN's process_infer_time-derived metric).
        "output_throughput": output_throughput,
        "total_token_throughput": total_token_throughput,
        "input_throughput": input_throughput,
        # Wall-clock aggregates (incl. prefill) for reference.
        "wall_clock_output_throughput": wall_clock_output_throughput,
        "wall_clock_output_throughput_per_chip": wall_clock_output_throughput_per_chip,
        "wall_clock_total_throughput":
            (total_input_tokens + total_output_tokens) / timed_s
            if timed_s > 0 else 0.0,
        # Distributions — required by summarize.py
        "mean_ttft_ms": ttft_stats["mean"],
        "median_ttft_ms": ttft_stats["median"],
        "p90_ttft_ms": ttft_stats["p90"],
        "p99_ttft_ms": ttft_stats["p99"],
        "p99.9_ttft_ms": ttft_stats["p99.9"],
        "std_ttft_ms": ttft_stats["std"],
        "mean_tpot_ms": tpot_stats["mean"],
        "median_tpot_ms": tpot_stats["median"],
        "p90_tpot_ms": tpot_stats["p90"],
        "p99_tpot_ms": tpot_stats["p99"],
        "p99.9_tpot_ms": tpot_stats["p99.9"],
        "std_tpot_ms": tpot_stats["std"],
        # decode aggregates (raw inputs to CANN-style metrics).
        "decode_iters_total": sum_decode_iters,
        "decode_tokens_total": sum_decode_tokens,
        "mean_e2el_ms": e2el_stats["mean"],
        "median_e2el_ms": e2el_stats["median"],
        "p90_e2el_ms": e2el_stats["p90"],
        "p99_e2el_ms": e2el_stats["p99"],
        "p99.9_e2el_ms": e2el_stats["p99.9"],
        "std_e2el_ms": e2el_stats["std"],
        "benchmark_input_len": args.infinitebench_input_len,
        "benchmark_output_len": args.infinitebench_output_len,
    }
    if tpot_fallback_ms is not None:
        result["tpot_fallback_ms"] = tpot_fallback_ms
    # Only emit per-step TPOT when we have real engine telemetry. Don't
    # write `null` for *_ms fields — process_result.py runs `1000/float(v)`
    # over every key matching `*_ms` containing `tpot`.
    if mean_tpot_per_step_ms is not None:
        result["mean_tpot_per_step_ms"] = mean_tpot_per_step_ms
    if median_tpot_per_step_ms is not None:
        result["median_tpot_per_step_ms"] = median_tpot_per_step_ms
    if spec_tokens_per_step is not None:
        result["spec_tokens_per_step_observed"] = spec_tokens_per_step

    for kv in args.metadata:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        result[k] = v

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.result_filename}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Also stdout the metric line that downstream collectors look for.
    print("============ Offline Benchmark Result ============")
    print(f"Engine:                            {args.engine}")
    print(f"Batch size (= concurrency):        {args.batch_size}")
    print(f"Warmup duration (s):               {metrics.get('warmup_seconds', 0):.3f}")
    print(f"Timed duration (s):                {timed_s:.3f}")
    print(f"Total output tokens:               {total_output_tokens}")
    # Decode-only is the canonical metric (also = result['output_throughput']).
    print(f"Decode tok/s (decode-only):        {decode_throughput:.2f}")
    print(f"Decode tok/s/chip (decode-only):   {decode_throughput_per_chip:.2f}")
    # Wall-clock aggregate kept for reference.
    print(f"Wall-clock tok/s (incl. prefill):  {wall_clock_output_throughput:.2f}")
    print(f"Wall-clock tok/s/chip:             {wall_clock_output_throughput_per_chip:.2f}")
    print(f"Latency metrics source:            {latency_metrics_source}")
    print(f"TTFT / TPOT sample counts:         "
          f"{len(ttft_samples)} / {len(tpot_samples)}")
    print(f"mean TTFT ms / median TTFT ms:     "
          f"{ttft_stats['mean']:.2f} / {ttft_stats['median']:.2f}")
    print(f"mean TPOT ms / median TPOT ms:     "
          f"{tpot_stats['mean']:.2f} / {tpot_stats['median']:.2f}  "
          f"(per output token, sum/sum aggregate)")
    if mean_tpot_per_step_ms is not None and spec_tokens_per_step is not None:
        print(f"mean / median per-step TPOT ms:    "
              f"{mean_tpot_per_step_ms:.2f} / {median_tpot_per_step_ms:.2f}  "
              f"(observed {spec_tokens_per_step:.2f} tokens/step, CANN-style)")
    else:
        print("mean / median per-step TPOT ms:    "
              "n/a (engine did not report per-request decode iters)")
    print(f"Interactivity (tok/s/user):        "
          f"{1000.0/tpot_stats['mean']:.2f}" if tpot_stats.get("mean") else "n/a")
    print(f"Result JSON:                       {out_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
