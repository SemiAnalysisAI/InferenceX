"""Offline lockstep benchmark for DeepSeek-V4 on Blackwell.

Runs an offline in-process decode-step workload:
  1. Load engine once (vLLM / SGLang / TRT-LLM offline mode).
  2. Build a single batch of `batch_size` InfiniteBench 8K prompts.
  3. Run one warmup `engine.generate(prompts)` call (doubles as the
     tokens-per-round calibration for spec runs).
  4. Run one timed `engine.generate(prompts)` call sized so each request
     executes ~`--decode-steps` main-model forward passes.
  5. Record per-decode-step latency and step throughput.
  6. Emit a result JSON consumable by utils/process_result.py.

Step semantics: the unit of work is the *decode step* (one main-model forward
pass). The timed run targets `--decode-steps` main-model passes per request.
`mean/median_tpot_ms` are per step, and `output_throughput` is steps/s. MTP
bonus tokens are deliberately excluded from headline metrics; observed MTP
acceptance is recorded separately (`spec_tokens_per_step_observed`) for anyone
who wants token throughput.

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
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes (for multi-node tp>8).")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="This node's rank (0=leader, >0=follower).")
    parser.add_argument("--max-model-len", type=int, default=9472)
    parser.add_argument("--mtp", type=int, default=3,
                        help="Number of MTP / EAGLE speculative tokens "
                        "(0 disables). Default 3 for offline decode-step runs.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--routing-sim-strategy", default="uniform_random",
                        choices=["uniform_random", "normal_routing", "none"],
                        help="Force simulated MoE routing for idealized expert "
                        "balance, replacing gate top-k expert ids while keeping "
                        "the benchmark workload identical. Model outputs become "
                        "invalid; performance measurement only. Only vLLM supports this "
                        "(VLLM_MOE_ROUTING_SIMULATION_STRATEGY); SGLang and "
                        "TRT runs warn and fall back to real routing, and "
                        "every result records what actually ran in "
                        "`moe_routing`. Use 'none' for honest routing.")
    # Workload
    parser.add_argument("--infinitebench-task", default=DEFAULT_INFINITEBENCH_TASK)
    parser.add_argument("--infinitebench-input-len", type=int, default=8192)
    parser.add_argument("--decode-steps", type=int, default=256,
                        help="Target main-model forward passes per request "
                        "(the prefill pass emits token 1, then "
                        "decode-steps-1 decode rounds). With MTP each round "
                        "emits 1+accepted tokens; the timed run's token "
                        "budget is calibrated from the warmup batch so the "
                        "step count still lands on this target.")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of prompts in the single warmup+timed batch "
                        "(the fixed lockstep batch size).")
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--dsv4", action="store_true", default=True)
    parser.add_argument("--dsv4-thinking-mode", default="chat",
                        choices=["chat", "thinking"])
    parser.add_argument("--moe-runner-backend", default="flashinfer_mxfp4",
                        help="SGLang MoE runner backend (flashinfer_mxfp4 for FP4/Blackwell, triton/deep_gemm/cutlass for FP8)")
    parser.add_argument("--dpa-moe-runner-backend", default=None,
                        help="SGLang MoE runner backend override used only with --dp-attn.")
    parser.add_argument("--dpa-size", type=int, default=None,
                        help="SGLang DP-attention data parallel size; defaults to TP.")
    parser.add_argument("--dpa-moe-a2a-backend", default="deepep",
                        choices=["none", "deepep", "mooncake", "nixl", "mori", "flashinfer"],
                        help="SGLang MoE A2A backend used only with --dp-attn.")
    parser.add_argument("--moe-dense-tp-size", type=int, default=None,
                        help="SGLang dense MoE TP size for DP-attention layouts.")
    parser.add_argument("--enable-dp-lm-head", action="store_true",
                        help="Enable SGLang DP LM head for DP-attention layouts.")
    parser.add_argument("--deepep-mode", default=None,
                        choices=["auto", "normal", "low_latency"],
                        help="SGLang DeepEP mode.")
    parser.add_argument("--sglang-dpa-env-preset", default="fp4",
                        choices=["fp4", "fp8", "none"],
                        help="SGLang DP-attention environment preset.")
    parser.add_argument("--cpu-offload-gb", type=int, default=0,
                        help="SGLang CPU offload budget in GiB.")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="SGLang KV cache dtype.")
    parser.add_argument("--quantization", default=None,
                        help="SGLang quantization mode.")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Disable SGLang CUDA graph capture.")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None,
                        help="Maximum SGLang batch size to capture with CUDA graphs.")
    parser.add_argument("--chunked-prefill-size", type=int, default=None,
                        help="SGLang chunked prefill size.")
    parser.add_argument("--max-running-requests", type=int, default=None,
                        help="SGLang max running requests.")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85,
                        help="SGLang static memory fraction.")
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

    # Engine context length must fit the worst-case timed run: prompt +
    # 1 prefill token + (decode_steps-1) rounds × (1+mtp) tokens, plus slack
    # for prompt-length mismatch. The sweep-provided MAX_MODEL_LEN is
    # isl+osl+256 which under-provisions MTP runs (every draft accepted).
    worst_case_len = (args.infinitebench_input_len + 64 + 1
                      + (args.mtp + 1) * (args.decode_steps - 1))
    args.engine_max_len = max(args.max_model_len, worst_case_len)
    if args.engine_max_len > args.max_model_len:
        print(f"[run_offline] Raising engine context length "
              f"{args.max_model_len} -> {args.engine_max_len} to fit "
              f"{args.decode_steps} decode steps at mtp={args.mtp}.")

    print(f"[run_offline] engine={args.engine} bs={args.batch_size} "
          f"isl={args.infinitebench_input_len} "
          f"decode_steps={args.decode_steps} "
          f"mtp={args.mtp} tp={args.tp} ep={args.ep} dp_attn={args.dp_attn} "
          f"nnodes={args.nnodes} node_rank={args.node_rank} "
          f"routing_sim={args.routing_sim_strategy}")

    if args.node_rank > 0 and not args.dp_attn:
        if args.engine == "vllm":
            from vllm_offline import run_follower
        else:
            raise ValueError(f"Multi-node follower not supported for engine={args.engine}")
        run_follower(args)
        sys.exit(0)

    tokenizer = _load_tokenizer(
        args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    # The prompt suffix's "in N words" uses the decode-step target.
    prompts = build_infinitebench_prompts(
        dataset_path=args.dataset_path,
        task=args.infinitebench_task,
        input_len=args.infinitebench_input_len,
        output_len=args.decode_steps,
        num_prompts=args.batch_size,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
        dsv4=args.dsv4,
        dsv4_thinking_mode=args.dsv4_thinking_mode,
    )

    metrics = _engine_run(args, prompts)

    if args.node_rank > 0:
        print(f"[run_offline] Follower node {args.node_rank} done.")
        sys.exit(0)

    timed_s = metrics["timed_seconds"]
    total_output_tokens = metrics["total_output_tokens"]

    def _stats_ms(samples: List[float], iqr_filter: bool = False
                  ) -> Dict[str, float]:
        """Compute mean/median/p* in milliseconds. When iqr_filter=True,
        drop samples above q3 + 1.5xIQR before computing mean. Percentiles are
        still computed on the full sorted set so the tail isn't hidden."""
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
            "min": full_sorted[0] * 1000.0,
            "p90": _pct_of(full_sorted, 90) * 1000.0,
            "p99": _pct_of(full_sorted, 99) * 1000.0,
            "p99.9": _pct_of(full_sorted, 99.9) * 1000.0,
            "std": (((sum((x - mean_v) ** 2 for x in mean_arr)
                      / n_filtered) ** 0.5) * 1000.0),
            "_n_full": n_full,
            "_n_iqr_filtered": n_filtered,
        }

    ttft_samples = metrics.get("ttfts_s") or []
    step_samples = metrics.get("tpot_steps_s") or []
    token_tpot_samples = metrics.get("tpots_s") or []
    e2el_samples = metrics.get("e2els_s") or []
    ttft_stats = _stats_ms(ttft_samples)
    e2el_stats = _stats_ms(e2el_samples)

    # Headline TPOT = per-DECODE-STEP latency, with IQR filtering on the mean.
    # The per-output-token distribution is kept as a secondary metric.
    tpot_unit = "decode_step"
    tpot_stats = _stats_ms(step_samples, iqr_filter=True)
    token_tpot_stats = _stats_ms(token_tpot_samples, iqr_filter=True)
    used_tpot_fallback = False
    if not tpot_stats and token_tpot_stats:
        # No step telemetry. For non-spec runs per-token == per-step so this
        # is exact; for spec runs it understates step time — label it.
        tpot_stats = token_tpot_stats
        tpot_unit = ("decode_step" if args.mtp <= 0
                     else "output_token_fallback")
        if args.mtp > 0:
            print("[run_offline] WARN: spec run without step telemetry; "
                  "headline TPOT is per OUTPUT TOKEN, not per decode step "
                  "(tpot_unit=output_token_fallback). Do not compare this "
                  "row against offline decode-step per-step numbers.")
    if not tpot_stats:
        used_tpot_fallback = True
        fallback_ms = timed_s * 1000.0 / max(args.decode_steps, 1)
        print("[run_offline] WARN: no engine latency samples; using full "
              f"wall-clock per-step fallback ({fallback_ms:.2f} ms).")
        tpot_stats = {
            "mean": fallback_ms, "median": fallback_ms,
            "p90": fallback_ms, "p99": fallback_ms,
            "p99.9": fallback_ms, "std": 0.0,
        }
        tpot_unit = "wall_clock_fallback"

    used_ttft_fallback = not ttft_stats
    used_e2el_fallback = not e2el_stats
    if used_ttft_fallback:
        ttft_stats = {"mean": 0.0, "median": 0.0, "p90": 0.0,
                      "p99": 0.0, "p99.9": 0.0, "std": 0.0}
    if used_e2el_fallback:
        e2el_stats = {"mean": timed_s * 1000.0, "median": timed_s * 1000.0,
                      "p90": timed_s * 1000.0, "p99": timed_s * 1000.0,
                      "p99.9": timed_s * 1000.0, "std": 0.0}

    engine_latency_source = metrics.get("latency_metrics_source") or "engine"
    if used_tpot_fallback:
        latency_metrics_source = "fallback_wall_clock"
    elif used_ttft_fallback or tpot_unit != "decode_step":
        latency_metrics_source = f"{engine_latency_source}_partial"
    else:
        latency_metrics_source = engine_latency_source

    # Offline decode-step throughput: requests_per_chip / step_time == decode
    # steps/s per chip. One main-model token per step; MTP bonus tokens
    # excluded by construction.
    mean_tpot_ms = tpot_stats["mean"]
    steps_per_user = (1000.0 / mean_tpot_ms) if mean_tpot_ms > 0 else 0.0
    decode_step_throughput = steps_per_user * args.batch_size
    decode_step_throughput_per_chip = decode_step_throughput / args.num_chips

    decode_tokens_total = metrics.get("decode_tokens_total") or 0
    decode_rounds_total = metrics.get("decode_rounds_total") or 0
    spec_tokens_per_step = None
    if decode_rounds_total > 0 and decode_tokens_total > 0:
        spec_tokens_per_step = decode_tokens_total / decode_rounds_total

    # Wall-clock aggregates (incl. prefill) kept for reference.
    wall_clock_output_throughput = (total_output_tokens / timed_s
                                    if timed_s > 0 else 0.0)
    total_input_tokens = metrics.get("total_input_tokens") or 0
    input_throughput = (total_input_tokens / timed_s if timed_s > 0 else 0.0)

    # output_throughput: what process_result divides by tp_size for
    # output_tput_per_gpu. Bound to the per-step rate so dashboard rows report
    # decode steps/s/chip for offline runs.
    output_throughput = decode_step_throughput
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
        "max_model_len": args.engine_max_len,
        "temperature": args.temperature,
        "dataset_name": "infinitebench",
        "infinitebench_task": args.infinitebench_task,
        "infinitebench_input_len": args.infinitebench_input_len,
        "dsv4_thinking_mode": args.dsv4_thinking_mode,
        "batch_size": args.batch_size,
        "max_concurrency": args.batch_size,
        "num_prompts": args.batch_size,
        "warmup_seconds": metrics.get("warmup_seconds"),
        "timed_seconds": timed_s,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "latency_metrics_source": latency_metrics_source,
        # What MoE routing actually ran: "simulated_<strategy>" when the
        # engine honored --routing-sim-strategy, "real" otherwise. Engines
        # without routing simulation always report "real" — never assume
        # the request was honored.
        "moe_routing": metrics.get(
            "moe_routing",
            "real" if args.routing_sim_strategy == "none" else "unknown"),
        # Step accounting (the benchmark unit).
        "tpot_unit": tpot_unit,
        "decode_steps_target": args.decode_steps,
        "decode_rounds_total": decode_rounds_total,
        "decode_tokens_total": decode_tokens_total,
        "step_sample_count": len(step_samples),
        "ttft_sample_count": len(ttft_samples),
        "e2el_sample_count": len(e2el_samples),
        "used_ttft_fallback": used_ttft_fallback,
        "used_tpot_fallback": used_tpot_fallback,
        "used_e2el_fallback": used_e2el_fallback,
        # decode_throughput: steps/s == requests × (1000/mean_step_ms).
        "decode_throughput": decode_step_throughput,
        "decode_step_throughput": decode_step_throughput,
        "decode_step_throughput_per_chip": decode_step_throughput_per_chip,
        "output_throughput": output_throughput,
        "total_token_throughput": total_token_throughput,
        "input_throughput": input_throughput,
        "wall_clock_output_token_throughput": wall_clock_output_throughput,
        "wall_clock_output_token_throughput_per_chip":
            wall_clock_output_throughput / args.num_chips,
        "wall_clock_total_throughput":
            (total_input_tokens + total_output_tokens) / timed_s
            if timed_s > 0 else 0.0,
        # Distributions — required by summarize.py. TPOT fields are PER
        # DECODE STEP (see tpot_unit).
        "mean_ttft_ms": ttft_stats["mean"],
        "median_ttft_ms": ttft_stats["median"],
        "p90_ttft_ms": ttft_stats["p90"],
        "p99_ttft_ms": ttft_stats["p99"],
        "p99.9_ttft_ms": ttft_stats["p99.9"],
        "std_ttft_ms": ttft_stats["std"],
        "mean_tpot_ms": tpot_stats["mean"],
        "median_tpot_ms": tpot_stats["median"],
        # Min over requests: prefill is staggered under chunked prefill, so
        # the last-prefilled request's decode window is the purest full-batch
        # lockstep measurement.
        "min_tpot_ms": tpot_stats.get("min", tpot_stats["median"]),
        "p90_tpot_ms": tpot_stats["p90"],
        "p99_tpot_ms": tpot_stats["p99"],
        "p99.9_tpot_ms": tpot_stats["p99.9"],
        "std_tpot_ms": tpot_stats["std"],
        "mean_e2el_ms": e2el_stats["mean"],
        "median_e2el_ms": e2el_stats["median"],
        "p90_e2el_ms": e2el_stats["p90"],
        "p99_e2el_ms": e2el_stats["p99"],
        "p99.9_e2el_ms": e2el_stats["p99.9"],
        "std_e2el_ms": e2el_stats["std"],
        "benchmark_input_len": args.infinitebench_input_len,
        "benchmark_output_len": args.decode_steps,
    }
    # Per-output-token distribution as secondary fields. Only written when
    # real samples exist — process_result.py runs `1000/float(v)` over
    # `*_ms` keys containing `tpot`, so no nulls.
    if token_tpot_stats:
        result["mean_tpot_per_token_ms"] = token_tpot_stats["mean"]
        result["median_tpot_per_token_ms"] = token_tpot_stats["median"]
    if spec_tokens_per_step is not None:
        result["spec_tokens_per_step_observed"] = spec_tokens_per_step
        result["mtp_accepted_per_step_observed"] = spec_tokens_per_step - 1.0
    if metrics.get("warmup_tokens_per_round") is not None:
        result["warmup_tokens_per_round"] = metrics["warmup_tokens_per_round"]
    if metrics.get("timed_max_tokens") is not None:
        result["timed_max_tokens"] = metrics["timed_max_tokens"]

    for kv in args.metadata:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        result[k] = v

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.result_filename}.json"
    out_path.write_text(json.dumps(result, indent=2))

    if args.nnodes > 1:
        (out_dir / ".offline_follower_exit").touch()

    # Also stdout the metric line that downstream collectors look for.
    print("============ Offline Benchmark Result ============")
    print(f"Engine:                            {args.engine}")
    print(f"Batch size (= concurrency):        {args.batch_size}")
    print(f"Decode steps target / observed:    {args.decode_steps} / "
          f"{decode_rounds_total / max(args.batch_size, 1):.1f} per request")
    print(f"Warmup duration (s):               {metrics.get('warmup_seconds', 0):.3f}")
    print(f"Timed duration (s):                {timed_s:.3f}")
    print(f"Total output tokens:               {total_output_tokens}")
    print(f"MoE routing:                       "
          f"{result['moe_routing']}")
    print(f"TPOT unit:                         {tpot_unit}")
    print(f"mean TPOT ms / median TPOT ms:     "
          f"{tpot_stats['mean']:.2f} / {tpot_stats['median']:.2f}  "
          f"(per decode step)")
    if spec_tokens_per_step is not None:
        print(f"Observed tokens/step (1+accepted): {spec_tokens_per_step:.3f}")
    print(f"Decode steps/s (batch):            {decode_step_throughput:.2f}")
    print(f"Decode steps/s/chip:               {decode_step_throughput_per_chip:.2f}  "
          f"(headline offline throughput unit)")
    print(f"Wall-clock out tok/s (incl. MTP):  {wall_clock_output_throughput:.2f}")
    print(f"Latency metrics source:            {latency_metrics_source}")
    print(f"TTFT / step sample counts:         "
          f"{len(ttft_samples)} / {len(step_samples)}")
    print(f"mean TTFT ms / median TTFT ms:     "
          f"{ttft_stats['mean']:.2f} / {ttft_stats['median']:.2f}")
    print(f"Interactivity (steps/s/user):      "
          f"{1000.0/tpot_stats['mean']:.2f}" if tpot_stats.get("mean") else "n/a")
    print(f"Result JSON:                       {out_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
