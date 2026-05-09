"""SGLang offline plugin: mirrors `sglang serve` flags from
benchmarks/single_node/dsv4_fp4_b300_sglang_mtp.sh, but uses `sgl.Engine`
in-process and runs `Engine.generate(prompts)`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

from latency_utils import to_seconds

# Re-imported below where used.


def _sgl_env_for_dsv4(dp_attn: bool, dpa_env_preset: str) -> None:
    """Mirror the SGLANG_* env exports the b300 sglang_mtp launch script
    sets before `sglang serve`. Apply before `sgl.Engine` init."""
    common = {
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT": "1",
        "SGLANG_OPT_USE_JIT_NORM": "1",
        "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
        "SGLANG_OPT_USE_TOPK_V2": "1",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "1",
    }
    for k, v in common.items():
        os.environ.setdefault(k, v)
    # Non-streaming SGLang only flushes intermediate token batches every
    # N tokens. Force per-token flushes so request timing has real decode
    # latency instead of only final wall-clock latency.
    os.environ["SGLANG_FORCE_STREAM_INTERVAL"] = "1"
    if not dp_attn:
        return

    if dpa_env_preset == "fp4":
        for k, v in {
            "SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN": "1",
            "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1",
            "SGLANG_OPT_FIX_HASH_MEGA_MOE": "1",
            "SGLANG_OPT_USE_FAST_MASK_EP": "1",
            "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": "1",
            "SGLANG_OPT_USE_JIT_EP_ACTIVATION": "1",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
            "SGLANG_OPT_FIX_NEXTN_MEGA_MOE": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "0",
            "SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2": "1",
        }.items():
            os.environ[k] = v
    elif dpa_env_preset == "fp8":
        os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "1")


def run(args: argparse.Namespace,
        prompts: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    _sgl_env_for_dsv4(args.dp_attn, args.sglang_dpa_env_preset)
    import sglang as sgl

    output_len = args.infinitebench_output_len
    sampling_params = {
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_new_tokens": output_len,
        "ignore_eos": args.ignore_eos,
    }

    # MTP-N via EAGLE chain (matches sglang_mtp.sh).
    eagle_steps = args.mtp
    eagle_draft = max(eagle_steps + 1, 1)
    spec_kwargs: Dict[str, Any] = {}
    if args.mtp > 0:
        spec_kwargs.update({
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": eagle_steps,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": eagle_draft,
        })

    # sgl.Engine supports DP-attn natively in single-process: it spawns
    # `dp_size` scheduler subprocesses internally (one per DP rank), each
    # replicating attention while sharing experts via EP.
    if args.dp_attn:
        deepep_config = (
            '{"normal_dispatch":{"num_sms":96},'
            '"normal_combine":{"num_sms":96}}'
        )
        parallel_kwargs: Dict[str, Any] = {
            "tp_size": args.tp,
            "dp_size": args.tp,           # DP across all TP ranks
            "ep_size": args.ep if args.ep > 1 else args.tp,
            "enable_dp_attention": True,
            "moe_a2a_backend": "deepep",
            "deepep_config": deepep_config,
            "chunked_prefill_size": (
                args.chunked_prefill_size
                if args.chunked_prefill_size is not None
                else 32768
            ),
        }
        if args.dpa_moe_runner_backend is not None:
            parallel_kwargs["moe_runner_backend"] = args.dpa_moe_runner_backend
        if args.moe_dense_tp_size is not None:
            parallel_kwargs["moe_dense_tp_size"] = args.moe_dense_tp_size
        if args.enable_dp_lm_head:
            parallel_kwargs["enable_dp_lm_head"] = True
        if args.deepep_mode is not None:
            parallel_kwargs["deepep_mode"] = args.deepep_mode
    else:
        parallel_kwargs = {
            "tp_size": args.tp,
            "moe_runner_backend": getattr(args, "moe_runner_backend", "flashinfer_mxfp4"),
            "disable_flashinfer_autotune": True,
            "chunked_prefill_size": (
                args.chunked_prefill_size
                if args.chunked_prefill_size is not None
                else 8192
            ),
        }
        if args.ep > 1:
            parallel_kwargs["ep_size"] = args.ep

    max_running_requests = (
        args.max_running_requests
        if args.max_running_requests is not None
        else args.batch_size
    )
    if args.dp_attn:
        dp_size = int(parallel_kwargs["dp_size"])
        if max_running_requests < dp_size:
            print(
                "[sglang_offline] Raising max_running_requests from "
                f"{max_running_requests} to {dp_size} for DP-attn."
            )
            max_running_requests = dp_size
    engine_kwargs: Dict[str, Any] = dict(
        model_path=args.model,
        trust_remote_code=args.trust_remote_code,
        disable_radix_cache=True,
        max_running_requests=max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
        swa_full_tokens_ratio=0.1,
        context_length=args.max_model_len,
        skip_tokenizer_init=False,
        enable_metrics=True,
        enable_metrics_for_all_schedulers=True,
        **parallel_kwargs,
        **spec_kwargs,
    )
    if args.disable_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.cpu_offload_gb > 0:
        engine_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.kv_cache_dtype is not None:
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.quantization is not None:
        engine_kwargs["quantization"] = args.quantization

    print(f"[sglang_offline] Engine kwargs: {engine_kwargs}")
    print(f"[sglang_offline] SamplingParams: {sampling_params}")

    t_load = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    print(f"[sglang_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]

    print(f"[sglang_offline] Warmup batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    _ = engine.generate(prompt_strs, sampling_params)
    warmup_s = time.perf_counter() - t0
    print(f"[sglang_offline] Warmup done in {warmup_s:.3f}s")
    time.sleep(2.0)

    print(f"[sglang_offline] Timed batch: {len(prompt_strs)} prompts...")
    t0 = time.perf_counter()
    outputs = engine.generate(prompt_strs, sampling_params)
    timed_s = time.perf_counter() - t0
    print(f"[sglang_offline] Timed done in {timed_s:.3f}s")

    # SGLang per-request fields surfaced in meta_info (probed in run
    # 25536749769): completion_tokens, e2e_latency, prefill_finished_ts,
    # decode_finished_ts, request_received_ts, spec_verify_ct,
    # spec_accept_rate, spec_accept_length, spec_accept_token_num,
    # spec_draft_token_num, decode_throughput.
    #
    # Decode-rate metric: use the engine's `decode_throughput` (per-req
    # internal decode-step rate). This is the apples-to-apples comparable
    # to CANN's process_infer_time on batched-prefill engines (Ascend),
    # where wall-clock per-req decode time = engine-internal decode time.
    # On chunked-prefill engines the wall-clock per-req decode windows are
    # staggered, so wall-clock-derived TPOT diverges from what CANN reports.
    # Falling back to wall-clock derivation (decode_finished_ts -
    # prefill_finished_ts) when `decode_throughput` is missing.
    total_output_tokens = 0
    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []
    decode_tokens_per_req: List[int] = []
    decode_iters_per_req: List[int] = []
    n_decode_tokens = 0
    for o in outputs:
        meta = o.get("meta_info") if isinstance(o, dict) else getattr(o, "meta_info", None)
        if not isinstance(meta, dict):
            total_output_tokens += output_len
            continue
        n_out = int(meta.get("completion_tokens", output_len))
        total_output_tokens += n_out
        decode_tokens = max(n_out - 1, 0)
        n_decode_tokens += decode_tokens
        decode_tokens_per_req.append(decode_tokens)

        # Decode-iter count for CANN-style spec acceptance (engine telemetry).
        verify_ct = meta.get("spec_verify_ct")
        if verify_ct is not None:
            decode_iters_per_req.append(int(verify_ct))

        prefill_done = to_seconds(meta.get("prefill_finished_ts"))
        decode_done = to_seconds(meta.get("decode_finished_ts"))

        # Direct TTFT = prefill_finished - request_received (sglang doesn't
        # publish a `ttft` field but the timestamps are unambiguous).
        req_received = to_seconds(meta.get("request_received_ts"))
        if req_received is not None and prefill_done is not None and prefill_done >= req_received:
            ttfts.append(prefill_done - req_received)

        # E2E from the engine's own field.
        e2e = to_seconds(meta.get("e2e_latency"))
        if e2e is not None and e2e >= 0:
            e2els.append(e2e)

        # TPOT: prefer engine's per-req decode_throughput field (CANN-style on
        # batched-prefill engines). Fall back to wall-clock per-req derivation.
        decode_tput = to_seconds(meta.get("decode_throughput"))
        if decode_tput is not None and decode_tput > 0:
            tpots.append(1.0 / decode_tput)
        elif (prefill_done is not None and decode_done is not None
              and decode_done > prefill_done and decode_tokens > 0):
            tpots.append((decode_done - prefill_done) / decode_tokens)

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
        "latency_metrics_source": "sglang_meta_info",
    }
