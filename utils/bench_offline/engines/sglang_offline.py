"""SGLang offline plugin: mirrors `sglang serve` flags from the b300
sglang_mtp launch script, but uses `sgl.Engine` in-process and runs
`Engine.generate(prompts)`.

Step semantics: the benchmark unit is *decode rounds* (main-model forward
passes). The warmup batch measures tokens-per-round (1 + MTP accepted), and the timed
batch's token budget is scaled so each request executes ~`--decode-steps`
forward passes. Per-request rounds come from `meta_info.spec_verify_ct`
(spec runs) or completion_tokens (non-spec: one token per round).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from latency_utils import (
    request_rounds,
    timed_max_tokens,
    to_seconds,
    tokens_per_round_estimate,
)


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


def _extract_batch(outputs: List[Any], mtp: int,
                   fallback_output_len: int) -> Dict[str, Any]:
    """Per-request telemetry from a generate() batch.

    SGLang per-request fields surfaced in meta_info (probed in run
    25536749769): completion_tokens, e2e_latency, prefill_finished_ts,
    decode_finished_ts, request_received_ts, spec_verify_ct,
    spec_accept_rate, spec_accept_length, spec_accept_token_num,
    spec_draft_token_num, decode_throughput.
    """
    total_output_tokens = 0
    ttfts: List[float] = []
    e2els: List[float] = []
    tpots: List[float] = []           # per output token
    tpot_steps: List[float] = []      # per decode round (offline decode-step)
    decode_tokens_per_req: List[int] = []
    decode_rounds_per_req: List[int] = []
    n_decode_tokens = 0
    n_rounds = 0
    for o in outputs:
        meta = o.get("meta_info") if isinstance(o, dict) else getattr(o, "meta_info", None)
        if not isinstance(meta, dict):
            total_output_tokens += fallback_output_len
            continue
        n_out = int(meta.get("completion_tokens", fallback_output_len))
        total_output_tokens += n_out
        decode_tokens = max(n_out - 1, 0)
        n_decode_tokens += decode_tokens
        decode_tokens_per_req.append(decode_tokens)

        verify_ct = meta.get("spec_verify_ct")
        rounds = request_rounds(
            decode_tokens,
            int(verify_ct) if verify_ct is not None else None,
            mtp,
        )
        if rounds is not None:
            decode_rounds_per_req.append(rounds)
            n_rounds += rounds

        prefill_done = to_seconds(meta.get("prefill_finished_ts"))
        decode_done = to_seconds(meta.get("decode_finished_ts"))

        # Direct TTFT = prefill_finished - request_received (sglang doesn't
        # publish a `ttft` field but the timestamps are unambiguous).
        req_received = to_seconds(meta.get("request_received_ts"))
        if req_received is not None and prefill_done is not None and prefill_done >= req_received:
            ttfts.append(prefill_done - req_received)

        e2e = to_seconds(meta.get("e2e_latency"))
        if e2e is not None and e2e >= 0:
            e2els.append(e2e)

        # Decode window for this request. Prefer wall timestamps; fall back
        # to the engine's per-req decode_throughput (tokens/s) to recover it.
        window: Optional[float] = None
        if (prefill_done is not None and decode_done is not None
                and decode_done > prefill_done):
            window = decode_done - prefill_done
        else:
            decode_tput = to_seconds(meta.get("decode_throughput"))
            if decode_tput is not None and decode_tput > 0 and decode_tokens > 0:
                window = decode_tokens / decode_tput

        if window is not None and decode_tokens > 0:
            tpots.append(window / decode_tokens)
            if rounds is not None and rounds > 0:
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
        print("[sglang_offline] WARN: --routing-sim-strategy="
              f"{args.routing_sim_strategy} requested, but SGLang has no "
              "MoE routing simulation as of v0.5.12. Running REAL routing; "
              "result is labeled moe_routing=real and should not be compared "
              "against idealized expert-balance rows.")
    _sgl_env_for_dsv4(args.dp_attn, args.sglang_dpa_env_preset)
    import sglang as sgl

    decode_steps = args.decode_steps

    def sampling(max_new_tokens: int) -> Dict[str, Any]:
        return {
            "temperature": args.temperature,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
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
        a2a_backend = args.dpa_moe_a2a_backend
        dp_size = args.dpa_size if args.dpa_size is not None else args.tp
        if dp_size < 1 or args.tp % dp_size != 0:
            raise ValueError(
                f"--dpa-size must be a positive divisor of --tp; got "
                f"dpa_size={dp_size}, tp={args.tp}"
            )
        deepep_config = (
            '{"normal_dispatch":{"num_sms":96},'
            '"normal_combine":{"num_sms":96}}'
        )
        parallel_kwargs: Dict[str, Any] = {
            "tp_size": args.tp,
            "dp_size": dp_size,
            "ep_size": args.ep if args.ep > 1 else args.tp,
            "enable_dp_attention": True,
            "disable_flashinfer_autotune": True,
            "chunked_prefill_size": (
                args.chunked_prefill_size
                if args.chunked_prefill_size is not None
                else 32768
            ),
        }
        if a2a_backend != "none":
            parallel_kwargs["moe_a2a_backend"] = a2a_backend
        if a2a_backend == "deepep":
            parallel_kwargs["deepep_config"] = deepep_config
        if args.dpa_moe_runner_backend is not None:
            parallel_kwargs["moe_runner_backend"] = args.dpa_moe_runner_backend
        if args.moe_dense_tp_size is not None:
            parallel_kwargs["moe_dense_tp_size"] = args.moe_dense_tp_size
        if args.enable_dp_lm_head:
            parallel_kwargs["enable_dp_lm_head"] = True
        if args.deepep_mode is not None and a2a_backend != "none":
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
        context_length=args.engine_max_len,
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

    t_load = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    print(f"[sglang_offline] Engine init: {time.perf_counter() - t_load:.2f}s")

    prompt_strs = [p for (p, _, _) in prompts]

    # Warmup batch doubles as the tokens-per-round calibration run.
    warmup_params = sampling(decode_steps)
    print(f"[sglang_offline] Warmup batch: {len(prompt_strs)} prompts, "
          f"params={warmup_params}")
    t0 = time.perf_counter()
    warmup_outputs = engine.generate(prompt_strs, warmup_params)
    warmup_s = time.perf_counter() - t0
    print(f"[sglang_offline] Warmup done in {warmup_s:.3f}s")
    warmup_stats = _extract_batch(warmup_outputs, args.mtp, decode_steps)
    tokens_per_round = tokens_per_round_estimate(
        warmup_stats["decode_tokens_total"],
        warmup_stats["decode_rounds_total"])
    if args.mtp > 0 and not warmup_stats["decode_rounds_per_req"]:
        print("[sglang_offline] WARN: spec run but no spec_verify_ct in "
              "meta_info; cannot calibrate decode-step budget. Timed run "
              "keeps max_new_tokens=decode_steps and step metrics will "
              "fall back to per-token accounting.")
    timed_budget = timed_max_tokens(decode_steps, tokens_per_round)
    time.sleep(2.0)

    timed_params = sampling(timed_budget)
    print(f"[sglang_offline] Timed batch: {len(prompt_strs)} prompts, "
          f"warmup tokens/round={tokens_per_round:.3f}, "
          f"max_new_tokens={timed_budget} (targeting {decode_steps} "
          f"forward passes/request)")
    t0 = time.perf_counter()
    outputs = engine.generate(prompt_strs, timed_params)
    timed_s = time.perf_counter() - t0
    print(f"[sglang_offline] Timed done in {timed_s:.3f}s")

    stats = _extract_batch(outputs, args.mtp, timed_budget)
    total_input_tokens = sum(plen for (_, plen, _) in prompts)
    stats.update({
        "warmup_seconds": warmup_s,
        "timed_seconds": timed_s,
        "total_input_tokens": total_input_tokens,
        "warmup_tokens_per_round": tokens_per_round,
        "timed_max_tokens": timed_budget,
        "latency_metrics_source": "sglang_meta_info",
        "moe_routing": "real",
    })
    return stats
