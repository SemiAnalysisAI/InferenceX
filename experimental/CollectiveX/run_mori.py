#!/usr/bin/env python3
"""CollectiveX spike — MoRI (AMD) MoE dispatch+combine, normal mode.

AMD counterpart to run_deepep.py, using ROCm MoRI's EpDispatchCombine op. One
decode-shaped dispatch+combine point, correctness-gated, CUDA-event timed,
emitting the same flat-JSON shape (family=moe, backend=mori).

  MoRI's Python API is VERSION-SENSITIVE. The config/dispatch/combine block below
  follows ROCm/mori examples/ops/dispatch_combine/test_dispatch_combine.py. The
  first MI355X run (image rocm/sgl-dev:...-mori-0227-2) confirmed the setup +
  config + dispatch path reach the MoRI kernel; it OOM'd the default 2 GiB
  symmetric heap, now sized up via MORI_SHMEM_HEAP_SIZE above. The correctness
  gate and timing are validated by the heap-sized re-run.

Launch (one process per GPU), e.g. single-node 8x MI355X:
    torchrun --nproc_per_node=8 run_mori.py \\
        --runner mi355x-amds --topology-class mi355x-xgmi --transport xgmi \\
        --env-json results/env.json --out results/mi355x_mori.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

# MoRI's symmetric-memory heap defaults to 2 GiB (static) — too small for the
# DeepSeek hidden size (7168) across 8 ranks: the dispatch/combine buffers
# overflow it ("Out of static heap memory ... Increase via MORI_SHMEM_HEAP_SIZE",
# observed on the first MI355X run). Size it generously here, BEFORE `import mori`
# (the heap is created at shmem init); MI355X HBM is ample. Layered override:
# explicit MORI_SHMEM_HEAP_SIZE > CX_MORI_HEAP_BYTES > 16 GiB default.
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE",
                      os.environ.get("CX_MORI_HEAP_BYTES", str(16 * 1024**3)))

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "mori-normal-v1"


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def comparison_key(meta: dict) -> str:
    parts = [
        meta["op"], meta["backend"], meta["mode"], str(meta["world_size"]),
        str(meta["nodes"]), meta["topology_class"], meta["comparison_class"],
        meta["measurement_contract"], str(meta["shape"]),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX MoRI dispatch+combine (normal mode)")
    ap.add_argument("--tokens-per-rank", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--dispatch-dtype", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--block-num", type=int, default=int(os.environ.get("CX_MORI_BLOCK_NUM", "80")))
    ap.add_argument("--dispatch-warps", type=int, default=int(os.environ.get("CX_MORI_DISPATCH_WARPS", "16")))
    ap.add_argument("--combine-warps", type=int, default=int(os.environ.get("CX_MORI_COMBINE_WARPS", "8")))
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--comparison-class", default="standardized")
    ap.add_argument("--mori-commit", default=os.environ.get("MORI_COMMIT", "unknown"))
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3
    try:
        import mori  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: mori import failed — needs the AMD MoRI image. {exc!r}", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size % 1 != 0 or args.experts % world_size != 0:
        if rank == 0:
            print(f"ERROR: experts ({args.experts}) must divide world_size ({world_size})", file=sys.stderr)
        return 2
    experts_per_rank = args.experts // world_size
    torch.manual_seed(args.seed + rank)

    # ===================== ADAPT HERE (MoRI API) =========================
    # init torch.distributed + MoRI shmem (per the MoRI dispatch/combine test).
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                                world_size=world_size, device_id=device)
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    mori.shmem.shmem_torch_process_group_init("default")

    n = args.tokens_per_rank
    H = args.hidden
    topk = args.topk
    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=H,
        scale_dim=0,
        scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
        max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
        max_num_inp_token_per_rank=max(4096, n),
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=topk,
        use_external_inp_buf=False,
        quant_type="none",
    )
    op = mori.ops.EpDispatchCombineOp(config)

    # Routing: each token -> topk distinct experts in [0, experts). MoRI expects
    # INT32 expert indices, and a real (n, scale_dim) fp8 scales tensor even when
    # scale_dim==0 (an (n,0) tensor) — not None (see the reference test).
    x = torch.randn((n, H), dtype=torch.bfloat16, device=device)
    indices = torch.stack([torch.randperm(args.experts, device=device)[:topk] for _ in range(n)]).to(torch.int32)
    weights = torch.rand((n, topk), dtype=torch.float32, device=device)
    scales = torch.empty((n, 0), dtype=torch.float8_e4m3fnuz, device=device)

    def run_once():
        (dispatch_output, dispatch_weights, _dispatch_scales,
         dispatch_indices, recv_num) = op.dispatch(
            x, weights, scales, indices,
            block_num=args.block_num, warp_per_block=args.dispatch_warps)
        # Zero-copy mode (use_external_inp_buf=False): combine reads from MoRI's
        # registered combine-input buffer, so stage the dispatched rows into it
        # first. (In a real MoE the expert FFN writes its outputs here; with no
        # expert compute we copy the dispatched activations straight through.)
        total_recv = int(recv_num[0].item())
        combine_input = dispatch_output.to(torch.bfloat16)
        combine_buf = op.get_registered_combine_input_buffer(
            torch.bfloat16, hidden_dim=combine_input.size(1))
        combine_buf[:total_recv, :].copy_(combine_input[:total_recv, :])
        combined, _combined_w = op.combine(
            combine_input, dispatch_weights, dispatch_indices,
            block_num=args.block_num, warp_per_block=args.combine_warps)
        return combined, recv_num
    # =====================================================================

    # ---- correctness gate ----
    combined, recv_num = run_once()
    torch.cuda.synchronize()
    # MoRI combine sums one copy per destination RANK, so combined[i] ≈
    # input[i] * (#unique destination ranks among the token's topk experts)
    # (see ROCm/mori .../test_dispatch_combine.py).
    pes = indices.long() // experts_per_rank
    unique_pes = torch.tensor(
        [len(set(row.tolist())) for row in pes], device=device, dtype=torch.float32
    ).unsqueeze(1)
    expected = x.float() * unique_pes
    max_abs = (combined.float() - expected).abs().max().item()
    max_rel = max_abs / (expected.abs().max().item() + 1e-6)
    # Validated tolerance from the reference test (bf16 + up-to-topk summation).
    combine_ok = bool(torch.allclose(combined.float(), expected.float(), atol=1e-2, rtol=1e-2))
    recv_ok = bool(int(recv_num[0].item()) > 0) if recv_num is not None else True
    correct = bool(combine_ok and recv_ok)

    def time_us(fn, warmup, iters) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            out.append(s.elapsed_time(e) * 1000.0)
        return out

    def dispatch_only():
        op.dispatch(x, weights, scales, indices,
                    block_num=args.block_num, warp_per_block=args.dispatch_warps)

    trials = []
    for _ in range(args.trials):
        rt = time_us(run_once, args.warmup, args.iters)
        dp = time_us(dispatch_only, args.warmup, args.iters)
        trials.append({"roundtrip_us_p50": _percentile(rt, 50), "roundtrip_us_p99": _percentile(rt, 99),
                       "dispatch_us_p50": _percentile(dp, 50)})

    local_rt_p50 = sum(t["roundtrip_us_p50"] for t in trials) / len(trials)
    t = torch.tensor([local_rt_p50], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    slowest_rank_us = float(t.item())

    if rank == 0:
        shape = {"tokens_per_rank": n, "hidden": H, "topk": topk, "experts": args.experts,
                 "experts_per_rank": experts_per_rank, "dispatch_dtype": args.dispatch_dtype}
        meta = {"op": "dispatch-combine", "backend": "mori", "mode": "normal",
                "world_size": world_size, "nodes": int(os.environ.get("SLURM_NNODES", "1")),
                "topology_class": args.topology_class, "comparison_class": args.comparison_class,
                "measurement_contract": MEASUREMENT_CONTRACT, "shape": shape}
        rt_p50 = sum(t["roundtrip_us_p50"] for t in trials) / len(trials)
        tokens_total = n * world_size
        env = None
        if args.env_json and os.path.exists(args.env_json):
            with open(args.env_json) as fh:
                env = json.load(fh)
        doc = {
            "schema_version": SCHEMA_VERSION, "family": "moe", "generated_by": "run_mori.py",
            "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
            "runner": args.runner, "transport": args.transport,
            "status": "valid" if correct else "invalid",
            "comparison_key": comparison_key(meta),
            "backend_provenance": {"mori_commit": args.mori_commit,
                                   "block_num": args.block_num,
                                   "dispatch_warps": args.dispatch_warps,
                                   "combine_warps": args.combine_warps},
            **meta,
            "correctness": {"passed": correct, "combine_within_tol": combine_ok,
                            "recv_nonzero": recv_ok, "max_abs_error": max_abs, "max_rel_error": max_rel},
            "metrics": {
                "roundtrip_us_p50": rt_p50,
                "roundtrip_us_p99": sum(t["roundtrip_us_p99"] for t in trials) / len(trials),
                "dispatch_us_p50": sum(t["dispatch_us_p50"] for t in trials) / len(trials),
                "slowest_rank_roundtrip_us": slowest_rank_us,
                "tokens_per_second": (tokens_total / (rt_p50 * 1e-6)) if rt_p50 else None,
            },
            "trials": trials, "environment": env,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"mori dispatch-combine: status={doc['status']} rt_p50={rt_p50:.1f}us "
              f"slowest_rank={slowest_rank_us:.1f}us correct={correct} -> {args.out}")

    try:
        mori.shmem.shmem_finalize()
    except Exception:
        pass
    dist.barrier()
    dist.destroy_process_group()
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
