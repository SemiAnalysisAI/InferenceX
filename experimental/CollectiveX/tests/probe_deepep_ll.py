#!/usr/bin/env python3
"""Go/No-Go: does DeepEP low-latency (LL) mode actually run on THIS fabric?

LL dispatch/combine require IBGDA ("all ranks visible via RDMA, IBGDA enabled" —
even intranode), with allow_nvlink_for_low_latency_mode as a possible NVLink escape
hatch. On a single-node NVLink-only box this may or may not initialize. Run under
torchrun (8 ranks). Prints LL_OK with shapes + reconstruction error, or LL_FAIL with
the exception — that verdict decides whether 'll' enters DeepEPBackend.SUPPORTED_MODES.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing  # noqa: E402


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12377")
    dist.init_process_group("nccl")

    from deep_ep import Buffer
    hidden, topk, experts = 7168, 8, 256
    T = 8                                   # decode-shaped
    num_max = 128                           # fixed LL cap (>= max T in a decode sweep)
    nle = experts // world                  # num local experts

    ok = True
    detail = ""
    try:
        rdma = Buffer.get_low_latency_rdma_size_hint(num_max, hidden, world, experts)
        if rank == 0:
            print(f"[ll] rdma_size_hint={rdma} bytes; nle={nle} num_max={num_max}")
        # LL buffer: nvl=0, rdma=hint, low_latency_mode=True. allow_nvlink default True.
        buf = Buffer(dist.group.WORLD, 0, rdma, low_latency_mode=True,
                     num_qps_per_rank=max(1, experts // world))
        # shared trace slice (same builder the harness uses)
        gi, gw = routing.build_global_routing(T * world, experts, topk, "uniform", 67, nle)
        si, sw = routing.rank_slice(gi, gw, rank, T)
        x = routing.rank_activations(T, hidden, 67, rank, device, torch.bfloat16)
        topk_idx = si.to(device).to(torch.int64)
        topk_w = sw.to(device).to(torch.float32)

        recv_x, recv_count, handle, event, hook = buf.low_latency_dispatch(
            x, topk_idx, num_max, experts, use_fp8=True, return_recv_hook=False)
        rfp8, rscale = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        if rank == 0:
            print(f"[ll] dispatch OK: recv_fp8={tuple(rfp8.shape)} dtype={rfp8.dtype} "
                  f"scale={None if rscale is None else tuple(rscale.shape)} "
                  f"recv_count={tuple(recv_count.shape)}")
        # dequant fp8 recv -> bf16 in the [nle, num_max*world, hidden] layout for combine
        R = rfp8.float()
        if rscale is not None:
            E, S, H = rfp8.shape
            R = (rfp8.float().view(E, S, H // 128, 128) * rscale.unsqueeze(-1)).view(E, S, H)
        comb_in = R.to(torch.bfloat16)
        combined, event2, hook2 = buf.low_latency_combine(comb_in, topk_idx, topk_w, handle)
        torch.cuda.synchronize()
        # reconstruction: combined[i] ~= dequant(x[i]) * sum_j w[i,j]  (weighted reduce)
        wsum = topk_w.sum(dim=1, keepdim=True)
        ref = x.float() * wsum
        err = (combined[:T].float() - ref[:T]).abs().max().item() / (ref[:T].abs().max().item() + 1e-6)
        buf.clean_low_latency_buffer(num_max, hidden, experts)
        detail = (f"combined={tuple(combined.shape)} max_rel_err={err:.4f} "
                  f"wsum[0]={wsum[0].item():.3f}")
        if rank == 0:
            print(f"[ll] combine OK: {detail}")
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
        if rank == 0:
            print(f"[ll] EXCEPTION: {detail}")
            traceback.print_exc()

    # reduce verdict across ranks
    v = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(v, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("LL_OK" if int(v.item()) == 1 else "LL_FAIL", detail)
    dist.destroy_process_group()
    return 0 if int(v.item()) == 1 else 7


if __name__ == "__main__":
    raise SystemExit(main())
