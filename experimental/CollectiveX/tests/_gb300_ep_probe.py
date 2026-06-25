#!/usr/bin/env python3
"""GB300 EP8 GO/NO-GO probe — does DeepEP work across 2 NVL72 trays (8 ranks / 2 nodes)?

Read-only spike (no artifacts). One PATH per process (CX_PROBE_PATH), because NVSHMEM
inits once per process and the internode/LL buffers each bootstrap it. Reports, on rank 0,
which Buffer construction + a 1-shot dispatch/combine round-trip actually runs on this fabric:

  intranode  Buffer(group, nvl, 0)                         (MNNVL-as-one-NVLink-domain hope)
  internode  Buffer(group, nvl, rdma>0)                    (DeepEP NVSHMEM path, over NVLink/IB)
  ll         Buffer(group, 0, rdma, low_latency_mode=True) (decode path; nvlink-LL allowed)

Env (set per-rank by the srun wrapper): RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
                                        CX_PROBE_PATH=intranode|internode|ll
"""
import os
import socket
import sys
import traceback

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LR = int(os.environ["LOCAL_RANK"])
PATH = os.environ.get("CX_PROBE_PATH", "intranode")
HOST = socket.gethostname()
H = 7168
TOPK = 8
EXPERTS = WORLD * 32          # 256 at world=8 — same as the real sweep
T = 8                          # tiny: this is a does-it-run probe, not a timing run


def log(msg):
    print(f"[r{RANK}@{HOST} {PATH}] {msg}", flush=True)


def main():
    torch.cuda.set_device(LR)
    dev = torch.device(f"cuda:{LR}")
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)

    import deep_ep
    from deep_ep import Buffer
    if RANK == 0:
        import inspect
        try:
            import importlib.metadata as md
            ver = md.version("deep_ep")
        except Exception:
            ver = getattr(deep_ep, "__version__", "?")
        log(f"deep_ep={ver} torch={torch.__version__} cuda={torch.version.cuda}")
        log(f"Buffer.__init__{inspect.signature(Buffer.__init__)}")
        log(f"caps: internode_dispatch={hasattr(Buffer,'internode_dispatch')} "
            f"get_dispatch_config={hasattr(Buffer,'get_dispatch_config')} "
            f"low_latency_dispatch={hasattr(Buffer,'low_latency_dispatch')} "
            f"ll_rdma_hint={hasattr(Buffer,'get_low_latency_rdma_size_hint')}")

    hosts = [None] * WORLD
    dist.all_gather_object(hosts, HOST)
    if RANK == 0:
        uniq = sorted(set(hosts))
        log(f"world={WORLD} over {len(uniq)} node(s): {uniq}")

    group = dist.group.WORLD
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    g = torch.Generator(device=dev).manual_seed(1234 + RANK)
    idx = torch.stack([torch.randperm(EXPERTS, device=dev, generator=g)[:TOPK]
                       for _ in range(T)]).to(torch.int64)
    w = torch.rand(T, TOPK, device=dev, generator=g).to(torch.float32)

    dist.barrier()
    try:
        if PATH == "intranode":
            buf = Buffer(group, 1 * 1024**3, 0)
            try:
                Buffer.set_num_sms(24)
            except Exception:
                pass
            ntr, ntrr, ntpe, itir, _ = buf.get_dispatch_layout(idx, EXPERTS)
            rx, _ri, rw, _nre, h, _ev = buf.dispatch(
                x, topk_idx=idx, topk_weights=w, num_tokens_per_rank=ntr,
                num_tokens_per_rdma_rank=ntrr, is_token_in_rank=itir,
                num_tokens_per_expert=ntpe)
            cx, _, _ = buf.combine(rx, h, topk_weights=rw)
            rxs = rx[0].shape if isinstance(rx, tuple) else rx.shape
            log(f"RESULT intranode OK: recv={tuple(rxs)} combine={tuple(cx.shape)} "
                f"rdma_rank_layout={'present' if ntrr is not None else 'None'}")

        elif PATH == "internode":
            buf = Buffer(group, 1 * 1024**3, 1 * 1024**3)
            try:
                Buffer.set_num_sms(24)
            except Exception:
                pass
            ntr, ntrr, ntpe, itir, _ = buf.get_dispatch_layout(idx, EXPERTS)
            rx, _ri, rw, _nre, h, _ev = buf.dispatch(
                x, topk_idx=idx, topk_weights=w, num_tokens_per_rank=ntr,
                num_tokens_per_rdma_rank=ntrr, is_token_in_rank=itir,
                num_tokens_per_expert=ntpe)
            cx, _, _ = buf.combine(rx, h, topk_weights=rw)
            rxs = rx[0].shape if isinstance(rx, tuple) else rx.shape
            log(f"RESULT internode OK: recv={tuple(rxs)} combine={tuple(cx.shape)} "
                f"rdma_rank_layout={'present' if ntrr is not None else 'None'}")

        elif PATH == "ll":
            num_max = 128
            rdma = Buffer.get_low_latency_rdma_size_hint(num_max, H, WORLD, EXPERTS)
            nq = max(1, EXPERTS // WORLD)
            buf = Buffer(group, 0, rdma, low_latency_mode=True, num_qps_per_rank=nq,
                         allow_nvlink_for_low_latency_mode=True)
            rx, rc, h, _ev, _hook = buf.low_latency_dispatch(
                x, idx, num_max, EXPERTS, use_fp8=False, return_recv_hook=False)
            cx, _ev2, _hook2 = buf.low_latency_combine(rx, idx, w, h)
            rxs = rx[0].shape if isinstance(rx, tuple) else rx.shape
            log(f"RESULT ll OK: recv={tuple(rxs)} combine={tuple(cx.shape)}")
        else:
            log(f"unknown CX_PROBE_PATH={PATH}")
            return 2
        dist.barrier()
    except Exception as exc:
        if RANK == 0:
            log(f"RESULT {PATH} FAIL: {exc!r}")
            tb = traceback.format_exc().strip().splitlines()
            for ln in tb[-8:]:
                log(f"  | {ln}")
        # let other ranks print their error too (often the real one is rank-specific)
        else:
            log(f"FAIL(non0): {exc!r}")
        try:
            dist.barrier()
        except Exception:
            pass
        return 1
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
