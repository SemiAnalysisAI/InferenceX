#!/usr/bin/env python3
"""Read-only MoRI capability probe (run under torchrun on MI355X, 8 ranks).

Decides whether 'fp8' enters MoRIBackend.SUPPORTED_PRECISIONS: inspects
EpDispatchCombineConfig for quant_type options + the scale plumbing, then attempts a
small fp8 dispatch/combine. Prints MORI_FP8_OK (with the working quant_type + recon
error) or MORI_FP8_FAIL (with the exception) — that verdict gates the reject matrix.
LL is not probed: MoRI exposes no separate low-latency entrypoint (caps exclude it).
"""
import inspect
import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing  # noqa: E402

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", os.environ.get("CX_MORI_HEAP_SIZE", "2G"))


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12399")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world,
                            device_id=device)
    import mori

    if rank == 0:
        p = torch.cuda.get_device_properties(0)
        print(f"[mori] device={p.name} cus={p.multi_processor_count}")
        print("[mori] EpDispatchCombineConfig sig:")
        try:
            print("   ", inspect.signature(mori.ops.EpDispatchCombineConfig))
        except Exception as e:
            print("    <no sig>", repr(e))
        # surface any quant enum the module exposes
        for name in dir(mori.ops):
            if "quant" in name.lower() or "Quant" in name:
                obj = getattr(mori.ops, name)
                print(f"[mori] ops.{name} = {obj}")
                if hasattr(obj, "__members__"):
                    print("     members:", list(obj.__members__))

    hidden, topk, experts = 7168, 8, 256
    T = 8
    epr = experts // world
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    mori.shmem.shmem_torch_process_group_init("default")

    # candidate fp8 quant_type values to try (string and enum forms)
    candidates = []
    QT = getattr(mori.ops, "EpDispatchCombineQuantType", None) or getattr(mori.ops, "QuantType", None)
    if QT is not None and hasattr(QT, "__members__"):
        for mname in QT.__members__:
            if "8" in mname or "fp8" in mname.lower() or "FP8" in mname:
                candidates.append((f"enum:{mname}", QT.__members__[mname]))
    for s in ("fp8", "fp8_e4m3", "e4m3"):
        candidates.append((f"str:{s}", s))

    if rank == 0:
        print(f"[mori] fp8 quant_type candidates: {[c[0] for c in candidates]}")

    gi, gw = routing.build_global_routing(T * world, experts, topk, "uniform", 67, epr)
    si, sw = routing.rank_slice(gi, gw, rank, T)
    x = routing.rank_activations(T, hidden, 67, rank, device, torch.bfloat16)
    indices = si.to(device).to(torch.int32)
    weights = sw.to(device).to(torch.float32)

    working = None
    detail = ""
    for label, qt in candidates:
        try:
            cfg = mori.ops.EpDispatchCombineConfig(
                data_type=torch.bfloat16, rank=rank, world_size=world,
                hidden_dim=hidden, scale_dim=hidden // 128,
                scale_type_size=torch.tensor([], dtype=torch.float32).element_size(),
                max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
                max_num_inp_token_per_rank=512, num_experts_per_rank=epr,
                num_experts_per_token=topk, use_external_inp_buf=False, quant_type=qt)
            op = mori.ops.EpDispatchCombineOp(cfg)
            scales = torch.ones((T, hidden // 128), dtype=torch.float32, device=device)
            out = op.dispatch(x, weights, scales, indices, block_num=80, warp_per_block=16)
            recv = int(out[-1][0].item())
            dist.barrier()
            working = label
            detail = f"quant_type={label} dispatched recv={recv}"
            if rank == 0:
                print(f"[mori] FP8 DISPATCH OK with {label}: recv={recv}")
            break
        except Exception as exc:
            if rank == 0:
                print(f"[mori] {label} failed: {type(exc).__name__}: {str(exc)[:160]}")
            detail = f"{type(exc).__name__}: {str(exc)[:160]}"

    v = torch.tensor([1 if working else 0], device=device)
    dist.all_reduce(v, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(("MORI_FP8_OK " + detail) if int(v.item()) == 1 else ("MORI_FP8_FAIL " + detail))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if int(v.item()) == 1 else 7)


if __name__ == "__main__":
    raise SystemExit(main())
