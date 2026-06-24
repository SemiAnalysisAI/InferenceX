#!/usr/bin/env python3
"""CollectiveX EP backend adapter — MoRI (AMD ROCm), normal mode.

Ports the validated dispatch/combine sequence from the old run_mori.py into the
ep_harness Backend protocol. The harness owns the token sweep + separated timing;
this file owns MoRI's API and the three ionic_rdma-fabric constraints found on
MI355X (all validated on-node, see CONTAINERS.md):
  1. MoRI registers the WHOLE symmetric heap as one RDMA MR at shmem init, and
     these NICs cap GPU-memory MRs at ~4 GiB — a 6 GiB heap fails (errno 22),
     2 GiB registers. So hold the heap at 2 GiB and bound the buffers via
     max_num_inp_token_per_rank (=> buffer_cap clamps the token sweep).
  2. combine() resets recv_num, so read it BEFORE combine; combine returns the
     full max_num_inp_token_per_rank buffer, so compare only the first T rows.
  3. MoRI's shmem teardown asserts (CheckStatusValid -> SIGABRT) when the op is
     destroyed after shmem_finalize(); finalize() hard-exits past it.

combine_needs_redispatch = True: combine consumes the dispatch state (recv_num),
so the harness re-dispatches (untimed) before each timed combine sample.
"""
from __future__ import annotations

import os
import sys
import types

# MoRI registers the WHOLE symmetric heap as one RDMA MR at shmem init — set this
# BEFORE `import mori`. 2 GiB registers cleanly on the MI355X ionic_rdma NICs;
# larger fails. Layered: explicit MORI_SHMEM_HEAP_SIZE > CX_MORI_HEAP_SIZE > 2G.
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE",
                      os.environ.get("CX_MORI_HEAP_SIZE", "2G"))

import torch
import torch.distributed as dist

try:
    import mori  # type: ignore
except Exception as exc:  # pragma: no cover - needs the AMD MoRI image
    print("ERROR: mori import failed — needs the AMD MoRI image "
          f"(rocm/sgl-dev:...-mori-...). {exc!r}", file=sys.stderr)
    raise


class MoRIBackend:
    name = "mori"
    mode = "normal"
    measurement_contract = "mori-normal-v1"
    combine_needs_redispatch = True
    # MoRI wedges on a COLD dispatch that jumps straight to a large token count
    # (validated on MI355X: a fresh-shmem sweep starting at T=128 hangs, while a
    # gradual sweep 1,2,4,...,512 runs every point fine — including 256/512). So
    # the harness ramps this backend's ladder geometrically from 1 up to its max,
    # turning any phase's sweep into the proven gradual ramp.
    needs_gradual_ramp = True

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.ep_size = world_size // max(1, args.num_ep_groups)
        self.experts_per_rank = args.experts // self.ep_size
        self.block_num = int(os.environ.get("CX_MORI_BLOCK_NUM", "80"))
        self.dispatch_warps = int(os.environ.get("CX_MORI_DISPATCH_WARPS", "16"))
        self.combine_warps = int(os.environ.get("CX_MORI_COMBINE_WARPS", "8"))
        if args.dispatch_dtype != "bf16":
            if rank == 0:
                print(f"WARN: mori adapter validated for bf16 (quant_type=none); "
                      f"'{args.dispatch_dtype}' not wired — using bf16.", file=sys.stderr)
            args.dispatch_dtype = "bf16"

        # init MoRI shmem on the torch process group (per the reference test).
        world_group = torch.distributed.group.WORLD
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        # Size the symmetric buffers to the registerable heap (see buffer_cap). The
        # op is built ONCE and reused for every T in the sweep; a T<=cap problem
        # just fills the first T rows of the fixed buffer.
        self._cap = self.buffer_cap(args)
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=torch.bfloat16, rank=rank, world_size=world_size,
            hidden_dim=args.hidden, scale_dim=0,
            scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
            max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
            max_num_inp_token_per_rank=max(512, self._cap),
            num_experts_per_rank=self.experts_per_rank,
            num_experts_per_token=args.topk,
            use_external_inp_buf=False, quant_type="none",
        )
        self.op = mori.ops.EpDispatchCombineOp(self.config)
        self.backend_provenance = {
            "mori_commit": os.environ.get("MORI_COMMIT", "unknown"),
            "heap_size": os.environ.get("MORI_SHMEM_HEAP_SIZE"),
            "max_num_inp_token_per_rank": max(512, self._cap),
            "block_num": self.block_num,
            "dispatch_warps": self.dispatch_warps, "combine_warps": self.combine_warps,
        }

    def buffer_cap(self, args):
        # Largest tokens/rank the 2 GiB registerable heap holds at this hidden size.
        # 512 was validated on-node at hidden=7168; override via CX_MORI_MAX_TOKENS
        # once a larger heap/ceiling is confirmed. Prefill ladders clamp to this.
        return int(os.environ.get("CX_MORI_MAX_TOKENS", "512"))

    def make_problem(self, T):
        a = self.args
        device, H, topk, E = self.device, a.hidden, a.topk, a.experts
        x = torch.randn((T, H), dtype=torch.bfloat16, device=device)
        # MoRI expects INT32 expert indices and a real (T, scale_dim) fp8 scales
        # tensor even when scale_dim==0 (an (T,0) tensor), not None.
        indices = torch.stack([
            torch.randperm(E, device=device)[:topk] for _ in range(T)
        ]).to(torch.int32)
        weights = torch.rand((T, topk), dtype=torch.float32, device=device)
        scales = torch.empty((T, 0), dtype=torch.float8_e4m3fnuz, device=device)
        return types.SimpleNamespace(T=T, x=x, indices=indices, weights=weights, scales=scales)

    def dispatch(self, p):
        (dispatch_output, dispatch_weights, _scales, dispatch_indices, recv_num) = self.op.dispatch(
            p.x, p.weights, p.scales, p.indices,
            block_num=self.block_num, warp_per_block=self.dispatch_warps)
        # Read total_recv BEFORE any combine — combine() resets recv_num (a later
        # read yields 0, a false "received nothing").
        total_recv = int(recv_num[0].item())
        return types.SimpleNamespace(
            dispatch_output=dispatch_output, dispatch_weights=dispatch_weights,
            dispatch_indices=dispatch_indices, total_recv=total_recv,
            combine_input=dispatch_output.to(torch.bfloat16))

    def stage(self, p, h):
        # Zero-copy mode (use_external_inp_buf=False): combine reads MoRI's
        # registered combine-input buffer, so stage the dispatched rows into it.
        # In a real MoE the expert FFN writes its outputs here; with no expert
        # compute we copy the dispatched activations straight through.
        buf = self.op.get_registered_combine_input_buffer(
            torch.bfloat16, hidden_dim=h.combine_input.size(1))
        buf[:h.total_recv, :].copy_(h.combine_input[:h.total_recv, :])

    def combine(self, p, h):
        combined, _w = self.op.combine(
            h.combine_input, h.dispatch_weights, h.dispatch_indices,
            block_num=self.block_num, warp_per_block=self.combine_warps)
        return combined

    def expected(self, p, h):
        # MoRI combine sums one copy per destination RANK, so combined[i] ≈
        # x[i] * (#unique destination ranks among the token's topk experts).
        pes = p.indices.long() // self.experts_per_rank
        unique_pes = torch.tensor(
            [len(set(row.tolist())) for row in pes], device=self.device, dtype=torch.float32
        ).unsqueeze(1)
        return p.x.float() * unique_pes, p.T

    def recv_tokens(self, h):
        return int(h.total_recv)

    def finalize(self, rc):
        # MoRI's shmem teardown asserts when the op is destroyed after
        # shmem_finalize() (CheckStatusValid -> SIGABRT on this build). The result
        # JSON is already written, so sync the ranks and hard-exit past it.
        try:
            dist.barrier()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if rc == 0 else 1)
