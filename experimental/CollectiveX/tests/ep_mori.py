#!/usr/bin/env python3
"""CollectiveX EP backend adapter — MoRI (AMD ROCm), normal mode.

The harness owns the deterministic shared routing trace and the comm-only timing;
this file owns MoRI's API and the ionic_rdma-fabric constraints found on MI355X
(validated on-node, see CONTAINERS.md): the whole symmetric heap is one RDMA MR
capped at ~4 GiB (hold at 2 GiB; bound buffers via max_num_inp_token_per_rank ⇒
buffer_cap); combine() resets recv_num (read it before combine; compare only the
first T rows); and the post-shmem_finalize teardown asserts (finalize hard-exits).

`make_problem` now materializes the harness-provided rank slice, so MoRI honors the
requested routing (it no longer always-uniform) and runs the identical workload to
the NVIDIA SKUs. combine_needs_redispatch=True: combine consumes recv_num, so the
harness re-dispatches (untimed) before each timed combine sample.
"""
from __future__ import annotations

import os
import sys
import types

# MoRI registers the WHOLE symmetric heap as one RDMA MR at shmem init — set BEFORE
# `import mori`. 2 GiB registers on the MI355X ionic_rdma NICs; larger fails.
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
    combine_needs_redispatch = True
    # MoRI wedges on a COLD dispatch jumping straight to a large T (validated on
    # MI355X); the harness ramps this backend's ladder geometrically from 1.
    needs_gradual_ramp = True
    # MoRI WEDGES under a sustained warm-up burst (the harness's Blackwell clock-ramp)
    # and is already steady at a short warm-up (~44us, reproducible) — so it opts out.
    wants_warm_burst = False
    # Capabilities — run_ep.py REJECTS anything outside these BEFORE construction (no
    # fallback/mislabel). Expanded as each path is implemented + hardware-validated.
    # MoRI exposes quant_type (fp8) in EpDispatchCombineConfig; added once validated.
    SUPPORTED_PRECISIONS = {"bf16"}        # + "fp8" once the fp8 quant_type path is wired
    SUPPORTED_MODES = {"normal"}           # MoRI has no separate low-latency entrypoint

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = args.mode
        assert args.dispatch_dtype in self.SUPPORTED_PRECISIONS and args.mode in self.SUPPORTED_MODES, \
            "run_ep.py must reject unsupported dtype/mode before constructing the backend"
        self.fp8_in_timing = None  # set when fp8 dispatch is used (whether the cast is timed)
        self.ep_size = world_size
        self.experts_per_rank = args.experts // self.ep_size
        dev_cus = torch.cuda.get_device_properties(device).multi_processor_count
        # Resource regime — map the comm budget onto CUs to mirror DeepEP's SM fraction.
        #   normalized: block_num ≈ sm_fraction · CUs (≈ the same device fraction);
        #   tuned: MoRI launch auto-tuning (API not present in this build — uses default,
        #          labeled tuned_source); default: the 80-block bring-up budget.
        # MoRI DEADLOCKS at T>=32 when block_num is reduced toward the normalized target
        # (validated on MI355X g15: block_num=46 wedges, 80 completes T=32/64 with the
        # realistic fan-out≈5.3 trace). So MoRI cannot be normalized down to DeepEP's
        # device fraction; floor it at a known-functional minimum and record that the
        # target fraction was NOT reached.
        rm = args.resource_mode
        floor = int(os.environ.get("CX_MORI_MIN_BLOCKS", "80"))  # functional minimum (deadlocks lower)
        env_blocks = os.environ.get("CX_MORI_BLOCK_NUM")
        self._block_floored = False
        if env_blocks:
            self.block_num = int(env_blocks)
            self._block_target = self.block_num
        elif rm == "normalized":
            self._block_target = max(1, round(args.sm_fraction * dev_cus))
            self.block_num = max(floor, self._block_target)
            self._block_floored = self.block_num > self._block_target
        else:  # tuned (no launch auto-tune API in mori-0227-2) / default
            self.block_num = 80
            self._block_target = 80
        self._tuned_source = ("default-80" if rm == "tuned" else
                              ("normalized-floored" if self._block_floored else "n/a"))
        self.dispatch_warps = int(os.environ.get("CX_MORI_DISPATCH_WARPS", "16"))
        self.combine_warps = int(os.environ.get("CX_MORI_COMBINE_WARPS", "8"))

        world_group = torch.distributed.group.WORLD
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

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
        # Provenance: MoRI has no pip version; pin via MORI_COMMIT, else the image tag
        # the launcher exported (COLLECTIVEX_IMAGE carries the mori build tag), so the
        # provenance gate has something real rather than "unknown".
        img = os.environ.get("COLLECTIVEX_IMAGE", "")
        mori_commit = os.environ.get("MORI_COMMIT") or (f"image:{img}" if img else "unknown")
        self.backend_provenance = {
            "mori_commit": mori_commit,
            "heap_size": os.environ.get("MORI_SHMEM_HEAP_SIZE"),
            "max_num_inp_token_per_rank": max(512, self._cap),
            "resource_mode": args.resource_mode, "block_num": self.block_num,
            "block_num_target": self._block_target, "block_num_floored": self._block_floored,
            "dispatch_warps": self.dispatch_warps, "combine_warps": self.combine_warps,
            "device_cus": dev_cus, "sm_fraction": (self.block_num / dev_cus),
            "tuned_source": self._tuned_source,
        }

    def buffer_cap(self, args):
        # Largest tokens/rank the 2 GiB registerable heap holds at hidden=7168 (512,
        # validated on-node). Override via CX_MORI_MAX_TOKENS.
        return int(os.environ.get("CX_MORI_MAX_TOKENS", "512"))

    def make_problem(self, T, idx, weights, x):
        # Shared-trace slice: idx[T,topk] -> int32 (MoRI expects int32 expert ids);
        # weights[T,topk] f32; x[T,hidden] bf16; scales is a real (T,0) fp8 tensor
        # (not None) since scale_dim==0.
        indices = idx.to(torch.int32)
        scales = torch.empty((T, 0), dtype=torch.float8_e4m3fnuz, device=self.device)
        return types.SimpleNamespace(T=T, x=x, indices=indices,
                                     weights=weights.to(torch.float32), scales=scales)

    def dispatch(self, p):
        (dispatch_output, dispatch_weights, _scales, dispatch_indices, recv_num) = self.op.dispatch(
            p.x, p.weights, p.scales, p.indices,
            block_num=self.block_num, warp_per_block=self.dispatch_warps)
        total_recv = int(recv_num[0].item())  # read BEFORE combine (combine resets recv_num)
        return types.SimpleNamespace(
            dispatch_output=dispatch_output, dispatch_weights=dispatch_weights,
            dispatch_indices=dispatch_indices, total_recv=total_recv,
            combine_input=dispatch_output.to(torch.bfloat16))

    def stage(self, p, h):
        # comm-only contract: stage the "expert outputs" into MoRI's registered
        # combine-input buffer UNTIMED (in a real MoE the expert FFN writes here).
        buf = self.op.get_registered_combine_input_buffer(
            torch.bfloat16, hidden_dim=h.combine_input.size(1))
        buf[:h.total_recv, :].copy_(h.combine_input[:h.total_recv, :])

    def combine(self, p, h):
        combined, _w = self.op.combine(
            h.combine_input, h.dispatch_weights, h.dispatch_indices,
            block_num=self.block_num, warp_per_block=self.combine_warps)
        return combined

    def expected(self, p, h):
        # MoRI combine sums one copy per destination RANK ⇒ combined[i] ≈
        # x[i] * (#unique destination ranks among the token's topk experts).
        pes = p.indices.long() // self.experts_per_rank
        unique_pes = torch.tensor(
            [len(set(row.tolist())) for row in pes], device=self.device, dtype=torch.float32
        ).unsqueeze(1)
        return p.x.float() * unique_pes, p.T

    def recv_tokens(self, h):
        return int(h.total_recv)

    def finalize(self, rc):
        # MoRI's shmem teardown asserts after shmem_finalize(); results are already
        # written, so sync and hard-exit past it.
        try:
            dist.barrier()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if rc == 0 else 1)
