#!/usr/bin/env python3
"""CollectiveX EP backend adapter — DeepEP `hybrid-ep` branch (NVIDIA TMA-based HybridEPBuffer).

The hybrid-ep branch (https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) is NVIDIA's TMA +
warp-pipeline implementation of expert-parallel all-to-all, exposing `deep_ep.HybridEPBuffer`
(distinct from the mainline `deep_ep.Buffer`). It supports intra-node NVLink AND inter-node
RDMA/NIXL; this adapter exercises the INTRANODE path (single NVLink domain, <=8 ranks), which needs
no multi-node/NVSHMEM bring-up. The container build is done by runtime/run_in_container.sh
`cx_build_deepep_hybrid` (CUDA-13 cccl include + libnvshmem symlink fixes).

API (pinned on B300, branch e0a5b1d):
  HybridEPBuffer(group, hidden_dim, max_num_of_tokens_per_rank, num_local_experts, use_fp8=False, ...)
  .dispatch(hidden, topk_idx=, topk_weights=, num_of_experts=) -> (recv_hidden, recv_x2, None, handle)
  .combine(hidden, handle=) -> [T, hidden]

CORRECTNESS: identity expert (no expert compute), combine WITHOUT probs -> each source token is
reconstructed as x * (distinct ranks among its top_k experts) — verified: an 8-rank uniform top_k=8
round trip gives relerr(combined, x) = 4.28, matching E[distinct ranks] ~ 5.26 exactly. So this uses
the SAME "ranks" factor as ep_flashinfer (per-rank-sum combine, no gate re-weight). bf16 tol 5e-2.

STATUS: bf16 / normal / layout-and-dispatch-v1, intranode NVLink (<=8 ranks). fp8 + internode are
further lift (use_fp8 path + a multi-node runner — the hybrid NVLink<->RDMA forwarding is the
branch's headline but needs >1 node; docs/gated.md rack-scale).
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

try:
    import deep_ep
    HybridEPBuffer = deep_ep.HybridEPBuffer
except Exception as exc:  # pragma: no cover - needs the hybrid-ep build
    print("ERROR: deep_ep.HybridEPBuffer import failed — the hybrid-ep branch must be built at job "
          "setup (cx_build_deepep_hybrid). "
          f"{exc!r}", file=sys.stderr)
    raise


def _deepep_hybrid_version() -> str:
    return os.environ.get("DEEPEP_COMMIT", getattr(deep_ep, "__version__", "hybrid-ep"))


class DeepEPHybridBackend:
    name = "deepep-hybrid"
    # HybridEPBuffer.combine consumes the recv payload + the dispatch handle (no re-dispatch needed
    # before a timed combine); the harness times dispatch and combine separately (like ep_deepep).
    combine_needs_redispatch = False
    wants_warm_burst = True
    # Capabilities — run_ep.py REJECTS anything outside these before construction.
    SUPPORTED_PRECISIONS = {"bf16"}        # fp8 = use_fp8 path, further lift
    SUPPORTED_MODES = {"normal"}
    SUPPORTED_CONTRACTS = {"layout-and-dispatch-v1"}
    SUPPORTED_COMBINE_DTYPES = {"bf16"}
    SUPPORTED_COMBINE_QUANT_MODES = {"none"}

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = args.mode
        self.contract = args.measurement_contract
        self.group = dist.group.WORLD
        assert args.dispatch_dtype in self.SUPPORTED_PRECISIONS and args.mode in self.SUPPORTED_MODES, \
            "run_ep.py must reject unsupported dtype/mode before constructing the backend"
        self.tolerance = 5e-2
        self.fp8_in_timing = None
        self.top_k = int(args.topk)
        self.num_experts = int(args.experts)
        self.hidden = int(args.hidden)
        self.local_experts = max(1, self.num_experts // world_size)
        # Token cap (per rank) for the symmetric buffer; the sweep is capped here (buffer_cap).
        self.max_tokens = int(os.environ.get("CX_HYBRIDEP_MAX_TOKENS", "4096"))
        dev_sms = torch.cuda.get_device_properties(device).multi_processor_count
        ver = _deepep_hybrid_version()

        # Construct the HybridEPBuffer. Intranode: all ranks in one NVLink domain. We let it default
        # num_of_hybrid_ep_ranks_per_nvlink_domain (== world_size intranode) and SM counts.
        try:
            self.buffer = HybridEPBuffer(
                self.group, hidden_dim=self.hidden,
                max_num_of_tokens_per_rank=self.max_tokens,
                num_local_experts=self.local_experts, use_fp8=False)
        except Exception as exc:
            raise RuntimeError(
                f"HybridEPBuffer construction failed (hidden={self.hidden} max_tokens={self.max_tokens} "
                f"local_experts={self.local_experts} world={world_size}): {exc!r}") from exc
        if rank == 0:
            print(f"[deepep-hybrid] HybridEPBuffer constructed (intranode NVLink, world={world_size}, "
                  f"local_experts={self.local_experts}, hidden={self.hidden})", file=sys.stderr)

        self.backend_provenance = {
            "deepep_commit": ver, "branch": "hybrid-ep",
            "impl": "deep_ep.HybridEPBuffer (NVIDIA TMA + warp-pipeline)",
            "mode": "normal", "transport": "intranode-nvlink",
            "resource_mode": args.resource_mode,
            "num_sms": None, "device_sms": dev_sms, "tuned_source": "fixed-kernel",
            "max_num_tokens": self.max_tokens, "top_k": self.top_k,
            "num_experts": self.num_experts, "local_experts": self.local_experts,
            "routing_factor": "ranks",
        }

    def buffer_cap(self, args):
        return self.max_tokens

    def make_problem(self, T, idx, weights, x):
        return types.SimpleNamespace(
            T=int(T), x=x,
            topk_idx=idx.to(torch.int64),
            topk_weights=weights.to(torch.float32),
        )

    def dispatch(self, p):
        # HybridEPBuffer.dispatch(hidden, topk_idx=, topk_weights=, num_of_experts=) ->
        #   (recv_hidden [n_recv, H], recv_x2, None, handle).
        out = self.buffer.dispatch(p.x, topk_idx=p.topk_idx, topk_weights=p.topk_weights,
                                   num_of_experts=self.num_experts)
        recv = out[0] if isinstance(out, (tuple, list)) else out
        handle = None
        if isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, tuple):
                    handle = o
        return types.SimpleNamespace(recv=recv, recv_payload=recv, handle=handle, combine_input=None)

    def stage(self, p, h):
        # Identity expert: the recv hidden IS the "expert output". combine reduces it per source token.
        h.combine_input = h.recv_payload
        return None

    def combine(self, p, h):
        # combine(hidden, handle=) -> [T, H] per-source-token reduction (no gate re-weight: "ranks").
        comb = self.buffer.combine(h.combine_input, handle=h.handle)
        return comb[0] if isinstance(comb, (tuple, list)) else comb

    def expected(self, p, h):
        # Round trip, identity expert, per-RANK-sum combine (no gate weights): each source token is
        # x * (distinct ranks among its top_k experts) — same as ep_flashinfer's "ranks" factor.
        ref = p.x.float()
        epr = max(1, self.num_experts // self.world_size)
        ranks = (p.topk_idx.long() // epr).clamp_(0, self.world_size - 1)        # [T, topk]
        present = torch.zeros(ranks.shape[0], self.world_size, device=ranks.device, dtype=torch.float32)
        present.scatter_(1, ranks, 1.0)
        factor = present.sum(dim=1, keepdim=True)                                # [T, 1] distinct ranks
        return ref * factor, p.T

    def recv_tokens(self, h):
        rp = h.recv_payload
        if torch.is_tensor(rp) and rp.dim() >= 1:
            return int(rp.shape[0])
        return 0

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
