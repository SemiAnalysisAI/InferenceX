#!/usr/bin/env python3
"""CollectiveX EP backend adapter — UCCL EP (NVIDIA), normal mode. SCAFFOLD — NOT yet
producing results (see docs/gated.md "UCCL EP").

IMPORTANT (empirically established on H100 via GHA): the LOW-LEVEL `uccl.ep.Buffer` is
NOT a drop-in DeepEP clone. Its constructor is
  Buffer(rank, num_ranks, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False, …)
— it takes rank/num_ranks ints, NOT a torch ProcessGroup, so the `Buffer(self.group, …)`
calls below raise `TypeError: incompatible function arguments`. The DeepEP-identical
`Buffer(group, …)` API is UCCL's separate ~1900-line `deep_ep_wrapper` package (packaged
as `deep_ep`, colliding with the container's real DeepEP), whose __init__ runs a proxy +
IPC-handle-exchange + runtime.sync + connect_atomic_buffer bootstrap. To finish UCCL:
vendor `deep_ep_wrapper` under a non-colliding name (it uses relative imports + only needs
`uccl.ep`) and import its Buffer here; then this file is a true ep_deepep.py clone. Until
then `benchmark=uccl` fails loudly (preserved failed-case), never faked. The build hook
cx_build_uccl + capability/schema wiring are in place as scaffolding.

The harness contract (make_problem/dispatch/stage/combine/expected/buffer_cap/recv_tokens/
finalize + backend_provenance + SUPPORTED_*) mirrors ep_deepep.py and is correct once the
wrapper Buffer is wired.

Install (see launchers/run_in_container.sh cx_build_uccl): `pip install uccl` ships a
prebuilt cp312 wheel; the UCCL EP kernels need a cu12 CUDA runtime on LD_LIBRARY_PATH
(pip install nvidia-cuda-runtime-cu12, prepend its lib dir) even on a cu13 image.

Correctness (identical to DeepEP's intranode test): a pure dispatch->combine round trip
with no expert compute reconstructs x only after dividing by the number of ranks each
token was sent to, so the harness expects combined ~= x * is_token_in_rank.sum(dim=1).
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

try:
    import uccl  # for version/provenance
    try:
        # PREFERRED: vendored deep_ep_wrapper (cx_build_uccl -> uccl_deepep). Buffer(group, ...)
        # takes a torch ProcessGroup (matches DeepEP + this adapter's calls) + runs UCCL's full
        # proxy/IPC/runtime.sync bootstrap. Fallback: low-level uccl.ep.Buffer(rank,num_ranks,...).
        from uccl_deepep import Buffer  # type: ignore
    except Exception:
        from uccl.ep import Buffer  # type: ignore
except Exception as exc:  # pragma: no cover - needs the installed uccl wheel + cu12 runtime
    print("ERROR: uccl.ep import failed — `pip install uccl nvidia-cuda-runtime-cu12` and "
          "prepend the cu12 lib dir to LD_LIBRARY_PATH at job setup (cx_build_uccl). "
          f"{exc!r}", file=sys.stderr)
    raise


def _uccl_version() -> str:
    try:
        import importlib.metadata as _md
        return _md.version("uccl")
    except Exception:
        return getattr(uccl, "__version__", "unknown")


# UCCL's normal-mode fp8 dispatch takes x as a (fp8, scales) tuple with a per-token
# block-128 scale — the SAME convention DeepEP's kernels expect (UCCL's ep.Buffer is a
# clone): scales [T, H//128] float32, e4m3, 448 = e4m3 max. Both directions of the cast
# run OUTSIDE the timed window (cast in make_problem, dequant in stage), so fp8
# quantization is NOT included in dispatch time (except under runtime-visible-v1).
_FP8_MAX = 448.0
_FP8_BLOCK = 128


def _per_token_cast_to_fp8(x):
    # x: [T, H] (H % 128 == 0) -> (x_fp8 [T,H] e4m3fn, scales [T, H//128] f32)
    T, H = x.shape
    xv = x.float().view(T, H // _FP8_BLOCK, _FP8_BLOCK)
    amax = xv.abs().amax(dim=2).clamp(min=1e-4)               # [T, H//128]
    x_fp8 = (xv * (_FP8_MAX / amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(T, H)
    return x_fp8, (amax / _FP8_MAX).contiguous()


def _per_block_dequant(x_fp8, scales):
    # inverse of the above: [R,H] e4m3 + [R, H//128] f32 -> [R,H] bf16
    R, H = x_fp8.shape
    xv = x_fp8.float().view(R, H // _FP8_BLOCK, _FP8_BLOCK)
    return (xv * scales.unsqueeze(2)).view(R, H).to(torch.bfloat16)


def _per_block_dequant_3d(x_fp8, scales):
    # LL recv layout: [E, S, H] e4m3 + [E, S, H//128] f32 -> [E, S, H] bf16
    E, S, H = x_fp8.shape
    xv = x_fp8.float().view(E, S, H // _FP8_BLOCK, _FP8_BLOCK)
    return (xv * scales.unsqueeze(-1)).view(E, S, H).to(torch.bfloat16)


class UCCLBackend:
    name = "uccl"
    combine_needs_redispatch = False  # UCCL combine reuses the handle (DeepEP-clone semantics)
    # Blackwell (B300) drops GPU clocks during the tiny small-T points, so the harness
    # re-ramps clocks at each shape before timing it. Harmless (just untimed iters) on H100/H200.
    wants_warm_burst = True
    # Capabilities — run_ep.py REJECTS anything outside these BEFORE construction (no
    # fallback/mislabel). Expanded as each path is implemented + hardware-validated.
    #   normal mode: bf16 + fp8 (per-token block-128 cast) — validated intranode NVLink on H200 (EP2).
    #   ll mode: low_latency_dispatch/combine via allow_nvlink_for_low_latency_mode — validated
    #   RUNNING intranode over NVLink on H200 (EP2); same DeepEP-clone LL kernel family.
    SUPPORTED_PRECISIONS = {"bf16", "fp8"}
    SUPPORTED_MODES = {"normal", "ll"}
    # Three contracts (mirror DeepEP — UCCL's Buffer is the same API):
    #   layout-and-dispatch-v1     — times get_dispatch_layout INSIDE dispatch; fp8 cast/dequant
    #                                OUTSIDE (preprocessing mirrors a producer handing quantized x).
    #   cached-layout-comm-only-v1 — layout hoisted out (untimed); dispatch = pure comm. normal only.
    #   runtime-visible-v1         — dispatch INCLUDES the fp8 quant (cast) + layout + comm + the
    #                                recv-dequant that makes expert input consumable; combine starts
    #                                from bf16 expert outputs. (normal mode; LL times all of it in-kernel.)
    SUPPORTED_CONTRACTS = {"layout-and-dispatch-v1", "cached-layout-comm-only-v1", "runtime-visible-v1"}

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = args.mode
        self.ll = (args.mode == "ll")
        self.contract = args.measurement_contract
        # hoist layout out of the timed dispatch only for the cached contract in normal mode.
        self.cache_layout = (self.contract == "cached-layout-comm-only-v1") and not self.ll
        # runtime-visible-v1: the fp8 cast + recv-dequant move INSIDE the timed dispatch (normal
        # mode). LL already times cast+layout+comm in its single kernel, so it's runtime-visible
        # by construction — the flag only changes normal mode's boundary.
        self.runtime_visible = (self.contract == "runtime-visible-v1") and not self.ll
        self.group = dist.group.WORLD
        assert args.dispatch_dtype in self.SUPPORTED_PRECISIONS and args.mode in self.SUPPORTED_MODES, \
            "run_ep.py must reject unsupported dtype/mode before constructing the backend"
        # fp8 e4m3 per-token-block round-trip caps reconstruction error near the largest
        # element at ~1/16 (3 mantissa bits); bf16 round-trip is ~5e-3. Tolerance is
        # recorded in the artifact so the looser fp8 gate is explicit, not hidden.
        self.fp8 = (args.dispatch_dtype == "fp8")
        self.tolerance = 1.25e-1 if self.fp8 else 5e-2
        dev_sms = torch.cuda.get_device_properties(device).multi_processor_count
        ver = _uccl_version()
        if self.ll:
            self._init_ll(args, dev_sms, ver)
        else:
            self._init_normal(args, rank, dev_sms, ver)

    def _init_normal(self, args, rank, dev_sms, ver):
        # fp8 cast: UNTIMED (make_problem) under layout-and-dispatch / cached-layout; TIMED (inside
        # dispatch) under runtime-visible-v1. So fp8_in_timing tracks the contract honestly.
        self.fp8_in_timing = (self.runtime_visible if self.fp8 else None)
        self.combine_needs_redispatch = False  # normal combine reuses the handle
        # Intranode normal mode: NVLink buffer only. ONE buffer size for ALL points (the shared
        # T=128 point must match between the decode and prefill sweeps). 4 GiB holds T up to 4096.
        num_nvl_bytes = int(os.environ.get("CX_UCCL_NVL_BYTES",
                                           os.environ.get("CX_DEEPEP_NVL_BYTES",
                                                          str(4 * 1024 * 1024 * 1024))))
        self.buffer = Buffer(self.group, num_nvl_bytes, 0)
        rm = args.resource_mode
        tuned_src = None
        if rm == "normalized":
            num_sms = max(1, round(args.sm_fraction * dev_sms))   # ~same device fraction as MoRI
        elif rm == "tuned":
            # Best-available for the installed UCCL: its OWN default SM count (Buffer.num_sms —
            # the library's analytic choice). get_dispatch_config(num_ranks) returns the
            # recommended Config but doesn't expose num_sms to Python; the default reflects it.
            num_sms = int(getattr(Buffer, "num_sms", args.num_sms))
            tuned_src = "uccl-default-num_sms"
        else:  # default — the bring-up budget
            num_sms = args.num_sms
        try:
            Buffer.set_num_sms(num_sms)
        except Exception as exc:  # pragma: no cover - version dependent
            if rank == 0:
                print(f"WARN: could not set num_sms={num_sms}: {exc!r}", file=sys.stderr)
        self.backend_provenance = {
            "uccl_version": ver,
            "uccl_commit": os.environ.get("UCCL_COMMIT") or f"pkg-{ver}",
            "mode": "normal", "resource_mode": rm, "num_sms": num_sms, "device_sms": dev_sms,
            "sm_fraction": (num_sms / dev_sms), "tuned_source": tuned_src or "n/a",
            "num_nvl_bytes": num_nvl_bytes,
        }

    def _init_ll(self, args, dev_sms, ver):
        # Low-latency mode: a distinct kernel family (IBGDA, but runs intranode over NVLink via
        # allow_nvlink_for_low_latency_mode). fp8 cast happens INSIDE low_latency_dispatch so for
        # fp8 the quantization IS inside the timed window (recorded honestly). The buffer is sized
        # for a FIXED num_max_dispatch_tokens_per_rank (all ranks identical), so LL is a
        # decode-shaped path; buffer_cap caps the sweep at num_max (no silent drop). set_num_sms
        # does NOT apply (the LL kernel picks its own occupancy) — recorded n/a.
        self.fp8_in_timing = (True if self.fp8 else None)
        self.combine_needs_redispatch = True   # re-dispatch (untimed) before each timed combine
        self.num_max = int(os.environ.get("CX_LL_MAX_TOKENS", "128"))
        self.experts = args.experts
        rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            self.num_max, args.hidden, self.world_size, args.experts)
        # one QP per local expert is the DeepEP/UCCL convention for LL
        self.num_qps = max(1, args.experts // self.world_size)
        self.buffer = Buffer(self.group, 0, rdma_bytes, low_latency_mode=True,
                             num_qps_per_rank=self.num_qps,
                             allow_nvlink_for_low_latency_mode=True)
        self.backend_provenance = {
            "uccl_version": ver,
            "uccl_commit": os.environ.get("UCCL_COMMIT") or f"pkg-{ver}",
            "mode": "ll", "resource_mode": args.resource_mode,
            "num_sms": None, "device_sms": dev_sms, "tuned_source": "ll-fixed-kernel",
            "num_max_dispatch_tokens_per_rank": self.num_max,
            "num_rdma_bytes": rdma_bytes, "num_qps_per_rank": self.num_qps,
            "low_latency_mode": True, "use_fp8": self.fp8,
        }

    def buffer_cap(self, args):
        # LL is sized for a fixed num_max; cap the sweep there (reported, not silent).
        return self.num_max if self.ll else None

    def make_problem(self, T, idx, weights, x):
        # idx[T,topk] int64, weights[T,topk] f32, x[T,hidden] bf16 — the shared trace slice.
        p = types.SimpleNamespace(T=T, x=x, topk_idx=idx.to(torch.int64),
                                  topk_weights=weights.to(torch.float32), layout=None)
        if self.fp8 and not self.ll and not self.runtime_visible:
            # layout-and-dispatch / cached-layout: per-token block-128 cast, UNTIMED (preprocessing,
            # mirrors the real producer that hands the dispatcher already-quantized activations).
            # runtime-visible does NOT pre-cast (the cast is timed inside dispatch); LL casts in-kernel.
            p.x_fp8, p.x_scales = _per_token_cast_to_fp8(x)
        if self.cache_layout:
            # cached-layout-comm-only-v1: compute the dispatch layout ONCE here (untimed) so the
            # timed dispatch is pure comm. (layout-and-dispatch-v1 leaves it None and dispatch
            # computes it inside the timed window.)
            ntr, _, ntpe, itir, _ = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
            p.layout = (ntr, ntpe, itir)
        return p

    def dispatch(self, p):
        if self.ll:
            return self._dispatch_ll(p)
        if p.layout is not None:                       # cached-layout-comm-only-v1
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = p.layout
        else:                                          # layout-and-dispatch / runtime-visible (timed layout)
            (num_tokens_per_rank, _, num_tokens_per_expert,
             is_token_in_rank, _) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        ref_fp8 = ref_scales = None
        if self.fp8:
            if self.runtime_visible:
                # runtime-visible: the per-token block-128 cast is INSIDE the timed dispatch.
                x_fp8, x_scales = _per_token_cast_to_fp8(p.x)
                ref_fp8, ref_scales = x_fp8, x_scales      # for the correctness reference
            else:
                x_fp8, x_scales = p.x_fp8, p.x_scales      # pre-cast (untimed)
            x_in = (x_fp8, x_scales)
        else:
            x_in = p.x
        recv_x, _recv_idx, recv_topk_weights, _, handle, _ = self.buffer.dispatch(
            x_in, topk_idx=p.topk_idx, topk_weights=p.topk_weights,
            num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert)
        out = types.SimpleNamespace(
            recv_x=recv_x, recv_topk_weights=recv_topk_weights, handle=handle,
            is_token_in_rank=is_token_in_rank, ref_fp8=ref_fp8, ref_scales=ref_scales)
        if self.fp8 and self.runtime_visible:
            # dispatch ENDS when expert input is consumable: dequant fp8 recv -> bf16 INSIDE the
            # timed window (the contract's "expert input genuinely consumable" boundary). stage()
            # then no-ops for this contract.
            recv_fp8, recv_scales = recv_x
            out.combine_input = _per_block_dequant(recv_fp8, recv_scales)
            out.rv_staged = True
        return out

    def _dispatch_ll(self, p):
        # x is bf16; the kernel casts to fp8 internally when use_fp8=True (so for fp8 the cast IS
        # inside this timed op — fp8_in_timing=True). recv is the expert-major 3D layout
        # [num_local_experts, num_max*world, hidden] (+scales when fp8).
        recv_x, recv_count, handle, _event, _hook = self.buffer.low_latency_dispatch(
            p.x, p.topk_idx, self.num_max, self.experts,
            use_fp8=self.fp8, return_recv_hook=False)
        return types.SimpleNamespace(recv_x=recv_x, recv_count=recv_count, handle=handle)

    def stage(self, p, h):
        # comm-only contract: "expert outputs" already exist as recv_x. Dequantize fp8 recv to
        # bf16 HERE (untimed) — the expert-compute boundary — so combine moves bf16 in both
        # precisions. Bf16 recv is staged as-is. (LL recv is 3D; normal recv is 2D.)
        if getattr(h, "rv_staged", False):
            return None   # runtime-visible already produced bf16 combine_input inside dispatch (timed)
        if self.ll:
            if self.fp8:
                recv_fp8, recv_scales = h.recv_x
                h.combine_input = _per_block_dequant_3d(recv_fp8, recv_scales)
            else:
                h.combine_input = h.recv_x
        elif self.fp8:
            recv_fp8, recv_scales = h.recv_x
            h.combine_input = _per_block_dequant(recv_fp8, recv_scales)
        else:
            h.combine_input = h.recv_x
        return None

    def combine(self, p, h):
        if self.ll:
            # weighted per-expert reduce; topk_idx/weights are the ORIGINAL per-token ones.
            combined_x, _event, _hook = self.buffer.low_latency_combine(
                h.combine_input, p.topk_idx, p.topk_weights, h.handle)
            return combined_x
        combined_x, _, _ = self.buffer.combine(h.combine_input, h.handle,
                                               topk_weights=h.recv_topk_weights)
        return combined_x

    def expected(self, p, h):
        if self.ll:
            # LL combine reduces each token's topk expert copies weighted by topk_weights; with no
            # expert compute each copy is (the kernel's fp8 cast of) x, so combined ~= x *
            # sum(topk_weights). fp8 quant error is covered by self.tolerance.
            wsum = p.topk_weights.sum(dim=1, keepdim=True)
            return p.x.float() * wsum, p.T
        # normal: round trip with no expert compute reconstructs x*(#destination ranks); for fp8
        # compare against the dequantized cast that was actually sent.
        ranks_per_token = h.is_token_in_rank.sum(dim=1, keepdim=True).clamp(min=1).float()
        ref = p.x.float()
        if self.fp8:
            # runtime-visible cast lives on the handle (no pre-cast on p); else use the pre-cast.
            x_fp8 = getattr(h, "ref_fp8", None)
            x_scales = getattr(h, "ref_scales", None)
            if x_fp8 is None:
                x_fp8, x_scales = p.x_fp8, p.x_scales
            ref = _per_block_dequant(x_fp8, x_scales).float()
        return ref * ranks_per_token, p.T

    def recv_tokens(self, h):
        if self.ll:
            return int(h.recv_count.sum().item())  # token-copies received across local experts
        rx = h.recv_x[0] if isinstance(h.recv_x, tuple) else h.recv_x
        return int(rx.shape[0])

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
