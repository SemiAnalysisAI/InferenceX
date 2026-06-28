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

# e4m3fnuz (the ROCm-native fp8) finite max. AMD's "fnuz" (finite, no -0/Inf/NaN-unsigned) e4m3
# saturates at 240.0 — the dispatch fp8 cast scales each block so its amax maps to this.
_FP8_FNUZ_MAX = 240.0
_FP8_BLOCK = 128  # MoRI/DeepSeek blockwise fp8: one scale per 128-elem hidden block (7168%128==0)


def _mori_quant_introspect():
    """Describe MoRI's quant API (enum members + ctor/dispatch signatures + quant/scale helpers).

    FNUZ fp8 dispatch on MoRI keys off EpDispatchCombineConfig.quant_type, which PR311 extended with
    QuantType::Fp8BlockwiseQuant — but how that value is EXPOSED to Python (enum attr vs accepted
    string vs int) differs by build. We print this to stderr at construction so a GHA run's log is
    self-documenting: even if the run wedges or the quant_type guess is wrong, the next iteration has
    MoRI's exact surface without needing interactive SSH (which stalls on the shared cluster)."""
    import inspect
    info = {}
    ops = getattr(mori, "ops", None)
    try:
        info["config_sig"] = str(inspect.signature(mori.ops.EpDispatchCombineConfig.__init__))
    except Exception as e:
        info["config_sig"] = f"<err {e!r}>"
    for meth in ("dispatch", "combine"):
        try:
            info[f"{meth}_sig"] = str(inspect.signature(getattr(mori.ops.EpDispatchCombineOp, meth)))
        except Exception as e:
            info[f"{meth}_sig"] = f"<err {e!r}>"
    # Any enum / helper whose name mentions quant or scale (the QuantType enum + any quantize fn).
    surface = {}
    for nm in (dir(ops) if ops else []):
        if nm.startswith("_"):
            continue
        if "quant" in nm.lower() or "scale" in nm.lower():
            obj = getattr(ops, nm)
            members = {}
            for m in dir(obj):
                if m.startswith("_"):
                    continue
                try:
                    members[m] = int(getattr(obj, m))
                except Exception:
                    members[m] = str(type(getattr(obj, m)).__name__)
            surface[nm] = members or str(type(obj).__name__)
    info["quant_surface"] = surface
    return info


def _fp8_quant_type_candidates():
    """Ordered (value, label) candidates for MoRI's blockwise-fp8 quant_type. The config currently
    accepts the STRING "none", so strings are viable; we still try the typed enum first (PR311's
    QuantType::Fp8BlockwiseQuant). __init__ keeps the first that constructs."""
    ops = mori.ops
    out = []
    for enum_name in ("EpDispatchCombineQuantType", "QuantType", "DispatchCombineQuantType"):
        enum = getattr(ops, enum_name, None)
        if enum is None:
            continue
        for member in dir(enum):
            ml = member.lower()
            if member.startswith("_") or "fp8" not in ml:
                continue
            try:
                out.append((getattr(enum, member), f"{enum_name}.{member}"))
            except Exception:
                pass
    # String fallbacks (best guess first) — mirror the PR311 naming.
    for s in ("fp8_blockwise", "Fp8BlockwiseQuant", "fp8", "Fp8"):
        out.append((s, f"str:{s}"))
    return out


def _quant_blockwise_fp8_fnuz(x, block=_FP8_BLOCK):
    """bf16 [T,H] -> (e4m3fnuz [T,H], f32 per-block scales [T,H//block]). Per-128-block amax scaling
    onto the fnuz finite range. Caller-side quantization (MoRI transports the fp8 payload + scales;
    the combine reduces and the harness dequantizes for the consistency-correctness gate)."""
    T, H = x.shape
    assert H % block == 0, f"hidden {H} not a multiple of fp8 block {block}"
    nb = H // block
    xb = x.float().view(T, nb, block)
    amax = xb.abs().amax(dim=2).clamp_min(1e-8)          # [T, nb]
    scale = amax / _FP8_FNUZ_MAX                          # f32 dequant scale
    xq = (xb / scale.unsqueeze(2)).clamp(-_FP8_FNUZ_MAX, _FP8_FNUZ_MAX).to(torch.float8_e4m3fnuz)
    return xq.view(T, H), scale


def _dequant_blockwise_fp8_fnuz(xq, scale, block=_FP8_BLOCK):
    """Inverse of _quant_blockwise_fp8_fnuz: e4m3fnuz [T,H] + f32 [T,H//block] -> bf16-range f32 [T,H]."""
    T, H = xq.shape
    nb = H // block
    return (xq.float().view(T, nb, block) * scale.unsqueeze(2)).view(T, H)


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
    # fallback/mislabel). DISPATCH precision and the SEPARATE combine path are distinct axes
    # (review: dispatch_dtype=fp8 must NOT imply quantized combine). bf16 is the default; fp8
    # routes the AMD-native blockwise path (QuantType::Fp8BlockwiseQuant, MoRI PR311) — caller-side
    # e4m3fnuz block-128 quantization transported through the MoRI A2A, dequantized for the
    # consistency-correctness gate. The combine OUTPUT stays bf16 (quant_type drives transport, the
    # reduction emits bf16) so SUPPORTED_COMBINE_DTYPES is unchanged. Keep in sync with
    # capability.py CAP["mori"].
    SUPPORTED_DISPATCH_DTYPES = {"bf16", "fp8"}  # fp8 = e4m3fnuz blockwise (FNUZ dispatch variant)
    SUPPORTED_COMBINE_DTYPES = {"bf16"}         # + "fp8" once the PR311 quant combine OUTPUT lands
    SUPPORTED_COMBINE_QUANT_MODES = {"none"}    # + the PR311 mode id once validated
    SUPPORTED_PRECISIONS = SUPPORTED_DISPATCH_DTYPES  # back-compat alias (run_ep.py / older refs)
    SUPPORTED_MODES = {"normal"}           # MoRI has no separate low-latency entrypoint
    # MoRI computes its routing layout INSIDE the dispatch kernel (block_num/warps launch);
    # it cannot be hoisted, so MoRI honors only the layout-and-dispatch contract. Cross-
    # vendor comparisons must therefore use layout-and-dispatch-v1 (the common contract).
    SUPPORTED_CONTRACTS = {"layout-and-dispatch-v1"}

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = args.mode
        assert (args.dispatch_dtype in self.SUPPORTED_DISPATCH_DTYPES
                and args.mode in self.SUPPORTED_MODES
                and getattr(args, "combine_dtype", "bf16") in self.SUPPORTED_COMBINE_DTYPES
                and getattr(args, "combine_quant_mode", "none") in self.SUPPORTED_COMBINE_QUANT_MODES), \
            "run_ep.py must reject unsupported dispatch/mode/combine before constructing the backend"
        self.fp8_in_timing = None  # set when fp8 dispatch is used (whether the cast is timed)
        # Combine-path quant timing (None today — no quant combine wired). PR311 sets these +
        # the combine_* dtype attrs ep_harness reads via getattr; until then ep_harness records
        # combine bf16 / none from the args defaults.
        self.combine_quant_in_timing = None
        self.combine_dequant_in_timing = None
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
        # Dispatch precision: bf16 (quant_type="none", scale_dim=0) or fp8 (e4m3fnuz blockwise — the
        # FNUZ variant). For fp8 we DUMP MoRI's quant API to stderr (the GHA log is then self-
        # documenting even if the run wedges or the guess is wrong — SSH inspection stalls on the
        # shared cluster) and resolve quant_type by trying candidates until the config constructs.
        self._fp8 = (args.dispatch_dtype == "fp8")
        self._quant_label = "none"
        scale_dim = 0
        quant_type = "none"
        if self._fp8:
            import json as _json
            print("MORI_QUANT_API " + _json.dumps(_mori_quant_introspect()), file=sys.stderr, flush=True)
            assert args.hidden % _FP8_BLOCK == 0, f"hidden {args.hidden} not divisible by fp8 block {_FP8_BLOCK}"
            scale_dim = args.hidden // _FP8_BLOCK
            cands = _fp8_quant_type_candidates()
            print(f"MORI_FP8_CANDIDATES {[l for _, l in cands]}", file=sys.stderr, flush=True)
            for val, label in cands:
                try:
                    mori.ops.EpDispatchCombineConfig(
                        data_type=torch.bfloat16, rank=rank, world_size=world_size,
                        hidden_dim=args.hidden, scale_dim=scale_dim,
                        scale_type_size=torch.tensor([], dtype=torch.float32).element_size(),
                        max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
                        max_num_inp_token_per_rank=max(512, self._cap),
                        num_experts_per_rank=self.experts_per_rank,
                        num_experts_per_token=args.topk,
                        use_external_inp_buf=False, quant_type=val)
                    quant_type, self._quant_label = val, label
                    break
                except Exception as e:
                    print(f"MORI_FP8_REJECT {label}: {e!r}", file=sys.stderr, flush=True)
            if quant_type == "none":
                raise RuntimeError("no MoRI quant_type candidate accepted for fp8 blockwise — see "
                                   "MORI_QUANT_API above for this build's actual quant surface")
            print(f"MORI_FP8_QUANT_TYPE {self._quant_label}", file=sys.stderr, flush=True)
            self.fp8_in_timing = True  # caller-side cast, cached on the problem (untimed steady state)
        # fp8 carries a per-block f32 scale; bf16 keeps the 1-byte sentinel the bring-up used.
        _scale_elt = torch.tensor([], dtype=(torch.float32 if self._fp8 else torch.float8_e4m3fnuz)).element_size()
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=torch.bfloat16, rank=rank, world_size=world_size,
            hidden_dim=args.hidden, scale_dim=scale_dim,
            scale_type_size=_scale_elt,
            max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
            max_num_inp_token_per_rank=max(512, self._cap),
            num_experts_per_rank=self.experts_per_rank,
            num_experts_per_token=args.topk,
            use_external_inp_buf=False, quant_type=quant_type,
        )
        self.op = mori.ops.EpDispatchCombineOp(self.config)
        # fp8 blockwise carries fp8 quant error -> loosen the correctness gate to the fp8 class
        # (the harness reads backend.tolerance; bf16 default 5e-2). The combine reduces the
        # (dequantized) payload per rank, compared against x*unique_ranks within this tolerance class.
        if self._fp8:
            self.tolerance = 1.5e-1
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
            "dispatch_dtype": args.dispatch_dtype,
            "quant_type": self._quant_label,
            "fp8_format": ("e4m3fnuz" if self._fp8 else None),
            "fp8_block": (_FP8_BLOCK if self._fp8 else None),
        }

    def buffer_cap(self, args):
        # Largest tokens/rank the 2 GiB registerable heap holds at hidden=7168 (512,
        # validated on-node). Override via CX_MORI_MAX_TOKENS.
        return int(os.environ.get("CX_MORI_MAX_TOKENS", "512"))

    def make_problem(self, T, idx, weights, x):
        # Shared-trace slice: idx[T,topk] -> int32 (MoRI expects int32 expert ids);
        # weights[T,topk] f32; x[T,hidden] bf16. bf16: scales is the (T,0) fp8 sentinel (scale_dim==0).
        # fp8: a sized [T, hidden/128] f32 scale buffer (scale_dim>0) the blockwise-fp8 kernel uses.
        indices = idx.to(torch.int32)
        if self._fp8:
            nb = x.size(1) // _FP8_BLOCK
            scales = torch.empty((T, nb), dtype=torch.float32, device=self.device)
        else:
            scales = torch.empty((T, 0), dtype=torch.float8_e4m3fnuz, device=self.device)
        return types.SimpleNamespace(T=T, x=x, indices=indices,
                                     weights=weights.to(torch.float32), scales=scales)

    def dispatch(self, p):
        (dispatch_output, dispatch_weights, out_scales, dispatch_indices, recv_num) = self.op.dispatch(
            p.x, p.weights, p.scales, p.indices,
            block_num=self.block_num, warp_per_block=self.dispatch_warps)
        total_recv = int(recv_num[0].item())  # read BEFORE combine (combine resets recv_num)
        # Form the bf16 combine input. If the blockwise-fp8 kernel returned an fp8 payload (+ its
        # per-block scales), dequant it; if it already dequantized to bf16, use it directly. Both
        # the bf16 path and the kernel-dequantized fp8 path land here as a plain .to(bf16).
        if dispatch_output.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            deq = _dequant_blockwise_fp8_fnuz(dispatch_output[:total_recv].contiguous(),
                                              out_scales[:total_recv].contiguous().to(torch.float32))
            combine_input = torch.zeros((dispatch_output.size(0), dispatch_output.size(1)),
                                        dtype=torch.bfloat16, device=self.device)
            combine_input[:total_recv] = deq.to(torch.bfloat16)
        else:
            combine_input = dispatch_output.to(torch.bfloat16)
        return types.SimpleNamespace(
            dispatch_output=dispatch_output, dispatch_weights=dispatch_weights,
            dispatch_indices=dispatch_indices, total_recv=total_recv,
            combine_input=combine_input)

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
