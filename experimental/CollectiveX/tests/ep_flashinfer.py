#!/usr/bin/env python3
"""CollectiveX EP backend adapter — FlashInfer EP (NVIDIA), normal mode.

This file owns ONLY FlashInfer's MoE-AllToAll API calls + its correctness reference;
the harness (ep_harness.py) owns the deterministic shared routing trace, the comm-only
timing, the correctness gate, and the provenance-tagged doc. The adapter protocol
(make_problem / dispatch / stage / combine / expected / buffer_cap / recv_tokens /
finalize + backend_provenance + SUPPORTED_*) mirrors ep_deepep.py exactly.

WHAT FLASHINFER PROVIDES (flashinfer 0.6.8.post1, NVIDIA container):
  * `flashinfer.comm.MoeAlltoAll(mapping, max_num_tokens, top_k, num_experts)` — a class
    holding an MNNVL symmetric workspace, with
      .dispatch(token_selected_experts, input_payloads: list[Tensor],
                runtime_max_tokens_per_rank, ...)  -> recv payload(s)
      .combine(payload, runtime_max_tokens_per_rank, payload_in_workspace=False) -> combined
  * module-level `flashinfer.comm.trtllm_moe_alltoall` and the lower-level
    `moe_a2a_dispatch` / `moe_a2a_combine` / `moe_a2a_initialize` /
    `get_workspace_size_per_rank` — the TensorRT-LLM one-sided path. Selected by
    env CX_FLASHINFER_TRTLLM=1 (provenance trtllm=True); covers goal's
    "TensorRT-LLM NVLink one-sided AllToAll EP".

The exact kwarg names for dispatch/combine and the Mapping constructor differ across
FlashInfer point releases. This adapter has NO GPU to validate against, so EVERY
FlashInfer API call is wrapped to fail LOUD + SPECIFIC (the call site, the kwargs
tried, and the underlying error) so the parent's GHA smoke shows precisely what to fix
rather than a bare TypeError. See `_call_variants` and `_build_mapping`.

CORRECTNESS (`expected`): FlashInfer's MoeAlltoAll is expert-centric (TensorRT-LLM MoE
A2A): `dispatch` sends each token to its top_k selected experts; `combine` gathers the
per-expert results back and reduces the top_k copies for each SOURCE token. With an
identity expert (the harness does NO expert compute) and a combine that does NOT apply
the gate weights (the public `combine(payload, ...)` takes no topk_weights — gate
weighting is the MoE epilogue, not the comm), the round trip yields:
      combined ≈ x * top_k          (sum of top_k identical copies of x)
This is structurally DeepEP-LL-like (per-expert reduce) but WITHOUT LL's weight multiply.
The alternative (combine applies softmax gate weights, like DeepEP LL) would give
`x * sum(topk_weights)`. We LEAD with `x * top_k` and document both; the parent's GHA
validates which FlashInfer actually implements and flips ONE constant (_ROUTING_FACTOR).
Tolerance bf16 ~5e-2 (FlashInfer dispatch keeps bf16 end-to-end; no fp8 round-trip yet).

STATUS: bf16 / normal / layout-and-dispatch-v1 only (fp8 is behind a clearly-marked
TODO below). The MoeAlltoAll workspace bootstraps inside the single torch.distributed
NCCL group of same-user ranks (MNNVL symmetric memory) — no special caps assumed here;
the launcher/image owns CAP_SYS_PTRACE / FABRIC plumbing (docs/gated.md).
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

try:
    import flashinfer  # for version/provenance
    import flashinfer.comm as fi_comm  # MoeAlltoAll / trtllm_moe_alltoall / moe_a2a_* live here
except Exception as exc:  # pragma: no cover - needs the FlashInfer wheel on the container
    print("ERROR: flashinfer import failed — FlashInfer must be present on the container at job "
          "setup (cx_build_flashinfer: `pip install flashinfer-python`). "
          f"{exc!r}", file=sys.stderr)
    raise


def _flashinfer_version() -> str:
    try:
        import importlib.metadata as _md
        return _md.version("flashinfer-python")
    except Exception:
        try:
            import importlib.metadata as _md
            return _md.version("flashinfer")
        except Exception:
            return getattr(flashinfer, "__version__", "unknown")


# --- The round-trip routing factor (see module docstring). LEAD = top_k (sum of top_k
# identical copies, combine does NOT weight). If GHA shows FlashInfer's combine applies
# the gate weights instead, flip this to "weight-sum" and the reference becomes
# x * sum(topk_weights). This is the ONE knob the parent edits after the first GHA run. ---
_ROUTING_FACTOR = os.environ.get("CX_FLASHINFER_ROUTING_FACTOR", "topk")  # "topk" | "weight-sum"


def _loud(where: str, attempted, exc: Exception) -> RuntimeError:
    """Build a LOUD + SPECIFIC error for a failed FlashInfer call so the parent's GHA smoke
    shows exactly which API/kwargs to fix (no GPU here to discover the right names)."""
    return RuntimeError(
        f"FlashInfer EP adapter: {where} failed against flashinfer {_flashinfer_version()}. "
        f"Attempted: {attempted}. Underlying error: {exc!r}. "
        f"FIX: inspect the installed flashinfer.comm signatures "
        f"(python3 -c 'import flashinfer.comm as c; help(c.MoeAlltoAll)') and adjust the "
        f"kwarg names / Mapping construction in tests/ep_flashinfer.py.")


def _call_variants(where: str, fn, variants):
    """Try a sequence of (args, kwargs) plausible signatures for one FlashInfer call.
    Returns (result, chosen_index). Raises a LOUD error listing EVERY attempt if all fail.
    Used so a renamed kwarg surfaces as a precise, actionable message in GHA — not a
    silent fallback (the harness contract forbids faking) and not a bare TypeError."""
    errors = []
    for i, (args, kwargs) in enumerate(variants):
        try:
            return fn(*args, **kwargs), i
        except TypeError as exc:        # wrong kwarg name / arity — try the next signature
            errors.append(f"  variant[{i}] args={_shape_repr(args)} kwargs={list(kwargs)} -> {exc!r}")
        # any non-TypeError (e.g. a real CUDA/runtime error) is NOT a signature problem —
        # re-raise immediately, wrapped, so it isn't masked by trying other signatures.
        except Exception as exc:
            raise _loud(where, _shape_repr(args) + f" kwargs={list(kwargs)}", exc)
    raise _loud(where, "all signature variants exhausted:\n" + "\n".join(errors),
                TypeError("no matching signature"))


def _shape_repr(args):
    out = []
    for a in args:
        if torch.is_tensor(a):
            out.append(f"Tensor{tuple(a.shape)}:{a.dtype}")
        elif isinstance(a, (list, tuple)):
            out.append("[" + ",".join(
                f"Tensor{tuple(t.shape)}:{t.dtype}" if torch.is_tensor(t) else repr(t) for t in a) + "]")
        else:
            out.append(repr(a))
    return "(" + ", ".join(out) + ")"


def _build_mapping(world_size, rank):
    """Construct the FlashInfer Mapping for PURE EP: tp_size=1, moe_ep_size=world_size,
    moe_tp_size=1. The Mapping kwarg set varies across releases, so try the plausible
    constructors defensively and record which one worked (logged at rank 0). Raises a LOUD
    error (listing every attempt) if none construct."""
    Mapping = getattr(fi_comm, "Mapping", None) or getattr(flashinfer, "Mapping", None)
    if Mapping is None:
        raise _loud("Mapping lookup",
                    "flashinfer.comm.Mapping / flashinfer.Mapping not found",
                    AttributeError("Mapping"))
    # Ordered most-specific (pure-EP, explicit moe_*) -> least. Each is a full kwargs dict.
    variants = [
        ((), dict(world_size=world_size, rank=rank, gpus_per_node=world_size,
                  tp_size=1, moe_ep_size=world_size, moe_tp_size=1)),
        ((), dict(world_size=world_size, rank=rank,
                  tp_size=1, moe_ep_size=world_size, moe_tp_size=1)),
        ((), dict(world_size=world_size, rank=rank, moe_ep_size=world_size, moe_tp_size=1)),
        ((), dict(world_size=world_size, rank=rank, tp_size=1, ep_size=world_size)),
        ((), dict(world_size=world_size, rank=rank, moe_ep_size=world_size)),
        # positional last-resort: (world_size, rank, gpus_per_node, tp_size, ...) shapes seen
        ((world_size, rank), dict(tp_size=1, moe_ep_size=world_size, moe_tp_size=1)),
        ((world_size, rank), {}),
    ]
    mapping, idx = _call_variants("Mapping(...)", Mapping, variants)
    return mapping, idx


class FlashInferBackend:
    name = "flashinfer"
    # FlashInfer combine reuses the dispatch workspace/handle (no re-dispatch needed before
    # a timed combine), mirroring DeepEP normal mode — combine consumes the recv payload.
    combine_needs_redispatch = False
    # Blackwell (B300/GB300) drops GPU clocks during the tiny small-T points, so the harness
    # re-ramps clocks at each shape before timing it. Harmless (just untimed iters) on H100/H200.
    wants_warm_burst = True
    # Capabilities — run_ep.py REJECTS anything outside these BEFORE construction (no
    # fallback/mislabel). Start bf16 / normal / layout-and-dispatch only.
    #   bf16: FlashInfer MoeAlltoAll keeps bf16 payloads end-to-end (no quant round trip).
    #   fp8 : TODO (see SUPPORTED_PRECISIONS note) — FlashInfer supports mxfp8/nvfp4 payloads via
    #         moe_a2a (PR3376/3643) but it is MNNVL-gated on x86_64; not wired here yet.
    SUPPORTED_PRECISIONS = {"bf16"}
    # TODO(fp8): add "fp8" once the per-token-block (or mx/nvfp4) payload path is wired AND
    # hardware-validated on an MNNVL-capable runner. FlashInfer's moe_a2a takes multiple input
    # payloads (x + scales) as the input_payloads list; the dispatch call already passes a list,
    # so fp8 = append the scale tensor + set the payload dtype, then dequant in stage() like
    # ep_deepep.py. Gated until then (docs/gated.md, goal.md "MXFP8 dispatch ⛔ gated").
    SUPPORTED_MODES = {"normal"}
    # Only the contract whose timing boundary FlashInfer can honor: layout (the dispatch
    # send-counts) is computed inside dispatch and cannot be hoisted to a separate untimed
    # step the way DeepEP's get_dispatch_layout can — so cached-layout-comm-only-v1 and
    # runtime-visible-v1 (fp8) are NOT offered.
    SUPPORTED_CONTRACTS = {"layout-and-dispatch-v1"}
    # Combine path is bf16 / none today (the harness default); declared explicitly so the
    # capability gate and run_ep.py agree (they getattr these with bf16/none defaults anyway).
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
        # bf16 round-trip reconstruction error is ~5e-3; 5e-2 leaves headroom (kept identical to
        # the other bf16 adapters so the gate is comparable). Recorded in the artifact.
        self.tolerance = 5e-2
        # No quant in the timed window today (bf16 end-to-end). Recorded honestly.
        self.fp8_in_timing = None

        # The TensorRT-LLM one-sided variant (env CX_FLASHINFER_TRTLLM=1) routes the SAME
        # interface through trtllm_moe_alltoall / moe_a2a_* instead of the MoeAlltoAll class.
        self.trtllm = os.environ.get("CX_FLASHINFER_TRTLLM", "0") == "1"

        self.top_k = int(args.topk)
        self.num_experts = int(args.experts)
        # Workspace/buffer ceiling. The MoeAlltoAll symmetric workspace is sized for
        # max_num_tokens per rank; the sweep is capped at this (buffer_cap) so a too-large T
        # is dropped (reported) rather than overflowing. 4096 holds the prefill ladder top.
        self.max_num_tokens = int(os.environ.get("CX_FLASHINFER_MAX_TOKENS", "4096"))

        dev_sms = torch.cuda.get_device_properties(device).multi_processor_count
        ver = _flashinfer_version()

        # Build the pure-EP Mapping (defensive over kwarg variants; logs which worked).
        self.mapping, map_variant = _build_mapping(world_size, rank)
        if rank == 0:
            print(f"[flashinfer] Mapping constructed via variant #{map_variant} "
                  f"(world={world_size} rank={rank} tp=1 moe_ep={world_size} moe_tp=1)",
                  file=sys.stderr)

        # Construct the comm object. The MoeAlltoAll class allocates its MNNVL symmetric
        # workspace internally; the trtllm path initializes via moe_a2a_initialize +
        # get_workspace_size_per_rank. Both are tried defensively and recorded.
        self.path = "moe_alltoall"
        self.a2a = None            # the MoeAlltoAll instance (class path)
        self.workspace = None      # the trtllm workspace tensor(s) (functional path)
        self.ws_size = None
        if self.trtllm:
            self._init_trtllm(ver)
        else:
            self._init_moe_alltoall(ver)

        self.backend_provenance = {
            "flashinfer_version": ver,
            "flashinfer_commit": os.environ.get("FLASHINFER_COMMIT") or f"pkg-{ver}",
            "mode": "normal", "path": self.path, "trtllm": self.trtllm,
            "resource_mode": args.resource_mode,
            # FlashInfer MoE A2A occupancy is fixed by the library (a symmetric-memory kernel, not
            # an SM/CU budget we set) — like DeepEP LL. Recorded as a fixed-kernel run so the
            # resource_profile maps it to resource_class=fixed-kernel (excluded from the Pareto).
            "num_sms": None, "device_sms": dev_sms, "tuned_source": "fixed-kernel",
            "max_num_tokens": self.max_num_tokens, "top_k": self.top_k,
            "num_experts": self.num_experts,
            "mapping_variant": map_variant,
            "routing_factor": _ROUTING_FACTOR,
            # MNNVL symmetric workspace — bootstraps within the NCCL group; the launcher owns
            # the CAP_SYS_PTRACE (x86_64) / FABRIC (aarch64) plumbing (docs/gated.md).
            "workspace": "mnnvl-symmetric",
        }

    def _init_moe_alltoall(self, ver):
        """Class path: flashinfer.comm.MoeAlltoAll(mapping, max_num_tokens, top_k, num_experts)."""
        MoeAlltoAll = getattr(fi_comm, "MoeAlltoAll", None)
        if MoeAlltoAll is None:
            raise _loud("MoeAlltoAll lookup", "flashinfer.comm.MoeAlltoAll not found",
                        AttributeError("MoeAlltoAll"))
        # kwarg names have drifted across releases; try the documented set + positional fallback.
        variants = [
            ((self.mapping,), dict(max_num_tokens=self.max_num_tokens,
                                   top_k=self.top_k, num_experts=self.num_experts)),
            ((self.mapping,), dict(max_num_tokens=self.max_num_tokens,
                                   top_k=self.top_k, ep_size=self.world_size,
                                   num_experts=self.num_experts)),
            ((self.mapping, self.max_num_tokens, self.top_k, self.num_experts), {}),
            ((self.mapping,), dict(max_num_tokens_per_rank=self.max_num_tokens,
                                   top_k=self.top_k, num_experts=self.num_experts)),
        ]
        self.a2a, idx = _call_variants("MoeAlltoAll(...)", MoeAlltoAll, variants)
        self.path = "moe_alltoall"
        if self.rank == 0:
            print(f"[flashinfer] MoeAlltoAll constructed via variant #{idx}", file=sys.stderr)

    def _init_trtllm(self, ver):
        """Functional one-sided path: moe_a2a_initialize + get_workspace_size_per_rank
        (the TensorRT-LLM NVLink one-sided AllToAll). dispatch/combine then go through
        moe_a2a_dispatch / moe_a2a_combine (or trtllm_moe_alltoall). Sizing the workspace
        here is best-effort + defensive; the per-call wiring is in _dispatch_trtllm."""
        self.path = "trtllm_moe_alltoall"
        get_ws = getattr(fi_comm, "get_workspace_size_per_rank", None)
        init = getattr(fi_comm, "moe_a2a_initialize", None)
        if get_ws is not None:
            try:
                self.ws_size, _ = _call_variants(
                    "get_workspace_size_per_rank(...)", get_ws,
                    [((), dict(max_num_tokens=self.max_num_tokens, top_k=self.top_k,
                               num_experts=self.num_experts, ep_size=self.world_size)),
                     ((self.max_num_tokens, self.top_k, self.num_experts, self.world_size), {}),
                     ((self.max_num_tokens, self.top_k, self.num_experts), {})])
            except Exception as exc:
                # not fatal at construction — surface at first dispatch if it actually blocks
                if self.rank == 0:
                    print(f"[flashinfer] WARN: get_workspace_size_per_rank probe failed: {exc!r}",
                          file=sys.stderr)
        if init is not None:
            try:
                self.workspace, _ = _call_variants(
                    "moe_a2a_initialize(...)", init,
                    [((self.mapping,), dict(max_num_tokens=self.max_num_tokens, top_k=self.top_k,
                                            num_experts=self.num_experts)),
                     ((self.mapping, self.max_num_tokens, self.top_k, self.num_experts), {})])
            except Exception as exc:
                if self.rank == 0:
                    print(f"[flashinfer] WARN: moe_a2a_initialize probe failed: {exc!r}",
                          file=sys.stderr)
        if self.rank == 0:
            print(f"[flashinfer] trtllm one-sided path initialized "
                  f"(ws_size={self.ws_size})", file=sys.stderr)

    def buffer_cap(self, args):
        # The symmetric workspace is sized for max_num_tokens per rank; cap the sweep there
        # (reported by the harness, never silently truncated).
        return self.max_num_tokens

    def make_problem(self, T, idx, weights, x):
        # idx[T,topk] int64, weights[T,topk] f32, x[T,hidden] bf16 — the shared trace slice.
        # FlashInfer's dispatch wants: token_selected_experts = idx (the per-token expert IDs),
        # input_payloads = [x] (a list — fp8 would append the scale tensor here, see TODO).
        # token_selected_experts is commonly int32 in TensorRT-LLM kernels; keep an int32 copy
        # alongside the int64 (the harness/expected use int64; the kernel call uses int32).
        p = types.SimpleNamespace(
            T=int(T), x=x,
            topk_idx=idx.to(torch.int64),
            topk_idx_i32=idx.to(torch.int32),
            topk_weights=weights.to(torch.float32),
        )
        return p

    def dispatch(self, p):
        if self.trtllm:
            return self._dispatch_trtllm(p)
        # MoeAlltoAll.dispatch(token_selected_experts, input_payloads, runtime_max_tokens_per_rank)
        # -> the recv payload(s) on this rank (the tokens routed to this rank's local experts).
        # The recv may be a single Tensor or a list (one per input payload); normalize below.
        variants = [
            ((p.topk_idx_i32, [p.x], p.T), {}),
            ((p.topk_idx_i32, [p.x]), dict(runtime_max_tokens_per_rank=p.T)),
            ((p.topk_idx_i32, [p.x]), dict(runtime_max_tokens=p.T)),
            ((p.topk_idx, [p.x], p.T), {}),                       # int64 idx fallback
            ((p.topk_idx_i32, p.x, p.T), {}),                     # single-tensor payload fallback
        ]
        recv, idx = _call_variants("MoeAlltoAll.dispatch(...)", self.a2a.dispatch, variants)
        recv_payload = self._first_payload(recv)
        return types.SimpleNamespace(recv=recv, recv_payload=recv_payload,
                                     dispatch_variant=idx, combine_input=None)

    def _dispatch_trtllm(self, p):
        # Functional one-sided path. Prefer the explicit moe_a2a_dispatch; fall back to the
        # bundled trtllm_moe_alltoall if that's the only entry point. Both are tried defensively.
        moe_a2a_dispatch = getattr(fi_comm, "moe_a2a_dispatch", None)
        trtllm_a2a = getattr(fi_comm, "trtllm_moe_alltoall", None)
        if moe_a2a_dispatch is not None:
            variants = [
                ((self.workspace, p.topk_idx_i32, [p.x], p.T), {}),
                ((self.workspace, p.topk_idx_i32, [p.x]), dict(runtime_max_tokens_per_rank=p.T)),
                ((p.topk_idx_i32, [p.x], p.T), {}),
            ]
            recv, idx = _call_variants("moe_a2a_dispatch(...)", moe_a2a_dispatch, variants)
        elif trtllm_a2a is not None:
            variants = [
                ((self.workspace, p.topk_idx_i32, [p.x], p.T), {}),
                ((p.topk_idx_i32, [p.x], p.T), {}),
            ]
            recv, idx = _call_variants("trtllm_moe_alltoall(...)", trtllm_a2a, variants)
        else:
            raise _loud("trtllm dispatch lookup",
                        "neither flashinfer.comm.moe_a2a_dispatch nor trtllm_moe_alltoall found",
                        AttributeError("moe_a2a_dispatch/trtllm_moe_alltoall"))
        recv_payload = self._first_payload(recv)
        return types.SimpleNamespace(recv=recv, recv_payload=recv_payload,
                                     dispatch_variant=idx, combine_input=None)

    @staticmethod
    def _first_payload(recv):
        """dispatch may return a Tensor, a (payloads, meta) tuple, or a list of payloads.
        Return the first payload Tensor (the routed x on this rank) for recv_tokens/staging."""
        if torch.is_tensor(recv):
            return recv
        if isinstance(recv, (list, tuple)) and recv:
            head = recv[0]
            if torch.is_tensor(head):
                return head
            if isinstance(head, (list, tuple)) and head and torch.is_tensor(head[0]):
                return head[0]
        return recv  # leave as-is; recv_tokens guards with is_tensor

    def stage(self, p, h):
        # No expert compute (identity expert). bf16 recv is the "expert output" as-is; FlashInfer's
        # combine reads back from the SAME workspace the dispatch populated, so combine() is told
        # the payload is already in the workspace (payload_in_workspace=True) when supported. We
        # stash the recv payload as combine_input so combine() can pass it explicitly if the API
        # wants the tensor handed back. (fp8 would dequant here, like ep_deepep.py — see TODO.)
        h.combine_input = h.recv_payload
        return None

    def combine(self, p, h):
        if self.trtllm:
            return self._combine_trtllm(p, h)
        # MoeAlltoAll.combine(payload, runtime_max_tokens_per_rank, payload_in_workspace=False)
        # -> the per-source-token reduced result on this rank ([T, hidden] bf16). Because the
        # dispatch populated the symmetric workspace, the data is already there: try
        # payload_in_workspace=True first (no payload re-copy), then the explicit-payload forms.
        variants = [
            ((h.combine_input, p.T), dict(payload_in_workspace=True)),
            ((h.combine_input, p.T), dict(payload_in_workspace=False)),
            ((h.combine_input, p.T), {}),
            ((h.combine_input,), dict(runtime_max_tokens_per_rank=p.T)),
            ((h.combine_input,), dict(runtime_max_tokens_per_rank=p.T, payload_in_workspace=True)),
        ]
        combined, idx = _call_variants("MoeAlltoAll.combine(...)", self.a2a.combine, variants)
        h.combine_variant = idx
        return self._as_tensor(combined)

    def _combine_trtllm(self, p, h):
        moe_a2a_combine = getattr(fi_comm, "moe_a2a_combine", None)
        if moe_a2a_combine is None:
            raise _loud("trtllm combine lookup",
                        "flashinfer.comm.moe_a2a_combine not found",
                        AttributeError("moe_a2a_combine"))
        variants = [
            ((self.workspace, h.combine_input, p.T), dict(payload_in_workspace=True)),
            ((self.workspace, h.combine_input, p.T), {}),
            ((h.combine_input, p.T), dict(payload_in_workspace=True)),
            ((h.combine_input, p.T), {}),
        ]
        combined, idx = _call_variants("moe_a2a_combine(...)", moe_a2a_combine, variants)
        h.combine_variant = idx
        return self._as_tensor(combined)

    @staticmethod
    def _as_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]):
            return x[0]
        raise _loud("combine result", f"expected a Tensor, got {type(x)}",
                    TypeError("non-tensor combine result"))

    def expected(self, p, h):
        # Round trip with identity expert: combine reduces the top_k copies of each SOURCE
        # token's x. See the module docstring for the full reasoning.
        #   _ROUTING_FACTOR == "topk"       -> combined ≈ x * top_k  (LEAD: combine does NOT weight)
        #   _ROUTING_FACTOR == "weight-sum" -> combined ≈ x * sum(topk_weights)  (combine weights)
        # The harness gate compares combined[:T] to this over the full [T, hidden] slice.
        ref = p.x.float()
        if _ROUTING_FACTOR == "weight-sum":
            factor = p.topk_weights.sum(dim=1, keepdim=True)        # [T, 1]
        else:  # "topk"
            factor = float(self.top_k)
        return ref * factor, p.T

    def recv_tokens(self, h):
        # Realized token-copies received on this rank (the routed payload's first dim). FlashInfer
        # pads to max_num_tokens-per-source-rank; the row count is the realistic recv-buffer size
        # the harness reports (it does NOT gate on this — recv_total>0 is the only liveness check).
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
