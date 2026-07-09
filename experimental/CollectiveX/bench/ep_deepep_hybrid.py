#!/usr/bin/env python3
"""CollectiveX EP backend adapter — DeepEP `hybrid-ep` branch (NVIDIA TMA-based HybridEPBuffer).

The hybrid-ep branch (https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) is NVIDIA's TMA +
warp-pipeline implementation of expert-parallel all-to-all, exposing `deep_ep.HybridEPBuffer`
(distinct from the mainline `deep_ep.Buffer`). HybridEP is NVIDIA's MoE backend built for NVL72
rack-scale (Megatron `moe_flex_dispatcher_backend="hybridep"`). This adapter binds the API's
"ranks per node" field to active ranks per NVLink/MNNVL communication domain, not physical host
GPUs: x86 EP16 is two 8-rank domains, while GB EP8/EP16 is one 8/16-rank MNNVL domain across hosts.
The container build is done by runtime/run_in_container.sh `cx_build_deepep_hybrid` (CUDA-13 CCCL
include path, without the V2 NVSHMEM overlay).

API (pinned on B300, branch e0a5b1d):
  HybridEPBuffer(group, hidden_dim, max_num_of_tokens_per_rank, num_local_experts, use_fp8=False, ...)
  .dispatch(hidden, topk_idx=, topk_weights=, num_of_experts=) -> (recv_hidden, recv_x2, None, handle)
  .combine(hidden, handle=) -> [T, hidden]

CORRECTNESS: identity expert (no expert compute), combine WITHOUT probs -> each source token is
reconstructed as x * (distinct ranks among its top_k experts) — verified: an 8-rank uniform top_k=8
round trip gives relerr(combined, x) = 4.28, matching E[distinct ranks] ~ 5.26 exactly. So this uses
the same per-rank-sum combine contract (no gate re-weight). BF16 tolerance is 5e-2.

STATUS: BF16 or native block-scaled FP8 dispatch, BF16 combine, normal mode. The v1 scope covers
one MNNVL domain or x86 scale-out between two eight-GPU NVLink domains.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types

import torch
import torch.distributed as dist
import ep_provenance
from ep_backend import EPBackend

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


def _validate_hybrid_build() -> None:
    for module_name in ("deep_ep_cpp", "hybrid_ep_cpp"):
        module = importlib.import_module(module_name)
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            raise RuntimeError(f"{module_name} has no loaded extension path")


HYBRID_CONFIG_FIELDS = (
    "hidden_dim", "max_num_of_tokens_per_rank", "num_of_experts_per_rank",
    "num_of_ranks_per_node", "num_of_nodes", "pad_multiple",
    "num_of_tokens_per_chunk_preprocessing_api",
    "num_of_threads_per_block_preprocessing_api", "num_of_blocks_preprocessing_api",
    "num_of_blocks_permute", "num_of_blocks_unpermute", "token_data_type",
    "num_of_stages_dispatch_api", "num_of_stages_permute_block_dispatch_api",
    "num_of_in_flight_s2g_dispatch_api",
    "num_of_in_flight_s2g_permute_block_dispatch_api",
    "num_of_additional_in_flight_s2g_dispatch_api",
    "num_of_tokens_per_chunk_dispatch_api", "num_of_blocks_dispatch_api",
    "forward_dispatch_api", "device_side_sync_dispatch_api",
    "num_of_stages_g2s_combine_api", "num_of_stages_s2g_combine_api",
    "num_of_tokens_per_chunk_combine_api", "num_of_tokens_per_group_combine_api",
    "num_of_blocks_combine_api", "num_of_additional_in_flight_s2g_combine_api",
    "backward_combine_api", "device_side_sync_combine_api",
)


def _hybrid_realized_config(config) -> dict[str, str | int | bool]:
    """Project the Python-visible, post-autotune HybridEP config to JSON scalars."""
    realized = {}
    for field in HYBRID_CONFIG_FIELDS:
        try:
            value = getattr(config, field)
        except AttributeError as exc:
            raise RuntimeError(f"HybridEP realized config omits {field}") from exc
        if field == "token_data_type":
            token_type = getattr(value, "name", None)
            if token_type not in {"UINT8", "UINT16"}:
                token_type = {"uint8_t": "UINT8", "uint16_t": "UINT16"}.get(str(value))
            if token_type is None:
                raise RuntimeError("HybridEP realized token_data_type is invalid")
            realized[field] = token_type
            continue
        if type(value) is bool:
            realized[field] = value
            continue
        try:
            realized[field] = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"HybridEP realized config {field} is not integral") from exc
    return realized


def _hybrid_jit_kernel_keys(root: Path) -> list[str]:
    if not root.is_dir():
        raise RuntimeError("DeepEP Hybrid produced no JIT cache directory")
    kernel_keys = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.suffix != ".so":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("DeepEP Hybrid JIT artifact is not a regular file")
        kernel_key = path.stem
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,511}", kernel_key):
            raise RuntimeError("DeepEP Hybrid JIT kernel key is invalid")
        if path.stat().st_size <= 0:
            raise RuntimeError("DeepEP Hybrid JIT artifact is empty")
        kernel_keys.append(kernel_key)
    if len(kernel_keys) != 3:
        raise RuntimeError(
            f"DeepEP Hybrid expected 3 final JIT libraries, found {len(kernel_keys)}"
        )
    return kernel_keys


def _require_cross_rank_equal(value, label: str) -> None:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    canonical = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in gathered}
    if len(canonical) != 1:
        raise RuntimeError(f"DeepEP Hybrid {label} differs across ranks")


def _hybrid_topology(args, world_size: int) -> dict[str, int | str]:
    """Translate physical placement into HybridEP communication-domain geometry."""
    gpus_per_node = int(args.gpus_per_node or world_size)
    scale_up_domain = int(args.scale_up_domain or gpus_per_node)
    key = (
        world_size, gpus_per_node, scale_up_domain, args.scope,
        args.scale_up_transport, args.scale_out_transport or None, args.transport,
    )
    fixed = {
        (8, 8, 8, "scale-up", "nvlink", None, "nvlink"): (8, 1),
        (16, 8, 8, "scale-out", "nvlink", "rdma", "nvlink-rdma"): (8, 2),
        (8, 4, 72, "scale-up", "mnnvl", None, "mnnvl"): (8, 1),
        (16, 4, 72, "scale-up", "mnnvl", None, "mnnvl"): (16, 1),
    }
    if key not in fixed:
        raise RuntimeError("DeepEP Hybrid topology is outside the fixed v1 matrix")
    domain_ranks, communication_domains = fixed[key]

    return {
        "communication_domains": communication_domains,
        "domain_ranks": domain_ranks,
        "physical_nodes": world_size // gpus_per_node,
        "transport": str(args.transport),
    }


class DeepEPHybridBackend(EPBackend):
    name = "deepep-hybrid"
    stage_device_work = False
    # HybridEPBuffer.combine consumes the recv payload + the dispatch handle (no re-dispatch needed
    # before a timed combine); the harness times dispatch and combine separately (like ep_deepep).
    combine_needs_redispatch = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        # deepep-hybrid is normal-mode only; base SUPPORTED_MODES=("normal",) enforces it.
        super().__init__(args, rank, world_size, local_rank, device)
        ep_provenance.require_keyword(
            HybridEPBuffer.__init__,
            "use_fp8",
            api="deep_ep.HybridEPBuffer.__init__",
        )
        ep_provenance.require_keyword(
            HybridEPBuffer.dispatch,
            "scaling_factor",
            api="deep_ep.HybridEPBuffer.dispatch",
        )
        self.group = dist.group.WORLD
        self.tolerance = 5e-2
        self.top_k = int(args.topk)
        self.num_experts = int(args.experts)
        self.hidden = int(args.hidden)
        self.local_experts = max(1, self.num_experts // world_size)
        topology = _hybrid_topology(args, world_size)
        self.domain_ranks = int(topology["domain_ranks"])
        self.communication_domains = int(topology["communication_domains"])
        build_mode = os.environ.get("DEEPEP_HYBRID_BUILD_MODE", "")
        if self.communication_domains > 1:
            if (
                os.environ.get("HYBRID_EP_MULTINODE") != "1"
                or build_mode != "multinode-doca"
                or os.environ.get("USE_NIXL", "0") != "0"
            ):
                raise RuntimeError("DeepEP Hybrid scale-out build mode is not realized")
        elif build_mode != "intradomain":
            raise RuntimeError("DeepEP Hybrid scale-up requires the intradomain build")
        if args.scale_up_transport == "mnnvl" and any(
            os.environ.get(name) != "1"
            for name in ("NCCL_CUMEM_ENABLE", "NCCL_MNNVL_ENABLE", "MC_FORCE_MNNVL")
        ):
            raise RuntimeError("DeepEP Hybrid MNNVL runtime enablement is incomplete")
        # Token cap (per rank) for the symmetric buffer; the sweep is capped here (buffer_cap).
        self.max_tokens = 4096
        # Stash the topology so the moved create_buffer provenance can read it back.
        self._topology = topology

    def create_buffer(self, spec):
        # Local aliases re-expose the __init__ names so the moved tempdir / buffer /
        # geometry / monkeypatch / provenance body below stays byte-verbatim.
        args, world_size, rank, device = self.args, self.world_size, self.rank, self.device
        topology = self._topology
        assert spec.max_tokens_per_rank <= self.max_tokens
        dev_sms = torch.cuda.get_device_properties(device).multi_processor_count
        ver = _deepep_hybrid_version()
        _validate_hybrid_build()

        # HybridEP's compiler uses a process-specific child of HYBRID_EP_CACHE_DIR. Give every
        # rank a fresh private base so stale kernels cannot enter this attempt's evidence.
        self._previous_jit_cache_dir = os.environ.get("HYBRID_EP_CACHE_DIR")
        self._previous_domain_ranks = os.environ.get(
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
        )
        self._jit_cache_dir = tempfile.mkdtemp(prefix=f"collectivex-hybrid-r{rank}-")
        os.environ["HYBRID_EP_CACHE_DIR"] = self._jit_cache_dir
        os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(self.domain_ranks)
        self._jit_root = (
            Path(self._jit_cache_dir) / ".deepep" / "hybrid_ep" / "jit"
            / f"proc-{os.getpid()}"
        )
        self._realized_config = None
        self._deferred_semantic_snapshot = None

        try:
            self.buffer = HybridEPBuffer(
                self.group, hidden_dim=self.hidden,
                max_num_of_tokens_per_rank=self.max_tokens,
                num_local_experts=self.local_experts,
                use_fp8=False,  # BF16 communication path.
            )
            realized_geometry = (
                int(self.buffer.num_of_hybrid_ep_ranks_per_nvlink_domain),
                int(self.buffer.num_of_nodes),
                int(self.buffer.local_rank),
                int(self.buffer.node_rank),
            )
            expected_geometry = (
                self.domain_ranks,
                self.communication_domains,
                rank % self.domain_ranks,
                rank // self.domain_ranks,
            )
            buffer_config = self.buffer.configurer.buffer_config
            if realized_geometry != expected_geometry or (
                int(buffer_config.num_of_ranks_per_node) != self.domain_ranks
                or int(buffer_config.num_of_nodes) != self.communication_domains
            ):
                raise RuntimeError(
                    "HybridEPBuffer communication-domain geometry differs from the case"
                )
        except Exception as exc:
            shutil.rmtree(self._jit_cache_dir, ignore_errors=True)
            if self._previous_jit_cache_dir is None:
                os.environ.pop("HYBRID_EP_CACHE_DIR", None)
            else:
                os.environ["HYBRID_EP_CACHE_DIR"] = self._previous_jit_cache_dir
            if self._previous_domain_ranks is None:
                os.environ.pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
            else:
                os.environ[
                    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
                ] = self._previous_domain_ranks
            raise RuntimeError(
                f"HybridEPBuffer construction failed (hidden={self.hidden} max_tokens={self.max_tokens} "
                f"local_experts={self.local_experts} world={world_size}): {exc!r}") from exc
        update_template_config = self.buffer.update_template_config

        def tracked_update_template_config(*call_args, **call_kwargs):
            config = update_template_config(*call_args, **call_kwargs)
            realized = _hybrid_realized_config(config)
            if (
                realized["num_of_ranks_per_node"] != self.domain_ranks
                or realized["num_of_nodes"] != self.communication_domains
            ):
                raise RuntimeError("DeepEP Hybrid realized topology changed within one case")
            # BF16 dispatch realizes the 2-byte UINT16 token wire type.
            if realized["token_data_type"] != "UINT16":
                raise RuntimeError(
                    "DeepEP Hybrid realized token dtype is not the BF16 UINT16 wire type"
                )
            if self._realized_config is not None and realized != self._realized_config:
                raise RuntimeError("DeepEP Hybrid realized autotune config changed within one case")
            self._realized_config = realized
            return config

        self.buffer.update_template_config = tracked_update_template_config
        self.domain_rank = int(self.buffer.local_rank)
        if rank == 0:
            print(
                "[deepep-hybrid] HybridEPBuffer constructed "
                f"(domains={self.communication_domains} ranks_per_domain={self.domain_ranks} "
                f"world={world_size} local_experts={self.local_experts} hidden={self.hidden})",
                file=sys.stderr,
            )

        self.backend_provenance = {
            "deepep_commit": ver, "branch": "hybrid-ep",
            "deepep_tree": os.environ.get("DEEPEP_TREE"),
            "backend_lineage": "deepep-hybrid",
            "impl": "deep_ep.HybridEPBuffer (NVIDIA TMA + warp-pipeline)",
            "mode": "normal", "transport": topology["transport"],
            "dispatch_dtype": "bf16",
            "combine_dtype": "bf16",
            "resource_mode": "fixed-profile",
            "num_sms": None, "device_sms": dev_sms,
            "tuned_source": "deepep-hybrid-configurer-autotune-v1",
            "realized_config": None, "jit_kernel_keys": [],
            "max_num_tokens": self.max_tokens, "top_k": self.top_k,
            "num_experts": self.num_experts, "local_experts": self.local_experts,
            "routing_factor": "ranks",
        }

    def buffer_cap(self, args):
        return self.max_tokens

    def make_problem(self, T, idx, weights, x):
        # BF16 dispatch sends x directly; scaling_factor stays None (no separate scale payload).
        return types.SimpleNamespace(
            T=int(T),
            x=x,
            dispatch_x=x,
            dispatch_scales=None,
            topk_idx=idx.to(torch.int64),
            topk_weights=weights.to(torch.float32),
        )

    def dispatch(self, p):
        recv, recv_probs, _scales, handle = self.buffer.dispatch(
            p.dispatch_x,
            scaling_factor=p.dispatch_scales,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_of_experts=self.num_experts,
        )
        return types.SimpleNamespace(
            recv=recv,
            recv_payload=recv,
            recv_probs=recv_probs,
            handle=handle,
            combine_input=None,
        )

    def stage(self, p, h):
        # Identity expert: the recv hidden IS the "expert output". combine reduces it per source token.
        h.combine_input = h.recv_payload
        return None

    def combine(self, p, h):
        # combine(hidden, handle=) -> [T, H] per-source-token reduction (no gate re-weight: "ranks").
        comb = self.buffer.combine(h.combine_input, handle=h.handle)
        return comb[0] if isinstance(comb, (tuple, list)) else comb

    def capture_deferred_provenance(self):
        torch.cuda.synchronize()
        dist.barrier()
        if self._realized_config is None:
            raise RuntimeError("DeepEP Hybrid autotune config was not materialized")
        kernel_keys = _hybrid_jit_kernel_keys(self._jit_root)
        semantic = {
            "jit_kernel_keys": kernel_keys,
            "realized_config": dict(self._realized_config),
        }
        _require_cross_rank_equal(semantic, "realized config/JIT kernel keys")
        if self._deferred_semantic_snapshot is not None and semantic != self._deferred_semantic_snapshot:
            raise RuntimeError("DeepEP Hybrid config/JIT kernel set changed after measurement")
        self._deferred_semantic_snapshot = semantic
        self.backend_provenance.update(semantic)

    def inspect_dispatch(self, p, h):
        count = self.recv_tokens(h)
        routing_map = h.handle[4][:count]
        rows, local_expert_ids = routing_map.nonzero(as_tuple=True)
        positions = routing_map.to(torch.int64).cumsum(dim=1)[rows, local_expert_ids] - 1
        probability_columns = self.domain_rank * self.local_experts + local_expert_ids
        if h.recv_probs.shape[1] < (self.domain_rank + 1) * self.local_experts:
            raise RuntimeError("HybridEPBuffer probability tensor omits this NVLink-domain rank")
        expert_ids = torch.full(
            (count, self.top_k), -1, dtype=torch.int64, device=self.device
        )
        weights = torch.zeros(
            (count, self.top_k), dtype=torch.float32, device=self.device
        )
        expert_ids[rows, positions] = local_expert_ids + self.rank * self.local_experts
        weights[rows, positions] = h.recv_probs[:count][rows, probability_columns]
        return types.SimpleNamespace(
            payload=h.recv_payload[:count],
            expert_ids=expert_ids,
            weights=weights,
            local_expert_counts=routing_map.sum(dim=0, dtype=torch.int64),
            ordering_contract="global-source-filter-stable-v1",
        )

    def combine_transformed(self, p, h, transformed):
        combined = self.buffer.combine(
            transformed.to(torch.bfloat16), handle=h.handle
        )
        return combined[0] if isinstance(combined, (tuple, list)) else combined

    def recv_tokens(self, h):
        return int(h.handle[3].item())

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        # create_buffer may not have run (e.g. make_inputs early-returned before it),
        # so the JIT tempdir/env state only needs unwinding when it was established.
        if hasattr(self, "_jit_cache_dir"):
            shutil.rmtree(self._jit_cache_dir, ignore_errors=True)
            if self._previous_jit_cache_dir is None:
                os.environ.pop("HYBRID_EP_CACHE_DIR", None)
            else:
                os.environ["HYBRID_EP_CACHE_DIR"] = self._previous_jit_cache_dir
            if self._previous_domain_ranks is None:
                os.environ.pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
            else:
                os.environ[
                    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
                ] = self._previous_domain_ranks
        return rc
