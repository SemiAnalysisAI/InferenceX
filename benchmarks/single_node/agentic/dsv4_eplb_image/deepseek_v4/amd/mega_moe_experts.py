# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MegaMoE expert layer for DeepSeek V4.

The AMD counterpart to ``nvidia/model.py``'s ``DeepseekV4MegaMoEExperts``. The
kernel is aiter's ``MegaMoEV2`` (FlyDSL), which fuses EP dispatch, both expert
GEMMs, and EP combine into a single launch chain. Unlike the CUDA path there is
no separate input-staging kernel: ``MegaMoEV2.forward`` quantizes the bf16
hidden states to fp8 with per-1x32 e8m0 scales itself.

Weight layout matches the existing ROCm MXFP4 fused-MoE path -- the packed fp4
weights and their e8m0 scales are run through aiter's ``shuffle_weight_a16w4``
and ``shuffle_scale_a16w4``, exactly as ``fused_moe/oracle/mxfp4.py`` does for
the CK kernels, so no new checkpoint handling is required.
"""

from __future__ import annotations

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.amd.mega_moe_runtime import (
    MegaMoERuntime,
    MegaMoEShape,
    estimate_symmetric_bytes,
    resolve_max_tok_per_rank,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

# mori's static symmetric heap defaults to 4 GiB and does NOT grow on demand.
# The mega path overruns that whenever the scheduler budget pushes
# max_tok_per_rank to 16384: FlyDSLDispatchGroupMajorOp's rx_em alone is
# num_valid_max * model_dim = 5.29 GiB, and the full footprint is ~7.7 GiB.
# The symptom is an AssertionError ("mori_shmem.shmem_malloc failed") from
# shmem_malloc during weight finalization, four minutes into a run and before
# the server ever binds.
#
# Sizing this belongs here rather than in a launcher env var: this is the only
# place that knows the MegaMoEShape, the heap is reserved VRAM (a blanket
# value would tax arms that do not need it), and mori reads the variable
# lazily at shmem_torch_process_group_init -- verified -- so setting it from
# Python before the collective init takes effect.
_HEAP_HEADROOM = 1.25
_HEAP_GRANULARITY = 2**30  # round up to a whole GiB; mori aligns internally


def _ensure_symmetric_heap_size(shape: MegaMoEShape) -> None:
    """Raise MORI_SHMEM_HEAP_SIZE to fit ``shape`` unless already set.

    Never lowers an explicit value: an operator who exported a larger heap for
    another consumer of the same heap (a mori all2all backend, say) must win.
    """
    if os.environ.get("MORI_SHMEM_HEAP_SIZE"):
        return

    want = int(estimate_symmetric_bytes(shape) * _HEAP_HEADROOM)
    want = -(-want // _HEAP_GRANULARITY) * _HEAP_GRANULARITY
    os.environ["MORI_SHMEM_HEAP_SIZE"] = str(want)
    logger.info_once(
        "MegaMoEV2: sized the mori symmetric heap at %.1f GiB for this shape "
        "(estimate %.2f GiB x %.2f headroom); mori's own default is 4 GiB and "
        "does not grow on demand",
        want / 2**30,
        estimate_symmetric_bytes(shape) / 2**30,
        _HEAP_HEADROOM,
    )


def shuffle_mega_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert checkpoint-layout fp4 expert weights into aiter's kernel layout.

    Split out of :meth:`DeepseekV4MegaMoEExperts.finalize_weights` so it can be
    tested against the reference preparation in aiter's own
    ``test_mega_moe_v2.py`` without standing up a distributed environment.

    Inputs are in checkpoint layout -- ``w13`` is ``(E, 2*inter, hidden//2)``
    packed two fp4 per byte with w1 occupying rows ``[0, inter)`` and w3 rows
    ``[inter, 2*inter)``; scales are per-1x32 e8m0 bytes. ``shuffle_scale``
    asserts a 2-D input and treats the leading axis as ``E * N``, so the scales
    are flattened before the call.
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    experts = w13.shape[0]

    # The shuffles are layout transforms driven by shape; the fp4 pair packing
    # only has to survive as opaque bytes.
    w13_packed = w13.view(torch.float4_e2m1fn_x2)
    w2_packed = w2.view(torch.float4_e2m1fn_x2)

    return (
        shuffle_weight_a16w4(w13_packed, 16, True).contiguous(),
        shuffle_scale_a16w4(
            w13_scale.reshape(-1, w13_scale.shape[-1]), experts, True
        ).contiguous(),
        shuffle_weight_a16w4(w2_packed, 16, False).contiguous(),
        shuffle_scale_a16w4(
            w2_scale.reshape(-1, w2_scale.shape[-1]), experts, False
        ).contiguous(),
    )


def make_deepseek_v4_mega_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Checkpoint -> parameter mapping for the mega-MoE expert layout.

    ``fused_moe_make_expert_params_mapping`` targets FusedMoE's parameter
    names, which the mega path does not have. This is the AMD copy of
    ``nvidia/model.py``'s ``make_deepseek_v4_expert_params_mapping``: w1/w3 land
    in the fused ``w13_`` parameters, w2 in ``w2_``, and the per-expert loader
    on :class:`DeepseekV4MegaMoEExperts` does the sharding.
    """
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in (("w1", "w1"), ("w2", "w2"), ("w3", "w3"))
    ]


def finalize_mega_moe_layers(layers) -> None:
    """Run mega weight finalization over a layer container.

    Tolerates the three shapes the DeepSeek V4 code uses -- a list of decoder
    layers, a ``ModuleDict`` of MTP layers, and MTP layers that wrap the real
    decoder layer in ``.mtp_block`` -- and is a no-op when the mega path is off.
    """
    if hasattr(layers, "values"):
        layers = layers.values()
    for layer in layers:
        layer = getattr(layer, "mtp_block", layer)
        ffn = getattr(layer, "ffn", None)
        finalize = getattr(ffn, "finalize_mega_moe_weights", None)
        if finalize is not None:
            finalize()


class DeepseekV4MegaMoEExperts(nn.Module):
    """Routed experts backed by aiter's fused MegaMoEV2 kernel."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        prefix: str = "",
        num_logical_experts: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit

        # ``num_experts`` counts *physical* slots; under EPLB that exceeds the
        # number of logical experts in the checkpoint by num_redundant_experts.
        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )

        # Populated by EplbState.add_model via the model's set_eplb_state, and
        # then by EplbState._propagate_shared_tensors -- which only recognises
        # a real EplbLayerState instance, so this must not be a plain object.
        # Stays all-None when EPLB is off, which is what forward() tests.
        self.eplb_state = EplbLayerState()

        ep_group = get_ep_group()
        self.ep_size = ep_group.world_size
        self.ep_rank = ep_group.rank_in_group

        # MegaMoEV2 rejects any forward pass carrying more tokens than
        # max_tok_per_rank, so this must round the scheduler budget *up* to the
        # next power of two. It is also what sizes the symmetric buffers.
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_tok_per_rank = resolve_max_tok_per_rank(self.max_num_tokens)

        weight_attrs = {"weight_loader": self.weight_loader}

        # Loader-side layout, matching the checkpoint: fp4 packed two-per-byte
        # with per-1x32 e8m0 scales. finalize_weights() replaces these with the
        # shuffled layout the kernel wants.
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._shuffled: tuple[torch.Tensor, ...] | None = None

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # --- weight loading ---------------------------------------------------

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Local slot offsets where logical expert ``expert_id`` should land.

        The initial EPLB arrangement is ``[0..L) + [i % L for i in range(R)]``
        (``EplbState.build_initial_global_physical_to_logical_map``), i.e.
        physical slot ``p`` holds logical expert ``p % L``. Without redundancy
        this returns at most one slot and matches the old single-slot loader
        exactly; with redundancy a replicated logical expert can land in two
        slots on this rank, so the caller must not stop at the first hit.
        """
        return [
            p - self.experts_start_idx
            for p in range(self.experts_start_idx, self.experts_end_idx)
            if p % self.num_logical_experts == expert_id
        ]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if not local_expert_ids:
            return False if return_success else None

        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_offset = 0 if shard_id == "w1" else self.intermediate_size
                expert_data = expert_data.narrow(
                    0, shard_offset, self.intermediate_size
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                    f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                    f"vs checkpoint {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight.view(expert_data.dtype))
            loaded_any = True

        return loaded_any if return_success else None

    def finalize_weights(self) -> None:
        """Shuffle the loaded weights into aiter's a16w4 kernel layout.

        Idempotent: safe to call again after dummy-weight loading, mirroring
        the CUDA path.
        """
        if self._shuffled is not None:
            return

        (
            w13_shuffled,
            w13_scale_shuffled,
            w2_shuffled,
            w2_scale_shuffled,
        ) = shuffle_mega_moe_weights(
            self.w13_weight.data,
            self.w13_weight_scale.data,
            self.w2_weight.data,
            self.w2_weight_scale.data,
        )

        self._shuffled = (
            w13_shuffled,
            w13_scale_shuffled,
            w2_shuffled,
            w2_scale_shuffled,
        )

        # Build the shared instance here rather than lazily in forward().
        # MegaMoEV2.__init__ allocates the symmetric buffers and ends in
        # shmem_barrier_all(), so it is collective and host-blocking: every EP
        # rank must reach it. Weight finalization runs on all ranks at model
        # load, whereas a forward pass may legitimately skip this layer on a
        # rank that has no tokens under DP attention -- which would deadlock.
        self._runtime().build(
            w1=w13_shuffled,
            w1_scale=w13_scale_shuffled,
            w2=w2_shuffled,
            w2_scale=w2_scale_shuffled,
        )

        # Drop the loader-side parameters; the shuffle helpers return fresh
        # tensors, so the original storage is no longer referenced.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        torch.cuda.empty_cache()

    # --- EPLB -------------------------------------------------------------

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        """The tensors EPLB rearranges, as ``(num_local_experts, -1)`` rows.

        Two constraints shape this:

        * The rows must alias the storage the kernel reads, because
          ``rebalance_execute`` mutates them in place with
          ``w[dst].copy_(buffer[dst])``. ``bind_weights`` re-reads
          ``self._shuffled`` on every forward, so a swap takes effect
          immediately with no rebuild.
        * The dtype must be one NCCL accepts. ``float4_e2m1fn_x2`` is not in
          ``ncclDataTypeEnum.supports_torch_dtype``, and
          ``create_eplb_communicator`` raises on unsupported dtypes -- so hand
          out uint8 views. The packed fp4 only has to survive as opaque bytes,
          which a whole-expert row copy guarantees.

        Row-swapping the *shuffled* layout is only valid because
        ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` are per-expert
        independent: shuffling one expert alone is byte-identical to the
        corresponding slice of shuffling all of them. Verified at the DSv4
        shape; see ``test_mega_moe_shuffle_invariance.py``.
        """
        self.finalize_weights()
        assert self._shuffled is not None

        def _to_eplb_view(name: str, t: torch.Tensor) -> torch.Tensor:
            u = t.view(torch.uint8)
            if not u.is_contiguous():
                raise AssertionError(
                    f"DSv4 mega EPLB {name}: non-contiguous expert tensor "
                    f"shape={tuple(u.shape)} stride={tuple(u.stride())}"
                )
            # The weights are (E, ...) but the shuffled scales come back 2-D as
            # (E*N, K), so reshape on the flat element count rather than
            # assuming a leading expert axis.
            if u.numel() % self.num_local_experts != 0:
                raise AssertionError(
                    f"DSv4 mega EPLB {name}: {u.numel()} elements do not divide "
                    f"across {self.num_local_experts} local experts"
                )
            return u.reshape(self.num_local_experts, -1)

        w13, w13_scale, w2, w2_scale = self._shuffled
        return [
            _to_eplb_view("w13", w13),
            _to_eplb_view("w13_scale", w13_scale),
            _to_eplb_view("w2", w2),
            _to_eplb_view("w2_scale", w2_scale),
        ]

    def update_expert_map(self) -> None:
        # No expert_map to update: this path derives the local shard from
        # experts_start_idx at load time and the physical ids the kernel sees
        # come straight from the EPLB remap in forward(). Present because
        # DeepseekV4MixtureOfExperts.update_physical_experts_metadata calls it
        # on every layer during an elastic-EP resize.
        pass

    # --- runtime ----------------------------------------------------------

    def _runtime(self) -> MegaMoERuntime:
        shape = MegaMoEShape(
            world_size=self.ep_size,
            model_dim=self.hidden_size,
            inter_dim=self.intermediate_size,
            experts=self.num_experts,
            topk=self.top_k,
            max_tok_per_rank=self.max_tok_per_rank,
            swiglu_limit=self.swiglu_limit,
        )
        # Before MegaMoERuntime.get, so the size is in the environment ahead of
        # build()'s shmem_torch_process_group_init. Idempotent, and a no-op on
        # the forward() call path where the heap is already up.
        _ensure_symmetric_heap_size(shape)
        return MegaMoERuntime.get(shape, self.ep_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tok_per_rank:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {num_tokens} tokens, but the "
                f"symmetric buffers were sized for {self.max_tok_per_rank}."
            )

        # Covers the dummy-weight-loading path, as on CUDA.
        self.finalize_weights()
        assert self._shuffled is not None
        w13, w13_scale, w2, w2_scale = self._shuffled

        runtime = self._runtime()

        # EPLB: rewrite logical expert ids to the physical replica this token
        # should use, and accumulate the per-physical-expert load counters the
        # rebalancer reads. No-op when EPLB is off (the layer state stays
        # all-None), and fused into one Triton kernel when it is on.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None

            # Padded tokens carry garbage routing. -1 makes the remap kernel
            # treat them as invalid rather than crediting load to a real
            # expert; the dispatch kernel already drops negative ids.
            is_padding = None
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                is_padding = get_forward_context().is_padding
                if is_padding is not None:
                    is_padding = is_padding[:num_tokens]
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)

            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=(
                    eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
                    if eplb_state.num_unpadded_tokens_tensors is not None
                    else None
                ),
            )

        # The kernel's contracts: contiguous bf16 activations, float32 routing
        # weights, int32 expert ids.
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wts = topk_weights.to(torch.float32).contiguous()
        ids = topk_ids.to(torch.int32).contiguous()

        with runtime.bind_weights(
            self, w1=w13, w1_scale=w13_scale, w2=w2, w2_scale=w2_scale
        ) as moe:
            out = moe.forward(x, wts, ids)

        # The kernel returns a view into the shared symmetric combine buffer,
        # which the next layer's call will overwrite. Copy out before returning.
        return out.clone()


DeepseekV4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
