#!/usr/bin/env bash
# Add pipeline parallelism to the Muse Spark decoder in the dev-muse-spark-open
# image, whose model builds every layer on every rank. Each stage builds only
# its own layers; the fp32 residual stream crosses the stage boundary as a
# bf16 hi/lo pair so it fits the hidden_states/residual buffers the PP runtime
# and decode CUDA graphs already allocate.
set -euo pipefail
SGLANG_SRT=${SGLANG_SRT:-/sgl-workspace/sglang-muse-spark/python/sglang/srt}
python3 - "$SGLANG_SRT" <<'PY'
import sys
from pathlib import Path

srt = Path(sys.argv[1])
MARK = "# musespark-pp patch applied"


def patch(path, edits):
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: already patched")
        return
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            sys.exit(f"{path.name}: expected 1 match, found {count}:\n{old}")
        text = text.replace(old, new)
    path.write_text(f"{MARK}\n{text}")
    print(f"{path.name}: patched")


patch(
    srt / "models/muse_spark.py",
    [
        (
            "from sglang.srt.model_executor.forward_batch_info import ForwardBatch\n",
            "from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors\n"
            "from sglang.srt.distributed.utils import get_pp_indices\n"
            "from sglang.srt.layers.utils import PPMissingLayer\n"
            "import re as _re\n"
            "_PP_LAYER_RE = _re.compile(r\"\\.layers\\.(\\d+)\\.\")\n"
            "\n"
            "\n"
            "def _resolve_pp_dflash_layer_ids():\n"
            "    # Under PP only the last stage builds the DFLASH worker, so earlier\n"
            "    # stages read the draft's target layers to know what to capture.\n"
            "    import json as _json, os as _os\n"
            "    override = _os.environ.get(\"SGLANG_MUSE_PP_DFLASH_TARGET_LAYERS\")\n"
            "    if override:\n"
            "        return [int(v) for v in override.split(\",\") if v.strip()]\n"
            "    from sglang.srt.runtime_context import get_spec as _get_spec\n"
            "    spec = _get_spec()\n"
            "    algo = str(spec.speculative_algorithm or \"\").upper()\n"
            "    if \"DFLASH\" not in algo or not spec.speculative_draft_model_path:\n"
            "        return []\n"
            "    path = spec.speculative_draft_model_path\n"
            "    if _os.path.isdir(path):\n"
            "        cfg_path = _os.path.join(path, \"config.json\")\n"
            "    else:\n"
            "        from huggingface_hub import hf_hub_download\n"
            "        cfg_path = hf_hub_download(path, \"config.json\")\n"
            "    with open(cfg_path) as f:\n"
            "        cfg = _json.load(f)\n"
            "    ids = (cfg.get(\"dflash_config\") or {}).get(\"target_layer_ids\") or cfg.get(\n"
            "        \"target_layer_ids\"\n"
            "    )\n"
            "    if not ids:\n"
            "        raise ValueError(f\"DFLASH draft {path} declares no target_layer_ids\")\n"
            "    return [int(v) for v in ids]\n",
        ),
        # Build only this stage's layers.
        (
            "        for layer_idx in range(config.num_hidden_layers):\n"
            "            layer_prefix = add_prefix(f\"layers.{layer_idx}\", prefix)\n",
            "        self.pp_group = get_parallel().pp_group\n"
            "        self.start_layer, self.end_layer = get_pp_indices(\n"
            "            config.num_hidden_layers,\n"
            "            self.pp_group.rank_in_group,\n"
            "            self.pp_group.world_size,\n"
            "        )\n"
            "        for layer_idx in range(config.num_hidden_layers):\n"
            "            if not self.start_layer <= layer_idx < self.end_layer:\n"
            "                self.layers.append(PPMissingLayer())\n"
            "                continue\n"
            "            layer_prefix = add_prefix(f\"layers.{layer_idx}\", prefix)\n",
        ),
        # Fuse boundaries only between layers on this stage; the stage's last
        # layer takes the unfused tail, as the model's last layer already does.
        (
            "        for layer, next_layer in zip(self.layers[:-1], self.layers[1:]):\n",
            "        local_layers = list(self.layers)[self.start_layer : self.end_layer]\n"
            "        for layer, next_layer in zip(local_layers[:-1], local_layers[1:]):\n",
        ),
        (
            "                moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]\n"
            "                comm = tp_group.ca_comm\n",
            "                moe_layers = [layer.mlp for layer in local_layers if layer.is_moe_layer]\n"
            "                comm = tp_group.ca_comm\n",
        ),
        (
            "            for layer in self.layers:\n"
            "                linears = [layer.self_attn.qkv_proj, layer.self_attn.o_proj]\n",
            "            for layer in local_layers:\n"
            "                linears = [layer.self_attn.qkv_proj, layer.self_attn.o_proj]\n",
        ),
        (
            "            moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]\n"
            "            if moe_layers and not getattr(\n",
            "            moe_layers = [layer.mlp for layer in local_layers if layer.is_moe_layer]\n"
            "            if moe_layers and not getattr(\n",
        ),
        # Earlier stages capture their DFLASH taps for the last stage, which
        # alone holds the draft worker that would otherwise request them.
        (
            "        # Layer indices whose output residual stream feeds the DFLASH draft context.\n"
            "        self.layers_to_capture: List[int] = []\n",
            "        # Layer indices whose output residual stream feeds the DFLASH draft context.\n"
            "        self.layers_to_capture: List[int] = []\n"
            "        self.pp_dflash_layer_ids = (\n"
            "            _resolve_pp_dflash_layer_ids() if self.pp_group.world_size > 1 else []\n"
            "        )\n"
            "        if not self.pp_group.is_last_rank:\n"
            "            self.layers_to_capture = list(self.pp_dflash_layer_ids)\n",
        ),
        # Decoder forward: receive the stream on later stages, send it on
        # earlier ones.
        (
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "    ) -> torch.Tensor:\n"
            "        # Prefill CUDA graphs capture this decoder directly, below the causal\n"
            "        # LM's scope; open it here as well (it nests) so they keep the defaults.\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward(input_ids, positions, forward_batch, input_embeds)\n",
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "        pp_proxy_tensors: Optional[PPProxyTensors] = None,\n"
            "    ) -> torch.Tensor:\n"
            "        # Prefill CUDA graphs capture this decoder directly, below the causal\n"
            "        # LM's scope; open it here as well (it nests) so they keep the defaults.\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward(\n"
            "                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors\n"
            "            )\n",
        ),
        (
            "        input_embeds: Optional[torch.Tensor],\n"
            "    ) -> torch.Tensor:\n"
            "        hidden_states = (\n"
            "            self.embed_tokens(input_ids) if input_embeds is None else input_embeds\n"
            "        )\n"
            "        if input_embeds is None and self.embed_norm is not None:\n"
            "            hidden_states = self.embed_norm(hidden_states)\n"
            "        sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode)\n"
            "        get_forward().set(\"sp_active\", sp_active)\n"
            "        if sp_active:\n"
            "            hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)\n"
            "        stream = hidden_states.to(self.residual_dtype)\n"
            "        attn_input = self.layers[0].self_attn_input_norm(stream)\n"
            "        aux_hidden_states = []\n"
            "        for layer_idx, layer in enumerate(self.layers):\n"
            "            stream, attn_input = layer(positions, stream, attn_input, forward_batch)\n",
            "        input_embeds: Optional[torch.Tensor],\n"
            "        pp_proxy_tensors: Optional[PPProxyTensors] = None,\n"
            "    ) -> torch.Tensor:\n"
            "        sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode)\n"
            "        if self.pp_group.world_size > 1 and sp_active:\n"
            "            raise ValueError(\"Muse Spark pipeline parallelism does not support layernorm SP\")\n"
            "        get_forward().set(\"sp_active\", sp_active)\n"
            "        if self.pp_group.is_first_rank:\n"
            "            hidden_states = (\n"
            "                self.embed_tokens(input_ids) if input_embeds is None else input_embeds\n"
            "            )\n"
            "            if input_embeds is None and self.embed_norm is not None:\n"
            "                hidden_states = self.embed_norm(hidden_states)\n"
            "            if sp_active:\n"
            "                hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)\n"
            "            stream = hidden_states.to(self.residual_dtype)\n"
            "            aux_hidden_states = []\n"
            "        else:\n"
            "            # The fp32 stream arrives as its bf16 head plus the bf16 rounding error.\n"
            "            stream = pp_proxy_tensors[\"hidden_states\"].to(\n"
            "                self.residual_dtype\n"
            "            ) + pp_proxy_tensors[\"residual\"].to(self.residual_dtype)\n"
            "            # Earlier stages' DFLASH taps, packed in layer order.\n"
            "            relayed = pp_proxy_tensors.tensors.get(\"dspark_hidden_states\")\n"
            "            aux_hidden_states = (\n"
            "                list(relayed.split(self.config.hidden_size, dim=-1))\n"
            "                if self.layers_to_capture and relayed is not None\n"
            "                else []\n"
            "            )\n"
            "        attn_input = self.layers[self.start_layer].self_attn_input_norm(stream)\n"
            "        for layer_idx in range(self.start_layer, self.end_layer):\n"
            "            layer = self.layers[layer_idx]\n"
            "            stream, attn_input = layer(positions, stream, attn_input, forward_batch)\n",
        ),
        (
            "        norm_input = stream.to(self.operator_dtype)\n"
            "        hidden_states = self.norm(norm_input).to(self.operator_dtype)\n",
            "        if not self.pp_group.is_last_rank:\n"
            "            head = stream.to(self.operator_dtype)\n"
            "            tail = (stream - head.to(self.residual_dtype)).to(self.operator_dtype)\n"
            "            proxy = {\"hidden_states\": head, \"residual\": tail}\n"
            "            if aux_hidden_states:\n"
            "                proxy[\"dspark_hidden_states\"] = torch.cat(aux_hidden_states, dim=-1)\n"
            "            return PPProxyTensors(proxy)\n"
            "        norm_input = stream.to(self.operator_dtype)\n"
            "        hidden_states = self.norm(norm_input).to(self.operator_dtype)\n",
        ),
        # Wrapper module: thread the proxy and expose the stage's layer range.
        (
            "    def forward(self, input_ids, positions, forward_batch, input_embeds=None):\n"
            "        return self.language_model(\n"
            "            input_ids, positions, forward_batch, input_embeds=input_embeds\n"
            "        )\n",
            "    def forward(\n"
            "        self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None\n"
            "    ):\n"
            "        return self.language_model(\n"
            "            input_ids,\n"
            "            positions,\n"
            "            forward_batch,\n"
            "            input_embeds=input_embeds,\n"
            "            pp_proxy_tensors=pp_proxy_tensors,\n"
            "        )\n"
            "\n"
            "    @property\n"
            "    def start_layer(self):\n"
            "        return self.language_model.start_layer\n"
            "\n"
            "    @property\n"
            "    def end_layer(self):\n"
            "        return self.language_model.end_layer\n",
        ),
        # Causal LM: earlier stages return the proxy instead of logits.
        (
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "    ):\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward(input_ids, positions, forward_batch, input_embeds)\n",
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "        pp_proxy_tensors: Optional[PPProxyTensors] = None,\n"
            "    ):\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward(\n"
            "                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors\n"
            "            )\n"
            "\n"
            "    @property\n"
            "    def start_layer(self):\n"
            "        return self.model.language_model.start_layer\n"
            "\n"
            "    @property\n"
            "    def end_layer(self):\n"
            "        return self.model.language_model.end_layer\n"
            "\n"
            "    def get_pp_proxy_dspark_hidden_size(self) -> int:\n"
            "        # Width of the DFLASH taps earlier stages relay to this one.\n"
            "        lm = self.model.language_model\n"
            "        return self.config.hidden_size * sum(\n"
            "            layer < lm.start_layer for layer in lm.pp_dflash_layer_ids\n"
            "        )\n",
        ),
        (
            "        input_embeds: Optional[torch.Tensor],\n"
            "    ):\n"
            "        hidden_states = self.model(\n"
            "            input_ids, positions, forward_batch, input_embeds=input_embeds\n"
            "        )\n",
            "        input_embeds: Optional[torch.Tensor],\n"
            "        pp_proxy_tensors: Optional[PPProxyTensors] = None,\n"
            "    ):\n"
            "        hidden_states = self.model(\n"
            "            input_ids,\n"
            "            positions,\n"
            "            forward_batch,\n"
            "            input_embeds=input_embeds,\n"
            "            pp_proxy_tensors=pp_proxy_tensors,\n"
            "        )\n"
            "        if isinstance(hidden_states, PPProxyTensors):\n"
            "            return hidden_states\n",
        ),
        # Weights for other stages' layers have no parameter here.
        (
            "        for raw_name, loaded_weight in weights:\n"
            "            name = raw_name\n",
            "        start_layer, end_layer = self.start_layer, self.end_layer\n"
            "        for raw_name, loaded_weight in weights:\n"
            "            name = raw_name\n"
            "            layer_match = _PP_LAYER_RE.search(name)\n"
            "            if layer_match and not start_layer <= int(layer_match.group(1)) < end_layer:\n"
            "                continue\n",
        ),
        (
            "        for layer in self.model.language_model.layers:\n"
            "            for residual in (\n",
            "        for layer in self.model.language_model.layers:\n"
            "            if isinstance(layer, PPMissingLayer):\n"
            "                continue\n"
            "            for residual in (\n",
        ),
    ],
)

patch(
    srt / "models/muse_spark_omni.py",
    [
        (
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "    ):\n"
            "        # Covers the text embedding norm too; the decoder's own scope nests.\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward_omni(input_ids, positions, forward_batch, input_embeds)\n",
            "        input_embeds: Optional[torch.Tensor] = None,\n"
            "        pp_proxy_tensors=None,\n"
            "    ):\n"
            "        # Covers the text embedding norm too; the decoder's own scope nests.\n"
            "        with expert_tp.default_fused_ops_for_single_request(forward_batch):\n"
            "            return self._forward_omni(\n"
            "                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors\n"
            "            )\n",
        ),
        (
            "        input_embeds: Optional[torch.Tensor],\n"
            "    ):\n"
            "        if input_ids is None:\n",
            "        input_embeds: Optional[torch.Tensor],\n"
            "        pp_proxy_tensors=None,\n"
            "    ):\n"
            "        if not self.model.language_model.pp_group.is_first_rank:\n"
            "            # Later stages read the stream from the proxy, not embeddings.\n"
            "            return super().forward(\n"
            "                input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors\n"
            "            )\n"
            "        if input_ids is None:\n",
        ),
    ],
)

# --- DFLASH under SGLANG_ENABLE_PP_SPEC -------------------------------------
# The draft lives on the last stage. Each round it verifies the block the
# relay carries, accepts, then tail-drafts the next block; earlier stages run
# their half of the same 8-token DFLASH verify from the relayed tokens.

helper = srt / "speculative/dflash_pp.py"
helper.write_text(
    '''# musespark-pp patch applied
"""DFLASH verify forwards for the stages before the last one under PP."""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput


def build_relayed_verify(batch, target_worker, tokens: torch.Tensor):
    """Verify ``tokens`` ([bs, block], column 0 the bonus) at the batch's
    committed lengths, laid out as the DFLASH worker lays out its own block."""
    runner = target_worker.model_runner
    bs, block = tokens.shape
    prefix_lens = batch.seq_lens
    offsets = torch.arange(block, dtype=prefix_lens.dtype, device=prefix_lens.device)
    positions = (prefix_lens.unsqueeze(1) + offsets).reshape(-1).to(torch.int64)
    verify_out_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + block,
        batch_size=bs,
        draft_token_num=block,
        device=prefix_lens.device,
    )
    seq_lens_cpu = batch.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = prefix_lens.to("cpu")
    kv_loc_plan = runner.kv_index_translator.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=prefix_lens,
        seq_lens_cpu=seq_lens_cpu,
        write_virtual=verify_out_cache_loc,
        read_extent=block,
    )
    verify_input = DFlashVerifyInput(
        draft_token=tokens.reshape(-1),
        positions=positions,
        draft_token_num=block,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        kv_loc_plan=kv_loc_plan,
    )
    batch.out_cache_loc = verify_out_cache_loc
    backup = (batch.seq_lens_cpu, batch.seq_lens_sum)
    # Verify host bound = committed prefix + one verify block.
    batch.seq_lens_cpu = seq_lens_cpu + block
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    try:
        forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            batch, target_worker
        )
    finally:
        batch.seq_lens_cpu, batch.seq_lens_sum = backup
    return forward_batch, can_run_cuda_graph, verify_out_cache_loc


def run_relayed_verify(batch, target_worker, pp_proxy_tensors):
    """An earlier stage's model chunk over the relayed block."""
    tokens = batch.spec_info.tokens.to(
        device=batch.seq_lens.device, dtype=torch.int64
    )
    forward_batch, can_run_cuda_graph, verify_out_cache_loc = build_relayed_verify(
        batch, target_worker, tokens
    )
    result = target_worker.forward_batch_generation(
        batch=None,
        forward_batch=forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
        is_verify=True,
        skip_attn_backend_init=True,
    )
    result.can_run_cuda_graph = can_run_cuda_graph
    return result, verify_out_cache_loc
'''
)
print("dflash_pp.py: written")

patch(
    srt / "managers/scheduler.py",
    [
        (
            "                if is_verify_round:\n"
            "                    # PP+spec decode: every stage rebuilds the same verify\n"
            "                    # input from relayed per-req state (draft lives on the\n"
            "                    # last stage only).\n"
            "                    self._pp_spec_rebuild_verify_input(batch)\n",
            "                if is_verify_round and batch.spec_algorithm.is_dflash_family():\n"
            "                    # DFLASH verifies the relayed block directly; label its\n"
            "                    # rows for the live composition.\n"
            "                    live_rids = [req.rid for req in batch.reqs]\n"
            "                    relay_rids = getattr(batch.spec_info, \"rids\", None)\n"
            "                    if relay_rids is not None and relay_rids != live_rids:\n"
            "                        batch.spec_info = batch.spec_info.reindex(live_rids)\n"
            "                elif is_verify_round:\n"
            "                    # PP+spec decode: every stage rebuilds the same verify\n"
            "                    # input from relayed per-req state (draft lives on the\n"
            "                    # last stage only).\n"
            "                    self._pp_spec_rebuild_verify_input(batch)\n",
        ),
        (
            "                    if is_verify_round:\n"
            "                        from sglang.srt.speculative.eagle_utils import (\n"
            "                            eagle_prepare_for_verify,\n"
            "                        )\n",
            "                    if is_verify_round and batch.spec_algorithm.is_dflash_family():\n"
            "                        from sglang.srt.speculative.dflash_pp import (\n"
            "                            run_relayed_verify,\n"
            "                        )\n"
            "\n"
            "                        with self._forward_isolation(batch, overlap=False):\n"
            "                            batch_result, verify_out_cache_loc = (\n"
            "                                run_relayed_verify(\n"
            "                                    batch, self.tp_worker, pp_proxy_tensors\n"
            "                                )\n"
            "                            )\n"
            "                        batch_result.spec_verify_out_cache_loc = (\n"
            "                            verify_out_cache_loc\n"
            "                        )\n"
            "                    elif is_verify_round:\n"
            "                        from sglang.srt.speculative.eagle_utils import (\n"
            "                            eagle_prepare_for_verify,\n"
            "                        )\n",
        ),
    ],
)

patch(
    srt / "managers/scheduler_pp_mixin.py",
    [
        # DFLASH relays a chain: no tree topology rides with its tokens.
        (
            "                tensor_dict[\"spec_next_chain\"] = result.next_verify_chain\n"
            "                tensor_dict[\"spec_next_parents\"] = result.next_verify_parent_list\n"
            "                tensor_dict[\"spec_next_top_scores\"] = (\n"
            "                    result.next_verify_top_scores_index\n"
            "                )\n",
            "                tensor_dict[\"spec_next_chain\"] = result.next_verify_chain\n"
            "                if result.next_verify_parent_list is not None:\n"
            "                    tensor_dict[\"spec_next_parents\"] = result.next_verify_parent_list\n"
            "                    tensor_dict[\"spec_next_top_scores\"] = (\n"
            "                        result.next_verify_top_scores_index\n"
            "                    )\n",
        ),
        (
            "            and not batch.spec_algorithm.is_dspark()\n"
            "            and draft_input.topk_p is not None\n",
            "            and not batch.spec_algorithm.is_dspark()\n"
            "            and not batch.spec_algorithm.is_dflash_family()\n"
            "            and draft_input.topk_p is not None\n",
        ),
    ],
)

patch(
    srt / "speculative/spec_utils.py",
    [
        (
            "    if batch.spec_algorithm.is_dflash_family():\n"
            "        batch.spec_info.prepare_for_decode(batch)\n",
            "    from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput\n"
            "\n"
            "    if batch.spec_algorithm.is_dflash_family() and not isinstance(\n"
            "        batch.spec_info, PPSpecRelayInput\n"
            "    ):\n"
            "        batch.spec_info.prepare_for_decode(batch)\n",
        ),
    ],
)

patch(
    srt / "arg_groups/validation_hook.py",
    [
        (
            "        assert (\n"
            "            cfg.speculative_algorithm.upper() == \"EAGLE\"\n"
            "            and not cfg.enable_multi_layer_eagle\n"
            "        ), (\n",
            "        assert (\n"
            "            cfg.speculative_algorithm.upper() == \"EAGLE\"\n"
            "            and not cfg.enable_multi_layer_eagle\n"
            "        ) or (\n"
            "            cfg.speculative_algorithm.upper() == \"DFLASH\"\n"
            "            and envs.SGLANG_ENABLE_PP_SPEC.get()\n"
            "        ), (\n",
        ),
    ],
)

patch(
    srt / "arg_groups/speculative_hook.py",
    [
        (
            "    if cfg.pp_size != 1:\n"
            "        raise ValueError(\n"
            "            \"Currently DFLASH speculative decoding only supports pp_size == 1.\"\n"
            "        )\n",
            "    if cfg.pp_size != 1 and not envs.SGLANG_ENABLE_PP_SPEC.get():\n"
            "        raise ValueError(\n"
            "            \"Currently DFLASH speculative decoding only supports pp_size == 1.\"\n"
            "        )\n",
        ),
    ],
)

patch(
    srt / "speculative/dflash_worker_v2.py",
    [
        (
            "        grammar_barrier=None,\n"
            "        pp_proxy_tensors=None,\n"
            "    ) -> GenerationBatchResult:\n"
            "        self._validate_phase1_sampling_support(batch)\n",
            "        grammar_barrier=None,\n"
            "        pp_proxy_tensors=None,\n"
            "        _pp_tail_draft=False,\n"
            "    ) -> GenerationBatchResult:\n"
            "        self._validate_phase1_sampling_support(batch)\n",
        ),
        # Under PP the block to verify arrives on the relay.
        (
            "        draft_input = batch.spec_info\n"
            "        if not isinstance(draft_input, DFlashDraftInputV2):\n",
            "        from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput\n"
            "\n"
            "        pp_relayed_block = None\n"
            "        if isinstance(batch.spec_info, PPSpecRelayInput):\n"
            "            pp_relayed_block = batch.spec_info.tokens.to(\n"
            "                device=self.device, dtype=torch.int64\n"
            "            )\n"
            "            batch.spec_info = make_draft_input_v2(\n"
            "                bonus_tokens=pp_relayed_block[:, 0].contiguous(),\n"
            "                new_seq_lens=batch.seq_lens,\n"
            "            )\n"
            "        draft_input = batch.spec_info\n"
            "        if not isinstance(draft_input, DFlashDraftInputV2):\n",
        ),
        (
            "        draft_tokens = self._draft_block_tokens_buf[:bs]\n"
            "        draft_tokens[:, 0].copy_(block_ids[:, 0])\n"
            "        draft_tokens[:, 1:].copy_(draft_next)\n",
            "        draft_tokens = self._draft_block_tokens_buf[:bs]\n"
            "        draft_tokens[:, 0].copy_(block_ids[:, 0])\n"
            "        draft_tokens[:, 1:].copy_(draft_next)\n"
            "        if _pp_tail_draft:\n"
            "            return draft_tokens.clone()\n"
            "        if pp_relayed_block is not None:\n"
            "            # Earlier stages already verified the relayed block.\n"
            "            draft_tokens.copy_(pp_relayed_block)\n",
        ),
        (
            "        target_out = self.target_worker.forward_batch_generation(\n"
            "            batch=None,\n"
            "            forward_batch=verify_forward_batch,\n"
            "            is_verify=True,\n"
            "            skip_attn_backend_init=True if not _is_npu else None,\n"
            "        )\n",
            "        target_out = self.target_worker.forward_batch_generation(\n"
            "            batch=None,\n"
            "            forward_batch=verify_forward_batch,\n"
            "            pp_proxy_tensors=pp_proxy_tensors,\n"
            "            is_verify=True,\n"
            "            skip_attn_backend_init=True if not _is_npu else None,\n"
            "        )\n",
        ),
        (
            "        return GenerationBatchResult(\n"
            "            logits_output=logits_output,\n"
            "            next_token_ids=out_tokens.reshape(-1),\n"
            "            accept_lens=commit_lens,\n",
            "        # PP tail draft: earlier stages need the next block before their\n"
            "        # half of the next verify, so draft it now from the state the\n"
            "        # next round would start from.\n"
            "        pp_next_chain = (\n"
            "            self._pp_tail_draft(batch, next_draft_input)\n"
            "            if get_parallel().pp_size > 1\n"
            "            else None\n"
            "        )\n"
            "        result = GenerationBatchResult(\n"
            "            logits_output=logits_output,\n"
            "            next_token_ids=out_tokens.reshape(-1),\n"
            "            accept_lens=commit_lens,\n",
        ),
        (
            "            indexer_topk_output=target_out.indexer_topk_output,\n"
            "        )\n",
            "            indexer_topk_output=target_out.indexer_topk_output,\n"
            "        )\n"
            "        result.next_verify_chain = pp_next_chain\n"
            "        return result\n"
            "\n"
            "    def _pp_tail_draft(self, batch, next_draft_input):\n"
            "        saved = (\n"
            "            batch.spec_info,\n"
            "            batch.seq_lens,\n"
            "            batch.seq_lens_cpu,\n"
            "            batch.seq_lens_sum,\n"
            "            batch.out_cache_loc,\n"
            "            batch.input_ids,\n"
            "            batch.forward_mode,\n"
            "        )\n"
            "        try:\n"
            "            batch.spec_info = next_draft_input\n"
            "            batch.seq_lens = next_draft_input.new_seq_lens.to(saved[1].dtype)\n"
            "            batch.seq_lens_cpu = batch.seq_lens.to(\"cpu\")\n"
            "            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())\n"
            "            batch.forward_mode = ForwardMode.DECODE\n"
            "            batch.input_ids = None\n"
            "            tokens = self.forward_batch_generation(batch, _pp_tail_draft=True)\n"
            "        finally:\n"
            "            (\n"
            "                batch.spec_info,\n"
            "                batch.seq_lens,\n"
            "                batch.seq_lens_cpu,\n"
            "                batch.seq_lens_sum,\n"
            "                batch.out_cache_loc,\n"
            "                batch.input_ids,\n"
            "                batch.forward_mode,\n"
            "            ) = saved\n"
            "        return tokens.reshape(-1)\n",
        ),
    ],
)
PY
