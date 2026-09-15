from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
t = p.read_text()
old = """        if self._align_mode:\n            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)\n            aligned_index_builders = []\n            for group_idx, group_id in enumerate(mamba_group_ids):\n                for group in attn_groups[group_id]:\n                    builder = group.get_metadata_builder(0)\n                    if hasattr(builder, \"mamba_aligned_state_indices\"):\n                        aligned_index_builders.append((group_idx, builder))\n            if aligned_index_builders:\n                ctx = self._ensure_align_ctx(\n                    kv_cache_config, mamba_group_ids, block_tables\n                )\n                all_group_indices = ctx.compute_aligned_state_indices(\n                    input_batch.seq_lens, num_reqs\n                )\n                for group_idx, builder in aligned_index_builders:\n                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]\n\n"""
if "# K3 hybrid PP alignment deferral" in t:
    print("[k3-hybrid-pp] already applied")
elif old not in t:
    raise SystemExit("[k3-hybrid-pp] prepare_attn alignment block not found")
else:
    t=t.replace(old, "        # K3 hybrid PP alignment is initialized in preprocess_state only.\n        # K3 hybrid PP alignment deferral\n\n", 1)
    p.write_text(t)
    print("[k3-hybrid-pp] applied")
