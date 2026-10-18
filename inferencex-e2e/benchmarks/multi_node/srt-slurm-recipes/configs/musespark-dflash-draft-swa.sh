#!/usr/bin/env bash
set -eo pipefail
package=$(python3 -P -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
echo "Patching SGLang at $package"
python3 - "$package/srt" <<'PY'
import sys
from pathlib import Path

package = Path(sys.argv[1])
edits = [
    (
        "model_executor/pool_configurator.py",
        """    unified_total_bytes: Optional[int] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
""",
        """    unified_total_bytes: Optional[int] = None

    dflash_draft_swa: bool = False

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
""",
    ),
    (
        "model_executor/pool_configurator.py",
        """def _get_dsa_cache_layer_ids(kvc: KVCacheConfigurator, num_layers: int) -> list[int]:
""",
        """def dflash_draft_in_swa_pool(kvc: KVCacheConfigurator) -> bool:
    if kvc.is_draft_worker or not kvc.spec_algorithm.is_dflash():
        return False
    if not kvc.is_hybrid_swa or getattr(kvc.model_config, "is_deepseek_v4_arch", False):
        return False
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    draft = kvc.spec_aux_config.draft_model_config
    layer_types = get_dflash_layer_types(draft.hf_config) if draft else None
    window = get_dflash_attention_sliding_window_size(draft.hf_config) if draft else None
    unmet = [
        reason
        for reason, failed in (
            ("no full-attention target layers", not kvc.layer_info.full_attention_layer_ids),
            ("--enable-unified-memory", get_memory().enable_unified_memory),
            ("--disable-radix-cache", get_memory().disable_radix_cache),
            ("DCP", get_parallel().attn_dcp_size > 1),
            ("disaggregation", get_disagg().disaggregation_mode != "null"),
            ("no draft KV cell size", _dflash_draft_cell_size(kvc) <= 0),
            (
                f"draft layer_types {layer_types} are not all sliding_attention",
                not layer_types or any(t != "sliding_attention" for t in layer_types),
            ),
            (
                f"draft window {window} exceeds target window {kvc.sliding_window_size}",
                window is None
                or kvc.sliding_window_size is None
                or window > kvc.sliding_window_size,
            ),
        )
        if failed
    ]
    if unmet:
        raise ValueError(f"DFLASH draft KV in the SWA pool is unsupported: {'; '.join(unmet)}")
    return True


def _get_dsa_cache_layer_ids(kvc: KVCacheConfigurator, num_layers: int) -> list[int]:
""",
    ),
    (
        "model_executor/pool_configurator.py",
        """        self._draft_cell_size = _dflash_draft_cell_size(kvc)
        self._fused_full_entry = kvc.fused_entry_bytes("full")
""",
        """        self._draft_cell_size = _dflash_draft_cell_size(kvc)
        self._draft_in_swa = dflash_draft_in_swa_pool(kvc)
        self._fused_full_entry = kvc.fused_entry_bytes("full")
""",
    ),
    (
        "model_executor/pool_configurator.py",
        """                + self._swa_full_tokens_ratio * self._swa_cell_bytes()
                + self._draft_cell_size
            )

    def _draft_pool_bytes_per_token(self) -> int:
""",
        """                + self._swa_full_tokens_ratio * self._swa_cell_bytes()
                + self._draft_cell_size
                * (self._swa_full_tokens_ratio if self._draft_in_swa else 1)
            )

    def _draft_pool_bytes_per_token(self) -> int:
""",
    ),
    (
        "mem_cache/kv_cache_configurator.py",
        """        config.mem_fraction_static = get_schedule().mem_fraction_static
        return config

    def config_from_budget(
""",
        """        config.mem_fraction_static = get_schedule().mem_fraction_static
        from sglang.srt.model_executor.pool_configurator import (
            dflash_draft_in_swa_pool,
        )

        config.dflash_draft_swa = dflash_draft_in_swa_pool(self)
        if config.dflash_draft_swa:
            logger.info("DFLASH draft KV is sized and indexed in the SWA pool.")
        return config

    def config_from_budget(
""",
    ),
    (
        "mem_cache/kv_cache_configurator.py",
        """            config = self.memory_pool_config
        else:
            config = self._resolve_memory_pool_config(pre_model_load_memory)
""",
        """            config = self.memory_pool_config
            if config.dflash_draft_swa and not self.is_hybrid_swa:
                self._bind_dflash_draft_swa()
        else:
            config = self._resolve_memory_pool_config(pre_model_load_memory)
""",
    ),
    (
        "mem_cache/kv_cache_configurator.py",
        """    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
""",
        """    def _bind_dflash_draft_swa(self) -> None:
        from sglang.srt.mem_cache.unified_draft_pool import draft_kv_layer_ids

        alloc = self.token_to_kv_pool_allocator
        if (
            not isinstance(alloc, SWATokenToKVPoolAllocator)
            or isinstance(alloc, UnifiedSWAAllocatorBase)
            or is_swa_req_ring(alloc)
            or getattr(alloc, "full_to_swa_index_mapping", None) is None
        ):
            raise ValueError(
                "DFLASH draft KV in the SWA pool needs the target's paged "
                f"SWATokenToKVPoolAllocator, got {type(alloc).__name__}"
            )
        layer_ids = draft_kv_layer_ids(self.model)
        self.is_hybrid_swa = True
        self.layer_info = msgspec.structs.replace(
            self.layer_info,
            swa_attention_layer_ids=layer_ids,
            full_attention_layer_ids=[],
        )
        logger.info(f"DFLASH draft KV layers {layer_ids} bound to the SWA pool.")

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
""",
    ),
    (
        "mem_cache/swa_memory_pool.py",
        """        if is_swa_layer:
            # swa_loc is the full->SWA translation, computed once per forward by
            # the attention backend; set_kv_buffer never translates internally.
            assert swa_loc is not None
            loc = swa_loc
""",
        """        if is_swa_layer:
            if swa_loc is None:
                swa_loc = self.translate_loc_from_full_to_swa(loc)
            loc = swa_loc
""",
    ),
    (
        "mem_cache/swa_memory_pool.py",
        """    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
""",
        """    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer.layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            mapping = self.full_to_swa_index_mapping
            assert mapping is not None
            loc_2d = mapping[loc_2d.to(device=mapping.device, dtype=torch.int64)]
        pool.set_kv_buffer_prefix_valid(
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=layer_id_pool,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
""",
    ),
]
for name, old, new in edits:
    path = package / name
    text = path.read_text()
    if text.count(new) == 1:
        print(f"already patched {path}")
    elif text.count(old) == 1:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
    else:
        sys.exit(f"{path}: anchor found {text.count(old)} times, expected 1:\n{old}")
PY
