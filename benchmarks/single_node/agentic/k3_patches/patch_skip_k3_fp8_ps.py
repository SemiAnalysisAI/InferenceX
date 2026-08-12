"""Skip FP8 ASM MLA prefill PS workspace on Kimi-K3 (fused FA prefill path)."""
import re

F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s = open(F).read()
orig = s

MARKER = "PATCH(skip-k3-fp8-ps)"
if MARKER in s and "_model_uses_fused_mla_prefill" in s:
    print(f"{MARKER} v2 already applied")
    raise SystemExit(0)

helper_v2 = '''
# PATCH(skip-k3-fp8-ps): Kimi-K3 prefill uses fused FA, not forward_mha FP8 ASM.
_FUSED_MLA_PREFILL_MODEL_TYPES = frozenset({"kimi_k3", "kimi_k3_mtp"})
_FUSED_MLA_PREFILL_ARCHITECTURES = frozenset({
    "KimiK3ForConditionalGeneration",
    "KimiK3MTPModel",
})


def _model_uses_fused_mla_prefill(vllm_config=None) -> bool:
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return False

    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    if getattr(hf_config, "model_type", None) in _FUSED_MLA_PREFILL_MODEL_TYPES:
        return True

    architectures = getattr(hf_config, "architectures", None) or []
    if any(arch in _FUSED_MLA_PREFILL_ARCHITECTURES for arch in architectures):
        return True

    hf_text_config = model_config.hf_text_config
    if getattr(hf_text_config, "model_type", None) in _FUSED_MLA_PREFILL_MODEL_TYPES:
        return True

    return False


def _uses_aiter_fp8_asm_mla_prefill(
    *,
    kv_cache_dtype: str,
    num_heads: int,
    vllm_config=None,
) -> bool:
    from vllm.utils.torch_utils import is_quantized_kv_cache

    if not _fp8_mla_prefill_supported():
        return False
    if not is_quantized_kv_cache(kv_cache_dtype):
        return False
    if not (num_heads % 16 == 0 or 0 < num_heads < 16):
        return False

    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()

    if _model_uses_fused_mla_prefill(vllm_config):
        hf_config = (
            vllm_config.model_config.hf_config
            if vllm_config is not None and vllm_config.model_config is not None
            else None
        )
        model_label = getattr(hf_config, "model_type", None) or (
            getattr(hf_config, "architectures", ["Kimi-K3"])[0]
            if hf_config is not None
            else "Kimi-K3"
        )
        logger.info_once(
            "Skipping FP8 ASM MLA prefill workspace for %s: prefill uses "
            "the fused MLA path (forward_mha is not called).",
            model_label,
        )
        return False

    return True
'''

if MARKER in s:
    s = re.sub(
        rf"# {re.escape(MARKER)}:.*?^class AiterMLABackend",
        helper_v2 + "\n\nclass AiterMLABackend",
        s,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
else:
    insert_after = "    return True\n\n\nclass AiterMLABackend"
    assert insert_after in s, "insert anchor (AiterMLABackend) not found"
    s = s.replace(insert_after, "    return True\n" + helper_v2 + "\n\nclass AiterMLABackend", 1)

    builder_old = """        self._fp8_prefill_enabled = (
            _fp8_mla_prefill_supported() and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
        )"""
    builder_new = f"""        self._fp8_prefill_enabled = _uses_aiter_fp8_asm_mla_prefill(
            kv_cache_dtype=kv_cache_dtype_str,
            num_heads=self.num_heads,
            vllm_config=vllm_config,
        )  # {MARKER}"""
    assert builder_old in s, "builder gate not found"
    s = s.replace(builder_old, builder_new, 1)

    impl_old = """        self._fp8_prefill_enabled = (
            _fp8_mla_prefill_supported() and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
        )
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1"""
    impl_new = f"""        self._fp8_prefill_enabled = _uses_aiter_fp8_asm_mla_prefill(
            kv_cache_dtype=kv_cache_dtype,
            num_heads=num_heads,
        )  # {MARKER}
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1"""
    assert impl_old in s, "impl gate not found"
    s = s.replace(impl_old, impl_new, 1)

assert s != orig
open(F, "w").write(s)
print(f"{MARKER} v2 applied")
