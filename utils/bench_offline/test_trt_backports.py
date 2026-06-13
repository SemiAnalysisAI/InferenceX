from pathlib import Path

import trt_backports
from trt_backports import (
    DSV4_PATCH_MARKER,
    apply_dsv4_skip_premoe_allreduce_backport,
    patch_dsv4_source_text,
    sha256_text,
)


PINNED_FRAGMENT = '''def _resolve_enable_fused_hc(config: PretrainedConfig) -> bool:
    """Resolve the DeepSeek-V4 fused HC boundary-fusion knob."""
    env = os.environ.get("TRTLLM_MHC_ENABLE_FUSED_HC")
    if env is not None:
        return env not in ("0", "false", "False")
    return bool(getattr(config, "enable_fused_hc", True))


def _copy_deepseek_v4_fused_a_weight_scale(

        self.enable_fused_hc = _resolve_enable_fused_hc(config)
        self.next_layer_layernorm: RMSNorm = None

        if self.enable_fused_hc:
            residual, post_mix, comb_mix, layer_input = self.hc_ffn.fused_hc(
                x_prev=x_attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
            )
        else:

        if self.fusion_config.PRE_MOE_FUSION:
            # In DeepSeek-V4 the external residual connection is handled by mHC
            # (hc_ffn.post_mapping), so there is no residual to add here.
            # Use fused allreduce + RMSNorm (no residual addition).
            hidden_states = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        else:
'''


def test_patch_dsv4_source_text_adds_guarded_optimization():
    patched = patch_dsv4_source_text(PINNED_FRAGMENT)
    assert DSV4_PATCH_MARKER in patched
    assert "norm_weight=norm_weight" in patched
    assert "if not self.skip_premoe_allreduce:" in patched


def test_apply_backport_checks_hash_and_is_idempotent(tmp_path, monkeypatch):
    source_path = tmp_path / "modeling_deepseekv4.py"
    source_path.write_text(PINNED_FRAGMENT, encoding="utf-8")
    monkeypatch.setattr(
        trt_backports,
        "locate_dsv4_source",
        lambda: Path(source_path),
    )
    monkeypatch.setattr(
        trt_backports,
        "DSV4_PINNED_SOURCE_SHA256",
        sha256_text(PINNED_FRAGMENT),
    )
    monkeypatch.setattr(
        trt_backports,
        "DSV4_PATCHED_SOURCE_SHA256",
        sha256_text(patch_dsv4_source_text(PINNED_FRAGMENT)),
    )

    applied = apply_dsv4_skip_premoe_allreduce_backport()
    repeated = apply_dsv4_skip_premoe_allreduce_backport()

    assert applied["status"] == "applied"
    assert repeated["status"] == "already_applied"
    assert repeated["after_sha256"] == applied["after_sha256"]
