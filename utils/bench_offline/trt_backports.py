"""Runtime backports for the pinned TensorRT-LLM offline image."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any


DSV4_SKIP_PREMOE_ALLREDUCE_COMMIT = (
    "f04f90e8b48641fc82f4e5db5bd608b7debbff55"
)
DSV4_PINNED_SOURCE_SHA256 = (
    "4f65eaefdb1cbdb20456415c55d041493d7e4f984cee74f5f91ebecb0e9d33f8"
)
DSV4_PATCHED_SOURCE_SHA256 = (
    "09986ecbc71467325e668c59e61d790e120036e102eefe7c6eae9e671e0af18f"
)
DSV4_PATCH_MARKER = "def _resolve_skip_premoe_allreduce() -> bool:"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def locate_dsv4_source() -> Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed tensorrt_llm package")
    roots = [Path(value) for value in spec.submodule_search_locations]
    matches = [
        root / "_torch" / "models" / "modeling_deepseekv4.py"
        for root in roots
    ]
    existing = [path.resolve() for path in matches if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            "Expected one installed modeling_deepseekv4.py, found "
            f"{[str(path) for path in existing]}"
        )
    return existing[0]


def inspect_dsv4_source() -> dict[str, Any]:
    path = locate_dsv4_source()
    source = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "sha256": sha256_text(source),
        "skip_premoe_allreduce_backport": DSV4_PATCH_MARKER in source,
        "upstream_commit": DSV4_SKIP_PREMOE_ALLREDUCE_COMMIT,
    }


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"DeepSeek-V4 backport anchor {label!r} matched {count} times"
        )
    return source.replace(old, new, 1)


def patch_dsv4_source_text(source: str) -> str:
    source = _replace_once(
        source,
        """def _resolve_enable_fused_hc(config: PretrainedConfig) -> bool:
    \"\"\"Resolve the DeepSeek-V4 fused HC boundary-fusion knob.\"\"\"
    env = os.environ.get("TRTLLM_MHC_ENABLE_FUSED_HC")
    if env is not None:
        return env not in ("0", "false", "False")
    return bool(getattr(config, "enable_fused_hc", True))


def _copy_deepseek_v4_fused_a_weight_scale(
""",
        """def _resolve_enable_fused_hc(config: PretrainedConfig) -> bool:
    \"\"\"Resolve the DeepSeek-V4 fused HC boundary-fusion knob.\"\"\"
    env = os.environ.get("TRTLLM_MHC_ENABLE_FUSED_HC")
    if env is not None:
        return env not in ("0", "false", "False")
    return bool(getattr(config, "enable_fused_hc", True))


def _resolve_skip_premoe_allreduce() -> bool:
    \"\"\"Resolve whether to skip the redundant PRE_MOE_FUSION allreduce.\"\"\"
    env = os.environ.get("TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE", "1")
    return env not in ("0", "false", "False")


def _copy_deepseek_v4_fused_a_weight_scale(
""",
        "resolver",
    )
    source = _replace_once(
        source,
        """        self.enable_fused_hc = _resolve_enable_fused_hc(config)
        self.next_layer_layernorm: RMSNorm = None
""",
        """        self.enable_fused_hc = _resolve_enable_fused_hc(config)
        self.skip_premoe_allreduce = (
            _resolve_skip_premoe_allreduce() and self.enable_fused_hc
        )
        self.next_layer_layernorm: RMSNorm = None
""",
        "layer flag",
    )
    source = _replace_once(
        source,
        """        if self.enable_fused_hc:
            residual, post_mix, comb_mix, layer_input = self.hc_ffn.fused_hc(
                x_prev=x_attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
            )
        else:
""",
        """        if self.enable_fused_hc:
            norm_weight = (
                self.post_attention_layernorm.weight
                if self.skip_premoe_allreduce
                and self.fusion_config.PRE_MOE_FUSION
                else None
            )
            norm_eps = (
                self.post_attention_layernorm.variance_epsilon
                if norm_weight is not None
                else 0.0
            )
            residual, post_mix, comb_mix, layer_input = self.hc_ffn.fused_hc(
                x_prev=x_attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        else:
""",
        "fused HC norm",
    )
    source = _replace_once(
        source,
        """        if self.fusion_config.PRE_MOE_FUSION:
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
""",
        """        if self.fusion_config.PRE_MOE_FUSION:
            if not self.skip_premoe_allreduce:
                # In DeepSeek-V4 the external residual connection is handled by
                # mHC, so there is no residual to add here.
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
""",
        "redundant allreduce",
    )
    return source


def apply_dsv4_skip_premoe_allreduce_backport() -> dict[str, Any]:
    path = locate_dsv4_source()
    source = path.read_text(encoding="utf-8")
    before_sha256 = sha256_text(source)
    if DSV4_PATCH_MARKER in source:
        if before_sha256 != DSV4_PATCHED_SOURCE_SHA256:
            raise RuntimeError(
                "DeepSeek-V4 source contains the backport marker with an "
                f"unexpected hash: {before_sha256} != "
                f"{DSV4_PATCHED_SOURCE_SHA256}"
            )
        return {
            **inspect_dsv4_source(),
            "status": "already_applied",
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
        }
    if before_sha256 != DSV4_PINNED_SOURCE_SHA256:
        raise RuntimeError(
            "Pinned DeepSeek-V4 TRT source hash mismatch: "
            f"{before_sha256} != {DSV4_PINNED_SOURCE_SHA256}"
        )

    patched = patch_dsv4_source_text(source)
    after_sha256 = sha256_text(patched)
    if after_sha256 != DSV4_PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "DeepSeek-V4 backport output hash mismatch: "
            f"{after_sha256} != {DSV4_PATCHED_SOURCE_SHA256}"
        )
    temporary = path.with_name(f".{path.name}.trt-bench-{os.getpid()}.tmp")
    try:
        temporary.write_text(patched, encoding="utf-8")
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "status": "applied",
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "sha256": after_sha256,
        "skip_premoe_allreduce_backport": True,
        "upstream_commit": DSV4_SKIP_PREMOE_ALLREDUCE_COMMIT,
    }
