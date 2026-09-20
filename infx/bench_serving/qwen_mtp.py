"""Prepare native BF16 MTP and measured acceptance for Qwen3.8-27B FP8 recipes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from infx.bench_serving.speedbench_acceptance import mtp_quantization_overrides


def build_configs(
    target: dict[str, Any],
    draft: dict[str, Any],
    draft_weights: dict[str, str],
    golden: dict[str, Any],
    *,
    draft_model: str,
    draft_revision: str,
    tokens: int,
    thinking_mode: str,
    eval_only: bool,
    run_eval: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use synthetic acceptance only when no accuracy evaluation will run."""
    if isinstance(tokens, bool) or not 1 <= tokens <= 4:
        raise ValueError("Qwen3.8-27B native MTP requires 1-4 draft tokens")
    if thinking_mode not in {"thinking_on", "thinking_off"}:
        raise ValueError(f"Invalid thinking mode: {thinking_mode}")
    if target.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("The target must use the measured FP8 checkpoint")
    draft_text = draft.get("text_config", draft)
    if (
        draft.get("quantization_config")
        or draft_text.get("dtype", draft_text.get("torch_dtype")) != "bfloat16"
    ):
        raise ValueError("The native MTP head must retain its original BF16 precision")
    overrides = mtp_quantization_overrides(target, draft_weights)
    spec: dict[str, Any] = {
        "method": "mtp",
        "model": draft_model,
        "revision": draft_revision,
        "num_speculative_tokens": tokens,
        "kv_cache_dtype": "auto",
        "rejection_sample_method": "standard",
    }
    if not (eval_only or run_eval):
        try:
            value = golden["qwen3.8-27b-fp8"][thinking_mode][tokens]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing golden AL for {thinking_mode}, {tokens} drafts") from error
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 1 <= value <= tokens + 1
        ):
            raise ValueError(f"Invalid golden AL: {value!r}")
        spec.update(rejection_sample_method="synthetic", synthetic_acceptance_length=value)
    return spec, overrides


def configuration_file(model: str, revision: str, filename: str) -> Path:
    """Read local metadata or download it at the caller's pinned revision."""
    if Path(model).is_dir():
        return Path(model) / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, filename, revision=revision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-model", "target-revision", "draft-model", "draft-revision"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--thinking-mode", required=True)
    parser.add_argument("--eval-only", required=True, choices=("true", "false"))
    parser.add_argument("--run-eval", required=True, choices=("true", "false"))
    parser.add_argument("--golden-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    target = json.loads(
        configuration_file(args.target_model, args.target_revision, "config.json").read_text()
    )
    draft = json.loads(
        configuration_file(args.draft_model, args.draft_revision, "config.json").read_text()
    )
    draft_index = json.loads(
        configuration_file(
            args.draft_model, args.draft_revision, "model.safetensors.index.json"
        ).read_text()
    )
    spec, overrides = build_configs(
        target,
        draft,
        draft_index["weight_map"],
        yaml.safe_load(args.golden_file.read_text()),
        draft_model=args.draft_model,
        draft_revision=args.draft_revision,
        tokens=args.tokens,
        thinking_mode=args.thinking_mode,
        eval_only=args.eval_only == "true",
        run_eval=args.run_eval == "true",
    )
    (args.output_dir / "speculative-config.json").write_text(json.dumps(spec))
    (args.output_dir / "hf-overrides.json").write_text(json.dumps(overrides))
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
