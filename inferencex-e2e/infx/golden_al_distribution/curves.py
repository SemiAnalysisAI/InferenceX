"""Load committed golden AL curves and resolve which curve a speculative config uses."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GOLDEN_DIR = Path(__file__).resolve().parent
THINKING_MODES = ("thinking_on", "thinking_off")


@dataclass(frozen=True)
class Curve:
    """One committed curve: ``{thinking_mode: {num_speculative_tokens: AL}}``."""

    name: str
    model: str
    modes: Mapping[str, Mapping[int, Any]]

    def tokens(self, thinking: str) -> list[int]:
        return sorted(self.modes.get(thinking) or {})

    def acceptance(self, thinking: str, tokens: int) -> float:
        try:
            value = float(self.modes[thinking][tokens])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"No golden acceptance for {self.name}/{thinking}/{tokens} draft tokens"
            ) from error
        if not math.isfinite(value) or not 1 <= value <= tokens + 1:
            raise ValueError(f"Invalid golden acceptance {value} in {self.name}.yaml")
        return value


def curve_name(model: str, spec: Mapping[str, Any]) -> str:
    """Map an InferenceX model prefix and speculative config to a curve file stem."""
    method = str(spec.get("method", "")).lower()
    # SGLang calls native model MTP EAGLE/NEXTN; the curve describes the model's head.
    if method in ("eagle", "nextn"):
        method = "eagle3" if model == "minimaxm3" else "mtp"
    curve = f"{model}_{method}"
    if model in ("dsv4", "dsv4dspark", "dsv4dsparkprob") and method == "dspark":
        curve = "dsv4-pro-0813-dspark"
    elif model == "minimaxm3" and method == "eagle3":
        if "gqa" in str(spec.get("model", "")).lower():
            curve += "_gqa"
    elif model == "kimik3" and method == "dspark":
        # Kimi has distinct measured curves; require the recipe to choose its sampler.
        sampling = spec.get("draft_sample_method")
        if sampling == "probabilistic":
            curve += "_probabilistic_sample_method_block_rejection_sample_method"
        elif sampling != "greedy":
            raise ValueError(f"No Kimi DSpark golden curve for draft sampling {sampling!r}")
    if not re.fullmatch(r"[a-z0-9_.-]+", curve):
        raise ValueError(f"Invalid golden curve identity: {curve!r}")
    return curve


def load_curve(name: str, golden_dir: Path = GOLDEN_DIR) -> Curve:
    path = golden_dir / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"No committed golden curve {path.name} in {golden_dir}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one model in golden curve {path.name}")
    [(model, modes)] = data.items()
    if not isinstance(modes, dict):
        raise ValueError(f"Expected thinking modes in golden curve {path.name}")
    return Curve(name=name, model=str(model), modes=modes)


def list_curves(golden_dir: Path = GOLDEN_DIR) -> list[Curve]:
    return [load_curve(path.stem, golden_dir) for path in sorted(golden_dir.glob("*.yaml"))]


def golden_length(
    model: str, spec: Mapping[str, Any], thinking: str, golden_dir: Path = GOLDEN_DIR
) -> float:
    """Return the committed AL for a model prefix, speculative config and thinking mode."""
    name = curve_name(model, spec)
    tokens = spec.get("num_speculative_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Speculative decoding requires a positive integer draft length")
    return load_curve(name, golden_dir).acceptance(thinking, tokens)
