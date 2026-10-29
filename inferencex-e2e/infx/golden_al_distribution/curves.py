"""Load committed golden AL curves and resolve which curve a speculative config uses."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infx import yamlio

from .identity import IDENTITIES_STEM, IDENTITY_DIR, select_curve

GOLDEN_DIR = IDENTITY_DIR
THINKING_MODES = ("thinking_on", "thinking_off")
CURVE_NAME = re.compile(r"[a-z0-9_.-]+")


@dataclass(frozen=True)
class Curve:
    """One committed curve: ``{thinking_mode: {num_speculative_tokens: AL}}``."""

    name: str
    model: str
    modes: Mapping[str, Mapping[int, Any]]

    def tokens(self, thinking: str) -> list[int]:
        """Return the sorted draft lengths measured for one thinking mode."""
        return sorted(self.modes.get(thinking) or {})

    def acceptance(self, thinking: str, tokens: int) -> float:
        """Return the AL for a thinking mode and draft length, or raise ValueError if invalid."""
        try:
            value = float(self.modes[thinking][tokens])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"No golden acceptance for {self.name}/{thinking}/{tokens} draft tokens"
            ) from error
        if not math.isfinite(value) or not 1 <= value <= tokens + 1:
            raise ValueError(f"Invalid golden acceptance {value} in {self.name}.yaml")
        return value


def curve_name(model: str, spec: Mapping[str, Any], golden_dir: Path = GOLDEN_DIR) -> str:
    """The curve file stem whose ``identities.yaml`` select block matches a model prefix and
    speculative config, or ValueError when none or several do."""
    name = select_curve(model, spec, golden_dir)
    if not CURVE_NAME.fullmatch(name):
        raise ValueError(f"Invalid golden curve identity: {name!r}")
    return name


def load_curve(name: str, golden_dir: Path = GOLDEN_DIR) -> Curve:
    """Load ``<name>.yaml`` from ``golden_dir``, or raise ValueError if missing or malformed."""
    if not CURVE_NAME.fullmatch(name) or name == IDENTITIES_STEM:
        raise ValueError(f"Invalid golden curve identity: {name!r}")
    path = golden_dir / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"No committed golden curve {path.name} in {golden_dir}")
    data = yamlio.safe_load(path.read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one model in golden curve {path.name}")
    [(model, modes)] = data.items()
    if not isinstance(modes, dict):
        raise ValueError(f"Expected thinking modes in golden curve {path.name}")
    for mode, cells in modes.items():
        if cells is not None and not isinstance(cells, dict):
            raise ValueError(f"Expected draft lengths in {mode} of golden curve {path.name}")
    return Curve(name=name, model=str(model), modes=modes)


def list_curves(golden_dir: Path = GOLDEN_DIR) -> list[Curve]:
    """Return every committed curve in ``golden_dir``, sorted by name."""
    paths = sorted(golden_dir.glob("*.yaml"))
    return [load_curve(path.stem, golden_dir) for path in paths if path.stem != IDENTITIES_STEM]


def golden_length(
    model: str, spec: Mapping[str, Any], thinking: str, golden_dir: Path = GOLDEN_DIR
) -> float:
    """Return the committed AL for a model prefix, speculative config and thinking mode."""
    name = curve_name(model, spec, golden_dir)
    tokens = spec.get("num_speculative_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Speculative decoding requires a positive integer draft length")
    return load_curve(name, golden_dir).acceptance(thinking, tokens)
