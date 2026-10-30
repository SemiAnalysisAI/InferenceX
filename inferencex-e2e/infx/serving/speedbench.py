"""SPEED-Bench on the serving client: row selection and request payloads.

SPEED-Bench (nvidia/SPEED-Bench) measures speculative-decoding acceptance on a fixed prompt set. The
public release masks about 60% of its qualitative rows, so the scenario reads the fully resolved copy
``Green-Lab/speed-bench-prepared`` (private; each config is one jsonl, pinned in datasets/lock.yaml).
Like ``greenspec`` it replays one selection through AgentX as a ``raw_payload`` file. SPEED-Bench is a
public set, so unlike greenspec nothing is filtered or withheld: the console, server log and result files
are uploaded as for any other scenario.

``settings``         the selection and sampling a serving YAML asks for (overrides:).
``write_selection``  select the rows and write each as a ready request body.

Each request is the first turn of a row alone, as the aiperf and vLLM SPEED-Bench loaders send it;
later turns of a multi-turn row are dropped. Every request streams and completes on its own (no
ignore_eos): acceptance only means something on natural output. Sampling defaults are the golden AL
matrices' (temperature 1.0, top_p 0.95, 4096 output tokens); ``chat_template_kwargs`` carries the
thinking switch of the old per-mode collectors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infx.serving.greenspec import (
    iter_rows,
)  # one jsonl line per row, same as the greenspec release

FAMILY = "speedbench"
TAG = "SpeedBench"
QUALITATIVE = "qualitative"
THROUGHPUTS = (
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
CONFIGS = (QUALITATIVE, *THROUGHPUTS)
CATEGORIES = {
    QUALITATIVE: (
        "coding",
        "humanities",
        "math",
        "multilingual",
        "qa",
        "rag",
        "reasoning",
        "roleplay",
        "stem",
        "summarization",
        "writing",
    ),
    **dict.fromkeys(THROUGHPUTS, ("low_entropy", "mixed", "high_entropy")),
}
OVERRIDES = (
    "config",
    "category",
    "max_tokens",
    "temperature",
    "top_p",
    "chat_template_kwargs",
    "requests",
)
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95


class SpeedBenchError(ValueError):
    """A selection problem, as one line."""


@dataclass(frozen=True)
class Settings:
    config: str = QUALITATIVE
    category: str | None = None  # None: every category of the config
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    chat_template_kwargs: tuple[
        tuple[str, Any], ...
    ] = ()  # sorted items, so Settings stays hashable
    requests: int | None = None  # None: the whole selection

    @property
    def dataset(self) -> str:
        """The name of the config's file in datasets/lock.yaml."""
        return f"speedbench-{self.config.replace('_', '-')}"


def _number(
    overrides: Mapping[str, Any], key: str, default: float, low: float, high: float
) -> float:
    value = overrides.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise SpeedBenchError(f"overrides.{key}: a number from {low} to {high}, got {value!r}")
    return float(value)


def normalize(overrides: Mapping[str, Any]) -> Settings:
    """The overrides of a speedbench serving YAML as ``Settings``; one line per problem."""
    if unknown := sorted(set(overrides) - set(OVERRIDES)):
        raise SpeedBenchError(
            f"overrides: speedbench takes {', '.join(OVERRIDES)}, not {', '.join(unknown)}"
        )
    config = overrides.get("config", QUALITATIVE)
    if config not in CONFIGS:
        raise SpeedBenchError(f"overrides.config: {config!r} is not one of {', '.join(CONFIGS)}")
    category = overrides.get("category")
    if category is not None and category not in CATEGORIES[config]:
        raise SpeedBenchError(
            f"overrides.category: {category!r} is not one of {', '.join(CATEGORIES[config])} "
            f"(the categories of {config})"
        )
    max_tokens, requests = (
        overrides.get("max_tokens", DEFAULT_MAX_TOKENS),
        overrides.get("requests"),
    )
    for field, value in (("max_tokens", max_tokens), ("requests", requests)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise SpeedBenchError(f"overrides.{field}: a positive integer, got {value!r}")
    kwargs = overrides.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
        raise SpeedBenchError(
            f"overrides.chat_template_kwargs: a mapping with string keys, got {kwargs!r}"
        )
    return Settings(
        config,
        category,
        max_tokens,
        _number(overrides, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
        _number(overrides, "top_p", DEFAULT_TOP_P, 0.0, 1.0),
        tuple(sorted(kwargs.items())),
        requests,
    )


def settings(env: Mapping[str, str]) -> Settings | None:
    """The job's ``Settings`` when its serving YAML runs the speedbench scenario, else None."""
    import yaml  # not imported at the top: the client's first pass runs before the AgentX venv exists

    try:
        raw = yaml.safe_load(env.get("SERVING_YAML") or "")
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict) or raw.get("scenario") != FAMILY:
        return None
    return normalize(raw.get("overrides") or {})


def label(chosen: Settings) -> str:
    """The result label of a selection: speedbench-qualitative, speedbench-throughput_8k-mixed-n20, ...

    Every non-default knob is in it (ctk-enable_thinking-false, mt2048, t0.6, p1): the run summary lines up
    results by label and concurrency.
    """
    parts = [FAMILY, chosen.config]
    if chosen.category:
        parts.append(chosen.category)
    for key, value in chosen.chat_template_kwargs:
        parts.append(
            "ctk-" + re.sub(r"[^0-9A-Za-z_.]+", "-", f"{key}-{json.dumps(value)}").strip("-")
        )
    if chosen.max_tokens != DEFAULT_MAX_TOKENS:
        parts.append(f"mt{chosen.max_tokens}")
    if chosen.temperature != DEFAULT_TEMPERATURE:
        parts.append(f"t{chosen.temperature:g}")
    if chosen.top_p != DEFAULT_TOP_P:
        parts.append(f"p{chosen.top_p:g}")
    if chosen.requests:
        parts.append(f"n{chosen.requests}")
    return "-".join(parts)


def prompt(row: dict[str, Any]) -> str:
    """The first turn of a row, stripped; empty when the row has none (such a row is skipped)."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return ""
    return str(messages[0].get("content") or "").strip()


def select(path: Path, chosen: Settings) -> list[dict[str, Any]]:
    return [
        row
        for row in iter_rows(path)
        if (chosen.category is None or row.get("category") == chosen.category) and prompt(row)
    ]


def payload(row: dict[str, Any], model: str, chosen: Settings) -> dict[str, Any]:
    """One row -> the request body the client sends for it."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(row)}],
        "max_tokens": chosen.max_tokens,
        "temperature": chosen.temperature,
        "top_p": chosen.top_p,
    }
    if chosen.chat_template_kwargs:
        body["chat_template_kwargs"] = dict(chosen.chat_template_kwargs)
    return {**body, "stream": True, "stream_options": {"include_usage": True}}


def write_selection(dataset: Path, chosen: Settings, model: str, out: Path) -> int:
    """Write the selected rows as one request body per line to ``out``; return how many."""
    rows = select(dataset, chosen)
    if not rows:
        raise SpeedBenchError(f"selection {label(chosen)}: no row of {chosen.config} matches")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(payload(row, model, chosen), ensure_ascii=False) + "\n" for row in rows
        )
    return len(rows)
