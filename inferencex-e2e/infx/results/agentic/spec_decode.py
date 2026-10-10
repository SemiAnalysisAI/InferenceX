"""Realized speculative acceptance from the engines' own Prometheus counters.

``AL = 1 + accepted draft tokens / verification steps``, from the generation engine's counters'
increase between a scrape before and one after the measured traffic.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

# family -> [(labels, increase)] of every scraped endpoint's series
Increase = dict[str, list[tuple[dict[str, str], float]]]
Keep = Callable[[dict[str, str]], bool]
Reading = tuple[float, float] | None

_SAMPLE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:]*)(?:\{((?:[^"}]|"(?:[^"\\]|\\.)*")*)\})?\s+(\S+)')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, dict[frozenset[tuple[str, str]], float]]:
    """``{family: {labels: value}}`` of a Prometheus text exposition, label values escaped.

    Families drop a ``_total`` suffix; ``_created`` timestamps are skipped. Other text (TRT-LLM's
    JSON ``/metrics``, say) raises ``ValueError``.
    """
    samples: dict[str, dict[frozenset[tuple[str, str]], float]] = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample = _SAMPLE.match(line)
        if sample is None:
            raise ValueError(f"line {number} is not Prometheus text exposition: {line[:80]!r}")
        name, labels, value = sample.groups()
        if not name.endswith("_created"):
            series = samples.setdefault(name.removesuffix("_total"), {})
            series[frozenset(_LABEL.findall(labels or ""))] = float(value)
    return samples


def increase(before: Mapping[str, str], after: Mapping[str, str]) -> Increase:
    """Each series' increase between two scrapes (``{url: text}``), as Prometheus counts it.

    A series that went backwards was reset, so its new value counts; a new series starts at 0.
    """
    families: Increase = {}
    for url, text in after.items():
        start = parse(before[url])
        for family, series in parse(text).items():
            for labels, value in series.items():
                old = start.get(family, {}).get(labels, 0.0)
                rise = value - old if value >= old else value
                families.setdefault(family, []).append((dict(labels), rise))
    return families


def _sum(counters: Increase, family: str, keep: Keep | None = None) -> float | None:
    """The kept series' summed increase of the families matching ``family``; None without any."""
    kept = []
    for name, series in counters.items():
        if re.fullmatch(family, name):
            kept += [value for labels, value in series if keep is None or keep(labels)]
    return sum(kept) if kept else None


def _read(counters: Increase, accepted: str, steps: str, keep: Keep | None = None) -> Reading:
    verifications = _sum(counters, steps, keep)
    return None if verifications is None else (_sum(counters, accepted) or 0.0, verifications)


def _sglang(counters: Increase) -> Reading:
    # Disaggregated prefill workers export the same families without verifying any draft.
    verifications, tokens, requests = (
        _sum(counters, f"sglang[:_]{name}", lambda s: s.get("engine_type") != "prefill")
        for name in ("spec_verify_calls", "generation_tokens", "num_requests")
    )
    if verifications is None or tokens is None or requests is None:
        return None
    # Each verification emits its accepted drafts and one token; prefill emits each first token.
    return tokens - requests - verifications, verifications


# ``(accepted draft tokens, verifications)`` or None, by synthetic_acceptance.ENGINES names.
READERS: dict[str, Callable[[Increase], Reading]] = {
    "vllm": lambda c: _read(
        c, "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts"
    ),
    "sglang": _sglang,
    # Every verification drafts a first token; deeper positions are drafted only sometimes.
    "trtllm": lambda c: _read(
        c,
        "trtllm_spec_decode_accepted_tokens",
        "trtllm_spec_decode_drafted_tokens",
        lambda s: s.get("token_position") == "0",
    ),
    "atom": lambda c: _read(c, "atom[:_]mtp_accepted_tokens", "atom[:_]mtp_decode_steps"),
}
