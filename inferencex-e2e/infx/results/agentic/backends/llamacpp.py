"""llama.cpp server metric adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from infx.results.agentic.common import rate, sum_server_log_capacities, sum_stat

from .base import ServerMetricsBackend, counter_int


class LlamaCppBackend(ServerMetricsBackend):
    """llama-server's ``llamacpp:`` counters.

    It counts tokens reused from a slot's KV cache and from its host-memory prompt cache alike,
    so only the overall hit rate is known. It exports no KV usage, so its KV pool comes from
    what it logs at startup.
    """

    name = "llamacpp"
    _SLOTS_RE = re.compile(
        r"initializing, n_slots = (?P<slots>\d+), n_ctx_slot = (?P<context>\d+), "
        r"kv_unified = '(?P<unified>true|false)'"
    )
    _SEQUENCE_CONTEXT_RES = (
        re.compile(r"n_ctx_seq \((?P<context>\d+)\) > n_ctx_train"),
        re.compile(r"the slot context \((?P<context>\d+)\) exceeds the training context"),
    )

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        """Return True for llamacpp: metric names, or for an empty scrape of a llamacpp server."""
        return any(name.startswith("llamacpp:") for name in metrics) or (
            not metrics and framework.lower() == "llamacpp"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        """Fill token totals and the overall cache hit rate from ``metrics``."""
        computed = sum_stat(metrics, "llamacpp:prompt_tokens")
        cached = sum_stat(metrics, "llamacpp:prompt_tokens_cached")
        prompt = None if computed is None else computed + (cached or 0.0)
        generated = sum_stat(metrics, "llamacpp:tokens_predicted")
        flat["total_prompt_tokens"] = counter_int(prompt)
        flat["total_generation_tokens"] = counter_int(generated)
        flat["server_overall_cache_hit_rate"] = rate(cached, prompt)
        nested["cache"].update(
            {
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "prefix_cache_hits": cached,
                "prefix_cache_queries": prompt,
            }
        )
        raw = {"cached": cached, "computed": computed}
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
                "prompt_by_source": {
                    "gpu_cache_hit": None,
                    "cpu_or_external_cache_hit": None,
                    "computed": computed,
                    "raw": {source: tokens for source, tokens in raw.items() if tokens is not None},
                },
            }
        )

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],  # noqa: ARG002
        server_logs: Iterable[str | None],
    ) -> int | None:
        """Return the llama-server KV pool parsed from ``server_logs``, ignoring ``metrics``."""
        return sum_server_log_capacities(server_logs, self.kv_cache_pool_tokens_from_server_log)

    @classmethod
    def kv_cache_pool_tokens_from_server_log(cls, server_log: str | None) -> int | None:
        """Return the KV cache tokens llama-server allocated, from ``server_log``, or None.

        In b11910, a unified cache gives every sequence ``ctx-size`` tokens of one shared cache.
        Without one, each sequence has its own cache of ``ctx-size / parallel`` tokens.
        ``n_ctx_slot`` is capped at the training context, so when that cap applies the sequence's
        real context comes from a ``_SEQUENCE_CONTEXT_RES`` line.
        """
        log = server_log or ""
        slot_line = cls._SLOTS_RE.search(log)
        if slot_line is None:
            return None
        sequence_context = int(slot_line["context"])
        for pattern in cls._SEQUENCE_CONTEXT_RES:
            if found := pattern.search(log):
                sequence_context = int(found["context"])
                break
        if slot_line["unified"] == "true":
            return sequence_context
        return sequence_context * int(slot_line["slots"])
