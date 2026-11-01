"""Bounded, external GLM-5.3-Flash HiCache diagnostic. Never flushes caches.

Exit 0: requested real-host-restore rounds retained every marker.
Exit 1: a proven host-restored response lost a marker after valid controls.
Exit 2: prerequisites, evidence, transport, or resource budget inconclusive.
This is an additional diagnostic, not canonical accuracy/performance approval.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn


class InconclusiveError(Exception):
    pass


class IncorrectError(Exception):
    pass


SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*?)\})?\s+"
    r"([-+]?(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"(?:\s+[0-9]+)?$"
)
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
PREFIX = "sglang:"
MetricMap = dict[tuple[str, tuple[tuple[str, str], ...]], float]


class Metrics(dict):
    def __init__(self) -> None:
        super().__init__()
        self.registered = {}


def parse_metrics(text: str) -> Metrics:
    result = Metrics()
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            result.registered[name] = kind
            if kind == "histogram":
                result.registered[name + "_count"] = kind
                result.registered[name + "_sum"] = kind
            continue
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        name, labels, value = match.groups()
        parsed = tuple(
            sorted((key, json.loads('"' + val + '"')) for key, val in LABEL.findall(labels or ""))
        )
        number = float(value)
        if math.isfinite(number):
            result[name, parsed] = number
    return result


def series(metrics: MetricMap, name: str, labels: dict[str, str]) -> MetricMap:
    return {
        key: value
        for key, value in metrics.items()
        if key[0] == PREFIX + name and all(dict(key[1]).get(k) == v for k, v in labels.items())
    }


def delta(
    before: MetricMap, after: MetricMap, name: str, labels: dict[str, str], *, required: bool = True
) -> float | None:
    """Single logical series only: never multiply logical counts by TP ranks."""
    a = series(before, name, labels)
    b = series(after, name, labels)
    if not a and not b and not required:
        return None
    if (
        not a
        and len(b) == 1
        and getattr(before, "registered", {}).get(PREFIX + name) in ("counter", "histogram")
    ):
        # Prometheus labeled counters create their first child lazily. The
        # family must already have been registered before this request.
        return next(iter(b.values()))
    if len(a) != 1 or len(b) != 1 or a.keys() != b.keys():
        raise InconclusiveError(
            f"{name}: expected one stable metric series, got {len(a)}/{len(b)}; "
            "provide --metric-label for rank/model disambiguation"
        )
    value = next(iter(b.values())) - next(iter(a.values()))
    if value < 0:
        raise InconclusiveError(f"{name}: metric counter reset")
    return value


def dropped_delta(before: MetricMap, after: MetricMap, labels: dict[str, str]) -> float:
    # Reasons may be created lazily. Sum reasons, but reject unspecified ranks.
    selector = {**labels, "pool": "kv"}
    a = series(before, "hicache_dropped_tokens_total", selector)
    b = series(after, "hicache_dropped_tokens_total", selector)
    identities = {
        tuple((k, v) for k, v in key[1] if k not in ("reason", "pool"))
        for key in a.keys() | b.keys()
    }
    if len(identities) > 1:
        raise InconclusiveError("drop metrics span multiple ranks/models")
    for key, value in a.items():
        if key not in b or b[key] < value:
            raise InconclusiveError("drop metric counter reset/disappeared")
    return sum(b.values()) - sum(a.values())


def validate_provenance(
    fingerprint: dict[str, Any], boot_log: str, expected_source: str
) -> dict[str, Any]:
    if not isinstance(fingerprint.get("env"), dict):
        raise InconclusiveError("fingerprint must contain the actual server env object")
    simulation = {
        k: v
        for k, v in fingerprint["env"].items()
        if k.startswith(("SGLANG_SIMULATE_", "TILERT_SIMULATE_")) and v not in ("", None)
    }
    if simulation:
        raise InconclusiveError("server fingerprint contains simulation variables")
    packages = json.dumps(fingerprint.get("pip_packages", {}))
    if f"sglang.git@{expected_source}#egg=sglang" not in packages:
        raise InconclusiveError("server fingerprint does not prove expected SGLang source SHA")
    # Pool ordering differs across supported versions.
    stack_lines = [
        line
        for line in boot_log.splitlines()
        if "pool" in line.lower()
        and all(re.search(r"\b" + name + r"\b", line) for name in ("KV", "INDEXER", "MAMBA"))
    ]
    if not stack_lines:
        raise InconclusiveError("startup log does not prove KV + INDEXER + MAMBA stack")
    if re.search(r"declares host pools.*did not assemble", boot_log):
        raise InconclusiveError("startup log reports an unassembled declared host pool")
    identities = set()
    for line in boot_log.splitlines():
        if "server_args={" in line:
            args = ast.literal_eval(line.split("server_args=", 1)[1])
            identities.add(
                tuple(
                    args.get(k)
                    for k in (
                        "model_path",
                        "revision",
                        "served_model_name",
                    )
                )
            )
    if len(identities) != 1:
        raise InconclusiveError("startup log lacks one unambiguous server model identity")
    return {
        "frameworks": fingerprint.get("frameworks"),
        "pool_lines": stack_lines,
        "resolved_boot_contexts": sorted(
            set(map(int, re.findall(r"\bcontext_len=(\d+)", boot_log)))
        ),
        "boot_identity": dict(
            zip(
                ("model_path", "revision", "served_model_name"),
                identities.pop(),
                strict=False,
            )
        ),
    }


def validate_server(
    info: dict[str, Any],
    model_info: dict[str, Any],
    expected_model: str,
    expected_revision: str,
    served_model: str | None = None,
    fingerprint: dict[str, Any] | None = None,
    boot_identity: dict[str, Any] | None = None,
    expected_host_fraction: float = 0.70,
    boot_contexts: list[int] | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if info.get("model_path") != expected_model or model_info.get("model_path") != expected_model:
        raise InconclusiveError("live server model identity differs from expected model")
    served_model = served_model or expected_model
    if (
        info.get("served_model_name") != served_model
        or model_info.get("served_model_name") != served_model
    ):
        raise InconclusiveError("live served model name differs from requested model")
    if boot_identity is not None and any(
        boot_identity.get(k) != info.get(k) for k in ("model_path", "revision", "served_model_name")
    ):
        raise InconclusiveError("live model identity differs from startup log")
    fingerprint_revision = (fingerprint or {}).get("model", {}).get("hf_revision")
    if info.get("revision") != expected_revision and not (
        info.get("revision") is None
        and expected_model.startswith("/")
        and fingerprint_revision == expected_revision
        and boot_identity is not None
    ):
        raise InconclusiveError("live server revision differs from expected checkpoint")
    if fingerprint_revision is not None and fingerprint_revision != expected_revision:
        raise InconclusiveError("fingerprint revision differs from expected checkpoint")
    for flag in ("enable_hierarchical_cache", "enable_metrics", "enable_cache_report"):
        if info.get(flag) is not True:
            raise InconclusiveError(f"live server must enable {flag}")
    if info.get("speculative_num_steps") != 5:
        raise InconclusiveError("diagnostic requires the intended actual MTP5 server")
    context = info.get("context_length")
    context_evidence = "explicit live context_length"
    if context is None:
        config = model_config or {}
        text_config = config.get("text_config", config)
        if (
            boot_contexts != [1048576]
            or text_config.get("max_position_embeddings") != 1048576
            or "Glm5NextForConditionalGeneration" not in config.get("architectures", [])
        ):
            raise InconclusiveError(
                "unset CLI context requires actual 1M boot context and pinned Flash model config"
            )
        context = 1048576
        context_evidence = "resolved startup context plus revision-verified Flash model config"
    if context != 1048576:
        raise InconclusiveError("resolved runtime context must equal 1048576")
    required_settings = {
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 6,
        "hicache_io_backend": "kernel",
        "hicache_mem_layout": "page_first",
        "hicache_write_policy": "write_through",
        "kv_cache_dtype": "fp8_e4m3",
    }
    for key, expected in required_settings.items():
        if info.get(key) != expected:
            raise InconclusiveError(f"live {key} must equal {expected!r} before qualification")
    fraction = info.get("hicache_host_memory_fraction")
    if not isinstance(fraction, (int, float)) or not math.isclose(
        fraction, expected_host_fraction, abs_tol=1e-9
    ):
        raise InconclusiveError("live HiCache host-memory fraction differs from expected budget")
    result = {
        key: info.get(key)
        for key in (
            "model_path",
            "revision",
            "context_length",
            "speculative_algorithm",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "kv_cache_dtype",
            "tp_size",
            "ep_size",
            "hicache_mem_layout",
            "hicache_io_backend",
            "hicache_write_policy",
            "hicache_size",
            "hicache_ratio",
            "hicache_host_memory_fraction",
            "max_total_num_tokens",
            "launch_command",
            "version",
            "startup_time",
        )
    }
    result.update(resolved_context_length=context, context_evidence=context_evidence)
    return result


class HTTP:
    def __init__(self, url: str, metrics_url: str | None, deadline: float, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.metrics_url = metrics_url or self.url + "/metrics"
        self.deadline, self.timeout = deadline, timeout

    def request(self, url: str, payload: dict[str, Any] | None = None) -> str:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise InconclusiveError("wall-clock budget exhausted")
        if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
            raise InconclusiveError("only HTTP(S) diagnostic endpoints are supported")
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310 - HTTP(S) scheme validated above.
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - HTTP(S) scheme validated above.
                req, timeout=min(remaining, self.timeout)
            ) as response:
                return response.read().decode()
        except (OSError, urllib.error.HTTPError) as exc:
            raise InconclusiveError(f"HTTP transport failure: {exc}") from exc

    def metrics(self) -> Metrics:
        return parse_metrics(self.request(self.metrics_url))

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.request(self.url + "/v1/chat/completions", payload))

    def tokenize(self, prompt: str, model: str) -> int:
        data = json.loads(
            self.request(
                self.url + "/v1/tokenize",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        )
        count = data.get("count")
        if not isinstance(count, int) or count <= 0:
            raise InconclusiveError("tokenizer did not return a positive scalar count")
        return count


@dataclass
class Limits:
    max_requests: int = 0
    max_prompt_tokens: int = 0
    max_pressure_requests: int = 0
    max_seconds: float = 1800
    request_timeout: float = 120
    pressure_chars: int = 16384
    pressure_batch: int = 4
    restore_rounds: int = 2
    output_tokens: int = 4096
    min_prefix_fraction: float = 0.8
    pressure_target_tokens: int = 128000
    marker_min_tokens: int = 16384


def capacity_plan(
    metrics: MetricMap, labels: dict[str, str], boot_log: str, limits: Limits
) -> dict[str, Any]:
    def gauge(name: str) -> int:
        samples = series(metrics, name, labels)
        if len(samples) != 1:
            raise InconclusiveError(f"capacity gauge {name} must select one rank/model")
        value = next(iter(samples.values()))
        if value < 0 or value != int(value):
            raise InconclusiveError(f"capacity gauge {name} is not a nonnegative integer")
        return int(value)

    capacities = {
        pool: sum(gauge(f"{pool}_{state}_tokens") for state in ("available", "evictable", "used"))
        for pool in ("kv", "mamba")
    }
    logged = set(map(int, re.findall(r"max_total_num_tokens=(\d+)", boot_log)))
    if logged != {capacities["kv"]} or capacities["mamba"] <= 0:
        raise InconclusiveError("live KV/Mamba capacities are invalid or disagree with startup log")
    size = capacities["kv"]
    pressure = (11 * size + 9) // 10
    per_round = math.ceil(pressure / limits.pressure_target_tokens)
    planned = (
        limits.restore_rounds * (pressure + limits.pressure_target_tokens + 2048) + 131072
    )  # conservative allowance for cold/L1/marker-probe traffic
    ceiling = 3 * size
    if planned > ceiling:
        raise InconclusiveError("requested rounds/pressure granularity exceed the three-pool cap")
    limits.max_prompt_tokens = limits.max_prompt_tokens or ceiling
    limits.max_pressure_requests = limits.max_pressure_requests or per_round * limits.restore_rounds
    limits.max_requests = (
        limits.max_requests or limits.max_pressure_requests + 2 + limits.restore_rounds
    )
    if limits.max_prompt_tokens > ceiling or limits.max_prompt_tokens < planned:
        raise InconclusiveError(
            "token budget must cover the plan without exceeding three logical pools"
        )
    if (
        limits.max_pressure_requests < per_round * limits.restore_rounds
        or limits.max_requests < per_round * limits.restore_rounds + 2 + limits.restore_rounds
    ):
        raise InconclusiveError("request budget cannot cover the uninterrupted pressure plan")
    return {
        "device_kv_logical_tokens": size,
        "device_mamba_states": capacities["mamba"],
        "pressure_tokens_per_round": pressure,
        "target_pressure_tokens_per_request": limits.pressure_target_tokens,
        "estimated_pressure_requests_per_round": per_round,
        "conservative_planned_prompt_tokens": planned,
        "hard_total_prompt_token_cap": limits.max_prompt_tokens,
        "hard_wall_seconds": limits.max_seconds,
        "qualification_scope": "one representative allocation; not all lanes",
        "selected_capacity_labels": dict(
            next(iter(series(metrics, "kv_available_tokens", labels)))[1]
        ),
    }


@dataclass
class Probe:
    http: Any
    model: str
    limits: Limits
    metric_labels: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    requests: int = 0
    prompt_tokens: int = 0
    pressure_requests: int = 0
    plan: dict | None = None

    def chat(self, prompt: str, stage: str, max_tokens: int) -> dict[str, Any]:
        if self.requests >= self.limits.max_requests:
            raise InconclusiveError("request budget exhausted")
        tokenized = self.http.tokenize(prompt, self.model)
        if self.prompt_tokens + tokenized > self.limits.max_prompt_tokens:
            raise InconclusiveError("prompt-token budget exhausted (actual tokenizer precheck)")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "return_cached_tokens_details": True,
            "return_spec_tokens_details": True,
        }
        response = self.http.chat(payload)
        self.requests += 1
        usage = response.get("usage") or {}
        count = usage.get("prompt_tokens")
        if not isinstance(count, int) or count <= 0:
            raise InconclusiveError("response lacks positive integer prompt_tokens")
        self.prompt_tokens += count
        event = {
            "stage": stage,
            "request": self.requests,
            "usage": usage,
            "tokenized_prompt_tokens": tokenized,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "sglext": response.get("sglext") or {},
        }
        self.events.append(event)
        if count != tokenized:
            raise InconclusiveError("inference prompt usage differs from exact pre-tokenization")
        if self.prompt_tokens > self.limits.max_prompt_tokens:
            raise InconclusiveError("observed prompt-token budget exceeded")
        choices = response.get("choices") or []
        if not choices:
            raise InconclusiveError("response has no choices")
        choice = choices[0]
        event["finish_reason"] = choice.get("finish_reason")
        event["content"] = (choice.get("message") or {}).get("content") or ""
        return event

    def facts(self, prompt: str, expected: list[str], stage: str) -> dict[str, Any]:
        result = self.chat(prompt, stage, self.limits.output_tokens)
        if result["finish_reason"] != "stop":
            raise InconclusiveError(f"{stage}: finish_reason={result['finish_reason']!r}")
        missing = [
            value
            for value in expected
            if not re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])",
                result["content"],
            )
        ]
        result["missing_markers"] = missing
        if missing and stage in ("cold", "device_hit"):
            raise InconclusiveError(f"{stage}: control missing {len(missing)} verbatim markers")
        return result

    def pressure_prompt(self) -> str:
        target = self.limits.pressure_target_tokens
        seed = uuid.uuid4().hex
        full = f"Unrelated pressure document {seed}.\n" + "".join(
            f"Record {seed}-{j}: committee {j * 19} filed motion "
            f"{j * 37} under independent seal {uuid.uuid4().hex}.\n"
            for j in range(target * 8 // 90 + 1)
        )
        length = target * 4
        for _ in range(8):
            text = full[:length] + "\nReply OK."
            count = self.http.tokenize(text, self.model)
            if target <= count <= target + 2048:
                return text
            length = math.ceil(length * target / count) + 8
            if length > len(full):
                raise InconclusiveError("bounded pressure-token calibration ran out of text")
        raise InconclusiveError("bounded pressure-token calibration did not converge")

    def run(self) -> dict[str, Any]:
        nonce = uuid.uuid4().hex
        row_count = 240
        for _ in range(6):
            rows = [
                f"Entry {i:04d}: sector {i * 3}, vault {i * 17}, "
                f"exact access marker MARK_{nonce}_{i:04d}; "
                "this registry entry is independent of every other entry."
                for i in range(row_count)
            ]
            selected = [
                row_count // 20,
                row_count // 3,
                row_count * 2 // 3,
                row_count - row_count // 20,
            ]
            expected = [f"MARK_{nonce}_{i:04d}" for i in selected]
            prompt = (
                f"Independent registry {nonce}. Read the following records.\n"
                + "\n".join(rows)
                + "\nReturn only the exact access markers for entries "
                + ", ".join(map(str, selected))
                + ", one marker per line. Copy them verbatim; do not explain."
            )
            measured = self.http.tokenize(prompt, self.model)
            if measured >= self.limits.marker_min_tokens:
                break
            row_count = math.ceil(row_count * self.limits.marker_min_tokens / measured * 1.05)
            if row_count > 4000:
                raise InconclusiveError("marker calibration exceeded bounded registry size")
        else:
            raise InconclusiveError(
                "marker calibration did not reach minimum logical prefix length"
            )
        # Fresh nonce prevents reuse of a previous run's entire prefix.
        cold = self.facts(prompt, expected, "cold")
        cold["expected_markers"] = expected
        cold["marker_entry_positions"] = selected
        before = self.http.metrics()
        warm = self.facts(prompt, expected, "device_hit")
        after = self.http.metrics()
        warm_cached = (warm["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        warm_details = warm["sglext"].get("cached_tokens_details") or {}
        required = self.limits.min_prefix_fraction * cold["usage"]["prompt_tokens"]
        if (
            warm_cached < required
            or warm_details.get("device", 0) < required
            or warm_details.get("host", 0)
            or warm_details.get("storage", 0)
        ):
            raise InconclusiveError("device-hit control lacks a pure device-resident prefix")
        if (
            delta(before, after, "load_back_tokens_total", {**self.metric_labels, "pool": "kv"})
            != 0
        ):
            raise InconclusiveError("device-hit control unexpectedly restored host tokens")
        completed = 0
        batch = self.limits.pressure_batch
        baseline = self.http.metrics()
        while completed < self.limits.restore_rounds:
            start_tokens, batch_count = self.prompt_tokens, 0
            target_tokens = self.plan["pressure_tokens_per_round"] if self.plan else None
            while (
                (self.prompt_tokens - start_tokens < target_tokens)
                if target_tokens is not None
                else batch_count < batch
            ):
                if self.pressure_requests >= self.limits.max_pressure_requests:
                    raise InconclusiveError(
                        "pressure budget exhausted without required host restores"
                    )
                self.chat(self.pressure_prompt(), "pressure", 8)
                self.pressure_requests += 1
                batch_count += 1
            before = self.http.metrics()
            result = self.facts(prompt, expected, f"restore_candidate_{completed + 1}")
            after = self.http.metrics()
            kv = delta(
                before, after, "load_back_tokens_total", {**self.metric_labels, "pool": "kv"}
            )
            duration = delta(before, after, "load_back_duration_seconds_count", self.metric_labels)
            byte_count = delta(before, after, "load_back_bytes_total", self.metric_labels)
            mamba = delta(
                before,
                after,
                "load_back_tokens_total",
                {**self.metric_labels, "pool": "mamba"},
                required=False,
            )
            host_hit = delta(
                before,
                after,
                "prefill_effective_tokens_total",
                {**self.metric_labels, "mode": "host_hit"},
                required=False,
            )
            dropped = dropped_delta(baseline, after, self.metric_labels)
            cached = (result["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            details = result["sglext"].get("cached_tokens_details") or {}
            request_host = details.get("host", 0)
            evidence = {
                "kv_loaded_tokens": kv,
                "load_back_events": duration,
                "loaded_bytes": byte_count,
                "mamba_loaded_checkpoints": mamba,
                "host_hit_tokens_log_interval_counter": host_hit,
                "destroyed_without_backup_tokens_since_pressure_start": dropped,
                "cached_tokens": cached,
                "minimum_prefix_tokens": required,
                "request_host_tokens": request_host,
                "request_storage_tokens": details.get("storage", 0),
            }
            result["host_evidence"] = evidence
            if dropped > 0:
                raise InconclusiveError("pressure destroyed device tokens without backup")
            if (
                kv >= required
                and duration > 0
                and byte_count > 0
                and cached >= required
                and request_host >= required
                and details.get("storage", 0) == 0
            ):
                if mamba is None or mamba <= 0:
                    raise InconclusiveError("host restore lacks Mamba checkpoint evidence")
                spec = result["sglext"].get("spec_tokens_details")
                spec_rows = spec if isinstance(spec, list) else [spec or {}]
                if sum(row.get("spec_verify_ct", 0) for row in spec_rows) <= 0:
                    raise InconclusiveError(
                        "host probe lacks actual speculative verification evidence"
                    )
                result["proven_host_restore"] = True
                if result["missing_markers"]:
                    raise IncorrectError(
                        f"proven host restore: missing {len(result['missing_markers'])} "
                        "verbatim markers after valid cold/device controls"
                    )
                completed += 1
            else:
                result["proven_host_restore"] = False
                if self.plan:
                    raise InconclusiveError(
                        "full planned pressure did not prove a retained host restore"
                    )
                # Re-probing refreshes L1; next pressure batch must grow.
                batch *= 2
        return {"proven_host_restores": completed, "markers_per_probe": len(expected)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--metrics-url")
    p.add_argument("--model", required=True)
    p.add_argument("--expected-model", required=True)
    p.add_argument("--expected-revision", required=True)
    p.add_argument("--expected-source", default="baa9befa381511cd4d56f83980951e5a45fffc6d")
    p.add_argument("--expected-image", required=True)
    p.add_argument("--expected-host-fraction", type=float, default=0.70)
    p.add_argument("--model-config", type=Path)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Execute the recorded capacity-derived plan; default only plans",
    )
    p.add_argument("--server-fingerprint", type=Path, required=True)
    p.add_argument("--server-boot-log", type=Path, required=True)
    p.add_argument(
        "--exclusive-server",
        action="store_true",
        required=True,
        help="Operator confirms no other clients during diagnostic",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric-label", action="append", default=[], metavar="NAME=VALUE")
    for name, item in Limits.__dataclass_fields__.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(item.default), default=item.default)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = {"diagnostic": "glm53flash-forced-host-restore", "status": "INCONCLUSIVE"}
    probe = None
    exit_code = 2
    previous_alarm = None
    try:
        limits = Limits(**{key: getattr(args, key) for key in Limits.__dataclass_fields__})
        auto = {"max_requests", "max_prompt_tokens", "max_pressure_requests"}
        if (
            any(
                value < 0 or (value == 0 and key not in auto) for key, value in vars(limits).items()
            )
            or limits.min_prefix_fraction > 1
            or limits.max_seconds > 1800
        ):
            raise InconclusiveError(
                "budgets must be positive (0 means auto only for request/token limits); "
                "prefix fraction <=1; wall deadline <=1800 seconds"
            )
        if limits.marker_min_tokens < 16384 or limits.output_tokens > 4096:
            raise InconclusiveError("marker prefix must be >=16384 tokens; output budget <=4096")
        labels = dict(value.split("=", 1) for value in args.metric_label)
        report["provenance"] = validate_provenance(
            (fingerprint := json.loads(args.server_fingerprint.read_text())),
            args.server_boot_log.read_text(),
            args.expected_source,
        )
        report["limits"] = vars(limits)
        report["metric_labels"] = labels
        report["operator_attests_exclusive_server"] = args.exclusive_server
        report["operator_supplied_expected_image"] = args.expected_image
        if not hasattr(signal, "setitimer"):
            raise InconclusiveError("strict wall-clock deadline requires POSIX setitimer")

        def expire(*_: object) -> NoReturn:
            raise InconclusiveError("wall-clock budget exhausted")

        previous_alarm = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, limits.max_seconds)
        http = HTTP(
            args.url,
            args.metrics_url,
            time.monotonic() + limits.max_seconds,
            limits.request_timeout,
        )
        config_path = args.model_config
        if config_path is None and args.expected_model.startswith("/"):
            config_path = Path(args.expected_model) / "config.json"
        model_config = None
        if config_path is not None:
            config_bytes = config_path.read_bytes()
            model_config = json.loads(config_bytes)
            report["checkpoint_config"] = {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
                "architectures": model_config.get("architectures"),
                "max_position_embeddings": model_config.get("text_config", model_config).get(
                    "max_position_embeddings"
                ),
            }
        report["live_server"] = validate_server(
            json.loads(http.request(http.url + "/server_info")),
            json.loads(http.request(http.url + "/model_info")),
            args.expected_model,
            args.expected_revision,
            args.model,
            fingerprint,
            report["provenance"]["boot_identity"],
            args.expected_host_fraction,
            report["provenance"]["resolved_boot_contexts"],
            model_config,
        )
        report["capacity_plan"] = capacity_plan(
            http.metrics(),
            labels,
            args.server_boot_log.read_text(),
            limits,
        )
        report["status"] = "PLAN_ONLY"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.execute:
            probe = Probe(http, args.model, limits, labels, plan=report["capacity_plan"])
            report["result"] = probe.run()
            report["status"] = "DIAGNOSTIC_PASS"
        exit_code = 0
    except IncorrectError as exc:
        report.update(status="DIAGNOSTIC_FAIL", reason=str(exc))
        exit_code = 1
    except (InconclusiveError, OSError, ValueError, KeyError, TypeError) as exc:
        report["reason"] = str(exc)
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if probe is not None:
            report.update(
                events=probe.events,
                requests=probe.requests,
                prompt_tokens=probe.prompt_tokens,
                pressure_requests=probe.pressure_requests,
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output), "reason": report.get("reason")}
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
