"""The AIPerf ``profile`` argv for one AgentX point."""

from __future__ import annotations

import urllib.parse
import subprocess
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from infx.bench import env as inputs
from infx.bench.agentic import traces

REQUIRED = (
    "MODEL",
    "MODEL_PREFIX",
    "FRAMEWORK",
    "CONC",
    "DURATION",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS",
    "AIPERF_EXPERIMENTAL_FAST",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING",
)

MIN_SCENARIO_DURATION_S = 900
FAST_DURATION_S = 1200
FAST_WARMUP_REQUESTS_PER_LANE = "1"


@dataclass(frozen=True)
class ReplayConfig:
    """Everything ``replay_argv`` needs, validated."""

    url: str
    model: str
    """Served name sent as ``--model``."""
    tokenizer: str
    """``TOKENIZER`` (a recipe's checkpoint path), else the Hugging Face id; a wire name is not
    necessarily a valid repo id."""
    concurrency: int
    duration: int
    warmup_requests_per_lane: str | None
    live_failed_request_threshold: str | None
    warmup_grace_period: str | None
    extra_inputs: tuple[str, ...]
    dynamo_session_timeout: str | None
    """Set only when Dynamo conversation-aware routing applies."""
    max_context_length: int | None
    server_metrics_urls: tuple[str, ...]
    artifact_dir: Path
    unsafe_override: bool
    loader: str
    dataset: str
    apply_chat_template: bool
    benchmark_grace_period: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str], result_dir: Path) -> ReplayConfig:
        """Validate the replay inputs in ``env``; artifacts go below ``result_dir``."""
        values = inputs.require(*REQUIRED, env=env)
        duration = inputs.parse_positive_int("DURATION", values["DURATION"])
        warmup = None
        if values["AIPERF_EXPERIMENTAL_FAST"] == "1":
            duration, warmup = FAST_DURATION_S, FAST_WARMUP_REQUESTS_PER_LANE
        conv_aware = (
            values["FRAMEWORK"].startswith("dynamo-")
            and values["AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING"] != "0"
            and values["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"] != "true"
        )
        max_context = inputs.optional("AIPERF_MAX_CONTEXT_LENGTH", env)
        loader, dataset = traces.resolve(
            values["MODEL_PREFIX"], inputs.optional("WEKA_LOADER_OVERRIDE", env)
        )
        extra_inputs = inputs.optional("AIPERF_EXTRA_INPUTS", env) or ""
        return cls(
            url=server_url(env),
            model=inputs.optional("SERVED_MODEL_NAME", env) or values["MODEL"],
            tokenizer=inputs.optional("TOKENIZER", env) or values["MODEL"],
            concurrency=inputs.parse_positive_int("CONC", values["CONC"]),
            duration=duration,
            warmup_requests_per_lane=warmup,
            live_failed_request_threshold=inputs.optional(
                "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD", env
            ),
            warmup_grace_period=inputs.optional("AGENTIC_WARMUP_GRACE_PERIOD", env),
            extra_inputs=tuple(extra_inputs.split()),
            dynamo_session_timeout=(
                values["AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS"] if conv_aware else None
            ),
            max_context_length=(
                inputs.parse_positive_int("AIPERF_MAX_CONTEXT_LENGTH", max_context)
                if max_context
                else None
            ),
            server_metrics_urls=metrics_urls(env),
            artifact_dir=result_dir / "aiperf_artifacts",
            unsafe_override=duration < MIN_SCENARIO_DURATION_S,
            loader=loader,
            dataset=dataset,
            apply_chat_template=env.get("AIPERF_APPLY_CHAT_TEMPLATE") == "true",
            benchmark_grace_period=inputs.optional("AIPERF_BENCHMARK_GRACE_PERIOD", env),
        )


def extra_frontend_urls(url: str) -> list[str]:
    """Diagnostic branch only: a second address for the SAME frontend process.

    Each destination IP:port has its own ephemeral-port space, and the 20k
    cache-pressure stage opens ~30k connections against a 28,232-port range.
    When the frontend runs on this host (the client node), 127.0.0.1:8000
    reaches the same process over the same loopback path as the NIC address,
    so only the TCP destination differs; aiperf's round-robin URL choice is
    made once per session start. Falls back to the single URL unless the
    loopback endpoint answers /v1/models.
    """
    parsed = urllib.parse.urlsplit(url)
    host, port = parsed.hostname, parsed.port
    if not host or not port or host in {"127.0.0.1", "localhost"}:
        return []
    local = {"127.0.0.1", "localhost", socket.gethostname(), socket.getfqdn()}
    try:
        local.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    if host not in local:
        print(f"Frontend {host} is not this host; single URL", flush=True)
        return []
    loopback = f"http://127.0.0.1:{port}"
    check = subprocess.run(
        ["curl", "-sf", "-m", "10", f"{loopback}/v1/models"], capture_output=True, text=True
    )
    if check.returncode:
        print(f"Loopback preflight {loopback}/v1/models failed (rc={check.returncode}); single URL", flush=True)
        return []
    print(f"Frontend is local: adding {loopback} as a second URL for the same frontend", flush=True)
    return [loopback]


def server_url(env: Mapping[str, str]) -> str:
    """srt-slurm's frontend, else ``AIPERF_SERVER_URL``, else ``http://localhost:$PORT``."""
    host = inputs.optional("SRT_FRONTEND_HOST", env)
    if host:
        return f"http://{host}:{inputs.positive_int('SRT_FRONTEND_PORT', env)}"
    return inputs.optional("AIPERF_SERVER_URL", env) or (
        f"http://localhost:{inputs.positive_int('PORT', env)}"
    )


def metrics_urls(env: Mapping[str, str], path: str = "/metrics") -> tuple[str, ...]:
    """Every Prometheus endpoint AIPerf scrapes; it keeps ``endpoint_url`` per series.

    ``path`` is the workers' Prometheus route when the endpoints come from ``SRT_*``.
    """
    urls = inputs.optional("AIPERF_SERVER_METRICS_URLS", env)
    if not urls and env.get("SRTCTL_FRONTEND_TYPE") != "dynamo":
        prefill = env.get("SRT_PREFILL_ENDPOINTS")
        endpoints = env.get("SRT_AGG_ENDPOINTS") or (
            (f"{prefill}," if prefill else "") + env.get("SRT_DECODE_ENDPOINTS", "")
        )
        endpoints = endpoints.removesuffix(",")
        if endpoints:
            urls = ",".join(f"http://{e}{path}" if e else "" for e in endpoints.split(","))
    if not urls:
        return ()
    parts = urls.split(",")
    if parts[-1] == "":
        parts.pop()
    if not parts or any(not url or any(c.isspace() for c in url) for url in parts):
        raise inputs.InputError(
            "AIPERF_SERVER_METRICS_URLS must be a comma-separated list of non-empty URLs, "
            f"got {urls!r}"
        )
    return tuple(parts)


def replay_argv(cfg: ReplayConfig, cli: str | Path) -> list[str]:
    """The ``agentx profile`` argv for ``cfg``; ``cli`` is the ``agentx`` executable."""
    argv = [str(cli), "profile", "--url", cfg.url, *extra_frontend_urls(cfg.url)]
    argv += ["--model", cfg.model, "--tokenizer", cfg.tokenizer]
    argv += ["--concurrency", str(cfg.concurrency), "--benchmark-duration", str(cfg.duration)]
    if cfg.live_failed_request_threshold is not None:
        argv += ["--failed-request-threshold", cfg.live_failed_request_threshold]
    if cfg.warmup_requests_per_lane is not None:
        argv += ["--warmup-requests-per-lane", cfg.warmup_requests_per_lane]
    if cfg.warmup_grace_period is not None:
        argv += ["--warmup-grace-period", cfg.warmup_grace_period]
    if cfg.extra_inputs:
        argv += ["--extra-inputs", *cfg.extra_inputs]
    if cfg.dynamo_session_timeout is not None:
        argv += ["--use-dynamo-conv-aware-routing"]
        argv += ["--dynamo-session-timeout-seconds", cfg.dynamo_session_timeout]
    argv.append("--tokenizer-trust-remote-code")
    if cfg.max_context_length is not None:
        argv += ["--max-context-length", str(cfg.max_context_length)]
    if cfg.server_metrics_urls:
        argv += ["--server-metrics", *cfg.server_metrics_urls]
    argv += ["--output-artifact-dir", str(cfg.artifact_dir)]
    if cfg.unsafe_override:
        argv.append("--unsafe-override")
    argv += ["--public-dataset", cfg.loader]
    if cfg.apply_chat_template:
        argv.append("--apply-chat-template")
    if cfg.benchmark_grace_period is not None:
        argv += ["--benchmark-grace-period", cfg.benchmark_grace_period]
    return argv
