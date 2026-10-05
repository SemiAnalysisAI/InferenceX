"""The AIPerf ``profile`` argv for one AgentX point."""

from __future__ import annotations

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

# The scenario rejects shorter profiles; smoke runs below it pass --unsafe-override, which
# marks them submission_valid=false.
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
    """Hugging Face id; a wire name is not necessarily a valid repo id."""
    concurrency: int
    duration: int
    # None keeps the agentx scenario's default.
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
        # Dynamo routes later turns to their prefix's prefill worker via nvext.session_control;
        # builds after #9920 reject that field, so their recipes route by header or opt out.
        conv_aware = (
            values["FRAMEWORK"].startswith("dynamo-")
            and values["AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING"] != "0"
            and values["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"] != "true"
        )
        # Replays keep the model's native context: an inherited MAX_MODEL_LEN is a workflow
        # value, so only this opt-in sets a smaller service limit.
        max_context = inputs.optional("AIPERF_MAX_CONTEXT_LENGTH", env)
        loader, dataset = traces.resolve(
            values["MODEL_PREFIX"], inputs.optional("WEKA_LOADER_OVERRIDE", env)
        )
        extra_inputs = inputs.optional("AIPERF_EXTRA_INPUTS", env) or ""
        return cls(
            url=_server_url(env),
            model=inputs.optional("SERVED_MODEL_NAME", env) or values["MODEL"],
            tokenizer=values["MODEL"],
            concurrency=inputs.parse_positive_int("CONC", values["CONC"]),
            duration=duration,
            warmup_requests_per_lane=warmup,
            # Recipes with correlated low-concurrency trajectories loosen the live abort.
            live_failed_request_threshold=inputs.optional(
                "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD", env
            ),
            # Saturation arms with a larger in-flight set need longer to drain warmup.
            warmup_grace_period=inputs.optional("AGENTIC_WARMUP_GRACE_PERIOD", env),
            # One ``key:value`` pair per word, as AIPerf's multi-value flag takes them.
            extra_inputs=tuple(extra_inputs.split()),
            dynamo_session_timeout=(
                values["AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS"] if conv_aware else None
            ),
            max_context_length=(
                inputs.parse_positive_int("AIPERF_MAX_CONTEXT_LENGTH", max_context)
                if max_context
                else None
            ),
            server_metrics_urls=_server_metrics_urls(env),
            artifact_dir=result_dir / "aiperf_artifacts",
            unsafe_override=duration < MIN_SCENARIO_DURATION_S,
            loader=loader,
            dataset=dataset,
            # Recipes whose legacy launch rendered prompts client-side keep doing so.
            apply_chat_template=env.get("AIPERF_APPLY_CHAT_TEMPLATE") == "true",
            # Lets this point's admitted responses finish before the deployment is torn down.
            benchmark_grace_period=inputs.optional("AIPERF_BENCHMARK_GRACE_PERIOD", env),
        )


def _server_url(env: Mapping[str, str]) -> str:
    """srt-slurm's frontend, else ``AIPERF_SERVER_URL``, else ``http://localhost:$PORT``."""
    host = inputs.optional("SRT_FRONTEND_HOST", env)
    if host:
        return f"http://{host}:{inputs.positive_int('SRT_FRONTEND_PORT', env)}"
    return inputs.optional("AIPERF_SERVER_URL", env) or (
        f"http://localhost:{inputs.positive_int('PORT', env)}"
    )


def _server_metrics_urls(env: Mapping[str, str]) -> tuple[str, ...]:
    """Every Prometheus endpoint AIPerf scrapes; it keeps ``endpoint_url`` per series."""
    urls = inputs.optional("AIPERF_SERVER_METRICS_URLS", env)
    if not urls and env.get("SRTCTL_FRONTEND_TYPE") != "dynamo":
        # A router frontend does not re-export engine metrics; read each worker's.
        prefill = env.get("SRT_PREFILL_ENDPOINTS")
        endpoints = env.get("SRT_AGG_ENDPOINTS") or (
            (f"{prefill}," if prefill else "") + env.get("SRT_DECODE_ENDPOINTS", "")
        )
        endpoints = endpoints.removesuffix(",")
        if endpoints:
            urls = ",".join(f"http://{e}/metrics" if e else "" for e in endpoints.split(","))
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
    """The ``aiperf profile`` argv for ``cfg``; ``cli`` is the ``aiperf`` executable."""
    # The agentx scenario owns the endpoint, streaming, seed, start ratios, warmup and failure
    # defaults, trace idle cap, token counts, telemetry, dataset size, and metric slices.
    argv = [str(cli), "profile", "--scenario", "agentx", "--url", cfg.url]
    argv += ["--model", cfg.model, "--tokenizer", cfg.tokenizer]
    argv += ["--concurrency", str(cfg.concurrency), "--benchmark-duration", str(cfg.duration)]
    if cfg.live_failed_request_threshold is not None:
        # Live abort only; run.FAILED_REQUEST_THRESHOLD gates the finished profile.
        argv += ["--failed-request-threshold", cfg.live_failed_request_threshold]
    if cfg.warmup_requests_per_lane is not None:
        argv += ["--warmup-requests-per-lane", cfg.warmup_requests_per_lane]
    if cfg.warmup_grace_period is not None:
        argv += ["--warmup-grace-period", cfg.warmup_grace_period]
    if cfg.extra_inputs:
        argv += ["--extra-inputs", *cfg.extra_inputs]
    if cfg.dynamo_session_timeout is not None:
        # The router's inactivity lease; the upstream 300 s is shorter than an overloaded request.
        argv += ["--use-dynamo-conv-aware-routing"]
        argv += ["--dynamo-session-timeout-seconds", cfg.dynamo_session_timeout]
    # The dataset manager loads the tokenizer anyway, and Kimi ships custom tokenizer code.
    argv.append("--tokenizer-trust-remote-code")
    if cfg.max_context_length is not None:
        # Longer traces would be deterministic 4xxs that still pressure the engine while queued.
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
