"""AgentX replay configuration: environment rules, AIPerf argv, and trace corpus choice."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from infx.bench.agentic import traces
from infx.bench.agentic.replay import ReplayConfig, replay_argv
from infx.bench.env import InputError

BASE_ENV = {
    "MODEL": "deepseek-ai/DeepSeek-V4-Pro",
    "MODEL_PREFIX": "test",
    "FRAMEWORK": "vllm",
    "CONC": "8",
    "DURATION": "3600",
    "PORT": "8000",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
    "AIPERF_EXPERIMENTAL_FAST": "0",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1",
}


def _argv(**overrides: str | None) -> list[str]:
    env = {name: value for name, value in {**BASE_ENV, **overrides}.items() if value is not None}
    return replay_argv(ReplayConfig.from_env(env, Path("/results")), "/venv/bin/aiperf")


def _value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_default_point_leaves_the_replay_policy_to_the_scenario():
    # A workflow MAX_MODEL_LEN is not the server's limit, so it never caps the replay.
    assert _argv(MAX_MODEL_LEN="131072") == [
        "/venv/bin/aiperf", "profile", "--scenario", "agentx", "--url", "http://localhost:8000",
        "--model", "deepseek-ai/DeepSeek-V4-Pro", "--tokenizer", "deepseek-ai/DeepSeek-V4-Pro",
        "--concurrency", "8", "--benchmark-duration", "3600", "--tokenizer-trust-remote-code",
        "--output-artifact-dir", "/results/aiperf_artifacts",
        "--public-dataset", traces.CAPPED,
    ]  # fmt: skip


def test_opt_in_inputs_add_their_flags_in_place():
    argv = _argv(
        SERVED_MODEL_NAME="DeepSeek-V4-Pro",
        FRAMEWORK="dynamo-vllm",
        AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="0.25",
        AGENTIC_WARMUP_GRACE_PERIOD="3600",
        AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS="14400",
        AIPERF_EXTRA_INPUTS="temperature:0.7 top_p:0.9",
        AIPERF_MAX_CONTEXT_LENGTH="262144",
        AIPERF_SERVER_METRICS_URLS="http://a:1/metrics,http://b:2/metrics,",
        WEKA_LOADER_OVERRIDE=traces.UNCAPPED,
        AIPERF_APPLY_CHAT_TEMPLATE="true",
        AIPERF_BENCHMARK_GRACE_PERIOD="1800",
    )

    assert argv == [
        "/venv/bin/aiperf", "profile", "--scenario", "agentx", "--url", "http://localhost:8000",
        "--model", "DeepSeek-V4-Pro", "--tokenizer", "deepseek-ai/DeepSeek-V4-Pro",
        "--concurrency", "8", "--benchmark-duration", "3600",
        "--failed-request-threshold", "0.25", "--warmup-grace-period", "3600",
        "--extra-inputs", "temperature:0.7", "top_p:0.9",
        "--use-dynamo-conv-aware-routing", "--dynamo-session-timeout-seconds", "14400",
        "--tokenizer-trust-remote-code",
        "--max-context-length", "262144",
        "--server-metrics", "http://a:1/metrics", "http://b:2/metrics",
        "--output-artifact-dir", "/results/aiperf_artifacts",
        "--public-dataset", traces.UNCAPPED,
        "--apply-chat-template", "--benchmark-grace-period", "1800",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("overrides", "duration", "warmup", "unsafe"),
    [
        ({"DURATION": "899"}, "899", None, True),
        ({"DURATION": "900"}, "900", None, False),
        # agentx-fast's 20-minute profile meets the scenario minimum the caller's 300 s misses.
        ({"AIPERF_EXPERIMENTAL_FAST": "1", "DURATION": "300"}, "1200", "1", False),
    ],
)
def test_profile_length_decides_the_unsafe_override(overrides, duration, warmup, unsafe):
    argv = _argv(**overrides)

    assert _value(argv, "--benchmark-duration") == duration
    assert _value(argv, "--warmup-requests-per-lane") == warmup
    assert ("--unsafe-override" in argv) is unsafe


@pytest.mark.parametrize(("conv_aware", "header_routing"), [("0", "false"), ("1", "true")])
def test_dynamo_opt_outs_disable_conversation_aware_routing(conv_aware, header_routing):
    argv = _argv(
        FRAMEWORK="dynamo-sglang",
        AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=conv_aware,
        AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=header_routing,
    )

    assert "--use-dynamo-conv-aware-routing" not in argv
    assert "--dynamo-session-timeout-seconds" not in argv


@pytest.mark.parametrize(
    ("overrides", "scraped"),
    [
        (
            {"SRT_PREFILL_ENDPOINTS": "p:1", "SRT_DECODE_ENDPOINTS": "d:2,d:3"},
            ["--server-metrics", "http://p:1/metrics", "http://d:2/metrics", "http://d:3/metrics"],
        ),
        (
            {"SRT_AGG_ENDPOINTS": "w:9,", "SRT_DECODE_ENDPOINTS": "d:2"},
            ["--server-metrics", "http://w:9/metrics"],
        ),
        (
            {"AIPERF_SERVER_METRICS_URLS": "http://x/metrics", "SRT_AGG_ENDPOINTS": "w:9"},
            ["--server-metrics", "http://x/metrics"],
        ),
        ({"SRTCTL_FRONTEND_TYPE": "dynamo", "SRT_AGG_ENDPOINTS": "w:9"}, []),
    ],
)
def test_server_metrics_come_from_the_caller_or_each_srt_worker(overrides, scraped):
    argv = _argv(**overrides)

    assert argv[argv.index("--tokenizer-trust-remote-code") + 1 : argv.index("--output-artifact-dir")] == scraped


@pytest.mark.parametrize(
    ("overrides", "url"),
    [
        (
            {"SRT_FRONTEND_HOST": "10.0.0.5", "SRT_FRONTEND_PORT": "8000",
             "AIPERF_SERVER_URL": "http://router:30000"},
            "http://10.0.0.5:8000",
        ),
        ({"AIPERF_SERVER_URL": "http://router:30000", "PORT": None}, "http://router:30000"),
    ],
)  # fmt: skip
def test_replay_targets_the_srt_frontend_then_the_explicit_url(overrides, url):
    assert _value(_argv(**overrides), "--url") == url


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SRT_FRONTEND_HOST": "10.0.0.5"}, "  - SRT_FRONTEND_PORT"),
        ({"PORT": None}, "  - PORT"),
        ({"AIPERF_MAX_CONTEXT_LENGTH": "256k"}, "AIPERF_MAX_CONTEXT_LENGTH must be a positive"),
        ({"AIPERF_SERVER_METRICS_URLS": "http://a/metrics,,http://b/metrics"}, "non-empty URLs"),
        ({"AIPERF_SERVER_METRICS_URLS": "http://a/metrics, http://b/metrics"}, "non-empty URLs"),
        ({"AIPERF_SERVER_METRICS_URLS": ","}, "non-empty URLs"),
        (
            {"WEKA_LOADER_OVERRIDE": "no_such_loader"},
            "unknown WEKA_LOADER_OVERRIDE='no_such_loader'",
        ),
    ],
)
def test_malformed_replay_inputs_are_rejected(overrides, message):
    with pytest.raises(InputError, match=re.escape(message)):
        _argv(**overrides)


@pytest.mark.parametrize(
    ("prefix", "override", "loader"),
    [
        # Families match by prefix, so a variant of a 1M-context family replays its corpus.
        ("big41flash", None, traces.UNCAPPED),
        ("other", None, traces.CAPPED),
        # A pin beats the family default, and the pre-download follows the pinned corpus.
        ("other", traces.UNCAPPED, traces.UNCAPPED),
    ],
)
def test_corpus_follows_the_model_family_context_unless_pinned(
    monkeypatch, prefix, override, loader
):
    monkeypatch.setattr(traces, "UNCAPPED_FAMILIES", ("big4",))

    assert traces.resolve(prefix, override) == (loader, traces.LOADERS[loader])
