"""The ``al-gate`` command against a fake deployment: an AIPerf stand-in and metrics endpoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.bench import al_gate, speedbench
from infx.bench.env import BenchError
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ["alpha", "beta gamma", "delta"]
# sha256sum of PROMPTS as one UTF-8 JSON array; they stand in for the golden prompts.
PROMPTS_SHA256 = "906f677089ff300758545039f784a7a09ded65f5686a7ac0491388156f2b4c2e"
CURVE = """\
testmodel:
  thinking_on:
    3: 3.2
collection:
  concurrency: 2
  renderer: chat_template
  modes:
    thinking_on:
      chat_template_kwargs: {enable_thinking: true}
      sampling: {temperature: 0.6, top_k: 20}
"""
METRICS = """\
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0"} %d
vllm:spec_decode_num_drafts_created{engine="0"} 1.7e+09
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0"} %d
"""
# Records its argv, writes AIPerf's aggregate export, and moves the engine's counters.
FAKE_AIPERF = r"""#!/bin/sh
printf '%s\n' "$@" > "$AIPERF_ARGV"
while [ $# -gt 0 ]; do
    [ "$1" = --output-artifact-dir ] && out="$2"
    shift
done
mkdir -p "$out"
printf '{"request_count": {"avg": 3}, "error_request_count": {"avg": %s}}\n' "$FAILED" \
    > "$out/profile_export_aiperf.json"
cp "$COUNTERS_AFTER" "$COUNTERS"
"""


class Tokenizer:
    """Renders the user turn between role tags and records each call's options."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **options):
        self.calls.append(options)
        return f"<user> {messages[0]['content']} <assistant>"


@pytest.fixture
def gate(tmp_path, http_server, monkeypatch):
    """A served engine whose spec-decode counters the replay moves, and the gate's inputs."""
    counters = tmp_path / "counters.txt"
    counters.write_text(METRICS % (100, 160))
    url = http_server(lambda method, path: (200, counters.read_bytes()))
    (tmp_path / "golden").mkdir()
    (tmp_path / "golden" / "testmodel_mtp.yaml").write_text(CURVE)
    dataset = tmp_path / "speedbench_coding.jsonl"
    rows = [{"question_id": f"q{index}", "prompt": text} for index, text in enumerate(PROMPTS)]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tokenizer = Tokenizer()
    monkeypatch.setattr(speedbench, "PROMPTS_SHA256", PROMPTS_SHA256)
    monkeypatch.setattr(al_gate, "SETTLE_INTERVAL_S", 0.01)
    monkeypatch.setattr(al_gate, "_tokenizer", {"test/model": tokenizer}.get)
    runtime = SimpleNamespace(
        active=lambda: True, aiperf=executable(tmp_path / "aiperf", FAKE_AIPERF)
    )
    env = {
        "AL_GATE": "true", "AL_GATE_CURVE": "testmodel_mtp", "AL_GATE_DRAFT_TOKENS": "3",
        "AL_GATE_DATASET": str(dataset), "AL_GATE_OUTPUT_DIR": str(tmp_path / "logs"),
        "THINKING_MODE": "thinking_on", "RESULT_FILENAME": "gate", "MODEL": "test/model",
        "SERVED_MODEL_NAME": "served/model", "FRAMEWORK": "vllm",
        "AIPERF_SERVER_URL": "http://frontend:8000", "AIPERF_SERVER_METRICS_URLS": f"{url}/metrics",
        "PATH": os.environ["PATH"], "AIPERF_ARGV": str(tmp_path / "argv"), "FAILED": "0",
        "COUNTERS": str(counters), "COUNTERS_AFTER": str(tmp_path / "after.txt"),
    }  # fmt: skip

    def run(after: tuple[int, int] = (1100, 2160), **overrides: str) -> int:
        (tmp_path / "after.txt").write_text(METRICS % after)
        return al_gate.run({**env, **overrides}, runtime, golden_dir=tmp_path / "golden")

    def document() -> dict[str, object]:
        return json.loads((tmp_path / "logs" / "al_gate_gate.json").read_text())

    raw, argv = tmp_path / "logs" / "al_gate", tmp_path / "argv"
    return SimpleNamespace(
        run=run, document=document, url=url, tokenizer=tokenizer, raw=raw, argv=argv
    )


@pytest.mark.parametrize(
    ("after", "verdict"),
    [
        # 2160 - 160 accepted drafts over 1100 - 100 verifications: AL 3.0, 0.9375 x golden 3.2.
        ((1100, 2160), {"acceptance_length": 3.0, "ratio": 0.9375, "status": "below",
                        "accepted_tokens": 2000, "verifications": 1000}),
        # 2360 - 160 accepted drafts over 1000 verifications: AL 3.2.
        ((1100, 2360), {"acceptance_length": 3.2, "ratio": 1.0, "status": "pass",
                        "accepted_tokens": 2200, "verifications": 1000}),
    ],
    ids=["below", "pass"],
)  # fmt: skip
def test_the_gate_replays_the_golden_collection_and_reports_the_verdict(gate, after, verdict):
    assert gate.run(after) == 0

    assert gate.document() == {
        "result_filename": "gate", "framework": "vllm", "curve": "testmodel_mtp",
        "thinking_mode": "thinking_on", "draft_tokens": 3, "golden_acceptance_length": 3.2,
        "error": None, "report_only": True, **verdict,
    }  # fmt: skip
    # In vllm bench serve's order (random.seed(0); random.shuffle), rendered client-side.
    sent = [json.loads(line) for line in (gate.raw / "prompts.jsonl").read_text().splitlines()]
    assert sent == [
        {"text": f"<user> {prompt} <assistant>", "output_length": 4096}
        for prompt in ("alpha", "delta", "beta gamma")
    ]
    options = {"add_generation_prompt": True, "tokenize": False, "enable_thinking": True}
    assert gate.tokenizer.calls == [options] * 3
    assert gate.argv.read_text().splitlines() == [
        "profile", "--endpoint-type", "completions", "--streaming",
        "--url", "http://frontend:8000", "--model", "served/model",
        "--tokenizer", "test/model", "--tokenizer-trust-remote-code",
        "--custom-dataset-type", "single_turn", "--input-file", str(gate.raw / "prompts.jsonl"),
        "--dataset-sampling-strategy", "sequential", "--request-count", "3", "--concurrency", "2",
        "--extra-inputs", '{"temperature": 0.6, "top_k": 20}',
        "--server-metrics", f"{gate.url}/metrics",
        "--output-artifact-dir", str(gate.raw / "aiperf_artifacts"),
    ]  # fmt: skip
    for when, counters in (("before", (100, 160)), ("after", after)):
        scrapes = json.loads((gate.raw / f"metrics_{when}.json").read_text())
        assert scrapes == {f"{gate.url}/metrics": METRICS % counters}


def test_failed_requests_make_the_measurement_an_error(gate):
    with pytest.raises(BenchError, match="1 of 3 requests failed"):
        gate.run(FAILED="1")

    document = gate.document()
    assert (document["status"], document["error"]) == ("error", "1 of 3 requests failed")


@pytest.mark.parametrize(("framework", "replayed"), [("sglang", True), ("tilert", False)])
def test_without_the_generation_engines_counters_the_verdict_is_no_counters(
    gate, framework, replayed
):
    # The served engine exports only vLLM's families; TileRT's engine has no counter reader.
    assert gate.run(FRAMEWORK=framework) == 0

    document = gate.document()
    assert (document["status"], document["verifications"]) == ("no-counters", None)
    assert (gate.raw / "prompts.jsonl").exists() == replayed


def test_prompts_other_than_the_golden_ones_are_an_error_before_any_traffic(gate, monkeypatch):
    monkeypatch.setattr(speedbench, "PROMPTS_SHA256", "0" * 64)

    with pytest.raises(BenchError, match=f"has sha256 {PROMPTS_SHA256}, not 0+$"):
        gate.run()

    document = gate.document()
    assert (document["status"], document["golden_acceptance_length"]) == ("error", None)
    assert PROMPTS_SHA256 in document["error"]
    assert not gate.raw.exists()


BOS, USER, ASSISTANT = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"
# DeepSeek-V4's system prefix for reasoning effort "max", from vLLM's deepseek_v4 encoder.
MAX_EFFORT = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
    "edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate "
    "step, considered alternative, and rejected hypothesis to ensure absolutely no assumption "
    "is left unchecked.\n\n"
)


@pytest.mark.parametrize(
    ("kwargs", "effort", "opening"),
    [
        ({"thinking": True, "reasoning_effort": "high"}, "", "<think>"),
        # vLLM v0.21-v0.25 renders chat mode when no kwargs are sent.
        ({}, "", "</think>"),
        ({"thinking": True, "reasoning_effort": "none"}, "", "</think>"),
        ({"enable_thinking": True, "reasoning_effort": "xhigh"}, MAX_EFFORT, "<think>"),
    ],
)
def test_deepseek_v4_prompts_render_as_vllms_tokenizer_mode_did(kwargs, effort, opening):
    prompt = al_gate.render("Fix it.", "deepseek_v4", kwargs, tokenizer=None)

    assert prompt == f"{BOS}{effort}{USER}Fix it.{ASSISTANT}{opening}"


def test_the_srt_shim_records_a_gate_whose_inputs_are_missing(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    # The serving image's python3: the inputs are checked before the client venv exists.
    (tools / "python3").symlink_to(sys.executable)
    env = {
        "PATH": f"{tools}:/usr/bin:/bin", "AL_GATE": "true", "AL_GATE_CURVE": "testmodel_mtp",
        "AL_GATE_DRAFT_TOKENS": "3", "AL_GATE_OUTPUT_DIR": str(tmp_path), "RESULT_FILENAME": "gate",
    }  # fmt: skip

    run = subprocess.run(
        ["bash", str(REPO_ROOT / "benchmarks" / "srt_al_gate.sh")],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip

    missing = "  - AL_GATE_DATASET\n  - THINKING_MODE\n  - MODEL\n  - FRAMEWORK"
    assert (run.returncode, missing in run.stderr) == (1, True)
    document = json.loads((tmp_path / "al_gate_gate.json").read_text())
    assert (document["curve"], document["draft_tokens"], document["status"]) == (
        "testmodel_mtp", 3, "error",
    )  # fmt: skip
    assert missing in document["error"]
