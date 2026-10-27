"""``al-gate``: replay a curve's golden SPEED-Bench collection, then judge the realized AL.

The generation engine's own Prometheus counters, scraped before and after the replay, give
``AL = 1 + accepted draft tokens / verification steps``; the verdict compares it with the
curve's golden AL. Report-only: every measured verdict exits 0. A measurement that could not
run exits non-zero, and its result document records the error.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.bench import env as inputs, proc, speedbench
from infx.bench.agentic import replay
from infx.bench.agentic.venv import Runtime, bootstrap
from infx.bench.env import BenchError, InputError

REQUIRED = (
    "AL_GATE", "AL_GATE_CURVE", "AL_GATE_DRAFT_TOKENS", "AL_GATE_DATASET", "AL_GATE_OUTPUT_DIR",
    "THINKING_MODE", "RESULT_FILENAME", "MODEL", "FRAMEWORK",
)  # fmt: skip
# Every golden collection (vllm bench serve) streamed each prompt once to /v1/completions with
# max_tokens 4096 and ignore_eos false, the engines' default, so it is not sent.
MAX_TOKENS = 4096
MIN_RATIO, MAX_RATIO = 0.95, 1.05
# AIPerf returns before the engines count the last verifications: scrape until unchanged.
SETTLE_INTERVAL_S, SETTLE_CAP_S = 2.0, 60.0


def main(argv: list[str]) -> int:
    """Run the ``al-gate`` command."""
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench al-gate", description=__doc__)
    parser.parse_args(argv)
    return run(os.environ, Runtime.for_job(os.environ))


def run(env: Mapping[str, str], runtime: Runtime, golden_dir: Path | None = None) -> int:
    """Measure the gate ``env`` describes in ``runtime``'s venv, which it prepares first."""
    tokens = env.get("AL_GATE_DRAFT_TOKENS", "")
    document: dict[str, Any] = {
        "result_filename": env.get("RESULT_FILENAME"), "framework": env.get("FRAMEWORK"),
        "curve": env.get("AL_GATE_CURVE"), "thinking_mode": env.get("THINKING_MODE"),
        "draft_tokens": int(tokens) if tokens.isdecimal() else None,
        "golden_acceptance_length": None, "acceptance_length": None, "ratio": None,
        "status": "error", "error": None, "accepted_tokens": None, "verifications": None,
        "report_only": True,
    }  # fmt: skip
    try:
        rc = _measure(env, runtime, document, golden_dir)
    except Exception as error:
        document["error"] = str(error) or type(error).__name__
        _write(env, document)
        raise
    _write(env, document)
    print(f"AL gate result (report-only): {json.dumps(document)}", flush=True)
    return rc


def _measure(
    env: Mapping[str, str], runtime: Runtime, document: dict[str, Any], golden_dir: Path | None
) -> int:
    values = inputs.require(*REQUIRED, env=env)
    if values["AL_GATE"] != "true":
        raise InputError("AL_GATE must be true to run the acceptance-length gate")
    tokens = inputs.parse_positive_int("AL_GATE_DRAFT_TOKENS", values["AL_GATE_DRAFT_TOKENS"])
    server = replay.server_url(env)
    # trtllm-serve serves Prometheus text at /prometheus/metrics; its /metrics is JSON.
    trtllm = env.get("SRTCTL_FRONTEND_TYPE") == "trtllm_serve"
    metrics_urls = replay.metrics_urls(env, "/prometheus/metrics" if trtllm else "/metrics")
    if not metrics_urls:
        raise InputError("no metrics endpoint: set AIPERF_SERVER_METRICS_URLS or SRT_*_ENDPOINTS")
    # Relative to the workspace root, /infmax-workspace in containers.
    dataset = proc.REPO_ROOT / values["AL_GATE_DATASET"]
    prompts = speedbench.load(dataset)
    if (actual := speedbench.digest(prompts)) != speedbench.PROMPTS_SHA256:
        raise BenchError(f"{dataset} has sha256 {actual}, not {speedbench.PROMPTS_SHA256}")
    if not runtime.active():
        print(f"Preparing the AIPerf runtime in {runtime.venv}", flush=True)
        rc = bootstrap(runtime)
        if rc:
            document["error"] = f"creating the AIPerf runtime exited with code {rc}"
            return rc
        # The venv interpreter needs the same safe sys.path as this one had.
        environ = {**env, "PYTHONPATH": proc.pythonpath(), "PYTHONSAFEPATH": "1"}
        runtime.exec_python(["-m", "infx.bench", "al-gate"], environ)

    # Client venv only: PyYAML for the curves, Python 3.11 for infx.results and infx.srt_slurm.
    from infx.golden_al_distribution import GOLDEN_DIR, load_curve
    from infx.results.agentic.spec_decode import READERS, increase
    from infx.srt_slurm.synthetic_acceptance import ENGINES

    thinking = values["THINKING_MODE"]
    curve = load_curve(values["AL_GATE_CURVE"], golden_dir or GOLDEN_DIR)
    document["golden_acceptance_length"] = golden = curve.acceptance(thinking, tokens)
    collection = curve.collection or {}
    mode = collection.get("modes", {}).get(thinking)
    if mode is None:
        raise BenchError(f"golden curve {curve.name} records no {thinking} collection to replay")
    # Only the generation engine's own counters measure its acceptance.
    reader = READERS.get(ENGINES.get(values["FRAMEWORK"]))
    if reader is None:
        document["status"] = "no-counters"
        return 0

    renderer, kwargs = collection["renderer"], mode["chat_template_kwargs"] or {}
    tokenizer = _tokenizer(values["MODEL"]) if renderer == "chat_template" else None
    # vllm bench serve's SPEED-Bench loader order: random.seed(0); random.shuffle(prompts).
    random.Random(0).shuffle(prompts)  # noqa: S311
    raw = Path(values["AL_GATE_OUTPUT_DIR"]) / "al_gate"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "prompts.jsonl").open("w", encoding="utf-8") as sink:
        for prompt in prompts:
            row = {"text": render(prompt, renderer, kwargs, tokenizer), "output_length": MAX_TOKENS}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    argv = [
        str(runtime.aiperf), "profile", "--endpoint-type", "completions", "--streaming",
        "--url", server, "--model", env.get("SERVED_MODEL_NAME") or values["MODEL"],
        "--tokenizer", values["MODEL"], "--tokenizer-trust-remote-code",
        "--custom-dataset-type", "single_turn", "--input-file", str(raw / "prompts.jsonl"),
        "--dataset-sampling-strategy", "sequential", "--request-count", str(len(prompts)),
        "--concurrency", str(collection["concurrency"]),
        # JSON keeps each value's type; AIPerf guesses the types of key:value pairs.
        "--extra-inputs", json.dumps(mode["sampling"]),
        "--server-metrics", *metrics_urls, "--output-artifact-dir", str(raw / "aiperf_artifacts"),
    ]  # fmt: skip
    before = {url: _scrape(url) for url in metrics_urls}
    (raw / "metrics_before.json").write_text(json.dumps(before, indent=2))
    rc = proc.tee(argv, raw / "aiperf.log", dict(env))
    if rc:
        raise BenchError(f"aiperf exited with code {rc}")
    export = json.loads((raw / "aiperf_artifacts" / "profile_export_aiperf.json").read_text())
    if failed := export.get("error_request_count", {}).get("avg"):
        raise BenchError(f"{failed:g} of {len(prompts)} requests failed")
    deadline, previous = time.monotonic() + SETTLE_CAP_S, object()
    while True:
        after = {url: _scrape(url) for url in metrics_urls}
        counted = reader(increase(before, after))
        if counted == previous or time.monotonic() >= deadline:
            break
        previous = counted
        time.sleep(SETTLE_INTERVAL_S)
    (raw / "metrics_after.json").write_text(json.dumps(after, indent=2))
    accepted, verifications = counted or (None, None)
    document.update(accepted_tokens=accepted, verifications=verifications)
    if not verifications:
        document["status"] = "no-counters" if counted is None else "no-drafting"
        return 0
    length = 1 + accepted / verifications
    ratio = round(length / golden, 4)
    status = "below" if ratio < MIN_RATIO else "pass" if ratio <= MAX_RATIO else "above"
    document.update(acceptance_length=round(length, 4), ratio=ratio, status=status)
    return 0


def render(prompt: str, renderer: str, kwargs: Mapping[str, Any], tokenizer: Any) -> str:
    """One user turn rendered client-side, as the golden collector rendered it."""
    turn = [{"role": "user", "content": prompt}]
    if renderer == "deepseek_v4":
        from infx.bench_serving.encoding_dsv4 import encode_messages

        return encode_messages(turn, **dsv4_options(kwargs))
    if renderer != "chat_template":
        raise BenchError(f"renderer {renderer!r} is neither chat_template nor deepseek_v4")
    return tokenizer.apply_chat_template(turn, add_generation_prompt=True, tokenize=False, **kwargs)


def dsv4_options(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """``encode_messages`` options for chat_template_kwargs, as vLLM v0.21-v0.25 map them.

    Thinking is off unless ``thinking`` or ``enable_thinking`` is truthy; reasoning effort
    ``none`` turns it off, ``max``/``xhigh`` mean ``max``, any other string ``high``.
    """
    mode = "thinking" if kwargs.get("thinking") or kwargs.get("enable_thinking") else "chat"
    effort = kwargs.get("reasoning_effort")
    if not isinstance(effort, str):
        effort = None
    elif effort == "none":
        mode, effort = "chat", None
    else:
        effort = "max" if effort in ("max", "xhigh") else "high"
    drop = kwargs.get("drop_thinking", True)
    return {"thinking_mode": mode, "drop_thinking": drop, "reasoning_effort": effort}


def _tokenizer(model: str) -> Any:
    """The served model's Hugging Face tokenizer; the client venv provides transformers."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _scrape(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except OSError as error:
        raise BenchError(f"GET {url} failed: {error}") from error


def _write(env: Mapping[str, str], document: Mapping[str, Any]) -> None:
    """``<AL_GATE_OUTPUT_DIR>/al_gate_<RESULT_FILENAME>.json``, where the launcher looks."""
    if env.get("AL_GATE_OUTPUT_DIR") and env.get("RESULT_FILENAME"):
        path = Path(env["AL_GATE_OUTPUT_DIR"]) / f"al_gate_{env['RESULT_FILENAME']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n")
