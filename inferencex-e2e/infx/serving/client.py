"""Benchmark client of one serving-YAML point, run in the engine container by bench.sh.

    python3 -m infx.serving.client prepare   # AgentX venv, file dataset, model weights
    python3 -m infx.serving.client run       # replay the scenario, write the result JSON

Both re-exec into the job's AgentX venv (upstream infx.bench.agentic.venv); the first call
builds it. Result JSONs come from the upstream infx.results CLIs, in the order
infx/bench/agentic/run.py runs them, or from infx.serving.sweep_result for a search.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.bench import proc
from infx.bench.agentic.venv import Runtime, bootstrap
from infx.serving import greenspec, specdec, speedbench

OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
DATASET_PATH = ".dataset_path"
FAILED_REQUEST_THRESHOLD = "0.10"
COUNT_SOURCE_MARKER = ".token_count"
SELECTION_DIR = "/tmp/greenspec-selection"  # noqa: S108 -- inside the job container; outside the result dir: request bodies are never uploaded
PRIVATE_LOGS = (
    ".greenspec-private-logs"  # beside the aggregate, on the node; not in any upload path
)
PROBE_BODY = {  # the shape agentx sends: streaming, usage requested, ignore_eos injected by the scenario
    "messages": [{"role": "user", "content": "Say OK."}],
    "stream": True,
    "stream_options": {"include_usage": True},
    "max_completion_tokens": 16,
    "ignore_eos": True,
}


def online(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the offline switches a serving YAML may set for its engine."""
    return {key: value for key, value in env.items() if key not in OFFLINE}


def datasets_cache(env: Mapping[str, str]) -> dict[str, str]:
    """Keep the datasets Arrow cache on the node, beside the hub cache, so later jobs skip the rebuild.

    The default (~/.cache in the container) dies with the job: every run re-read the 1.8 GB AgentX
    trace JSON and rewrote ~4.5 GB of Arrow, and a cold read on a busy /mnt stalled a run 13 min.
    """
    hub = env.get("HF_HUB_CACHE")
    if not hub or env.get("HF_DATASETS_CACHE"):
        return {}
    return {"HF_DATASETS_CACHE": f"{hub.rstrip('/')}/.datasets-arrow"}


def served_model(env: Mapping[str, str]) -> str:
    return env.get("SERVED_MODEL_NAME") or env["MODEL"]


def authorization(api_key: str) -> str:
    """A key with a space is a full header value ("Basic <b64>"): gateways that
    only accept scheme auth reject "Bearer Basic <b64>"."""
    return api_key if " " in api_key else f"Bearer {api_key}"


def agentx_argv(agentx: str, scenario: Path, env: Mapping[str, str]) -> list[str]:
    """The ``aiperf profile`` argv; ``agentx`` is the venv's ``aiperf`` executable.

    Kept the historical name: the argv (and ``benchmark_command.txt``) is the
    AgentX-shaped scenario replay; only the binary changed in the aiperf
    migration."""
    argv = [agentx, "profile", "--config", str(scenario)]
    argv += ["--url", env.get("SERVING_REMOTE_URL") or f"http://localhost:{env['PORT']}"]
    argv += ["--model", served_model(env)]
    argv += ["--tokenizer", env.get("SERVING_TOKENIZER") or env["MODEL"]]
    if key := env.get("SERVING_API_KEY"):
        if " " in key:
            argv += ["--header", f"Authorization:{key}"]
        else:
            argv += ["--api-key", key]
    if env.get("SERVING_SEARCH") != "1":
        argv += ["--concurrency", env["CONC"]]
    if telemetry := env.get("AIPERF_GPU_TELEMETRY_URL"):
        argv += ["--gpu-telemetry", telemetry]
    else:
        argv.append("--no-gpu-telemetry")
    if urls := env.get("AIPERF_SERVER_METRICS_URLS"):
        argv += ["--server-metrics", *urls.split(",")]
    argv += ["--output-artifact-dir", f"{env['RESULT_DIR']}/aiperf_artifacts"]
    return argv


def redacted(argv: list[str]) -> list[str]:
    """``argv`` with the --api-key value and Authorization header replaced, for
    benchmark_command.txt (an uploaded artifact)."""
    hidden = []
    for index, arg in enumerate(argv):
        if (index and argv[index - 1] == "--api-key") or (
            index and argv[index - 1] == "--header" and arg.lower().startswith("authorization:")
        ):
            hidden.append("***")
        else:
            hidden.append(arg)
    return hidden


class ProbeError(RuntimeError):
    """The remote endpoint refused the pre-flight request; the message names the fix."""


def _probe_hint(status: int, detail: str, has_key: bool, secret: str) -> str:
    if status in (401, 403):
        if has_key:
            return f"HTTP {status}: key rejected; check GitHub secret {secret or '(unnamed)'}"
        return f"HTTP {status}: the endpoint wants a key; set endpoint.api-key-secret"
    if status == 404:
        return f"HTTP 404: check endpoint.url and endpoint.model ({detail})"
    if status == 400:
        return (
            f"HTTP 400: {detail} -- the endpoint may reject ignore_eos or max_completion_tokens "
            "(see the serving-bench skill, remote endpoint)"
        )
    return f"HTTP {status}: {detail}"


def _error_detail(error: urllib.error.HTTPError) -> str:
    """The response body, unwrapped from an OpenAI-style {"error": {"message": ...}} when it is one."""
    detail = error.read(2000).decode(errors="replace").strip()
    try:
        parsed = json.loads(detail)["error"]
    except (ValueError, KeyError, TypeError):
        return detail
    return parsed.get("message", str(parsed)) if isinstance(parsed, dict) else str(parsed)


def _tls_context(verify: bool) -> ssl.SSLContext | None:
    """None verifies as usual; an unverified context serves endpoint.tls-verify: false
    (self-signed gateway certificates)."""
    if verify:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def probe(
    url: str, model: str, api_key: str, secret: str = "", timeout: float = 60.0, verify: bool = True
) -> str:
    """One streaming request shaped like agentx's: "server" if a chunk carries usage, else "client"."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = authorization(api_key)
    body = json.dumps({"model": model, **PROBE_BODY}).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)  # noqa: S310 -- http(s) only, checked by the schema
    try:
        with urllib.request.urlopen(  # noqa: S310 -- http(s) only, checked by the schema
            request, timeout=timeout, context=_tls_context(verify)
        ) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except ValueError:
                    continue
                if isinstance(chunk, dict) and chunk.get("usage"):
                    return "server"
    except urllib.error.HTTPError as error:
        raise ProbeError(
            _probe_hint(error.code, _error_detail(error), bool(api_key), secret)
        ) from None
    except (urllib.error.URLError, OSError) as error:
        raise ProbeError(f"{url} unreachable: {getattr(error, 'reason', error)}") from None
    return "client"


def with_token_source(scenario_yaml: str, source: str) -> str:
    """The scenario with server-side token counting off when the endpoint reports no usage."""
    if source != "client":
        return scenario_yaml
    import yaml  # in the AgentX venv (agentx-harness depends on PyYAML)

    doc = yaml.safe_load(scenario_yaml)
    doc["benchmark"]["endpoint"]["useServerTokenCount"] = False
    return yaml.safe_dump(doc, sort_keys=False)


def selection_family(env: Mapping[str, str]) -> tuple[Any, Any] | None:
    """The (module, Settings) of the selection family the job's serving YAML runs, else None.

    greenspec and speedbench replay a selection of a pinned file, one request per row, and report
    acceptance length. Only greenspec's file is internal: it alone goes through ``guard``.
    """
    for module in (greenspec, speedbench):
        if (chosen := module.settings(env)) is not None:
            return module, chosen
    return None


def with_requests(scenario_yaml: str, requests: int) -> str:
    """The scenario with its profiling phase set to ``requests`` requests (the size of the selection)."""
    import yaml  # in the AgentX venv

    doc = yaml.safe_load(scenario_yaml)
    [phase] = [p for p in doc["benchmark"]["phases"] if p.get("kind") == "profiling"]
    phase["requests"] = requests
    return yaml.safe_dump(doc, sort_keys=False)


def label_result(path: Path, label: str, token_count_source: str | None = None) -> None:
    """File the aggregate under the legacy scenario label result-report reads first."""
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result["scenario"] = label
    if token_count_source:
        result["token_count_source"] = token_count_source
    path.write_text(json.dumps(result, indent=2))


def prepare(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if env.get("SERVING_DATASET_KIND") == "file":
        from infx.serving.datasets import DatasetError, fetch

        try:
            path = fetch(env["SERVING_DATASET"])
            if found := selection_family(env):
                module, chosen = found
                selection = Path(SELECTION_DIR) / "selection.jsonl"
                count = module.write_selection(path, chosen, served_model(env), selection)
                print(
                    f"{module.TAG} selection {module.label(chosen)}: {count} requests", flush=True
                )
                path = selection
            (result_dir / DATASET_PATH).write_text(str(path))
        except (DatasetError, greenspec.GreenSpecError, speedbench.SpeedBenchError) as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 1
    if url := env.get("SERVING_REMOTE_URL"):
        try:
            source = probe(
                url,
                env["MODEL"],
                env.get("SERVING_API_KEY", ""),
                env.get("SERVING_API_KEY_SECRET", ""),
                verify=env.get("SERVING_TLS_VERIFY", "1") != "0",
            )
        except ProbeError as error:
            print(f"ERROR remote endpoint: {error}", file=sys.stderr)
            return 1
        (result_dir / COUNT_SOURCE_MARKER).write_text(source)
        print(f"Remote endpoint {url} answered; token counts from the {source}", flush=True)
        if source == "client":
            print(
                "WARNING remote endpoint streams no usage; ISL/OSL are counted client-side with "
                f"tokenizer {env.get('SERVING_TOKENIZER')} (approximate unless it is the model's own; "
                "hidden reasoning tokens are not counted)",
                file=sys.stderr,
            )
        return 0  # no weights to download; aiperf fetches an HF tokenizer itself
    return proc.call([str(runtime.hf), "download", env["MODEL"]], online(env))


def run(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    artifacts = result_dir / "aiperf_artifacts"
    output_dir = Path(env.get("AGENTIC_OUTPUT_DIR") or "/workspace")
    scenario = result_dir / "scenario.yaml"
    token_marker = result_dir / COUNT_SOURCE_MARKER
    source = token_marker.read_text().strip() if token_marker.exists() else None
    found = selection_family(env)
    chosen = found[1] if found else None
    internal = found is not None and found[0] is greenspec
    scenario_yaml = with_token_source(env["SERVING_SCENARIO_YAML"], source or "server")
    child = online(env) | {
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
        "RESULT_DIR": str(result_dir),
        "AGENTIC_OUTPUT_DIR": str(output_dir),
        "PYTHONPATH": proc.pythonpath(env),
        **datasets_cache(env),
    }
    if env.get("SERVING_TLS_VERIFY", "1") == "0":
        child["AIPERF_HTTP_SSL_VERIFY"] = "false"  # aiperf's own switch for self-signed gateways
    if (marker := result_dir / DATASET_PATH).exists():
        child["IX_CUSTOM_DATASET_FILE"] = marker.read_text().strip()
    if chosen:
        with open(child["IX_CUSTOM_DATASET_FILE"], encoding="utf-8") as selection:
            size = sum(1 for _ in selection)
        scenario_yaml = with_requests(scenario_yaml, min(chosen.requests or size, size))
    scenario.write_text(scenario_yaml)
    argv = agentx_argv(str(runtime.aiperf), scenario, env)
    (result_dir / "benchmark_command.txt").write_text(shlex.join(redacted(argv)) + "\n")
    metrics_urls = [url for url in env.get("AIPERF_SERVER_METRICS_URLS", "").split(",") if url]
    if chosen:
        specdec.save(result_dir / "spec_metrics_before.json", specdec.snapshot(metrics_urls))
    replay = greenspec.tee_filtered if internal else proc.tee
    replay_rc = replay(argv, result_dir / "benchmark.log", child)
    if chosen:
        specdec.save(result_dir / "spec_metrics_after.json", specdec.snapshot(metrics_urls))
    python = str(runtime.python)
    if env.get("SERVING_SEARCH") == "1":
        return replay_rc or proc.call([python, "-m", "infx.serving.sweep_result"], child)
    aggregate_rc = proc.call([python, "-m", "infx.results.agentic.process_agentic_result"], child)
    proc.call(
        [
            python,
            "-m",
            "infx.results.agentic.analyze_benchmark_distributions",
            str(artifacts),
            "-o",
            str(result_dir),
        ],
        child,
    )
    validate_rc = proc.call(
        [
            python,
            "-m",
            "infx.results.agentic.validate_agentic_result",
            str(artifacts),
            "--failed-request-threshold",
            FAILED_REQUEST_THRESHOLD,
        ],
        child,
    )
    aggregate = output_dir / f"{env['RESULT_FILENAME']}.json"
    label_result(aggregate, env["SERVING_SCENARIO_LABEL"], token_count_source=source)
    guard_rc = 0
    if chosen:
        spec = specdec.summarize(
            result_dir / "spec_metrics_before.json", result_dir / "spec_metrics_after.json"
        )
        if spec is None:
            print(
                "WARNING no speculative-decoding series on the engine metrics; the result has no spec_decode block",
                file=sys.stderr,
            )
        else:
            specdec.merge(aggregate, spec)
        if internal:
            guard_rc = guard(env, result_dir, aggregate, output_dir)
    return replay_rc or aggregate_rc or validate_rc or guard_rc


def guard(env: Mapping[str, str], result_dir: Path, aggregate: Path, output_dir: Path) -> int:
    """Prune, redact and scan what a greenspec job uploads (the benchmark repo is public); 1 on a problem."""
    from infx.serving.datasets import fetch

    shutil.rmtree(SELECTION_DIR, ignore_errors=True)
    problems = greenspec.guard(
        fetch(env["SERVING_DATASET"]),
        result_dir,
        [aggregate],
        [output_dir / "server.log"],
        keep_log_dir=output_dir / PRIVATE_LOGS,
    )
    for problem in problems:
        print(f"GreenSpec guard: {problem}", file=sys.stderr)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["prepare"], ["run"]):
        print("usage: python3 -m infx.serving.client prepare|run", file=sys.stderr)
        return 2
    runtime = Runtime.for_job(os.environ)
    if not runtime.active():
        if not runtime.python.exists():
            print(f"Preparing the AgentX runtime in {runtime.venv}", flush=True)
            if rc := bootstrap(runtime):
                return rc
        environ = {**os.environ, "PYTHONPATH": proc.pythonpath(), "PYTHONSAFEPATH": "1"}
        runtime.exec_python(["-m", "infx.serving.client", *argv], environ)
    return prepare(os.environ, runtime) if argv == ["prepare"] else run(os.environ, runtime)


if __name__ == "__main__":
    sys.exit(main())
