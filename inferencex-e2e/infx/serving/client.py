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
import sys
from collections.abc import Mapping
from pathlib import Path

from infx.bench import proc
from infx.bench.agentic.venv import Runtime, bootstrap

OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
DATASET_PATH = ".dataset_path"
FAILED_REQUEST_THRESHOLD = "0.10"


def online(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the offline switches a serving YAML may set for its engine."""
    return {key: value for key, value in env.items() if key not in OFFLINE}


def agentx_argv(agentx: str, scenario: Path, env: Mapping[str, str]) -> list[str]:
    argv = [agentx, "profile", "--config", str(scenario)]
    argv += ["--url", f"http://localhost:{env['PORT']}"]
    argv += ["--model", env.get("SERVED_MODEL_NAME") or env["MODEL"], "--tokenizer", env["MODEL"]]
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


def label_result(path: Path, label: str) -> None:
    """File the aggregate under the legacy scenario label result-report reads first."""
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result["scenario"] = label
    path.write_text(json.dumps(result, indent=2))


def prepare(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if env.get("SERVING_DATASET_KIND") == "file":
        from infx.serving.datasets import DatasetError, fetch

        try:
            (result_dir / DATASET_PATH).write_text(str(fetch(env["SERVING_DATASET"])))
        except DatasetError as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 1
    return proc.call([str(runtime.hf), "download", env["MODEL"]], online(env))


def run(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    artifacts = result_dir / "aiperf_artifacts"
    output_dir = Path(env.get("AGENTIC_OUTPUT_DIR") or "/workspace")
    scenario = result_dir / "scenario.yaml"
    scenario.write_text(env["SERVING_SCENARIO_YAML"])
    child = online(env) | {
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        "RESULT_DIR": str(result_dir),
        "AGENTIC_OUTPUT_DIR": str(output_dir),
        "PYTHONPATH": proc.pythonpath(env),
    }
    if (marker := result_dir / DATASET_PATH).exists():
        child["IX_CUSTOM_DATASET_FILE"] = marker.read_text().strip()
    argv = agentx_argv(str(runtime.agentx), scenario, env)
    (result_dir / "benchmark_command.txt").write_text(shlex.join(argv) + "\n")
    replay_rc = proc.tee(argv, result_dir / "benchmark.log", child)
    python = str(runtime.python)
    if env.get("SERVING_SEARCH") == "1":
        return replay_rc or proc.call([python, "-m", "infx.serving.sweep_result"], child)
    aggregate_rc = proc.call([python, "-m", "infx.results.agentic.process_agentic_result"], child)
    proc.call([python, "-m", "infx.results.agentic.analyze_benchmark_distributions",
               str(artifacts), "-o", str(result_dir)], child)
    validate_rc = proc.call([python, "-m", "infx.results.agentic.validate_agentic_result", str(artifacts),
                             "--failed-request-threshold", FAILED_REQUEST_THRESHOLD], child)
    label_result(output_dir / f"{env['RESULT_FILENAME']}.json", env["SERVING_SCENARIO_LABEL"])
    return replay_rc or aggregate_rc or validate_rc


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
