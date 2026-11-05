"""The serve driver on the fake backend.

The real binder, driver and ``infx.bench serve`` run; only the server and the benchmark and eval
clients are stubs on the container PATH. PYTHONSAFEPATH makes ``infx.bench`` import from the
checkout only, and ``infx.bench`` clears it for the processes it starts.
"""

import json
import shlex
import sys
import tarfile
from functools import partial
from pathlib import Path

import pytest
import yaml

from infx.clusters import SCHEDULERS, load_inventory
from infx.golden_al_distribution.identity import FACTS
from infx.launch import drivers
from infx.launch.__main__ import launch
from infx.launch.backends import BACKENDS
from infx.launch.backends.base import BackendError
from infx.launch.context import LaunchError
from infx.launch.drivers import serve
from infx.launch.event import JobEventBuilder
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest
from infx.srt_slurm import al_gate
from infx.srt_slurm.synthetic_acceptance import acceptance_length
from infx.tests.launch.fake_backend import FakeBackend, FakeSettings, containers_dir
from infx.tests.processes import free_port, gone

ROOT = Path(__file__).resolve().parents[3]
RECIPE = "benchmarks/single_node/srt-slurm-recipes/m/llamacpp/m5/agentic.yaml"
SHARED_AGENTIC = {
    "benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/srt_agentic.sh"}
}
NATIVE_TELEMETRY = {"telemetry": {"enabled": True, "storage_subdir": "power"}}
FRAGMENT = {
    "schema": 2,
    "name": "m-gguf-m5",
    "frontend": {"type": "llamacpp"},
    "engine": "llamacpp",
    "health_check": {"interval_seconds": 10, "max_attempts": 60},
    "environment": {"LLAMA_ONLY": "recipe", "BENCH_KNOB": "recipe"},
    "roles": {"agg": {
        "nodes": 1, "workers": 1, "gpus": 1,
        "args": {"model": "Q4/M-00001-of-00002.gguf", "parallel": 4, "ctx_size": 36864,
                 "no-kv-unified": True, "no-mmap": False, "lazy-mode": "on", "flash-attn": "on"},
        "env": {"LLAMA_ONLY": "1", "BENCH_KNOB": "role", "DROPPED": None},
    }},
    "benchmark": {"env": {"BENCH_KNOB": "7", "TOKENIZER": "org/M"}},
}  # fmt: skip

SERVER = r"""#!{python}
import http.server, json, os, signal, sys
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
names = ("LLAMA_ONLY", "DROPPED", "BENCH_KNOB", "HF_HUB_CACHE", "CLUSTER_KNOB", "HF_TOKEN",
         "PYTHONPATH", "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE")
with open(os.environ["FAKE_LOG"] + ".server", "w") as out:
    json.dump({{"argv": argv, "env": {{n: os.environ.get(n) for n in names}}}}, out)
print("engine up", flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ready = self.path == "/health" and not os.environ.get("FAKE_NEVER_READY")
        self.send_response(200 if ready else 503)
        self.end_headers()
    def log_message(self, *_):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", port), Health).serve_forever()
"""

CLIENT = r"""#!{python}
import json, os, sys, time
argv = sys.argv[1:]
names = ("LLAMA_ONLY", "BENCH_KNOB", "RESULT_DIR", "USE_CHAT_TEMPLATE", "MODEL", "RESULT_FILENAME",
         "PYTHONPATH", "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE")
gate = argv[:2] == ["-m", "infx.bench.al_gate_stub"]
if gate:
    names = ("THINKING_MODE", *(n for n in os.environ if n.startswith("AL_GATE")))
with open(os.environ["FAKE_LOG"] + ".clients", "a") as out:
    out.write(json.dumps({{"argv": argv, "env": {{n: os.environ.get(n) for n in names}}}}) + "\n")
if gate:
    verdict = os.path.join(os.environ["AL_GATE_OUTPUT_DIR"], "al_gate_m_point.json")
    with open(verdict, "w") as out:
        out.write('{{"status": "pass"}}')
    raw = os.path.join(os.environ["AL_GATE_OUTPUT_DIR"], "al_gate")
    os.makedirs(raw)
    with open(os.path.join(raw, "aiperf.log"), "w") as out:
        out.write("replayed")
    sys.exit(int(os.environ.get("FAKE_GATE_RC", "0")))
if argv[:2] == ["-m", "infx.bench.eval_stub"]:
    stage = argv[argv.index("--stage-to") + 1]
    for name in ("results_gsm8k.json", "meta_env.json"):
        with open(os.path.join(stage, name), "w") as out:
            out.write("{{}}")
    sys.exit(0)
names = ("RESULT_DIR", "AGENTIC_OUTPUT_DIR", "AIPERF_SERVER_URL", "AIPERF_SERVER_METRICS_URLS",
         "ENABLE_AGENTX_POWER", "CONC", "RUNNER_TYPE", "GITHUB_RUN_ATTEMPT", "GITHUB_TOKEN")
seen = {{n: os.environ.get(n) for n in names}}
log = os.path.join(os.environ["RESULT_DIR"], "server.log")
seen["server.log"] = open(log).read() if os.path.isfile(log) else None
with open(os.environ["FAKE_LOG"] + ".agentic", "w") as out:
    json.dump(seen, out)
rc = int(os.environ.get("FAKE_CLIENT_RC", "0"))
if rc:
    sys.exit(rc)
artifacts = os.path.join(os.environ["RESULT_DIR"], "aiperf_artifacts")
os.makedirs(artifacts)
with open(os.path.join(artifacts, "profile_export.json"), "w") as out:
    out.write("{{}}")
if not os.environ.get("FAKE_NO_RESULT"):
    with open(os.path.join(os.environ["AGENTIC_OUTPUT_DIR"], os.environ["RESULT_FILENAME"] + ".json"), "w") as out:
        out.write("{{}}")
window_dir = os.environ.get("SRT_MEASUREMENT_WINDOW_DIR")
if window_dir and not os.environ.get("FAKE_SUDO_RC"):
    samples = os.path.join(os.path.dirname(window_dir), "samples.csv")
    rows = []
    for _ in range(1500):
        time.sleep(0.02)
        rows = open(samples).read().splitlines() if os.path.exists(samples) else []
        if len(rows) >= 4:
            break
    start, end = float(rows[1].split(",")[0]), float(rows[3].split(",")[0])
    with open(os.path.join(os.environ["RESULT_DIR"], "agentic_power_concurrency_1.json"), "w") as out:
        json.dump({{"benchmark_start_time_unix": start, "benchmark_end_time_unix": end,
                    "duration": end - start, "completed": 2, "total_input_tokens": 100,
                    "total_output_tokens": 10}}, out)
"""

SUDO = r"""#!{python}
import os, plistlib, sys, time
if os.environ.get("FAKE_SUDO_RC"):
    sys.exit("sudo: a password is required")
with open(os.environ["FAKE_LOG"] + ".sudo", "w") as out:
    out.write(str(os.getpid()))
sample = plistlib.dumps({{"processor": {{"combined_power": 20000.0, "cpu_power": 5000.0,
                                        "gpu_power": 15000.0, "ane_power": 0.0}}}})
while True:
    sys.stdout.buffer.write(sample + b"\0")
    sys.stdout.buffer.flush()
    time.sleep(0.02)
"""

PYTHON = """#!/bin/sh
case "$1 $2" in
  "-m infx.bench") if [ "$3" = eval ]; then shift 3; exec {client} -m infx.bench.eval_stub "$@"; fi
                   if [ "$3" = agentic ]; then exec {client} -m infx.bench.agentic_stub; fi
                   if [ "$3" = al-gate ]; then exec {client} -m infx.bench.al_gate_stub; fi ;;
  "-m infx.bench.powermetrics") echo "$@" >> "$FAKE_LOG.sampler" ;;
esac
exec {python} "$@"
"""


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake cluster whose containers see the stub server and clients; returns its inventory."""
    monkeypatch.setitem(SCHEDULERS, "fake", FakeSettings)
    monkeypatch.setitem(BACKENDS, "fake", "infx.tests.launch.fake_backend:FakeBackend")
    monkeypatch.setattr(serve, "PORT", free_port())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = shlex.quote(sys.executable)
    for name, text in {
        "llama-server": SERVER.format(python=sys.executable),
        "client": CLIENT.format(python=sys.executable),
        "python3": PYTHON.format(client=bin_dir / "client", python=python),
    }.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    record = {
        "gpus-per-node": 1,
        "arch": "aarch64",
        "env": {"CLUSTER_KNOB": "c"},
        "models": {"entries": {"M-GGUF": {"root": "hf-hub-cache", "dir": "snap"}}},
        "scheduler": "fake",
        "fake": {
            "root": str(tmp_path / "fake"), "namespace": "bench", "bin_path": str(bin_dir),
            "volumes": {"hf-hub-cache": {"claim": "hub"}},
        },
    }  # fmt: skip
    return {"labels": {"cluster:mac": ["mac_00"]}, "clusters": {"mac": record}}


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    write_yaml(root / "configs/srt-recipes/agentic.yaml", SHARED_AGENTIC)
    write_yaml(root / "configs/srt-recipes/telemetry-native.yaml", NATIVE_TELEMETRY)
    write_yaml(root / RECIPE, FRAGMENT)
    (root / "infx").symlink_to(ROOT / "infx")
    return root


def request(workspace, tmp_path, **overrides: str) -> LaunchRequest:
    return LaunchRequest.from_env({
        "RUNNER_NAME": "mac_00", "GITHUB_WORKSPACE": str(workspace), "IMAGE": "https://example.invalid/llama.tar.gz",
        "FRAMEWORK": "llamacpp", "MODEL": "org/M-GGUF", "MODEL_PREFIX": "qwen3.827b", "PRECISION": "q4_k_m",
        "SPEC_DECODING": "none", "RESULT_FILENAME": "m_point", "IS_AGENTIC": "1", "IS_MULTINODE": "false",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "SRT_RECIPE": RECIPE, "TP": "1", "PP_SIZE": "1",
        "DCP_SIZE": "1", "PCP_SIZE": "1", "EP_SIZE": "1", "DP_ATTENTION": "false", "GPU_COUNT": "1",
        "CONC": "1", "ISL": "0", "OSL": "0", "RANDOM_RANGE_RATIO": "0.8", "THINKING_MODE": "thinking_on",
        "KV_OFFLOADING": "none",
        "HF_HUB_CACHE": "/mnt/hf_hub_cache/", "SALLOC_TIME_LIMIT": "30", "HF_TOKEN": "hf_secret",
        "FAKE_LOG": str(tmp_path / "calls"), **overrides,
    })  # fmt: skip


def calls(tmp_path, kind: str) -> list[dict]:
    path = tmp_path / f"calls.{kind}"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def flag_pairs(argv: list[str]) -> dict[str, str | None]:
    """Each ``--flag`` of ``argv`` with the value after it, or None for a bare flag."""
    return {
        token: None if following.startswith("--") else following
        for token, following in zip(argv, [*argv[1:], "--"], strict=True)
        if token.startswith("--")
    }


@pytest.mark.parametrize(("overrides", "clients"), [
    ({}, ["agentic"]),
    ({"RUN_EVAL": "true"}, ["agentic", "eval"]),
    ({"EVAL_ONLY": "true"}, ["eval"]),
], ids=["throughput", "with-eval", "eval-only"])  # fmt: skip
def test_a_point_serves_the_named_gguf_of_its_bound_recipe_and_stages_what_the_workflow_reads(
    fake, workspace, tmp_path, overrides, clients
):
    cluster = load_inventory(fake).clusters["mac"]
    event = JobEventBuilder()

    assert launch(cluster, request(workspace, tmp_path, **overrides), event) == 0

    [server] = calls(tmp_path, "server")
    port = str(serve.PORT)
    assert flag_pairs(server["argv"]) == {
        "--model": "/models/snap/Q4/M-00001-of-00002.gguf", "--host": "127.0.0.1", "--port": port,
        "--alias": "org/M-GGUF", "--ctx-size": "36864", "--flash-attn": "on", "--lazy-mode": "on",
        "--no-kv-unified": None, "--parallel": "4",
    }  # fmt: skip
    [workdir] = containers_dir(cluster.scheduler_settings).iterdir()
    python = {"PYTHONPATH": str(workdir), "PYTHONSAFEPATH": None, "PYTHONDONTWRITEBYTECODE": "1"}
    assert server["env"] == {
        "LLAMA_ONLY": "1", "DROPPED": None, "BENCH_KNOB": "7", "HF_HUB_CACHE": str(tmp_path / "fake/volumes/hub"),
        "CLUSTER_KNOB": "c", "HF_TOKEN": "hf_secret", **python,
    }  # fmt: skip
    runs = calls(tmp_path, "clients")
    assert [
        "eval" if run["argv"][1] == "infx.bench.eval_stub" else "agentic" for run in runs
    ] == clients
    [root] = [path for path in workspace.iterdir() if path.name.startswith("serve.")]
    for run in runs:
        assert run["env"] == {
            "LLAMA_ONLY": "1", "BENCH_KNOB": "7", "RESULT_DIR": f"{root.name}/agentic",
            "USE_CHAT_TEMPLATE": None, "MODEL": "org/M-GGUF", "RESULT_FILENAME": "m_point", **python,
        }  # fmt: skip
        if run["argv"][1] == "infx.bench.eval_stub":
            assert run["argv"][2:] == [
                "--endpoint", f"http://127.0.0.1:{port}", "--concurrency", "1", "--stage-to", f"{root.name}/eval",
            ]  # fmt: skip
    assert (workspace / "server.log").read_text() == "engine up\n"
    assert (workspace / "m_point.json").exists() == ("agentic" in clients)
    assert (workspace / "results_gsm8k.json").exists() == ("eval" in clients)
    assert not (workspace / "al_gate_m_point.json").exists()
    assert not (workspace / "LOGS/power").exists()
    assert event.event.recipe == RECIPE


@pytest.mark.parametrize(("env", "rc", "kind"), [
    ({"FAKE_CLIENT_RC": "3"}, 3, None),
    ({"FAKE_NO_RESULT": "1"}, 1, "ResultMissing"),
])  # fmt: skip
def test_a_point_without_its_result_fails_and_still_stages_the_server_log(
    fake, workspace, tmp_path, env, rc, kind
):
    cluster = load_inventory(fake).clusters["mac"]
    event = JobEventBuilder()

    assert launch(cluster, request(workspace, tmp_path, **env), event) == rc

    assert (workspace / "server.log").read_text() == "engine up\n"
    assert not (workspace / "m_point.json").exists()
    if kind is not None:
        assert event.finish(rc).error.type == kind


def test_a_launch_ended_by_an_error_still_stages_what_the_job_left(
    fake, workspace, tmp_path, monkeypatch
):
    def lost(_backend, job):
        job.process.wait()
        raise BackendError("the scheduler lost the job")

    monkeypatch.setattr(FakeBackend, "stream_logs", lost)

    assert launch(load_inventory(fake).clusters["mac"], request(workspace, tmp_path)) == 1

    assert (workspace / "server.log").read_text() == "engine up\n"


def edit_recipe(workspace: Path, changes: dict[str, object]) -> None:
    recipe = yaml.safe_load((workspace / RECIPE).read_text())
    role = recipe["roles"]["agg"]
    for key, value in changes.items():
        if key.startswith("args."):
            role["args"][key.removeprefix("args.")] = value
        elif key.startswith("role."):
            role[key.removeprefix("role.")] = value
        else:
            recipe[key] = value
    write_yaml(workspace / RECIPE, recipe)


def test_a_server_not_ready_within_the_budget_of_its_recipe_fails_the_point(
    fake, workspace, tmp_path, monkeypatch, capfd
):
    edit_recipe(workspace, {"health_check": {"max_attempts": 2, "interval_seconds": 1}})
    monkeypatch.setattr(FakeBackend, "stream_logs", lambda _self, job: job.process.wait(timeout=60))
    point = request(workspace, tmp_path, FAKE_NEVER_READY="1")

    assert launch(load_inventory(fake).clusters["mac"], point) == 1

    assert "was not ready within 2s" in capfd.readouterr().err
    assert calls(tmp_path, "clients") == []


AGENTIC_ENV = {
    "ENABLE_AGENTX_POWER": "1", "RUNNER_TYPE": "cluster:mac", "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_TOKEN": "ghs_runner_token",
}  # fmt: skip


def test_an_agentx_point_replays_against_its_server_with_no_power(fake, workspace, tmp_path):
    assert (
        launch(load_inventory(fake).clusters["mac"], request(workspace, tmp_path, **AGENTIC_ENV))
        == 0
    )

    [root] = [path for path in workspace.iterdir() if path.name.startswith("serve.")]
    url = f"http://127.0.0.1:{serve.PORT}"
    assert json.loads((tmp_path / "calls.agentic").read_text()) == {
        "RESULT_DIR": f"{root.name}/agentic", "AGENTIC_OUTPUT_DIR": f"{root.name}/results",
        "AIPERF_SERVER_URL": url, "AIPERF_SERVER_METRICS_URLS": f"{url}/metrics",
        "ENABLE_AGENTX_POWER": "0", "CONC": "1", "RUNNER_TYPE": "cluster:mac", "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_TOKEN": None, "server.log": "engine up\n",
    }  # fmt: skip
    assert (workspace / "m_point.json").read_text() == "{}"
    assert (workspace / "results/aiperf_artifacts/profile_export.json").is_file()
    assert (workspace / "server.log").read_text() == "engine up\n"
    assert not (workspace / "results/server.log").exists()


def test_artifacts_that_cannot_be_staged_fail_the_point(fake, workspace, tmp_path):
    (workspace / "results").write_text("a stale file where the AgentX artifacts go")
    event = JobEventBuilder()

    assert (
        launch(
            load_inventory(fake).clusters["mac"], request(workspace, tmp_path, **AGENTIC_ENV), event
        )
        == 1
    )

    assert event.finish(1).error.type == "ArtifactError"
    assert (workspace / "server.log").is_file()


@pytest.mark.parametrize(("env", "changes", "message"), [
    ({"IS_AGENTIC": "0", "ISL": "1024", "OSL": "128"}, {}, "runs only AgentX points"),
    ({"REQUIRE_POWER": "1"}, {}, "measures no power, which REQUIRE_POWER demands"),
    ({}, {"setup_script": "patch-engine.sh", "services": []}, "does not run services, setup_script"),
    ({}, {"role.restart": {"max": 1}}, "does not run roles.agg.restart"),
    ({}, {"frontend": {"type": "dynamo"}}, "runs no dynamo frontend"),
    ({}, {"frontend": {"type": "llamacpp", "enable_multiple_frontends": False, "env": {"X": "1"}}},
     "does not run frontend.env"),
    ({}, {"frontend": {"type": "llamacpp", "enable_multiple_frontends": True}},
     "does not run frontend.enable_multiple_frontends"),
    ({}, {"args.port": 9000}, "the launcher sets port"),
    ({}, {"args.lazy-mode": False}, "set lazy-mode to the string on, off or auto, not False"),
    ({}, {"args.flash-attn": True}, "set flash-attn to the string on, off or auto, not True"),
    ({}, {"role.args": {"lazy-mode": "on", "flash-attn": "on"}}, "roles.agg.args.model"),
    ({"MODEL": "org/Unstaged-GGUF"}, {}, "stages no 'org/Unstaged-GGUF'"),
    ({"PREFILL_ADDITIONAL_SETTINGS": '["MODEL_PATH=/data/m"]', "MODEL_PATH": "/data/m"}, {},
     "serves MODEL or its staged checkpoint"),
    ({"FRAMEWORK": "vllm"}, {"frontend": {"type": "vllm"}, "engine": "vllm", "role.args": {"tensor-parallel-size": 1}},
     "runs no vllm server"),
], ids=["fixed-sequence", "power", "unsupported-keys", "unsupported-role-key", "frontend", "frontend-env",
        "multiple-frontends", "owned-arg", "lazy-mode-read-as-false", "flash-attn-read-as-true", "no-model",
        "unstaged-checkpoint", "model-path", "no-server"])  # fmt: skip
def test_what_the_serve_path_cannot_honor_is_refused_before_any_container(
    fake, workspace, tmp_path, env, changes, message
):
    edit_recipe(workspace, changes)
    cluster = load_inventory(fake).clusters["mac"]

    with pytest.raises(LaunchError, match=message), Lifecycle() as life:
        drivers.run(cluster, request(workspace, tmp_path, **env), life)

    assert not containers_dir(cluster.scheduler_settings).exists()
    assert calls(tmp_path, "server") == []


@pytest.fixture
def mac(fake, tmp_path):
    """The fake cluster as a Mac, whose stub sudo streams powermetrics samples."""
    fake["clusters"]["mac"]["os"] = "darwin"
    sudo = tmp_path / "bin" / "sudo"
    sudo.write_text(SUDO.format(python=sys.executable))
    sudo.chmod(0o755)
    return fake


def test_a_mac_point_records_its_power_beside_its_result(mac, workspace, tmp_path):
    assert (
        launch(load_inventory(mac).clusters["mac"], request(workspace, tmp_path, **AGENTIC_ENV))
        == 0
    )

    result = json.loads((workspace / "m_point.json").read_text())
    assert result["power_valid"] == 1
    assert result["avg_power_w"] == pytest.approx(15.0)
    assert result["power_audit"]["source"] == "power_validation_m_point.json"
    assert (
        json.loads((workspace / "power_validation_m_point.json").read_text())["power_valid"] is True
    )
    assert (workspace / "LOGS/power/samples.csv").read_text().startswith("timestamp_unix,")
    assert not (workspace / "power").exists()
    # The stub python3 first on the job's PATH runs it, then hands it to the real interpreter.
    [root] = [path for path in workspace.iterdir() if path.name.startswith("serve.")]
    assert (tmp_path / "calls.sampler").read_text().split() == [
        "-m", "infx.bench.powermetrics", "--output", f"{root.name}/power/samples.csv",
    ]  # fmt: skip
    assert (workspace / "results/agentic_power_concurrency_1.json").is_file()
    assert json.loads((tmp_path / "calls.agentic").read_text())["ENABLE_AGENTX_POWER"] == "1"
    assert gone(int((tmp_path / "calls.sudo").read_text()))


def test_power_that_a_mac_could_not_record_costs_the_point_nothing(mac, workspace, tmp_path):
    env = {**AGENTIC_ENV, "FAKE_SUDO_RC": "1"}

    assert launch(load_inventory(mac).clusters["mac"], request(workspace, tmp_path, **env)) == 0

    assert json.loads((workspace / "m_point.json").read_text())["power_valid"] == 0
    sidecar = json.loads((workspace / "power_validation_m_point.json").read_text())
    assert "invalid_benchmark_result" in sidecar["reasons"]


def test_require_power_fails_a_mac_point_without_valid_power(mac, workspace, tmp_path):
    env = {**AGENTIC_ENV, "FAKE_SUDO_RC": "1", "REQUIRE_POWER": "1"}
    event = JobEventBuilder()

    assert (
        launch(load_inventory(mac).clusters["mac"], request(workspace, tmp_path, **env), event) == 1
    )

    assert event.finish(1).error.type == "PowerValidationFailed"


def test_an_eval_only_point_on_a_mac_samples_no_power(mac, workspace, tmp_path):
    env = {**AGENTIC_ENV, "EVAL_ONLY": "true", "RUN_EVAL": "true"}

    assert launch(load_inventory(mac).clusters["mac"], request(workspace, tmp_path, **env)) == 0

    assert not (tmp_path / "calls.sudo").exists()
    assert not (workspace / "power_validation_m_point.json").exists()
    assert not (workspace / "LOGS/power").exists()


@pytest.fixture
def golden(tmp_path, monkeypatch):
    """A golden curve for the point's model, so that no test depends on a committed curve."""
    curves = tmp_path / "golden"
    fact = {"value": None, "source": "default"}
    write_yaml(curves / "identities.yaml", {"m_mtp": {
        "select": {"model_prefixes": ["qwen3.827b"], "method": "mtp"},
        "collection": {"engine": "vllm", "draft": "native", **dict.fromkeys(FACTS, fact)},
    }})  # fmt: skip
    write_yaml(curves / "m_mtp.yaml", {
        "m": {"thinking_on": {1: 1.1, 2: 1.2, 3: 1.25}, "thinking_off": {1: 1.3, 2: 1.4, 3: 1.75}},
        "collection": {"renderer": "chat_template", "modes": {}},
    })  # fmt: skip
    monkeypatch.setattr(serve, "acceptance_length", partial(acceptance_length, golden_dir=curves))
    monkeypatch.setattr(al_gate, "plan", partial(al_gate.plan, golden_dir=curves))


MTP = {"args.spec-type": "draft-mtp", "args.spec-draft-n-max": 3, "args.model-draft": "MTP/head.gguf"}


def server_flag(tmp_path, name: str) -> str | None:
    [server] = calls(tmp_path, "server")
    argv = server["argv"]
    return argv[argv.index(name) + 1] if name in argv else None


@pytest.mark.parametrize(("thinking", "acceptance"), [("thinking_on", "1.25"), ("thinking_off", "1.75")])
@pytest.mark.usefixtures("golden")
def test_an_mtp_point_serves_its_head_at_the_golden_acceptance(fake, workspace, tmp_path, thinking, acceptance):
    edit_recipe(workspace, MTP)
    env = {"SPEC_DECODING": "mtp", "THINKING_MODE": thinking}

    assert launch(load_inventory(fake).clusters["mac"], request(workspace, tmp_path, **env)) == 0

    assert server_flag(tmp_path, "--spec-type") == "draft-mtp"
    assert server_flag(tmp_path, "--spec-draft-n-max") == "3"
    assert server_flag(tmp_path, "--model-draft") == "/models/snap/MTP/head.gguf"
    assert server_flag(tmp_path, "--spec-synth-len") == acceptance


EVAL_MTP = {"SPEC_DECODING": "mtp", "EVAL_ONLY": "true", "RUN_EVAL": "true"}
PROMPTS = "al_gate_dataset/speedbench_coding.jsonl"


def test_an_eval_only_mtp_point_without_prompts_records_why_its_gate_was_skipped(
    fake, workspace, tmp_path
):
    edit_recipe(workspace, MTP)

    assert (
        launch(load_inventory(fake).clusters["mac"], request(workspace, tmp_path, **EVAL_MTP)) == 0
    )

    assert server_flag(tmp_path, "--spec-type") == "draft-mtp"
    assert server_flag(tmp_path, "--spec-synth-len") is None
    gate = json.loads((workspace / "al_gate_m_point.json").read_text())
    assert (gate["status"], gate["error"]) == ("skipped", "no SPEED-Bench prompts at ''")
    assert [run["argv"][1] for run in calls(tmp_path, "clients")] == ["infx.bench.eval_stub"]


@pytest.mark.parametrize("gate_status", ["0", "3"], ids=["pass", "gate-fails"])
@pytest.mark.usefixtures("golden")
def test_an_eval_only_mtp_point_runs_its_gate_on_its_server_before_its_eval(
    fake, workspace, tmp_path, gate_status
):
    edit_recipe(workspace, MTP)
    (workspace / PROMPTS).parent.mkdir()
    (workspace / PROMPTS).write_text("{}\n")
    env = {**EVAL_MTP, "AL_GATE_DATASET": PROMPTS, "FAKE_GATE_RC": gate_status}

    assert launch(load_inventory(fake).clusters["mac"], request(workspace, tmp_path, **env)) == 0

    gate, evaluation = calls(tmp_path, "clients")
    assert gate["argv"][1] == "infx.bench.al_gate_stub"
    assert evaluation["argv"][1] == "infx.bench.eval_stub"
    [root] = [path for path in workspace.iterdir() if path.name.startswith("serve.")]
    assert gate["env"] == {
        "AL_GATE": "true", "AL_GATE_CURVE": "m_mtp", "AL_GATE_DRAFT_TOKENS": "3",
        "AL_GATE_DATASET": PROMPTS, "AL_GATE_OUTPUT_DIR": f"{root.name}/agentic",
        "AL_GATE_TIMEOUT_S": "1800", "THINKING_MODE": "thinking_on",
    }  # fmt: skip
    assert json.loads((workspace / "al_gate_m_point.json").read_text()) == {"status": "pass"}
    assert (workspace / "results_gsm8k.json").is_file()
    assert not list((workspace / "results").rglob("*"))
    with tarfile.open(workspace / "srt-single-node-logs.tar.gz") as logs:
        assert logs.extractfile("./aiperf.log").read() == b"replayed"


@pytest.mark.parametrize(("changes", "message"), [
    ({**MTP, "args.spec-synth-len": 9}, "the launcher sets spec-synth-len"),
    ({**MTP, "args.spec-synth-rates": "1,1"}, "the launcher sets spec-synth-rates"),
    ({"args.spec-type": "draft-mtp"}, "requires a positive integer draft length"),
], ids=["own-length", "own-rates", "no-draft-length"])  # fmt: skip
@pytest.mark.usefixtures("golden")
def test_an_mtp_point_cannot_choose_its_own_acceptance(fake, workspace, tmp_path, changes, message):
    edit_recipe(workspace, changes)
    cluster = load_inventory(fake).clusters["mac"]

    with pytest.raises(LaunchError, match=message), Lifecycle() as life:
        drivers.run(cluster, request(workspace, tmp_path, SPEC_DECODING="mtp"), life)

    assert calls(tmp_path, "server") == []
