"""A new scheduler backend needs no driver edits: the script driver runs on any of them."""

import pytest
import yaml

from infx.clusters import SCHEDULERS, load_inventory
from infx.launch import drivers
from infx.launch.__main__ import launch, main
from infx.launch.backends import BACKENDS
from infx.launch.backends.base import BackendError, JobState, JobStatus
from infx.launch.context import LaunchError
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest
from infx.tests.launch.fake_backend import FakeBackend, FakeSettings, containers_dir

COLLECTOR = "benchmarks/single_node/speedbench/fixture.sh"
SCRIPT = r"""printf 'model=%s\ncwd=%s\ngit=%s\nworkload=%s\ncluster=%s\ntoken=%s\n' \
    "$MODEL_PATH" "$PWD" "$([ -e .git ] && echo yes || echo no)" "${SPEEDBENCH_KNOB:-unset}" \
    "${UCX_NET_DEVICES:-unset}" "${GITHUB_TOKEN:-unset}" > "$OUT_YAML"
mkdir -p speedbench_results && echo serving > speedbench_results/server_0.log
mkdir -p draft_models && echo weights > draft_models/draft.bin
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Register the fake scheduler and backend; return a runner config with one fake cluster."""
    monkeypatch.setitem(SCHEDULERS, "fake", FakeSettings)
    monkeypatch.setitem(BACKENDS, "fake", "infx.tests.launch.fake_backend:FakeBackend")
    monkeypatch.setattr(FakeBackend, "cleaned", [])
    checkpoint = tmp_path / "fake/volumes/ckpt/Kimi-K3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    record = {
        "gpus-per-node": 8,
        "arch": "x86_64",
        "env": {"UCX_NET_DEVICES": "eth0"},
        "models": {"entries": {"Kimi-K3": {"root": "checkpoints", "dir": "Kimi-K3"}}, "download-root": "downloads"},
        "scheduler": "fake",
        "fake": {
            "root": str(tmp_path / "fake"),
            "namespace": "bench",
            "volumes": {
                "checkpoints": {"claim": "ckpt", "visibility": "node-local"},
                "downloads": {"claim": "downloads"},
                "hf-home": {"claim": "hf"},
            },
        },
    }  # fmt: skip
    return {"labels": {"cluster:local": ["local_00"]}, "clusters": {"local": record}}


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / COLLECTOR).parent.mkdir(parents=True)
    (root / COLLECTOR).write_text(SCRIPT)
    (root / ".git").mkdir()
    return root


def request(workspace, **overrides: str) -> LaunchRequest:
    return LaunchRequest.from_env({
        "RUNNER_NAME": "local_00", "GITHUB_WORKSPACE": str(workspace), "MODEL": "org/Kimi-K3",
        "IMAGE": "vllm/vllm-openai:v0.21.0", "GPU_COUNT": "8", "IS_MULTINODE": "false",
        "BENCH_SCRIPT_OVERRIDE": COLLECTOR, "SALLOC_TIME_LIMIT": "30",
        "OUT_YAML": "/workspace/speedbench-reference-al.yaml", "SPEEDBENCH_KNOB": "7",
        "GITHUB_TOKEN": "ghs_secret", **overrides,
    })  # fmt: skip


def result(workspace) -> dict[str, str]:
    lines = (workspace / "speedbench-reference-al.yaml").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def test_a_script_point_runs_on_a_new_backend_through_its_volumes(fake, workspace, tmp_path):
    cluster = load_inventory(fake).clusters["local"]

    with Lifecycle() as life:
        rc = drivers.run(cluster, request(workspace), life)

    assert rc == 0
    seen = result(workspace)
    assert seen["model"] == str(tmp_path / "fake/volumes/ckpt/Kimi-K3")
    assert seen["cwd"] != str(workspace) and seen["git"] == "no"
    assert (seen["workload"], seen["cluster"], seen["token"]) == ("7", "eth0", "unset")
    assert (workspace / "speedbench_results/server_0.log").read_text() == "serving\n"
    assert not (workspace / "draft_models").exists()
    assert (tmp_path / "fake/volumes/hf").is_dir()


def _script_fails(workspace, monkeypatch):
    (workspace / COLLECTOR).write_text(SCRIPT + "exit 7\n")


def _log_stream_lost(workspace, monkeypatch):
    def lost(self, job):
        job.process.wait()
        raise BackendError("log stream lost")

    monkeypatch.setattr(FakeBackend, "stream_logs", lost)


def _cancelled_with_exit_0(workspace, monkeypatch):
    monkeypatch.setattr(
        FakeBackend, "state", lambda self, job: JobStatus(JobState.CANCELLED, "cancelled|0:0", 0)
    )


@pytest.mark.parametrize(
    ("failure", "rc"), [(_script_fails, 7), (_log_stream_lost, 1), (_cancelled_with_exit_0, 1)]
)
def test_a_failed_point_fails_the_launch_and_still_returns_its_result(fake, workspace, monkeypatch, failure, rc):
    failure(workspace, monkeypatch)

    assert launch(load_inventory(fake).clusters["local"], request(workspace)) == rc
    assert result(workspace)["workload"] == "7"


def test_points_without_a_script_refuse_a_non_slurm_cluster_before_any_work(fake, workspace):
    cluster = load_inventory(fake).clusters["local"]
    srt_point = request(workspace, BENCH_SCRIPT_OVERRIDE="")

    with pytest.raises(LaunchError, match="needs a slurm cluster; 'local' uses scheduler 'fake'"), Lifecycle() as life:
        drivers.run(cluster, srt_point, life)
    assert not containers_dir(cluster.scheduler_settings).exists()


@pytest.mark.parametrize(("runner", "namespace"), [("local_00", "bench"), ("unlisted_00", None)])
def test_cleanup_hands_the_backend_the_runners_settings_when_it_has_a_record(
    fake, tmp_path, monkeypatch, runner, namespace
):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(fake))
    monkeypatch.setenv("RUNNER_NAME", runner)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert main(["--runner-config", str(config), "cleanup"]) == 0

    [(settings, cleaned)] = FakeBackend.cleaned
    assert (getattr(settings, "namespace", None), cleaned) == (namespace, runner)
