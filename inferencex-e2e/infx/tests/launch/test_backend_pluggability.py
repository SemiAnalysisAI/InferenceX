"""A new backend is its own modules, a cluster record and two registry entries: no driver edits.

The fake backend (fake_backend.py) reaches volumes by claim, delivers the checkout by copy
and fetches only declared outputs, unlike Slurm/Pyxis. On a cluster of any scheduler but
Slurm, the script driver is the one that runs.
"""

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
# Records what the script saw into its result, writes a log the workflow uploads, and
# leaves a file that is neither.
SCRIPT = r"""printf 'model=%s\ncwd=%s\ngit=%s\nworkload=%s\ncluster=%s\nhost=%s\ntoken=%s\n' \
    "$MODEL_PATH" "$PWD" "$([ -e .git ] && echo yes || echo no)" "${SPEEDBENCH_KNOB:-unset}" \
    "${UCX_NET_DEVICES:-unset}" "${FAKE_HOST_STATE:-unset}" "${GITHUB_TOKEN:-unset}" > "$OUT_YAML"
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
    """A checkout whose collector writes its result under the container workspace."""
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
        "FAKE_HOST_STATE": "host", "GITHUB_TOKEN": "ghs_secret", **overrides,
    })  # fmt: skip


def result(workspace) -> dict[str, str]:
    """What the script recorded, as brought back into the checkout."""
    lines = (workspace / "speedbench-reference-al.yaml").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def test_a_script_point_runs_on_a_new_backend_through_its_volumes(fake, workspace, tmp_path):
    cluster = load_inventory(fake).clusters["local"]

    with Lifecycle() as life:
        rc = drivers.run(cluster, request(workspace), life)

    assert rc == 0
    seen = result(workspace)
    # MODEL_PATH named the checkpoint inside the volume the backend resolved the claim to.
    assert seen["model"] == str(tmp_path / "fake/volumes/ckpt/Kimi-K3")
    # It ran in the backend's own copy of the checkout, delivered without git metadata.
    assert seen["cwd"] != str(workspace) and seen["git"] == "no"
    # Workload and cluster env reach the container; host state of either kind does not.
    assert (seen["workload"], seen["cluster"]) == ("7", "eth0")
    assert (seen["host"], seen["token"]) == ("unset", "unset")
    # What the workflow reads came back; nothing else did.
    assert (workspace / "speedbench_results/server_0.log").read_text() == "serving\n"
    assert not (workspace / "draft_models").exists()
    assert (tmp_path / "fake/volumes/hf").is_dir()


def test_the_result_comes_back_when_the_script_fails(fake, workspace):
    (workspace / COLLECTOR).write_text(SCRIPT + "exit 7\n")
    cluster = load_inventory(fake).clusters["local"]

    with Lifecycle() as life:
        assert drivers.run(cluster, request(workspace), life) == 7

    assert result(workspace)["workload"] == "7"


def test_the_result_comes_back_when_following_the_job_fails(fake, workspace, monkeypatch):
    def lost(self, job):
        job.process.wait()
        raise BackendError("log stream lost")

    monkeypatch.setattr(FakeBackend, "stream_logs", lost)

    assert launch(load_inventory(fake).clusters["local"], request(workspace)) == 1
    assert result(workspace)["workload"] == "7"


def test_a_job_that_did_not_succeed_fails_even_when_it_reports_exit_0(fake, workspace, monkeypatch):
    monkeypatch.setattr(FakeBackend, "state", lambda self, job: JobStatus(JobState.CANCELLED, "cancelled|0:0", 0))

    with Lifecycle() as life:
        assert drivers.run(load_inventory(fake).clusters["local"], request(workspace), life) == 1


def test_points_without_a_script_refuse_a_non_slurm_cluster_before_any_work(fake, workspace):
    cluster = load_inventory(fake).clusters["local"]
    srt_point = request(workspace, BENCH_SCRIPT_OVERRIDE="")

    with pytest.raises(LaunchError, match="needs a slurm cluster; 'local' uses scheduler 'fake'"), Lifecycle() as life:
        drivers.run(cluster, srt_point, life)
    assert not containers_dir(cluster.scheduler_settings).exists()


def test_cleanup_hands_the_backend_its_settings(fake, tmp_path, monkeypatch):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(fake))
    monkeypatch.setenv("RUNNER_NAME", "local_00")

    assert main(["--runner-config", str(config), "cleanup"]) == 0

    [(received, runner)] = FakeBackend.cleaned
    assert (received.namespace, runner) == ("bench", "local_00")


def test_cleanup_without_the_runners_record_passes_no_settings(fake, tmp_path, monkeypatch):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(fake))
    monkeypatch.setenv("RUNNER_NAME", "unlisted_00")
    monkeypatch.setenv("PATH", str(tmp_path))  # no Slurm here either

    assert main(["--runner-config", str(config), "cleanup"]) == 0

    assert FakeBackend.cleaned == [(None, "unlisted_00")]
