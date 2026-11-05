"""The native backend runs points as host process groups, from digest-pinned image archives."""

import contextlib
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import pytest
import yaml

from infx.bench import proc
from infx.clusters import Cluster, load_inventory
from infx.launch import drivers
from infx.launch.__main__ import main
from infx.launch.backends import native
from infx.launch.backends.base import BackendError, Container, JobState
from infx.launch.backends.native import NativeBackend
from infx.launch.context import LaunchError
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest
from infx.tests.processes import alive, free_port, gone

ROOT = Path(__file__).resolve().parents[3]
RUNNER = "mac_00"
COLLECTOR = "benchmarks/single_node/speedbench/fixture.sh"
SCRIPT = r"""printf 'cwd=%s\nmodel=%s\nhub=%s\nresults=%s\ntool=%s\npython=%s\nruns=%s\nuv=%s\nhome=%s\ntmp=%s\npath=%s\nrunner=%s\ngh=%s\ntoken=%s\nhw=%s\n' \
    "$PWD" "$MODEL_PATH" "$HF_HUB_CACHE" "$RESULT_DIR" "$(greet)" "$(command -v python3)" "$(python3 -c 'print(6 * 7)')" "$(command -v uv)" \
    "$HOME" "$TMPDIR" "$PATH" "${RUNNER_NAME:-unset}" "${GITHUB_TOKEN:-unset}" "$HF_TOKEN" "$RUNNER_TYPE" > "$OUT_YAML"
"""


def archive(tmp_path: Path, members: dict[str, str], name: str = "tool.tar.gz") -> str:
    """A gzipped tar of executable ``members`` (path to text), as a native image reference."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for member, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(member)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    path = tmp_path / name
    path.write_bytes(buffer.getvalue())
    return f"{path.as_uri()}@sha256:{hashlib.sha256(buffer.getvalue()).hexdigest()}"


def runner_config(tmp_path: Path) -> dict:
    """A runner config whose one native cluster matches this host and owns ``mac_00``."""
    record = {
        "gpus-per-node": 1,
        "arch": "aarch64",
        "os": sys.platform,
        "models": {"download-root": "models"},
        "scheduler": "native",
        "native": {
            "root": str(tmp_path / "native"),
            "volumes": {
                "models": {"path": str(tmp_path / "models")},
                "hf-home": {"path": str(tmp_path / "hf-home")},
            },
        },
    }
    return {"labels": {"cluster:mac": [RUNNER]}, "clusters": {"mac": record}}


def mac_cluster(tmp_path: Path) -> Cluster:
    return load_inventory(runner_config(tmp_path)).clusters["mac"]


def job_record(tmp_path: Path) -> Path:
    return tmp_path / "native" / "jobs" / f"{RUNNER}.jsonl"


FAKE_UV = r"""#!/bin/sh
echo "HOME=${{HOME:-}} UV_CACHE_DIR=${{UV_CACHE_DIR:-}} HF_TOKEN=${{HF_TOKEN:-}}" >> {log}
case " $* " in
*" sync "*" --frozen "*)
    rc=$(cat {rc} 2>/dev/null || echo 0)
    [ "$rc" = 0 ] || exit "$rc"
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    printf '#!/bin/sh\nexec {python} "$@"\n' > "$UV_PROJECT_ENVIRONMENT/bin/python3"
    chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python3"
    ;;
*)
    echo "uv: this call would re-resolve the clients instead of installing uv.lock" >&2
    exit 64
    ;;
esac
"""


@pytest.fixture(autouse=True)
def uv(tmp_path, monkeypatch):
    """Put the fake uv first on the ``PATH`` that launch requests receive.

    It logs each call with what it was given of the environment. A frozen sync exits with the status
    in uv.rc, if any, or makes the venv that UV_PROJECT_ENVIRONMENT names, whose python3 runs this
    interpreter, and installs nothing. Any other call fails.
    """
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    paths = {name: tmp_path / f"uv.{name}" for name in ("log", "rc")}
    fake.write_text(FAKE_UV.format(**paths, python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def uv_calls(tmp_path: Path) -> list[str]:
    """The fake uv's calls so far."""
    log = tmp_path / "uv.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / COLLECTOR).parent.mkdir(parents=True)
    (root / COLLECTOR).write_text(SCRIPT)
    return root


@pytest.fixture
def image(tmp_path):
    """A native image whose archive holds one stub executable."""
    return archive(tmp_path, {"tool-1/greet": "#!/bin/sh\n"})


def request(workspace: Path, image: str, **overrides: str) -> LaunchRequest:
    return LaunchRequest.from_env({
        "RUNNER_NAME": RUNNER, "GITHUB_WORKSPACE": str(workspace), "PATH": os.environ["PATH"],
        "MODEL": "unsloth/Qwen3.8-27B-GGUF", "IMAGE": image, "GPU_COUNT": "1",
        "BENCH_SCRIPT_OVERRIDE": COLLECTOR, "SALLOC_TIME_LIMIT": "5",
        "OUT_YAML": "/workspace/out.yaml", "RESULT_DIR": "/workspace/results", **overrides,
    })  # fmt: skip


def backend(tmp_path: Path, life: Lifecycle, workspace: Path, image: str) -> NativeBackend:
    cluster = mac_cluster(tmp_path)
    return NativeBackend(cluster, request(workspace, image), life)


def container(workspace: Path, image, script: str = "", **overrides) -> Container:
    fields = {
        "image": image, "command": ("bash", "-c", script), "gpus": 1, "time_limit_min": 5,
        "workspace": workspace, "workdir": PurePosixPath("/workspace"), **overrides,
    }  # fmt: skip
    return Container(**fields)


STUB_SERVER = r"""import http.server, os, signal, sys
port, pid_file = int(sys.argv[1]), sys.argv[2]
os.kill(os.getppid(), signal.SIGKILL)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_):
        pass

httpd = http.server.HTTPServer(("127.0.0.1", port), Health)
with open(pid_file + ".tmp", "w") as out:
    out.write(f"{os.getpid()}\n")
os.replace(pid_file + ".tmp", pid_file)
httpd.serve_forever()
"""


def serving(tmp_path: Path, workspace: Path, image) -> Container:
    """The serve path's container: ``infx.bench serve`` with the stub server and a long client.

    The stub server runs in its own session and kills its supervisor as it starts.
    """
    stub = tmp_path / "stub_server.py"
    stub.write_text(STUB_SERVER)
    port = free_port()
    command = (
        "python3", "-m", "infx.bench", "serve", "--ready-url", f"http://127.0.0.1:{port}/health",
        "--ready-timeout", "60", "--log", "/workspace/server.log", "--client", json.dumps(["sleep", "300"]),
        "--", "python3", str(stub), str(port), "/workspace/server.pid",
    )  # fmt: skip
    return container(workspace, image, command=command, env={"PYTHONPATH": str(ROOT)})


def wait_for(path: Path, timeout_s: float = 20.0) -> str:
    """The text of ``path`` once a writer has finished a line of it."""
    deadline = time.monotonic() + timeout_s
    while not (path.exists() and path.read_text().endswith("\n")):
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return path.read_text()


def test_a_script_point_runs_in_the_checkout_with_container_paths_and_a_home_of_its_own(
    tmp_path, workspace
):
    image = archive(tmp_path, {"tool-1/greet": "#!/bin/sh\necho hello from the image\n"})
    cluster = mac_cluster(tmp_path)
    home, cache = tmp_path / "runner-home", tmp_path / "uv-cache"
    point = request(
        workspace, image, HOME=str(home), UV_CACHE_DIR=str(cache), HF_TOKEN="hf_secret",
        RUNNER_TYPE="cluster:mac", GITHUB_TOKEN="ghs_runner",
    )  # fmt: skip

    with Lifecycle() as life:
        rc = drivers.run(cluster, point, life)

    assert rc == 0
    job_dir = tmp_path / "native" / "jobs" / RUNNER
    tools, venv = (
        tmp_path / "native" / "images" / image.rsplit("@sha256:", 1)[1],
        job_dir / "venv" / "bin",
    )
    seen = dict(line.split("=", 1) for line in (workspace / "out.yaml").read_text().splitlines())
    assert seen.pop("path").split(os.pathsep)[:2] == [str(tools), str(venv)]
    assert seen == {
        "cwd": str(workspace), "model": str(tmp_path / "models" / "Qwen3.8-27B-GGUF"),
        "hub": str(tmp_path / "hf-home" / "hub"), "results": str(workspace / "results"),
        "tool": "hello from the image", "python": str(venv / "python3"), "runs": "42",
        "uv": str(venv / "uv"), "home": str(job_dir / "home"), "tmp": str(job_dir / "tmp"),
        "runner": "unset", "gh": "unset", "token": "hf_secret", "hw": "cluster:mac",
    }  # fmt: skip
    assert set(uv_calls(tmp_path)) == {f"HOME={home} UV_CACHE_DIR={cache} HF_TOKEN="}
    assert not job_dir.exists()
    assert not job_record(tmp_path).exists()


def test_an_image_is_reused_without_fetching_it_again(tmp_path, workspace, image):
    with Lifecycle() as life:
        first = backend(tmp_path, life, workspace, image).prepare_image(image)
        (tmp_path / "tool.tar.gz").unlink()
        assert backend(tmp_path, life, workspace, image).prepare_image(image) == first


@pytest.mark.parametrize(("uv_rc", "command", "message"), [
    ("3", ("touch", "started"), r"building the clients' venv failed \(exit 3\)"),
    ("0", ("no-such-server",), "cannot run no-such-server: "),
], ids=["venv-not-installed", "command-not-runnable"])  # fmt: skip
def test_a_job_that_cannot_start_never_starts_and_leaves_the_host_free(
    tmp_path, workspace, image, uv_rc, command, message
):
    marker = workspace / "started"
    (tmp_path / "uv.rc").write_text(uv_rc)

    with pytest.raises(BackendError, match=message), Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        run.run_container(container(workspace, run.prepare_image(image), command=command))
    assert not marker.exists()
    assert [entry.name for entry in (tmp_path / "native" / "jobs").iterdir()] == [native.HOST_LOCK]

    (tmp_path / "uv.rc").unlink()
    with Lifecycle() as life:
        again = backend(tmp_path, life, workspace, image)
        job = again.run_container(
            container(workspace, again.prepare_image(image), f"touch {marker}")
        )
        again.stream_logs(job)
        assert again.state(job).succeeded
    assert marker.exists()


def _wrong_digest(tmp_path):
    return archive(tmp_path, {"tool-1/greet": "#!/bin/sh\n"}).rsplit(":", 1)[0] + ":" + "0" * 64


def _unfetchable(tmp_path):
    return f"{(tmp_path / 'absent.tar.gz').as_uri()}@sha256:{'a' * 64}"


def _tar_of(tmp_path: Path, *members: tarfile.TarInfo) -> str:
    """A tar of ``members`` with no content, as a native image reference."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for member in members:
            tar.addfile(member)
    path = tmp_path / "crafted.tar"
    path.write_bytes(buffer.getvalue())
    return f"{path.as_uri()}@sha256:{hashlib.sha256(buffer.getvalue()).hexdigest()}"


def _directory(name: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type, member.mode = tarfile.DIRTYPE, 0o755
    return member


def _escaping_member(tmp_path):
    """A member four levels up from the unpack directory, which leaves the images directory."""
    return _tar_of(tmp_path, _directory("tool-1"), tarfile.TarInfo("tool-1/../../../../escaped"))


def _escaping_link(tmp_path):
    link = tarfile.TarInfo("tool-1/escaped")
    link.type, link.linkname = tarfile.SYMTYPE, "../../../../escaped"
    return _tar_of(tmp_path, _directory("tool-1"), link)


def _two_top_directories(tmp_path):
    return _tar_of(tmp_path, _directory("tool-1"), _directory("tool-2"))


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (_wrong_digest, LaunchError, "has sha256 [0-9a-f]{64}, not 0{64}"),
        (_escaping_member, LaunchError, "cannot unpack image"),
        (_escaping_link, LaunchError, "cannot unpack image"),
        (_two_top_directories, LaunchError, "must unpack to one directory"),
        (
            lambda tmp_path: "https://example.invalid/llama.tar.gz",
            LaunchError,
            "<archive url>@sha256:<hex>",
        ),
        (_unfetchable, BackendError, "fetching image .*/absent.tar.gz failed: "),
    ],
    ids=[
        "wrong-digest",
        "escaping-member",
        "escaping-link",
        "two-top-directories",
        "unpinned",
        "unfetchable",
    ],
)
def test_an_image_that_does_not_match_its_reference_is_refused_and_nothing_of_it_kept(
    tmp_path, workspace, make, error, message
):
    image = make(tmp_path)

    with pytest.raises(error, match=message), Lifecycle() as life:
        backend(tmp_path, life, workspace, image).prepare_image(image)
    images = tmp_path / "native" / "images"
    assert not (images.exists() and any(images.iterdir()))
    assert not list(tmp_path.rglob("escaped"))


def test_a_host_runs_one_native_job_at_a_time(tmp_path, workspace, image):
    cluster = mac_cluster(tmp_path)

    with Lifecycle() as life:
        first = NativeBackend(cluster, request(workspace, image), life)
        prepared = first.prepare_image(image)
        first.run_container(container(workspace, prepared, "sleep 300"))
        second = NativeBackend(cluster, request(workspace, image, RUNNER_NAME="mac_01"), life)
        with pytest.raises(BackendError, match="another native job runs on this host"):
            second.run_container(container(workspace, prepared, "true"))

    with Lifecycle() as life:
        second = NativeBackend(cluster, request(workspace, image, RUNNER_NAME="mac_01"), life)
        job = second.run_container(container(workspace, prepared, "true"))
        second.stream_logs(job)
        assert second.state(job).succeeded


def test_a_job_a_killed_launcher_left_running_keeps_other_native_jobs_off_the_host(
    tmp_path, workspace, image
):
    cluster = mac_cluster(tmp_path)
    orphan = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        record = tmp_path / "native" / "jobs" / "mac_01.jsonl"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"pgid": orphan.pid, "started_at": time.time()}) + "\n")

        with (
            pytest.raises(BackendError, match="mac_01's native job still runs on this host"),
            Lifecycle() as life,
        ):
            run = NativeBackend(cluster, request(workspace, image), life)
            run.run_container(container(workspace, run.prepare_image(image), "true"))
    finally:
        orphan.kill()
        orphan.wait()


def test_ending_the_launch_stops_every_process_the_job_started(tmp_path, workspace, image):
    pids = workspace / "pids"

    with Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        job = run.run_container(
            container(workspace, run.prepare_image(image), f"sleep 300 & echo $$ $! > {pids}; wait")
        )
        leader, child = map(int, wait_for(pids).split())
        assert proc.recorded_groups(job_record(tmp_path))[-1][0] == leader

    assert gone(leader) and gone(child)
    assert run.state(job).state is JobState.FAILED
    assert not job_record(tmp_path).exists()


def test_a_job_that_ignores_sigterm_is_killed_when_the_launch_ends(
    tmp_path, workspace, image, monkeypatch
):
    monkeypatch.setattr(native, "TERM_GRACE_S", 0.2)
    pids = workspace / "pids"
    script = f"trap '' TERM; sleep 300 & echo $$ $! > {pids}; wait"

    with Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        job = run.run_container(container(workspace, run.prepare_image(image), script))
        leader, child = map(int, wait_for(pids).split())

    assert life.returncode == 0
    assert job.process.returncode == -signal.SIGKILL
    assert gone(leader) and gone(child)


@pytest.fixture
def kill_stub_server(workspace):
    """Kill whatever stub server a test leaves behind, once it is done with it."""
    yield
    pid_file = workspace / "server.pid"
    if pid_file.exists():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


def test_ending_the_launch_stops_a_server_whose_supervisor_was_killed_as_it_started(
    tmp_path, workspace, image, kill_stub_server
):
    with Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        job = run.run_container(serving(tmp_path, workspace, run.prepare_image(image)))
        run.stream_logs(job)
        server = int(wait_for(workspace / "server.pid"))
        assert job.process.returncode == -signal.SIGKILL
        assert alive(server)

    assert gone(server)
    assert not job_record(tmp_path).exists()


def test_ending_the_launch_after_the_job_exited_unreaped_cleans_up_without_error(
    tmp_path, workspace, image
):
    with Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        job = run.run_container(container(workspace, run.prepare_image(image), "exit 0"))
        deadline = time.monotonic() + 20
        while (
            subprocess.run(
                ["ps", "-o", "stat=", "-p", job.id], capture_output=True, text=True
            ).stdout[:1]
            != "Z"
        ):
            assert time.monotonic() < deadline, "the job never became a zombie"
            time.sleep(0.01)

    assert life.returncode == 0
    assert run.state(job).succeeded


def test_a_job_past_its_time_limit_is_stopped_and_fails_the_launch_though_it_exits_zero(
    tmp_path, workspace, image
):
    ready = workspace / "ready"
    script = f"trap 'exit 0' TERM; echo ready > {ready}; sleep 300 & wait"

    with Lifecycle() as life:
        run = backend(tmp_path, life, workspace, image)
        prepared = run.prepare_image(image)
        job = run.run_container(container(workspace, prepared, script, time_limit_min=0))
        wait_for(ready)
        run.stream_logs(job)
        status = run.state(job)

    assert (status.state, status.exit_code) == (JobState.CANCELLED, 0)
    assert life.event.finish(1).error.type == "JobTimeout"


def test_cleanup_stops_the_job_of_a_launcher_that_was_killed(
    tmp_path, workspace, image, monkeypatch
):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(runner_config(tmp_path)))
    pid_file = workspace / "pid"
    (workspace / COLLECTOR).write_text(f"echo $$ > {pid_file}\nexec sleep 300\n")
    launcher = subprocess.Popen(
        [sys.executable, "-m", "infx.launch", "--runner-config", str(config), "run"],
        cwd=ROOT, env={**request(workspace, image).env, "PYTHONPATH": str(ROOT)},
    )  # fmt: skip
    try:
        job = int(wait_for(pid_file))
        assert proc.recorded_groups(job_record(tmp_path))[-1][0] == job
    finally:
        launcher.kill()
        launcher.wait()
    assert alive(job)
    assert (tmp_path / "native" / "jobs" / RUNNER / "home").is_dir()
    monkeypatch.setenv("RUNNER_NAME", RUNNER)

    assert main(["--runner-config", str(config), "cleanup"]) == 0

    assert gone(job)
    assert not job_record(tmp_path).exists()
    assert not (tmp_path / "native" / "jobs" / RUNNER).exists()


def test_cleanup_stops_a_group_whose_leader_has_exited(tmp_path, monkeypatch):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(runner_config(tmp_path)))
    member_pid = tmp_path / "member"
    leader = subprocess.Popen(
        ["bash", "-c", f"sleep 300 & echo $! > {member_pid}"], start_new_session=True
    )
    leader.wait()
    member = int(member_pid.read_text())
    try:
        record = job_record(tmp_path)
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"pgid": leader.pid, "started_at": time.time()}) + "\n")
        monkeypatch.setenv("RUNNER_NAME", RUNNER)

        assert main(["--runner-config", str(config), "cleanup"]) == 0

        assert gone(member)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(member, signal.SIGKILL)


def test_cleanup_leaves_alone_a_process_that_reused_the_recorded_id(tmp_path, monkeypatch):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(runner_config(tmp_path)))
    stranger = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        record = job_record(tmp_path)
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"pgid": stranger.pid, "started_at": 0.0}) + "\n")
        monkeypatch.setenv("RUNNER_NAME", RUNNER)

        assert main(["--runner-config", str(config), "cleanup"]) == 0

        assert stranger.poll() is None
        assert not record.exists()
    finally:
        stranger.kill()
        stranger.wait()
