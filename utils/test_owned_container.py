"""Run the shipped task-shell lifecycle with a persistent recording runtime."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

HELPER = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/multi_node/amd_utils/run_owned_container.sh"
)


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    binary = tmp_path / "runtime"
    binary.write_text("""#!/usr/bin/env python3
import os
import signal
import sys
import time
from pathlib import Path
root = Path(os.environ["TEST_RUNTIME_ROOT"])
with (root / "calls").open("a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "run":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "ready").touch()
    if os.environ.get("PERSIST") == "1":
        while not (root / "stopped").exists():
            time.sleep(.01)
    sys.exit(int(os.environ.get("RUN_RC", "0")))
if sys.argv[1] == "stop":
    (root / "stopped").touch()
    sys.exit(int(os.environ.get("STOP_RC", "0")))
""")
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize("run_rc", [0, 7, 125])
@pytest.mark.parametrize("router", ["", "owned-router"])
def test_normal_completion_preserves_status(tmp_path, runtime, run_rc, router):
    result = subprocess.run(
        [
            "bash",
            str(HELPER),
            str(runtime),
            "owned-main",
            router,
            "--name",
            "owned-main",
            "image",
            "command",
        ],
        env={**os.environ, "TEST_RUNTIME_ROOT": str(tmp_path), "RUN_RC": str(run_rc)},
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == run_rc
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[:2] == [
        "run --name owned-main image command",
        "stop --time 10 owned-main",
    ]
    assert calls[2:] == (["rm -f owned-router"] if router else [])


@pytest.mark.parametrize(
    "sig,expected", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)]
)
@pytest.mark.parametrize("stop_rc", [0, 19])
def test_signal_interrupts_wait_and_cleans_only_owned_container(
    tmp_path,
    runtime,
    sig,
    expected,
    stop_rc,
):
    process = subprocess.Popen(
        [
            "bash",
            str(HELPER),
            str(runtime),
            "owned-main",
            "owned-router",
            "--name",
            "owned-main",
            "image",
        ],
        env={
            **os.environ,
            "TEST_RUNTIME_ROOT": str(tmp_path),
            "PERSIST": "1",
            "STOP_RC": str(stop_rc),
        },
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(sig)
        assert process.wait(timeout=5) == expected
        calls = (tmp_path / "calls").read_text().splitlines()
        assert calls[-2:] == ["stop --time 10 owned-main", "rm -f owned-router"]
        assert (tmp_path / "stopped").exists()
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
