"""Exercise the embedded store launcher without GPU or Mooncake dependencies."""
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_setup(tmp_path: Path, budget: int, master_failure: bool = False) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "KV_OFFLOADING": "dram",
        "KV_OFFLOAD_BACKEND": "mooncake",
        "TOTAL_CPU_DRAM_GB": str(budget),
        "GPU_COUNT": "4",
        "MOONCAKE_HOST_RESERVE_GB": "40",
        "RESULT_DIR": str(tmp_path),
        "MASTER_FAILURE": "1" if master_failure else "0",
    }
    return subprocess.run(
        ["bash", "-c", r'''
set -eo pipefail
source "$1/benchmarks/benchmark_lib.sh"
agentic_pip_install() { touch "$RESULT_DIR/install-called"; }
python3() {
    case "$*" in
        *'import torch;'*) echo 13 ;;
        *'from mooncake.store import'*) return 0 ;;
        *) command python3 "$@" ;;
    esac
}
mooncake_master() {
    if [[ "$MASTER_FAILURE" == 1 ]]; then
        echo "controlled master failure" >&2
        return 17
    fi
    command python3 -c '
import os, socket, sys, time
port = int(next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--port=")))
with socket.socket() as sock:
    sock.bind(("127.0.0.1", port))
    sock.listen()
    conn, _ = sock.accept()
    conn.close()
    time.sleep(120)
' "$@"
}
cleanup() {
    stop_background_process_tree "${MOONCAKE_MASTER_PID:-}" "test master" 1
}
trap cleanup EXIT
setup_agentic_mooncake
printf '%s\n' "$PYTHONHASHSEED" > "$RESULT_DIR/hash-seed"
printf '%s\n' "${OFFLOAD_ARGS[@]}" > "$RESULT_DIR/connector-args"
''', "bash", str(REPO_ROOT)],
        env=env, capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize("budget", [108, 109])
def test_store_reserves_model_and_transfer_buffers(tmp_path: Path, budget: int) -> None:
    result = run_setup(tmp_path, budget)
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "mooncake_config.json").read_text())
    # Independently calculated byte budgets, including the 4 GiB buffers.
    assert config["global_segment_size"] == {108: 12_705_032_704, 109: 12_955_032_704}[budget]
    assert config["local_buffer_size"] == 4_294_967_296
    assert 40_000_000_000 + 4 * (config["global_segment_size"] + config["local_buffer_size"]) <= budget * 1_000_000_000
    assert config["mode"] == "embedded"
    assert config["enable_offload"] is False
    assert config["protocol"] == "rdma"
    assert 0 < int(config["master_server_address"].rsplit(":", 1)[1]) < 65536
    flag, raw = (tmp_path / "connector-args").read_text().splitlines()
    assert flag == "--kv-transfer-config"
    assert json.loads(raw) == {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
    assert (tmp_path / "hash-seed").read_text().strip() == "0"


def test_insufficient_memory_fails_before_install(tmp_path: Path) -> None:
    result = run_setup(tmp_path, 56)
    assert result.returncode != 0
    assert "Host budget cannot fit" in result.stderr
    assert not (tmp_path / "install-called").exists()


def test_master_exit_prevents_serving(tmp_path: Path) -> None:
    result = run_setup(tmp_path, 108, master_failure=True)
    assert result.returncode != 0
    assert "controlled master failure" in result.stderr
    assert not (tmp_path / "connector-args").exists()
