import os
import shlex
import subprocess
from pathlib import Path

SIDECAR = Path(__file__).resolve().parents[3] / "benchmarks/single_node/serving/sidecar.sh"


def run(mode: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    full = {"PATH": os.environ["PATH"], "RESULT_DIR": str(tmp_path), **env}
    return subprocess.run(["bash", str(SIDECAR), mode], env=full, capture_output=True, text=True, check=False)


def test_generic_router_expands_engine_and_router_ports(tmp_path):
    command = shlex.join(["echo", "--worker-urls", "{{engine_urls}}", "--port={{router_port}}", "{{ keep }}"])
    out = run("router", tmp_path, ROUTER_CMD=command, ROUTER_PROFILE="generic", PORT="21000",
              ENGINE_PORT="21001", ENGINE_PORTS="21001 21002", ROUTER_ENV="TOKEN={{engine_port}}")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "--worker-urls http://127.0.0.1:21001 http://127.0.0.1:21002 --port=21000 {{ keep }}"
    assert "router env: TOKEN" in out.stderr and "21001" not in out.stderr  # names only, never values
    assert (tmp_path / "router_command.txt").read_text().startswith("echo ")


def test_dynamo_router_gets_the_http_flags(tmp_path):
    out = run("router", tmp_path, ROUTER_CMD="echo frontend", ROUTER_PROFILE="dynamo", PORT="21000")
    assert out.stdout.strip() == "frontend --http-host 0.0.0.0 --http-port 21000"


def test_storage_expands_its_own_ports(tmp_path):
    out = run("storage", tmp_path, STORAGE_CMD=shlex.join(["echo", "--port={{storage_master_port}}"]),
              STORAGE_MASTER_PORT="22000", STORAGE_META_PORT="22001")
    assert out.stdout.strip() == "--port=22000"


def test_shell_syntax():
    for name in ("job.sh", "bench.sh", "sidecar.sh"):
        assert subprocess.run(["bash", "-n", str(SIDECAR.with_name(name))], check=False).returncode == 0
