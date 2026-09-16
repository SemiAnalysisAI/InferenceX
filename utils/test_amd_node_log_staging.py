from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPO_ROOT / "benchmarks/multi_node/amd_utils/stage_node_logs.sh"


def _stub_sudo(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/sh\nexec "$@"\n')
    sudo.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_stage_node_logs_merges_prefill_and_decode_nodes(tmp_path: Path) -> None:
    prefill = tmp_path / "prefill-node"
    decode = tmp_path / "decode-node"
    shared = tmp_path / "shared"
    prefill.mkdir()
    decode.mkdir()
    (prefill / "prefill_host-a.log").write_text("prefill output\n")
    (prefill / "server_host-a.log").write_text("frontend output\n")
    (decode / "decode_host-b.log").write_text("decode output\n")
    (decode / "server_host-b.log").write_text("decode wrapper output\n")
    env = _stub_sudo(tmp_path)

    for node_logs in (prefill, decode):
        subprocess.run(
            ["bash", str(STAGE_SCRIPT), str(node_logs), str(shared)],
            check=True,
            env=env,
        )

    assert sorted(path.name for path in shared.iterdir()) == [
        "decode_host-b.log",
        "prefill_host-a.log",
        "server_host-a.log",
        "server_host-b.log",
    ]
    assert (shared / "decode_host-b.log").read_text() == "decode output\n"


def test_stage_node_logs_rejects_a_node_without_logs(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    shared = tmp_path / "shared"

    completed = subprocess.run(
        ["bash", str(STAGE_SCRIPT), str(missing), str(shared)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "no node-local logs found" in completed.stderr
    assert not shared.exists()
