import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.dsv41 import vllm_cohort_sweep


def test_sweep_keeps_partial_results_and_stops_after_failed_case(tmp_path, monkeypatch):
    monkeypatch.setenv("CONC", "384")
    monkeypatch.setenv("ISL", "131072")
    monkeypatch.setenv("RESULT_FILENAME", "ci-result")
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep",
            "--output",
            str(tmp_path),
            "--gpu-count",
            "8",
            "--dp-size",
            "8",
            "--profile-steps",
            "8",
            "--concurrencies",
            "384",
            "1536",
            "2560",
        ],
    )

    def child(command, *, env, check):
        directory = Path(command[command.index("--output") + 1])
        batch = int(env["CONC_LIST"])
        (directory / f"{env['RESULT_FILENAME']}.json").write_text(
            json.dumps({"measured_batch": batch})
        )
        return SimpleNamespace(returncode=0 if batch == 384 else 7)

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(subprocess.CalledProcessError) as error:
        vllm_cohort_sweep.main()
    assert error.value.returncode == 7
    out = tmp_path / "research" / "cohort-sweep"
    manifest = json.loads((out / "manifest.json").read_text())
    assert [
        (r["global_batch"], r["returncode"], r["result_present"]) for r in manifest
    ] == [(384, 0, True), (1536, 7, True)]
    assert json.loads((out / "result_batch384.json").read_text()) == {
        "measured_batch": 384
    }
    assert json.loads((out / "result_batch1536.json").read_text()) == {
        "measured_batch": 1536
    }
    assert not (out / "batch2560").exists()


def test_capacity_requires_all_replicas_and_scales_input_only_bound():
    logs = [
        "(EngineCore_DP0 pid=1) GPU KV cache size: 400,000 tokens, Maximum concurrency for 4,000 tokens per request: 100.00x"
    ]
    assert vllm_cohort_sweep.capacity_bounds(logs, 2, 2000) is None
    logs.append(
        "(EngineCore_DP1 pid=2) GPU KV cache size: 320,000 tokens, Maximum concurrency for 4,000 tokens per request: 80.00x"
    )
    assert vllm_cohort_sweep.capacity_bounds(logs, 2, 2000) == {0: 200.0, 1: 160.0}


def test_sweep_records_capacity_skips_without_launching_large_cases(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONC", "384")
    monkeypatch.setenv("ISL", "131072")
    monkeypatch.setenv("RESULT_FILENAME", "ci-result")
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep",
            "--output",
            str(tmp_path),
            "--gpu-count",
            "8",
            "--dp-size",
            "8",
            "--profile-steps",
            "8",
            "--concurrencies",
            "384",
            "1536",
            "2560",
        ],
    )
    (tmp_path / "node_agg_w0.out").write_text(
        "\n".join(
            f"(EngineCore_DP{rank} pid=1) Maximum concurrency for 131,072 tokens per request: 100.00x"
            for rank in range(8)
        )
    )

    def child(command, *, env, check):
        directory = Path(command[command.index("--output") + 1])
        (directory / f"{env['RESULT_FILENAME']}.json").write_text('{"completed":384}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    vllm_cohort_sweep.main()
    out = tmp_path / "research" / "cohort-sweep"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest[0]["returncode"] == 0
    assert [
        (r["global_batch"], r["status"], r["per_dp_requested"]) for r in manifest[1:]
    ] == [(1536, "not_run_capacity_bound", 192), (2560, "not_run_capacity_bound", 320)]
    assert sorted(p.name for p in out.glob("result_batch*.json")) == [
        "result_batch384.json"
    ]
