import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.dsv41 import vllm_cohort_sweep


def test_sweep_keeps_partial_results_and_stops_after_failed_case(tmp_path, monkeypatch):
    monkeypatch.setenv("CONC", "384")
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
