"""CPU admission checks for a newly sealed NVIDIA entry; no runtime is entered."""
import os
from pathlib import Path
import subprocess

import pytest

import ci


@pytest.fixture
def entry(tmp_path):
    source = Path(ci.__file__).parent
    helper = tmp_path / "benchmark_lib.sh"
    helper.write_bytes((source.parents[1] / "benchmarks/benchmark_lib.sh").read_bytes())
    script = tmp_path / "entry.sh"
    script.write_text((source / "runtime-entry.example.sh").read_text().replace(
        "REPLACE_WITH_BENCHMARK_LIB_SHA256", ci.digest(helper)))
    return script, helper


def test_entry_reports_required_inputs_before_runtime_admission(entry):
    script, _ = entry
    result = subprocess.run(["bash", str(script)], env={"PATH": os.environ["PATH"]},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "SLURM_JOB_ID" in result.stdout
    assert "H3_EXPECTED_GPU_MODEL" in result.stdout


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_entry_rejects_unsealed_helper_before_sourcing(entry, fault):
    script, helper = entry
    sentinel = helper.with_name("must-not-source-changed-helper")
    if fault == "missing":
        helper.unlink()
    else:
        helper.write_text(f'touch "{sentinel}"\n')
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validation helper missing or changed" in result.stderr
    assert not sentinel.exists()
