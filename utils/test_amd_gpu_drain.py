"""The actual GPU admission function must not interpret missing telemetry as idle."""

import os
import subprocess
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_lib.sh"


def run_gate(tmp_path, output, status=0, threshold="10", *, no_cli=False):
    source = LIBRARY.read_text()
    start = source.index("_amd_gpu_vram_max_percent() {")
    gate = source.index("wait_for_amd_gpu_clean() {")
    end = source.index("\n}\n", gate) + 3
    # Extract the unchanged function text, avoiding unrelated benchmark setup.
    script = tmp_path / "gate.sh"
    probe = ""
    if no_cli:
        probe = '\ncommand() { if [[ "$1 $2" == "-v rocm-smi" ]]; then return 1; else builtin command "$@"; fi; }\n'
    script.write_text(
        source[start:end] + probe + '\nwait_for_amd_gpu_clean "$1" "$2"\n'
    )
    binary = tmp_path / "rocm-smi"
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$SMI_OUTPUT"\nexit "$SMI_STATUS"\n')
    binary.chmod(0o755)
    sleeper = tmp_path / "sleep"
    sleeper.write_text('#!/bin/bash\nprintf "wait\\n" >> "$WAIT_LOG"\n')
    sleeper.chmod(0o755)
    return subprocess.run(
        ["bash", str(script), threshold, str(tmp_path / "drm")],
        cwd=tmp_path,
        env={
            **os.environ,
            # The gate needs only Linux coreutils and these two stubs. Avoid
            # repeated probes of inherited Windows/network PATH entries during
            # the 90-iteration no-sleep test; production wait bounds are intact.
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "SMI_OUTPUT": output,
            "SMI_STATUS": str(status),
            "WAIT_LOG": str(tmp_path / "wait.log"),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(
    "output,status",
    [
        ("", 127),
        ("driver unavailable", 1),
        ("unexpected new output format", 0),
        ("GPU[0] : GPU Memory Allocated (VRAM%): 0", 1),
        ("GPU[0] : GPU Memory Allocated (VRAM%): N/A", 0),
        ("GPU[0] : GPU Memory Allocated (VRAM%): 101", 0),
    ],
)
def test_bad_telemetry_never_passes(tmp_path, output, status):
    result = run_gate(tmp_path, output, status)
    assert result.returncode != 0, result.stdout
    assert "GPUs clean" not in result.stdout


def test_valid_idle_preserves_default_threshold(tmp_path):
    result = run_gate(
        tmp_path,
        "GPU[0] : GPU Memory Allocated (VRAM%): 0\nGPU[1] : GPU Memory Allocated (VRAM%): 10",
    )
    assert result.returncode == 0, result.stderr
    assert "GPUs clean" in result.stdout
    assert not (tmp_path / "wait.log").exists()


def test_busy_preserves_original_wait_budget(tmp_path):
    result = run_gate(tmp_path, "GPU[0] : GPU Memory Allocated (VRAM%): 11")
    assert result.returncode != 0
    assert len((tmp_path / "wait.log").read_text().splitlines()) == 90


def test_explicit_stricter_threshold(tmp_path):
    result = run_gate(
        tmp_path, "GPU[0] : GPU Memory Allocated (VRAM%): 2", threshold="1"
    )
    assert result.returncode != 0


@pytest.mark.parametrize("threshold", ["bad", "-1", "101", "1.5", "01"])
def test_invalid_threshold_is_rejected(tmp_path, threshold):
    result = run_gate(
        tmp_path, "GPU[0] : GPU Memory Allocated (VRAM%): 0", threshold=threshold
    )
    assert result.returncode != 0
    assert "GPUs clean" not in result.stdout


def sysfs_gpu(tmp_path, index, used, total="1000", vendor="0x1002"):
    device = tmp_path / "drm" / f"card{index}" / "device"
    device.mkdir(parents=True)
    for name, value in {
        "vendor": vendor,
        "mem_info_vram_used": used,
        "mem_info_vram_total": total,
    }.items():
        if value is not None:
            (device / name).write_text(f"{value}\n")


def test_no_cli_uses_all_amd_cards_and_ignores_other_vendors(tmp_path):
    sysfs_gpu(tmp_path, 0, "100")
    sysfs_gpu(tmp_path, 1, "0")
    sysfs_gpu(tmp_path, 2, None, None, vendor="0x10de")
    result = run_gate(tmp_path, "", no_cli=True)
    assert result.returncode == 0, result.stderr
    assert "vram%max=10" in result.stdout


def test_sysfs_rounds_up_and_checks_nonfirst_gpu(tmp_path):
    sysfs_gpu(tmp_path, 0, "0")
    sysfs_gpu(tmp_path, 1, "101")
    result = run_gate(tmp_path, "", no_cli=True)
    assert result.returncode != 0
    assert "vram%max=11" in result.stdout
    assert len((tmp_path / "wait.log").read_text().splitlines()) == 90


@pytest.mark.parametrize(
    "used,total",
    [
        (None, "1000"),
        ("N/A", "1000"),
        ("0", None),
        ("0", "0"),
        ("1001", "1000"),
        ("-1", "1000"),
    ],
)
def test_bad_sysfs_card_never_passes_even_if_another_is_idle(tmp_path, used, total):
    sysfs_gpu(tmp_path, 0, "0")
    sysfs_gpu(tmp_path, 1, used, total)
    result = run_gate(tmp_path, "", no_cli=True)
    assert result.returncode != 0
    assert "GPUs clean" not in result.stdout


def test_no_cli_no_cards_is_unknown_not_idle(tmp_path):
    result = run_gate(tmp_path, "", no_cli=True)
    assert result.returncode != 0
    assert "GPUs clean" not in result.stdout
