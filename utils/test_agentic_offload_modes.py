"""Exercise the shared offload gate with explicit recipe capabilities."""

import os
import subprocess
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_lib.sh"


@pytest.mark.parametrize(
    ("mode", "capacity", "extra_modes", "expected"),
    [
        ("dram", "128", "", 0),
        ("nvme", "0", "", 1),
        ("nvme", "0", "nvme", 0),
        ("dram+nvme", "128", "dram+nvme", 0),
        ("dram+nvme", "0", "dram+nvme", 1),
    ],
)
def test_recipe_offload_capabilities(mode, capacity, extra_modes, expected):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; '
            'require_agentic_kv_offload_backend vllm-native "$2"',
            "bash",
            str(LIBRARY),
            extra_modes,
        ],
        env={
            **os.environ,
            "IS_AGENTIC": "0",
            "KV_OFFLOADING": mode,
            "KV_OFFLOAD_BACKEND": "vllm-native",
            "TOTAL_CPU_DRAM_GB": capacity,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stderr
