"""AMD-specific runtime probes."""
from __future__ import annotations


def collect() -> dict[str, str]:
    return {}  # FIXME: rocm-smi --showdriverversion or /sys/module/amdgpu/version
