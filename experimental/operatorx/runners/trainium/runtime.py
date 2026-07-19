"""Trainium-specific runtime probes."""
from __future__ import annotations


def collect() -> dict[str, str]:
    return {}  # FIXME: neuron-ls -j (driver/firmware) or dpkg -s aws-neuronx-runtime-lib
