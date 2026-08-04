"""NVIDIA-specific runtime probes. ``collect()`` returns software/driver
versions to merge into RunInfo.software. Empty dict when not on NVIDIA or
when probes fail."""
from __future__ import annotations


def collect() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            out["nvidia_driver"] = v.decode() if isinstance(v, bytes) else str(v)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return out
