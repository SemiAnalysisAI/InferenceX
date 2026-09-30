"""AMD-specific runtime probes."""

from __future__ import annotations


def collect() -> dict[str, str]:
    import torch

    if not torch.version.hip:
        return {}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":")[0]
    return {
        "rocm": str(torch.version.hip),
        "gpu_name": str(props.name),
        "gpu_arch": arch,
        "l2_cache_bytes": str(props.L2_cache_size),
        "fp8_dtype": "float8_e4m3fnuz" if arch == "gfx942" else "float8_e4m3fn",
    }
