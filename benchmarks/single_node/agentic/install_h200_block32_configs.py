"""Install the H200 V4.1 tiling configs into the imported SGLang kernel package."""  # noqa: INP001 -- Standalone recipe executable.

import hashlib
import json
import shutil
import sys
from pathlib import Path

from sglang.kernels.ops.quantization import fp8_kernel


def main() -> None:
    source = Path(sys.argv[1])
    artifacts = Path(sys.argv[2]) / "fp8_kernel_configs"
    device = fp8_kernel.get_device_name().replace(" ", "_")
    if device != "NVIDIA_H200":
        raise RuntimeError(f"H200 tiling configs cannot be installed on {device}")
    destination = Path(fp8_kernel.__file__).resolve().parent / "configs"
    destination.mkdir(exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    for n, k in [(5120, 1024), (5120, 288), (576, 5120)]:
        name = f"N={n},K={k},device_name={device},dtype=fp8_w8a8,block_shape=[32, 32].json"
        path = source / name
        expected = {int(m): config for m, config in json.loads(path.read_text()).items()}
        shutil.copyfile(path, destination / name)
        shutil.copyfile(path, artifacts / name)
        fp8_kernel.get_w8a8_block_fp8_configs.cache_clear()
        actual = fp8_kernel.get_w8a8_block_fp8_configs(n, k, 32, 32)
        if actual != expected:
            raise RuntimeError(f"SGLang did not resolve the installed config: {name}")
        print(
            f"Installed {destination / name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
        )


if __name__ == "__main__":
    main()
