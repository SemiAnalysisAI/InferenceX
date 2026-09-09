"""Prepare one task-owned cached ROCm runtime; generation is a separate gate."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import ci

REVISION = "71de97b264b04dcd514cf904003028aefe9775c8"
IMAGE = Path("/var/lib/squash/lmsysorg_sglang-rocm_v0.5.18-rocm720-mi35x-20260828.sqsh")
CONTAINER = "wenyao-minimax-h3-rocm"


def prepare_source(workspace: Path) -> Path:
    source = workspace / ("runtime-sglang-" + REVISION)
    if not source.exists():
        subprocess.run(["git", "init", "-b", "feat/h3-runtime-preparation", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", "https://github.com/sgl-project/sglang.git"], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", REVISION], check=True, timeout=300)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    ci.need(ci.command(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() == REVISION, "Prepared AMD source has a different revision")
    ci.need(not ci.command(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"]).strip(), "Prepared AMD source is not clean")
    return source


PROBE = r'''
import ctypes, importlib, importlib.metadata as metadata, json, os, sys
from pathlib import Path
out = Path(sys.argv[1])
result = {"python": sys.executable, "packages": {}, "imports": {}, "generation_executed": False}
for name in ("torch", "torchvision", "av", "numpy", "diffusers", "transformers", "sglang", "aiter", "triton"):
    try:
        module = importlib.import_module(name)
        result["imports"][name] = {"path": getattr(module, "__file__", None)}
        try: result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError: pass
    except Exception as error:
        result["imports"][name] = {"error": str(error)}
try:
    import torch
    result["torch_hip"] = torch.version.hip
    result["torch_devices"] = [{"ordinal": i, "name": torch.cuda.get_device_name(i),
        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    assert hip.hipGetDeviceCount(ctypes.byref(count)) == 0
    devices = []
    for i in range(count.value):
        bdf = ctypes.create_string_buffer(32)
        assert hip.hipDeviceGetPCIBusId(bdf, len(bdf), i) == 0
        devices.append({"hip_ordinal": i, "pci_bdf": bdf.value.decode().lower()})
    result["hip_devices"] = devices
except Exception as error:
    result["device_error"] = str(error)
result["environment"] = {k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
'''


def prepare_on_node(workspace: Path, run_dir: Path) -> None:
    # Called only after inspect_amd_node has verified this job, node and all 8 GPUs.
    root = workspace.parent
    enroot = Path("/usr/local/bin/enroot")
    ci.need(enroot.is_file(), "Expected approved /usr/local/bin/enroot is unavailable")
    rootfs = root / "enroot-data" / CONTAINER
    record = {"rootfs": str(rootfs), "existed": rootfs.is_dir(), "image": str(IMAGE),
              "source_revision": REVISION, "generation_executed": False, "started_at": ci.now()}
    ci.write(run_dir / "runtime-preparation.json", record)
    env = os.environ.copy()
    env.update(PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               ENROOT_DATA_PATH=str(root / "enroot-data"), ENROOT_CACHE_PATH=str(root / "cache"),
               ENROOT_RUNTIME_PATH=str(root / "runtime-preparation"), ENROOT_TEMP_PATH=str(root / "tmp-preparation"),
               ENROOT_MAX_PROCESSORS="8", PYTHONDONTWRITEBYTECODE="1")
    for key in ("ENROOT_DATA_PATH", "ENROOT_CACHE_PATH", "ENROOT_RUNTIME_PATH", "ENROOT_TEMP_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    origin = rootfs.with_suffix(".image.json")
    if rootfs.is_dir():
        ci.need(origin.is_file() and ci.read(origin) == {"image": str(IMAGE), "status": "created"},
                "Existing AMD rootfs has no completed task-owned image receipt; inspect before reuse")
    if not rootfs.is_dir():
        ci.need(IMAGE.is_file(), "Validated cached ROCm image is missing on this node")
        record.update(create_reason="Task-owned named rootfs is missing", image_size_bytes=IMAGE.stat().st_size)
        ci.write(run_dir / "runtime-preparation.json", record)
        ci.write(origin, {"image": str(IMAGE), "status": "creating"})
        subprocess.run([str(enroot), "create", "--name", CONTAINER, str(IMAGE)], env=env, check=True, timeout=2400)
        ci.write(origin, {"image": str(IMAGE), "status": "created"})
    rc = rootfs / "etc/rc"
    record["entrypoint"] = rc.read_text() if rc.is_file() else None
    ci.write(run_dir / "runtime-preparation.json", record)
    probe = run_dir / "runtime-probe.py"
    probe.write_text(PROBE)
    container_dir = Path("/work") / run_dir.relative_to(workspace)
    source = Path("/work") / ("runtime-sglang-" + REVISION)
    # Discover the image's Python and retained version module before importing the
    # pinned checkout. No package upgrades or model inference occur in this step.
    script = 'set -eu; command -v python3; python3 "$1" "$2"'
    entry = ["bash", "-c", script, "probe", str(container_dir / probe.name), str(container_dir / "runtime-probe.json")]
    if record["entrypoint"] and 'exec bash "$@"' in record["entrypoint"]:
        entry = entry[1:]
    argv = [str(enroot), "start", "--rw", "--mount", str(workspace) + ":/work",
            "--mount", "/dev/kfd:/dev/kfd", "--mount", "/dev/dri:/dev/dri",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "SGLANG_USE_AITER=1",
            "--env", "PYTHONPATH=" + str(source / "python"),
            "--env", "ROCR_VISIBLE_DEVICES=" + os.environ["ROCR_VISIBLE_DEVICES"], CONTAINER, *entry]
    ci.write(run_dir / "runtime-command.json", argv)
    subprocess.run(argv, env=env, check=True, timeout=300)
    result = ci.read(run_dir / "runtime-probe.json")
    record.update(finished_at=ci.now(), probe=result,
                  status="inspected", compatibility="Imports and device enumeration only; H3 generation untested")
    ci.write(run_dir / "runtime-preparation.json", record)
    ci.write(workspace / "campaigns/h3-cross-hardware/runtime-inspected.json", record)
