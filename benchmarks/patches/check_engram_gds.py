"""Verify real cuFile reads into GPU memory before loading the Engram model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    # Configure cuFile before importing CUDA. Never mistake a POSIX fallback
    # through host memory for a successful GPUDirect experiment.
    config_path = args.result_dir / "cufile-gds.json"
    config = {}
    system_config = Path("/etc/cufile.json")
    if system_config.exists():
        # NVIDIA ships JSON with comments on some images; our strict minimal
        # per-process configuration intentionally does not parse that file.
        print("System cuFile config:", system_config.read_text(), flush=True)
    config["properties"] = {"allow_compat_mode": False}
    config["logging"] = {"level": "INFO", "dir": str(args.result_dir.resolve())}
    config_path.write_text(json.dumps(config, indent=2))
    os.environ["CUFILE_ENV_PATH_JSON"] = str(config_path.resolve())
    os.environ["CUFILE_FORCE_COMPAT_MODE"] = "false"
    for command in (["nvidia-smi"], ["findmnt", "-T", str(args.directory)],
                    ["ls", "-l", "/dev/nvidia-fs"],
                    ["sh", "-c", "ls /usr/local/cuda*/gds/tools/gdscheck* 2>/dev/null"]):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        print("DIAGNOSTIC", command, result.returncode, result.stdout, result.stderr, flush=True)
    # Import only after fallback has been disabled.
    import torch
    from torch.cuda import gds

    print("torch", torch.__version__, "CUDA", torch.version.cuda, flush=True)
    print("strict cuFile configuration", config, flush=True)
    nbytes = 16 * 1024 * 1024
    # Every page and every byte position carries a different pattern so both
    # addressing errors and incorrect payloads are detected.
    indexes = torch.arange(nbytes, dtype=torch.int64)
    expected = ((indexes // 4096 + indexes % 4096) % 251).to(torch.uint8)
    del indexes
    report = {"compatibility_fallback": False, "checks": []}
    with tempfile.TemporaryDirectory(prefix="engram-gds-probe-", dir=args.directory) as tmp:
        filename = Path(tmp) / "pages.bin"
        with filename.open("wb") as output:
            output.write(expected.numpy().tobytes())
            output.flush()
            os.fsync(output.fileno())
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                storage = buffer.untyped_storage()
                torch.cuda.synchronize()
                gds.gds_register_buffer(storage)
                handle = None
                try:
                    handle = gds.GdsFile(str(filename), os.O_RDONLY)
                    start = time.perf_counter()
                    handle.load_storage(storage, offset=0)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if not torch.equal(buffer.cpu(), expected):
                        raise RuntimeError("GDS bulk payload mismatch")
                    page = torch.empty(4096, dtype=torch.uint8, device="cuda")
                    page_storage = page.untyped_storage()
                    gds.gds_register_buffer(page_storage)
                    try:
                        latencies = []
                        for page_id in (0, 4095, 17, 2048, 1, 1023, 3071, 53):
                            offset = page_id * 4096
                            start = time.perf_counter()
                            handle.load_storage(page_storage, offset=offset)
                            torch.cuda.synchronize()
                            latencies.append(time.perf_counter() - start)
                            if not torch.equal(page.cpu(), expected[offset:offset + 4096]):
                                raise RuntimeError(f"GDS page payload mismatch at {offset}")
                    finally:
                        gds.gds_deregister_buffer(page_storage)
                    report["checks"].append({"device": device, "bytes": nbytes,
                        "bulk_seconds": elapsed, "page_read_seconds": latencies})
                    print("GDS payload verification PASSED", report["checks"][-1], flush=True)
                finally:
                    if handle is not None:
                        # GdsFile owns the descriptor and releases it on destruction.
                        del handle
                    gds.gds_deregister_buffer(storage)
    if not report["checks"]:
        raise RuntimeError("No CUDA devices were visible to the GDS probe")
    (args.result_dir / "gds_probe.json").write_text(json.dumps(report, indent=2))
    print("GDS_PROBE_PASS: direct reads verified on every visible GPU; no model benchmark ran", flush=True)


if __name__ == "__main__":
    main()
