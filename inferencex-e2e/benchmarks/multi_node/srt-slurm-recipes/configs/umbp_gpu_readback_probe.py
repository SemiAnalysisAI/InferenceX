"""Bounded auxiliary GPU probe matching the image connector client/retry setup.

Never interprets RPC False as success. Exit42 means byte corruption (fatal).
Other failures are recorded separately so real connector verification may run.
"""

import ctypes
import json
import os
from pathlib import Path
import socket
import time

import torch
from mori.cpp import UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig
from mori.io import MemoryLocationType
from mori.umbp import UMBPHostBufferBacking, UMBPHostMemAllocator


class IntegrityError(RuntimeError):
    pass


def make_client():
    cfg = UMBPConfig()
    standalone = UMBPStandaloneProcessConfig()
    standalone.address = "unix://" + os.environ["UMBP_ENDPOINT"]
    standalone.auto_start = False
    standalone.startup_timeout_ms = 30000
    cfg.standalone_process = standalone
    return UMBPClient(cfg)


node = socket.gethostname().split(".")[0]
logs = Path("/logs")
peer = json.loads((logs / f"pool-gate-{node}.json").read_text())["peer"]
job = os.environ["SLURM_JOB_ID"]
size = 8 << 20
report = {
    "passed": False,
    "node": node,
    "peer": peer,
    "bytes": size,
    "devices": [],
    "scope": "Auxiliary GPU-only client; connector-native two store retries. Peer origin is known; exact placement is reported by native range traces.",
}
target = logs / f"gpu-readback-{node}.json"


def save():
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(target)


def get_checked(client, key, tensor, gpu_ptr, expected):
    tensor.zero_()
    torch.cuda.synchronize(tensor.device)
    result = list(client.batch_get_ranges_into_ptr([key], [[gpu_ptr]], [[size]], [[0]]))
    if result == [True] and not bool(torch.all(tensor == expected).item()):
        raise IntegrityError(f"GPU byte mismatch for {key}")
    return result == [True]


host = None
allocator = None
handle = None
ptr = None
exit_code = 2
try:
    host = make_client()
    allocator = UMBPHostMemAllocator()
    handle = allocator.alloc(size, UMBPHostBufferBacking.AnonymousShm, 2 << 20, -1, True)
    assert handle
    ptr = int(handle.ptr)
    assert host.register_memory(ptr, size, MemoryLocationType.CPU, -1)
    assert torch.cuda.device_count() == 8
    for device in range(8):
        row = {
            "device": device,
            "registered": False,
            "put_attempts": [],
            "put_ok": False,
            "gpu_roundtrip_ok": False,
            "host_to_gpu_ok": False,
            "peer_to_gpu_ok": False,
        }
        report["devices"].append(row)
        with torch.cuda.device(device):
            tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{device}")
            storage = tensor.untyped_storage()
            gpu_ptr = storage.data_ptr()
            assert gpu_ptr == tensor.data_ptr() and storage.nbytes() == size
            client = make_client()
            byte = (41 if node < peer else 141) + device * 7
            gpu_key = f"gp-{job}-{node[-3:]}-{device}"
            host_key = f"hg-{job}-{node[-3:]}-{device}"
            try:
                row.update(
                    registered=bool(
                        client.register_memory(
                            gpu_ptr, storage.nbytes(), MemoryLocationType.GPU, device
                        )
                    ),
                    gpu_pointer=hex(gpu_ptr),
                    storage_bytes=storage.nbytes(),
                )
                print("GPU_REGISTER", json.dumps(row), flush=True)
                if not row["registered"]:
                    save()
                    continue
                tensor.fill_(byte)
                torch.cuda.synchronize(device)
                # Initial call plus exactly the two retries in image83d42f09.
                for attempt in range(3):
                    if attempt:
                        row.setdefault("retry_flushes", []).append(bool(client.flush()))
                        time.sleep(0.01 * attempt)
                    result = list(
                        client.batch_put_ranges_from_ptr(
                            [gpu_key], [size], [[gpu_ptr]], [[size]], [[0]]
                        )
                    )
                    row["put_attempts"].append(result)
                    print("GPU_PUT_RESULT", device, attempt, result, flush=True)
                    if result == [True]:
                        row["put_ok"] = True
                        break
                row["final_flush_ok"] = bool(client.flush())
                row["put_ok"] = row["put_ok"] and row["final_flush_ok"]
                if row["put_ok"]:
                    row["gpu_roundtrip_ok"] = get_checked(client, gpu_key, tensor, gpu_ptr, byte)
                # Independent known-correct host seed isolates the unexercised
                # host->GPU direction even if this generic GPU put is refused.
                ctypes.memset(ptr, byte, size)
                seed_result = list(
                    host.batch_put_ranges_from_ptr([host_key], [size], [[ptr]], [[size]], [[0]])
                )
                assert seed_result == [True] and host.flush()
                marker = logs / f"gpu-seed-{job}-{node}-{device}.json"
                temporary = marker.with_suffix(".tmp")
                temporary.write_text(json.dumps({"key": host_key, "byte": byte}))
                temporary.replace(marker)
                row["host_to_gpu_ok"] = get_checked(client, host_key, tensor, gpu_ptr, byte)
                remote_marker = logs / f"gpu-seed-{job}-{peer}-{device}.json"
                deadline = time.monotonic() + 60
                while not remote_marker.exists():
                    assert time.monotonic() < deadline, "Peer GPU seed timed out"
                    time.sleep(0.25)
                remote = json.loads(remote_marker.read_text())
                row["peer_to_gpu_ok"] = get_checked(
                    client, remote["key"], tensor, gpu_ptr, remote["byte"]
                )
                print("GPU_DEVICE_RESULT", json.dumps(row), flush=True)
            finally:
                if row["registered"]:
                    try:
                        row["deregister_return"] = client.deregister_memory(gpu_ptr)
                    except Exception as cleanup_error:
                        row["cleanup_error"] = repr(cleanup_error)
                del client
                del storage
                del tensor
                torch.cuda.empty_cache()
            save()
    report["all_gpu_reads_passed"] = len(report["devices"]) == 8 and all(
        r["host_to_gpu_ok"] and r["peer_to_gpu_ok"] for r in report["devices"]
    )
    report["passed"] = report["all_gpu_reads_passed"] and all(
        r["put_ok"] and r["gpu_roundtrip_ok"] for r in report["devices"]
    )
    exit_code = 0 if report["passed"] else 2
except IntegrityError as error:
    report["fatal_byte_corruption"] = str(error)
    exit_code = 42
except Exception as error:
    report["error"] = repr(error)
finally:
    if host is not None and ptr is not None:
        try:
            host.deregister_memory(ptr)
        except Exception as cleanup_error:
            report["host_cleanup_error"] = repr(cleanup_error)
    if allocator is not None and handle is not None:
        try:
            allocator.free(handle)
        except Exception as cleanup_error:
            report["allocation_cleanup_error"] = repr(cleanup_error)
    report["exit_code"] = exit_code
    save()
    print(json.dumps(report), flush=True)
raise SystemExit(exit_code)
