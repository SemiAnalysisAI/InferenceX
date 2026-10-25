"""Native UMBP byte gate: own local seed then read the other node's key."""

import ctypes
import json
import os
from pathlib import Path
import socket
import time

from mori.cpp import UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig
from mori.io import MemoryLocationType
from mori.umbp import UMBPHostBufferBacking, UMBPHostMemAllocator

node = socket.gethostname().split(".")[0]
logs = Path("/logs")
prefix = "umbp-gate-" + os.environ["SLURM_JOB_ID"] + "-"
cfg = UMBPConfig()
standalone = UMBPStandaloneProcessConfig()
standalone.address = "unix://" + os.environ["UMBP_ENDPOINT"]
standalone.auto_start = False
cfg.standalone_process = standalone
client = UMBPClient(cfg)
# StandaloneProcessClient::IsDistributed() is hard-coded false: it describes
# this IPC proxy, not the remote server backend. Cross-node bytes are the gate.
print("PROXY_MODE", client.get_backend_mode(), flush=True)
allocator = UMBPHostMemAllocator()
size = 8 << 20
handle = allocator.alloc(size, UMBPHostBufferBacking.AnonymousShm, 2 << 20, -1, True)
assert handle
ptr = int(handle.ptr)
assert client.register_memory(ptr, size, MemoryLocationType.CPU, -1)
key = prefix + node
value = bytes((sum(node.encode()) % 251 + 1,)) * size
try:
    ctypes.memmove(ptr, value, size)
    assert list(client.batch_put_ranges_from_ptr([key], [size], [[ptr]], [[size]], [[0]])) == [True]
    assert client.flush()
    marker = logs / f"{prefix}{node}.seed.json"
    marker.write_text(json.dumps({"key": key, "node": node, "byte": value[0]}))
    print("LOCAL_SEED_OK", node, flush=True)
    deadline = time.monotonic() + 120
    while True:
        peers = [p for p in logs.glob(prefix + "*.seed.json") if p != marker]
        if len(peers) == 1:
            try:
                peer = json.loads(peers[0].read_text())
                break
            except json.JSONDecodeError:
                pass
        assert time.monotonic() < deadline, "Peer did not complete local seed"
        time.sleep(0.25)
    assert peer["node"] != node
    assert list(client.batch_exists([peer["key"]])) == [True]
    for i in range(3):
        ctypes.memset(ptr, 0, size)
        begin = time.monotonic()
        assert list(client.batch_get_ranges_into_ptr([peer["key"]], [[ptr]], [[size]], [[0]])) == [
            True
        ]
        assert ctypes.string_at(ptr, size) == bytes((peer["byte"],)) * size
        print("PEER_READ_BYTES_OK", peer["node"], i, time.monotonic() - begin, flush=True)
    # Exercise the native model data path explicitly, without waiting for
    # a large trace corpus to evict GPU prefixes. All bytes go through the
    # same public ranged API as the connector; engine/package files stay intact.
    gpu_count = int(os.environ.get("UMBP_VERIFY_GPU_COUNT", "0"))
    if gpu_count:
        import torch

        assert 0 < gpu_count <= torch.cuda.device_count()
        for device in range(gpu_count):
            with torch.cuda.device(device):
                tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{device}")
                gpu_ptr = tensor.data_ptr()
                assert client.register_memory(gpu_ptr, size, MemoryLocationType.GPU, device)
                try:
                    byte = (value[0] + device) % 251 + 1
                    tensor.fill_(byte)
                    torch.cuda.synchronize(device)
                    gpu_key = key + f"-gpu-{device}"
                    assert list(
                        client.batch_put_ranges_from_ptr(
                            [gpu_key], [size], [[gpu_ptr]], [[size]], [[0]]
                        )
                    ) == [True]
                    assert client.flush()
                    # Destroy the source value before verifying host-pool readback.
                    tensor.zero_()
                    torch.cuda.synchronize(device)
                    assert list(
                        client.batch_get_ranges_into_ptr([gpu_key], [[gpu_ptr]], [[size]], [[0]])
                    ) == [True]
                    assert bool(torch.all(tensor == byte).item())
                    gpu_marker = logs / f"{prefix}{node}-gpu-{device}.json"
                    temp_marker = gpu_marker.with_suffix(".tmp")
                    temp_marker.write_text(json.dumps({"key": gpu_key, "byte": byte}))
                    temp_marker.replace(gpu_marker)
                    peer_marker = logs / f"{prefix}{peer['node']}-gpu-{device}.json"
                    deadline = time.monotonic() + 90
                    while not peer_marker.exists():
                        assert time.monotonic() < deadline, "Peer GPU seed missing"
                        time.sleep(0.25)
                    remote = json.loads(peer_marker.read_text())
                    tensor.zero_()
                    torch.cuda.synchronize(device)
                    assert list(
                        client.batch_get_ranges_into_ptr(
                            [remote["key"]], [[gpu_ptr]], [[size]], [[0]]
                        )
                    ) == [True]
                    assert bool(torch.all(tensor == remote["byte"]).item())
                    print("GPU_LOCAL_AND_PEER_BYTES_OK", device, size, flush=True)
                finally:
                    client.deregister_memory(gpu_ptr)
                    del tensor
                    torch.cuda.empty_cache()
    pid = int((logs / f"umbp-{node}.pid").read_text())
    os.kill(pid, 0)
    report = {
        "passed": True,
        "node": node,
        "peer": peer["node"],
        "bytes": size,
        "reads": 3,
        "daemon_pid": pid,
        "gpu_devices_verified": gpu_count,
        "gpu_local_and_peer_reads": gpu_count * 2,
        "scope": "Native host shared-buffer ranged API; local_first seed followed by other-node key read. Cache may make repeats local.",
    }
    (logs / f"pool-gate-{node}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
finally:
    client.deregister_memory(ptr)
    allocator.free(handle)
