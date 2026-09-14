"""Enumerate RDMA devices through each libibverbs a Mooncake process loads.

With --init DEVICE, also create a Mooncake transfer engine on that device,
which exercises provider loading, port query and RDMA context creation
without a model or a GPU.
"""
import ctypes
import ctypes.util
import importlib.metadata
import socket
import sys
from pathlib import Path


def enumerate_devices(lib):
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    count = ctypes.c_int()
    devices = lib.ibv_get_device_list(ctypes.byref(count))
    names = [lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
    if devices:
        lib.ibv_free_device_list(devices)
    return names


print("system_devices", enumerate_devices(ctypes.CDLL(ctypes.util.find_library("ibverbs"))), flush=True)
from mooncake.engine import TransferEngine  # noqa: E402

print("mooncake_version", importlib.metadata.version("mooncake-transfer-engine-cuda13"), flush=True)
loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/" in line and any(s in line for s in ("libibverbs", "libmlx5", "libefa", "mooncake"))})
for library in loaded:
    print("loaded", library, flush=True)
    if "libibverbs" in library:
        print("devices", enumerate_devices(ctypes.CDLL(library)), flush=True)

if "--init" in sys.argv:
    device = sys.argv[sys.argv.index("--init") + 1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = TransferEngine()
    rc = engine.initialize(f"127.0.0.1:{port}", "P2PHANDSHAKE", "rdma", device)
    print("transfer_engine_initialize", device, "rc", rc, flush=True)
    sys.exit(0 if rc == 0 else 4)
