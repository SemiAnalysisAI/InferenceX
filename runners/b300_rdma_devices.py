import ctypes
import ctypes.util
import importlib.metadata
import os
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
from mooncake.store import MooncakeDistributedStore
print("mooncake_version", importlib.metadata.version("mooncake-transfer-engine-cuda13"), flush=True)
loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/" in line and any(s in line for s in ("libibverbs", "libmlx5", "libefa", "mooncake"))})
for library in loaded:
    print("loaded", library, flush=True)
    if "libibverbs" in library:
        print("devices", enumerate_devices(ctypes.CDLL(library)), flush=True)
print("store_setup_signature", MooncakeDistributedStore.setup.__doc__, flush=True)
print("selected_device", os.environ["MOONCAKE_RAIL"], flush=True)
