"""Resolve HIP's device alias for an already-registered V4.1 host table."""

import ctypes
from functools import lru_cache


@lru_cache(maxsize=1)
def _hip_runtime():
    # cudaHostRegister already loaded PyTorch's HIP runtime. Resolve its symbol
    # from the process instead of loading a second runtime via ldconfig: the
    # ROCm10 image's cache still resolves amdhip64 to an incompatible .so.5,
    # while PyTorch uses the SDK's .so.7.
    runtime = ctypes.CDLL(None)
    runtime.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    runtime.hipHostGetDevicePointer.restype = ctypes.c_int
    return runtime


def hip_host_device_pointer(host_pointer: int) -> int:
    """Keep CPU tensors unchanged; kernels must use the mapped device address."""
    mapped = ctypes.c_void_p()
    status = _hip_runtime().hipHostGetDevicePointer(
        ctypes.byref(mapped), ctypes.c_void_p(host_pointer), 0
    )
    if status != 0 or mapped.value is None:
        raise RuntimeError(f"hipHostGetDevicePointer failed: {status}")
    return mapped.value
