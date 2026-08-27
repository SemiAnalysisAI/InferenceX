#!/usr/bin/env python3
"""Pool allocators for the KV suite.

The rdma lanes use plain torch (cudaMalloc) pools. The mnnvl lane needs cuMem
FABRIC allocations: UCX's cross-node cuda_ipc only engages on fabric-mappable
memory (cudaMalloc pools silently ride the IB rails instead), and fabric
handles need a live nvidia-imex domain. Both expose the same surface: raw
``ptr``/``nbytes``/``device``, pattern fill, byte fill, and ``read8`` for
kv_workload.verify_transfer. Adapters register raw pointers, never tensors.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_int, c_size_t, c_ulonglong, c_void_p

import numpy as np

import kv_workload

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_FABRIC = 0x8
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class TorchPool:
    def __init__(self, nbytes: int, device: int):
        import torch

        self._t = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device}")
        self._torch = torch
        self.ptr, self.nbytes, self.device = self._t.data_ptr(), nbytes, device

    def fill_pattern(self) -> None:
        kv_workload.fill_pattern(self._t)
        self._torch.cuda.synchronize()

    def fill_byte(self, value: int) -> None:
        self._t.fill_(value)
        self._torch.cuda.synchronize()

    def read8(self, offset: int):
        return self._t[offset : offset + 8].cpu().numpy().tobytes()


class _AllocProp(ctypes.Structure):
    _fields_ = [("type", c_int), ("requestedHandleTypes", c_int),
                ("location_type", c_int), ("location_id", c_int),
                ("win32HandleMetaData", c_void_p),
                ("compressionType", ctypes.c_ubyte),
                ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort),
                ("reserved", ctypes.c_ubyte * 4)]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location_type", c_int), ("location_id", c_int), ("flags", c_int)]


_PATTERNS: dict[int, np.ndarray] = {}


def _pattern(nbytes: int) -> np.ndarray:
    if nbytes not in _PATTERNS:
        chunks = nbytes // 256
        vals = ((np.arange(chunks, dtype=np.int64) * 131 + 7) & 0xFF).astype(np.uint8)
        _PATTERNS[nbytes] = np.repeat(vals, 256)
    return _PATTERNS[nbytes]


class FabricPool:
    def __init__(self, nbytes: int, device: int):
        cu = self._cu = ctypes.CDLL("libcuda.so.1")
        self._check(cu.cuInit(0), "cuInit")
        dev = c_int()
        self._check(cu.cuDeviceGet(byref(dev), device), "cuDeviceGet")
        ctx = c_void_p()
        self._check(cu.cuDevicePrimaryCtxRetain(byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self._check(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        prop = _AllocProp(type=CU_MEM_ALLOCATION_TYPE_PINNED,
                          requestedHandleTypes=CU_MEM_HANDLE_TYPE_FABRIC,
                          location_type=CU_MEM_LOCATION_TYPE_DEVICE,
                          location_id=device, gpuDirectRDMACapable=1)
        gran = c_size_t()
        self._check(cu.cuMemGetAllocationGranularity(byref(gran), byref(prop), 0), "granularity")
        size = (nbytes + gran.value - 1) // gran.value * gran.value
        handle = c_ulonglong()
        code = cu.cuMemCreate(byref(handle), c_size_t(size), byref(prop), 0)
        if code != 0:
            raise RuntimeError(
                f"cuMemCreate(FABRIC) -> CUresult {code}: no IMEX fabric access on this "
                "allocation; the mnnvl lane cannot run here")
        ptr = c_ulonglong()
        self._check(cu.cuMemAddressReserve(byref(ptr), c_size_t(size), 0, 0, 0), "reserve")
        self._check(cu.cuMemMap(ptr, c_size_t(size), 0, handle, 0), "map")
        access = _AccessDesc(location_type=CU_MEM_LOCATION_TYPE_DEVICE, location_id=device,
                             flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        self._check(cu.cuMemSetAccess(ptr, c_size_t(size), byref(access), 1), "setAccess")
        self.ptr, self.nbytes, self.device = ptr.value, size, device

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} -> CUresult {code}")

    def _h2d(self, host: np.ndarray) -> None:
        self._check(self._cu.cuMemcpyHtoD_v2(
            c_ulonglong(self.ptr), host.ctypes.data_as(c_void_p), c_size_t(host.nbytes)), "h2d")

    def fill_pattern(self) -> None:
        self._h2d(_pattern(self.nbytes))

    def fill_byte(self, value: int) -> None:
        self._h2d(np.full(self.nbytes, value, dtype=np.uint8))

    def read8(self, offset: int):
        out = np.empty(8, dtype=np.uint8)
        self._check(self._cu.cuMemcpyDtoH_v2(
            out.ctypes.data_as(c_void_p), c_ulonglong(self.ptr + offset), c_size_t(8)), "d2h")
        return out.tobytes()


def create(fabric: str, nbytes: int, device: int):
    return FabricPool(nbytes, device) if fabric == "mnnvl" else TorchPool(nbytes, device)
