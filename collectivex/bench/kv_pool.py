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

    def fill_pattern(self, salt: int = 0) -> None:
        kv_workload.fill_pattern(self._t, salt)
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

    def _sync(self) -> None:
        # Device-side memset/copy are asynchronous to the host; the pool must be
        # painted before the barrier that lets the peer transfer.
        self._check(self._cu.cuCtxSynchronize(), "sync")

    def fill_pattern(self, salt: int = 0) -> None:
        """Upload one pattern period, then double it in place: the pattern is
        PATTERN_PERIOD-periodic and every copy's destination sits at a multiple
        of the period, so a copy of [0, n) lands the right bytes. No host
        buffer of pool size is ever built."""
        tile = kv_workload.pattern_tile(salt)
        filled = min(tile.nbytes, self.nbytes)
        self._check(self._cu.cuMemcpyHtoD_v2(
            c_ulonglong(self.ptr), tile.ctypes.data_as(c_void_p), c_size_t(filled)), "h2d")
        while filled < self.nbytes:
            n = min(filled, self.nbytes - filled)
            self._check(self._cu.cuMemcpyDtoD_v2(
                c_ulonglong(self.ptr + filled), c_ulonglong(self.ptr), c_size_t(n)), "d2d")
            filled += n
        self._sync()

    def fill_byte(self, value: int) -> None:
        self._check(self._cu.cuMemsetD8_v2(
            c_ulonglong(self.ptr), ctypes.c_ubyte(value), c_size_t(self.nbytes)), "memset")
        self._sync()

    def read8(self, offset: int):
        out = np.empty(8, dtype=np.uint8)
        self._check(self._cu.cuMemcpyDtoH_v2(
            out.ctypes.data_as(c_void_p), c_ulonglong(self.ptr + offset), c_size_t(8)), "d2h")
        return out.tobytes()


def create(fabric: str, nbytes: int, device: int):
    return FabricPool(nbytes, device) if fabric == "mnnvl" else TorchPool(nbytes, device)
