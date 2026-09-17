"""Experimental GPU row gather backed by batched, direct cuFile page reads.

Only page offsets cross to the host. Table payloads go directly from storage to
registered CUDA buffers, then a Triton kernel selects the requested row bytes.
The caller must disable cuFile compatibility fallback before importing CUDA.
"""

import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import CUDA_HOME, load

_native_module = None
_native_lock = threading.Lock()


def native_module():
    global _native_module
    with _native_lock:
        if _native_module is None:
            if CUDA_HOME is None:
                raise RuntimeError(
                    "GDS batch reader requires the CUDA development toolkit"
                )
            package_root = Path(torch.__file__).resolve().parent.parent
            headers = list((package_root / "nvidia").rglob("cufile.h"))
            libraries = list((package_root / "nvidia").rglob("libcufile.so*"))
            include_dirs = [str(Path(CUDA_HOME) / "include")] + [
                str(p.parent) for p in headers
            ]
            cufile_link = str(libraries[0]) if libraries else "-lcufile"
            _native_module = load(
                name="infx_engram_gds",
                sources=[str(Path(__file__).with_suffix(".cpp"))],
                extra_include_paths=include_dirs,
                extra_ldflags=[f"-L{CUDA_HOME}/lib64", cufile_link, "-lcudart"],
                extra_cflags=["-O3"],
                with_cuda=True,
                verbose=True,
            )
    return _native_module


@triton.jit
def _gather_page_rows(
    pages,
    row_ids,
    page_slots,
    output,
    nrows,
    start,
    stop,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(page_slots + row, mask=row < nrows, other=-1).to(tl.int64)
    valid = (row < nrows) & (slot >= start) & (slot < stop)
    row_id = tl.load(row_ids + row, mask=valid, other=0).to(tl.int64)
    column = tl.arange(0, WIDTH)
    address = (slot - start) * 4096 + (row_id % (4096 // WIDTH)) * WIDTH
    value = tl.load(
        pages + address[:, None] + column[None, :], mask=valid[:, None], other=0
    )
    tl.store(
        output + row[:, None] * WIDTH + column[None, :], value, mask=valid[:, None]
    )


class GDSRows:
    """One immutable page-aligned table and a bounded reusable GPU page buffer."""

    def __init__(
        self,
        path: str,
        width: int,
        max_rows: int,
        page_capacity: int,
        device: torch.device,
    ):
        if width <= 0 or width > 4096 or 4096 % width:
            raise ValueError("GDS rows must divide a 4096-byte page")
        if max_rows <= 0 or page_capacity <= 0:
            raise ValueError("GDS buffer capacities must be positive")
        size = Path(path).stat().st_size
        if size == 0 or size % 4096:
            raise ValueError("GDS table must be padded to complete 4096-byte pages")
        self.width = width
        self.max_rows = max_rows
        self.page_capacity = page_capacity
        self.device = device
        self._lock = threading.Lock()
        self._allocation = torch.empty(
            page_capacity * 4096 + 4095, dtype=torch.uint8, device=device
        )
        alignment = (-self._allocation.data_ptr()) % 4096
        self.pages = self._allocation[alignment : alignment + page_capacity * 4096]
        self.output = torch.empty((max_rows, width), dtype=torch.uint8, device=device)
        self._reader = native_module().PageReader(
            path, self.pages.data_ptr(), page_capacity, device.index, self.pages
        )
        self.read_bytes = 0
        self.lookup_count = 0

    def read_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.device != self.device or rows.dtype != torch.int64 or rows.ndim != 1:
            raise ValueError(
                "Expected a one-dimensional int64 row tensor on the reader GPU"
            )
        if rows.numel() > self.max_rows:
            raise ValueError("GDS row output capacity exceeded")
        if not rows.numel():
            return self.output[:0]
        with self._lock, torch.cuda.device(self.device):
            # GPU deduplication; only page descriptors cross to the CPU.
            page_ids, slots = torch.unique(
                rows // (4096 // self.width), sorted=True, return_inverse=True
            )
            descriptors = page_ids.cpu().tolist()
            if descriptors[0] < 0:
                raise ValueError("GDS row IDs must be nonnegative")
            for start in range(0, len(descriptors), self.page_capacity):
                stop = min(start + self.page_capacity, len(descriptors))
                # The next DMA must not overwrite pages still used by the last
                # GPU gather. Synchronize this stream, not every stream/device.
                torch.cuda.current_stream().synchronize()
                self.read_bytes += self._reader.read(descriptors[start:stop])
                _gather_page_rows[(triton.cdiv(rows.numel(), 16),)](
                    self.pages,
                    rows,
                    slots,
                    self.output,
                    rows.numel(),
                    start,
                    stop,
                    WIDTH=self.width,
                    BLOCK=16,
                )
            torch.cuda.current_stream().synchronize()
            self.lookup_count += 1
            return self.output[: rows.numel()]

    def close(self) -> None:
        with self._lock, torch.cuda.device(self.device):
            torch.cuda.current_stream().synchronize()
            # Release the registered native pointer before its tensor owner.
            self._reader = None
