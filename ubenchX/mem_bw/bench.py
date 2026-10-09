"""Device memory copy bandwidth from 8 B to 16 GiB.

Copies a float32 tensor with ``b.copy_(a)`` at every power-of-two size and
times it with ``triton.testing.do_bench``. Bandwidth counts one read and one
write per byte. Results go to ``results/<gpu>.csv``.

Usage: python ubenchX/mem_bw/bench.py
"""

import csv
import re
from pathlib import Path

import torch
import triton

MIN_BYTES = 8
MAX_BYTES = 16 * 1024**3
DTYPE = torch.float32
ELEMENT_BYTES = torch.finfo(DTYPE).bits // 8


def main():
    sizes = [1 << p for p in range(MIN_BYTES.bit_length() - 1, MAX_BYTES.bit_length())]

    # Allocate the largest pair that fits once; smaller sizes reuse prefix views.
    free_bytes, _ = torch.cuda.mem_get_info()
    fitting = [s for s in sizes if 2 * s < free_bytes]
    skipped = [s for s in sizes if s not in fitting]
    max_elements = fitting[-1] // ELEMENT_BYTES
    a = torch.randn(max_elements, device="cuda", dtype=DTYPE)
    b = torch.empty_like(a)

    gpu = torch.cuda.get_device_name()
    rows = []
    for size in fitting:
        n = size // ELEMENT_BYTES
        src, dst = a[:n], b[:n]
        time_ms = triton.testing.do_bench(lambda src=src, dst=dst: dst.copy_(src))
        bandwidth_gbps = (size * 2) / (time_ms * 1e-3) / 1e9
        rows.append(
            {"bytes": size, "time_ms": time_ms, "bandwidth_gbps": bandwidth_gbps}
        )
        print(f"{size:>14} B  {time_ms:10.4f} ms  {bandwidth_gbps:10.2f} GB/s")
    for size in skipped:
        print(
            f"{size:>14} B  skipped: source and destination do not fit in free memory"
        )

    out = (
        Path(__file__).parent
        / "results"
        / f"{re.sub(r'[^A-Za-z0-9]+', '_', gpu).strip('_')}.csv"
    )
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bytes", "time_ms", "bandwidth_gbps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
