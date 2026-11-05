"""``powermetrics``: append a Mac's power to a CSV about once a second until stopped.

Only root can read the power rails, so the account needs passwordless sudo for powermetrics.
"""

from __future__ import annotations

import argparse
import plistlib
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

COMMAND = (
    "sudo", "-n", "/usr/bin/powermetrics",
    "-i", "1000", "--samplers", "cpu_power,gpu_power", "-f", "plist",
)  # fmt: skip
HEADER = "timestamp_unix,combined_w,cpu_w,gpu_w,ane_w\n"
RAILS_MW = ("combined_power", "cpu_power", "gpu_power", "ane_power")


def _documents(stream: BinaryIO) -> Iterator[bytes]:
    """Each finished plist in ``stream``; powermetrics ends each with a NUL byte."""
    pending = b""
    while chunk := stream.read1(1 << 16):
        *finished, pending = (pending + chunk).split(b"\0")
        yield from finished


def sample(output: Path) -> int:
    """Write one row to ``output`` per powermetrics sample, and return its status."""
    with (
        output.open("w", buffering=1) as sink,
        subprocess.Popen(COMMAND, stdout=subprocess.PIPE) as sudo,
    ):
        sink.write(HEADER)
        for document in _documents(sudo.stdout):
            rails = plistlib.loads(document.strip())["processor"]
            watts = [f"{rails[name] / 1000:.3f}" for name in RAILS_MW]
            sink.write(",".join([f"{time.time():.6f}", *watts]) + "\n")
    return sudo.returncode


def main(argv: list[str]) -> int:
    """Sample power into ``--output`` until stopped, and return powermetrics' status."""
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench.powermetrics", description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="the CSV to write")
    return sample(parser.parse_args(argv).output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
