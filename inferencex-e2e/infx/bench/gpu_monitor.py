"""GPU telemetry around a workload, in the files ``infx.results.power`` reads."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO, final

from infx.bench import proc

NVIDIA_QUERY = (
    "timestamp,index,power.draw,temperature.gpu,clocks.current.sm,"
    "clocks.current.memory,utilization.gpu,utilization.memory"
)
NVIDIA_IDENTITY = (
    "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version", "--format=csv",
)  # fmt: skip
NVIDIA_SAMPLE = ("nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader")
AMD_ENERGY = ("amd-smi", "metric", "-E", "--csv")
AMD_IDENTITY = ("amd-smi", "static", "--json")
STOP_TIMEOUT_S = 10


def _say(message: str, *, error: bool = False) -> None:
    print(f"[GPU Monitor] {message}", file=sys.stderr if error else sys.stdout, flush=True)


def _write(path: Path, command: Sequence[str], mode: str = "wb") -> bool:
    """Send ``command``'s stdout to ``path``; return whether it succeeded."""
    try:
        with path.open(mode) as out:
            done = subprocess.run(command, stdout=out, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return False
    return done.returncode == 0


def _copy_amd_rows(source: IO[bytes], sink: IO[bytes]) -> None:
    """Forward whole ``amd-smi`` rows under one header, dropping its banner and repeated headers."""
    with source, sink:
        header_seen = False
        for line in source:
            if not line.endswith(b"\n"):
                break
            if line.startswith(b"timestamp,"):
                if header_seen:
                    continue
                header_seen = True
            if header_seen:
                sink.write(line)
                sink.flush()


def _repair_tail(path: Path) -> bool:
    """Drop a partial final row; return whether ``path`` now ends on a row boundary."""
    try:
        with path.open("r+b") as stream:
            size = keep = stream.seek(0, os.SEEK_END)
            while keep:
                start = max(0, keep - 4096)
                stream.seek(start)
                newline = stream.read(keep - start).rfind(b"\n")
                if newline != -1:
                    keep = start + newline + 1
                    break
                keep = start
            if keep == size:
                return True
            stream.truncate(keep)
    except FileNotFoundError:
        return True
    except OSError:
        _say("Warning: could not repair truncated trailing sample", error=True)
        return False
    _say("Dropped truncated trailing sample")
    return True


def _count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1 << 16), b""))


@final
class GpuMonitor:
    """Sample ``nvidia-smi``, else ``amd-smi``, into output; ``vendor`` is None without either."""

    def __init__(self, output: Path, interval: int) -> None:
        self.output = output
        self.interval = interval
        self.vendor: str | None = None
        self._stem = str(output).removesuffix(".csv")
        self._sampler: subprocess.Popen[bytes] | None = None
        self._copier: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        interval = str(self.interval)
        if shutil.which("nvidia-smi"):
            self.vendor = "nvidia"
            self._sidecar("_identity.csv", NVIDIA_IDENTITY, "NVIDIA identity")
            with self.output.open("wb") as out:
                self._sampler = subprocess.Popen(
                    ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv", "-l", interval],
                    stdout=out,
                    stderr=subprocess.DEVNULL,
                )
        elif shutil.which("amd-smi"):
            self.vendor = "amd"
            sink = self.output.open("wb")
            # amd-smi is Python and block-buffers a pipe; unbuffered, no tick waits for exit.
            self._sampler = subprocess.Popen(
                ["amd-smi", "metric", "-p", "-c", "-t", "-u", "-w", interval, "--csv"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._copier = threading.Thread(
                target=_copy_amd_rows, args=(self._sampler.stdout, sink), daemon=True
            )
            self._copier.start()
            # Accumulator snapshots bracket the stream so auditors can cross-check its energy.
            self._sidecar("_energy_start.csv", AMD_ENERGY, "amd-smi metric")
            self._sidecar("_identity.json", AMD_IDENTITY, "amd-smi static")
        else:
            _say("No GPU monitoring tool found (nvidia-smi or amd-smi), skipping")
            return self
        _say(
            f"Started {self.vendor.upper()} (PID={self._sampler.pid}, "
            f"interval={self.interval}s, output={self.output})"
        )
        return self

    def __exit__(self, *_: object) -> None:
        sampler, self._sampler = self._sampler, None
        if sampler is None:
            return
        if sampler.poll() is not None:
            self._join_copier()
            rc = proc.status(sampler.returncode)
            _say(f"Warning: sampler exited early (rc={rc})", error=True)
            return
        if self.vendor == "amd":
            # The stream must bracket the workload's end, and amd-smi stamps whole seconds.
            time.sleep(self.interval + 2)
        sampler.terminate()
        try:
            sampler.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        self._join_copier()
        whole = _repair_tail(self.output)
        if self.vendor == "nvidia":
            # The stream can stop just before the workload ends; one more sample brackets it.
            if whole and not _write(self.output, NVIDIA_SAMPLE, "ab"):
                _say("Warning: final NVIDIA sample failed", error=True)
        else:
            self._sidecar("_energy_end.csv", AMD_ENERGY, "amd-smi metric")
        _say(f"Stopped (PID={sampler.pid})")
        if self.output.is_file():
            _say(f"Collected {_count_lines(self.output)} rows -> {self.output}")

    def _sidecar(self, suffix: str, command: Sequence[str], what: str) -> None:
        """Snapshot ``command`` beside the stream; a failure costs only this file."""
        path = Path(f"{self._stem}{suffix}")
        if not _write(path, command):
            path.unlink(missing_ok=True)
            _say(f"Warning: {what} sidecar failed", error=True)

    def _join_copier(self) -> None:
        copier, self._copier = self._copier, None
        if copier is not None:
            copier.join(timeout=STOP_TIMEOUT_S)
            if copier.is_alive():
                _say("Warning: amd-smi output did not close after the sampler stopped", error=True)


def run(output: Path, interval: int, command: Sequence[str]) -> int:
    """Run ``command`` under the monitor, relaying signals; return its status."""
    with proc.RelaySignals() as relay, GpuMonitor(output, interval):
        rc = relay.run(command)
    # A signal while the sampler stops still fails a clean run.
    return 128 + relay.received if rc == 0 and relay.received else rc
