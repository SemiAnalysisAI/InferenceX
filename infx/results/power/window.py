"""Write srt-slurm's power measurement window for one fixed-sequence result.

srt-slurm samples GPU power for the whole job; the window tells it which
interval belongs to one concurrency point, and which result it measured.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeasurementWindow:
    """One completed ``custom`` benchmark window in srt-slurm's schema version 1."""

    result_path: str
    concurrency: int
    benchmark_start_time_unix: float
    benchmark_end_time_unix: float
    duration: float
    schema_version: int = 1
    benchmark_type: str = "custom"
    clock_source: str = "head_node_unix_clock"
    status: str = "completed"
    reason: str | None = None


def write_window(result: Path, concurrency: int, windows: Path) -> None:
    """Publish the client's measured boundary as a ``custom`` benchmark window."""
    data = json.loads(result.read_text())
    window = MeasurementWindow(
        # srt-slurm resolves this against the log directory, <log dir>/<power dir>/windows/../..
        result_path=result.relative_to(windows.parent.parent).as_posix(),
        concurrency=concurrency,
        benchmark_start_time_unix=data["benchmark_start_time_unix"],
        benchmark_end_time_unix=data["benchmark_end_time_unix"],
        duration=data["duration"],
    )
    temporary = windows / f".{result.stem}.json.tmp"
    temporary.write_text(json.dumps(asdict(window), indent=2))
    temporary.replace(windows / f"{result.stem}.json")


if __name__ == "__main__":
    write_window(Path(sys.argv[1]), int(sys.argv[2]), Path(os.environ["SRT_MEASUREMENT_WINDOW_DIR"]))
