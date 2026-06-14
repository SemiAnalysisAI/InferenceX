import json
import multiprocessing
from pathlib import Path

from io_utils import append_json_line


def _append_marker_rows(path: str, writer: int) -> None:
    marker_path = Path(path)
    for sequence in range(16):
        append_json_line(
            marker_path,
            {
                "payload": "x" * 4096,
                "sequence": sequence,
                "writer": writer,
            },
        )


def test_append_json_line_serializes_concurrent_processes(tmp_path):
    marker = tmp_path / "marker.jsonl"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_append_marker_rows,
            args=(str(marker), writer),
        )
        for writer in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 128
    assert {
        (int(row["writer"]), int(row["sequence"])) for row in rows
    } == {(writer, sequence) for writer in range(8) for sequence in range(16)}
