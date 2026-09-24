"""The fixed-sequence client's power window matches srt-slurm's custom-benchmark contract."""

import json

from infx.results.power.window import write_window


def test_window_brackets_the_measured_result(tmp_path):
    result = tmp_path / "sa-bench_isl_8192_osl_1024" / "results_concurrency_16_gpus_16_ctx_8_gen_8.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps({"benchmark_start_time_unix": 1000.0, "benchmark_end_time_unix": 1060.5, "duration": 60.5})
    )
    windows = tmp_path / "power" / "windows"
    windows.mkdir(parents=True)

    write_window(result, 16, windows)

    assert json.loads((windows / f"{result.stem}.json").read_text()) == {
        "schema_version": 1,
        "benchmark_type": "custom",
        "result_path": f"sa-bench_isl_8192_osl_1024/{result.name}",
        "concurrency": 16,
        "benchmark_start_time_unix": 1000.0,
        "benchmark_end_time_unix": 1060.5,
        "duration": 60.5,
        "clock_source": "head_node_unix_clock",
        "status": "completed",
        "reason": None,
    }
    assert [p.name for p in windows.iterdir()] == [f"{result.stem}.json"]
