import json

from bench_offline.summarize import (
    discover_results,
    markdown,
    renderer_row,
    renderer_rows,
    result_row,
)


def successful_result(global_batch_size=64):
    return {
        "status": "success",
        "benchmark": {
            "experiment_id": f"gbs{global_batch_size}",
            "global_batch_size": global_batch_size,
            "concurrency": global_batch_size,
            "local_batch_size": global_batch_size // 8,
            "active_gpu_count": 8,
            "effective_parallelism": "DEP8",
            "input_tokens": 8192,
            "generated_output_tokens": 1025,
        },
        "provenance": {
            "image": "ghcr.io#semianalysisai/trtllm-dsv4:test",
        },
        "aggregate": {
            "decode_round_tpot_ms": 20.0,
            "median_decode_round_tpot_ms": 19.0,
            "p90_decode_round_tpot_ms": 22.0,
            "p99_decode_round_tpot_ms": 24.0,
            "decode_step_tput_per_gpu": 400.0,
            "observed_tokens_per_step": 2.5,
            "equivalent_output_tpot_ms": 8.0,
            "output_tput_per_gpu": 1000.0,
            "wall_output_tput_per_gpu": 700.0,
            "local_batch_size": global_batch_size // 8,
            "measured_decode_rounds": 256,
            "effective_acceptance_rate": 0.5,
            "token_yield_source": "iteration_spec_decoding_stats",
            "filter": {
                "retained_rounds": 254,
                "outlier_rounds": 1,
            },
            "mean_ttft_ms": 5000.0,
            "median_ttft_ms": 4900.0,
            "p90_ttft_ms": 6000.0,
            "p99_ttft_ms": 6500.0,
            "mean_e2e_ms": 25000.0,
            "median_e2e_ms": 24000.0,
            "p90_e2e_ms": 28000.0,
            "p99_e2e_ms": 30000.0,
        },
        "huawei": {
            "decode_round_tpot_ms": 19.03,
            "decode_step_tput_per_chip": 210.16,
            "b300_to_huawei_decode_step_ratio": 400.0 / 210.16,
            "b300_to_huawei_output_ratio": 1.9,
        },
    }


def test_summary_explains_huawei_round_units():
    row = result_row("gbs64", None, successful_result())
    rendered = markdown([row])
    assert "Decode round TPOT ms" in rendered
    assert "256 consecutive full-local-batch decode iterations" in rendered
    assert "GBS / decode_round_TPOT / 8" in rendered
    assert "eight B300 GPUs" in rendered
    assert "| 64 | 8 | 8 | success |" in rendered


def test_results_are_keyed_by_global_batch_id(tmp_path):
    path = tmp_path / "offline_result_gbs64.json"
    path.write_text(json.dumps(successful_result()), encoding="utf-8")
    discovered = discover_results(tmp_path)
    assert set(discovered) == {"gbs64"}


def test_failed_row_includes_specific_failure_kind():
    rendered = markdown(
        [
            {
                "experiment_id": "gbs128",
                "global_batch_size": 128,
                "status": "failed",
                "failure_kind": "fixed_batch_validation",
                "error": "decode started with 15 local requests",
            }
        ]
    )
    assert "fixed_batch_validation: decode started" in rendered


def test_renderer_row_uses_output_metrics_and_keeps_decode_fields():
    row = renderer_row(successful_result())
    assert row is not None
    assert row["hw"] == "b300"
    assert row["infmax_model_prefix"] == "dsv4"
    assert row["framework"] == "trt"
    assert row["precision"] == "fp4"
    assert row["isl"] == 8192
    assert row["osl"] == 1025
    assert row["conc"] == 64
    assert row["decode_tp"] == 8
    assert row["decode_ep"] == 8
    assert row["decode_dp_attention"] is True
    assert row["output_tput_per_gpu"] == 1000.0
    assert row["mean_tpot"] == 0.008
    assert row["decode_round_tpot_ms"] == 20.0
    assert row["decode_step_tput_per_gpu"] == 400.0
    assert row["local_batch_size"] == 8
    assert row["median_e2el"] == 24.0
    assert row["p99_ttft"] == 6.5


def test_renderer_rows_skip_failed_results(tmp_path):
    failed = successful_result(128)
    failed["status"] = "failed"
    rows = renderer_rows(
        {
            "success": (tmp_path / "success.json", successful_result()),
            "failed": (tmp_path / "failed.json", failed),
        }
    )
    assert [row["conc"] for row in rows] == [64]
