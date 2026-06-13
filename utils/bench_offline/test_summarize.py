import json

from bench_offline.summarize import (
    _row,
    discover_results,
    markdown,
    renderer_row,
    renderer_rows,
)


def test_summary_explains_offline_units():
    rendered = markdown(
        [
            {
                "experiment_id": "c8-lmtp",
                "concurrency": 8,
                "active_gpu_count": 8,
                "effective_parallelism": "DEP8",
                "status": "success",
                "candidate": "wait30",
                "measured_passes": 1,
                "mean_token_tpot_ms": 10.0,
                "mean_step_tpot_ms": 24.0,
                "derived_output_tput_per_gpu": 100.0,
                "derived_step_tput_per_gpu": 41.67,
                "wall_output_tput_per_gpu": 90.0,
                "observed_tokens_per_step": 2.4,
                "effective_acceptance_rate": 0.4667,
                "raw_speculative_metrics_available": False,
                "acceptance_rate": None,
                "mean_ttft_ms": 1000.0,
                "huawei_published_dataset_token_tput_per_chip": 110.0,
                "huawei_step_tput_per_chip": 45.0,
                "huawei_estimated_token_tput_per_chip": 120.0,
                "b300_to_huawei_ratio": 0.833,
                "b300_to_huawei_published_output_ratio": 0.91,
                "b300_to_huawei_step_rate_ratio": 0.73,
                "huawei_gate_passed": False,
            }
        ]
    )
    assert "Token TPOT ms" in rendered
    assert "last token time - first token time" in rendered
    assert "raw accepted/proposed counters" in rendered.lower()
    assert "Huawei output tok/s/chip" in rendered
    assert "B300/Huawei step" in rendered
    assert "| wait30 | 1 |" in rendered
    assert "full-shape warmup is separate" in rendered
    assert "| FAIL |" in rendered
    assert "| 8 | DEP8 |" in rendered


def test_results_are_keyed_by_experiment_not_concurrency(tmp_path):
    for experiment_id in ("c32-control", "c32-lmtp"):
        path = tmp_path / f"offline_result_{experiment_id}.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark": {
                        "experiment_id": experiment_id,
                        "concurrency": 32,
                    },
                    "status": "success",
                }
            ),
            encoding="utf-8",
        )
    discovered = discover_results(tmp_path)
    assert set(discovered) == {"c32-control", "c32-lmtp"}


def test_row_exposes_profile_optimization_identity():
    row = _row(
        "c128-tp4",
        None,
        {
            "benchmark": {"concurrency": 128},
            "status": "success",
            "winner": {
                "name": "tp4-skip",
                "kind": "tp4-skip-redundant-allreduce",
                "moe_autotune_dummy_distribution": "balanced",
                "dsv4_skip_premoe_allreduce": True,
            },
            "final": {
                "runtime_backports": {
                    "dsv4_skip_premoe_allreduce": {
                        "status": "applied",
                        "after_sha256": "patched-sha",
                    }
                }
            },
        },
    )
    assert row["candidate_kind"] == "tp4-skip-redundant-allreduce"
    assert row["moe_autotune_dummy_distribution"] == "balanced"
    assert row["dsv4_skip_premoe_allreduce"] is True
    assert row["dsv4_backport_status"] == "applied"
    assert row["dsv4_backport_sha256"] == "patched-sha"


def test_failed_row_includes_specific_failure_kind():
    rendered = markdown(
        [
            {
                "experiment_id": "c8-dsl",
                "concurrency": 8,
                "status": "failed",
                "failure_kind": "kernel_dtype",
                "error": "Executor worker returned error",
            }
        ]
    )
    assert "kernel_dtype: Executor worker returned error" in rendered


def test_renderer_row_uses_canonical_schema_and_seconds():
    row = renderer_row(
        {
            "status": "success",
            "benchmark": {
                "active_gpu_count": 4,
                "concurrency": 32,
                "effective_parallelism": "TP4",
                "input_tokens": 8192,
                "generated_output_tokens": 625,
            },
            "provenance": {
                "image": "ghcr.io#semianalysisai/trtllm-dsv4:test",
            },
            "aggregate": {
                "derived_output_tput_per_gpu": 489.17,
                "mean_ttft_ms": 5263.0,
                "median_ttft_ms": 5000.0,
                "p90_ttft_ms": 9000.0,
                "p99_ttft_ms": 9600.0,
                "mean_token_tpot_ms": 16.35,
                "median_token_tpot_ms": 16.8,
                "p90_token_tpot_ms": 21.98,
                "p99_token_tpot_ms": 23.35,
                "mean_intvty": 61.2,
                "median_intvty": 59.5,
                "p90_intvty": 70.0,
                "p99_intvty": 75.0,
                "mean_e2e_ms": 15468.0,
                "median_e2e_ms": 15000.0,
                "p90_e2e_ms": 17500.0,
                "p99_e2e_ms": 18124.0,
            },
        }
    )
    assert row is not None
    assert row["hw"] == "b300"
    assert row["infmax_model_prefix"] == "dsv4"
    assert row["framework"] == "trt"
    assert row["precision"] == "fp4"
    assert row["isl"] == 8192
    assert row["osl"] == 625
    assert row["conc"] == 32
    assert row["decode_tp"] == 4
    assert row["decode_ep"] == 1
    assert row["decode_dp_attention"] is False
    assert row["num_decode_gpu"] == 4
    assert row["output_tput_per_gpu"] == 489.17
    assert row["mean_tpot"] == 0.01635
    assert row["median_e2el"] == 15.0
    assert row["p99_ttft"] == 9.6
    assert row["median_intvty"] == 59.5


def test_renderer_row_records_attention_dp_topology():
    row = renderer_row(
        {
            "status": "success",
            "benchmark": {
                "active_gpu_count": 8,
                "concurrency": 128,
                "effective_parallelism": "DEP8",
            },
            "aggregate": {
                "derived_output_tput_per_gpu": 1000.0,
            },
        }
    )
    assert row is not None
    assert row["decode_tp"] == 8
    assert row["decode_ep"] == 8
    assert row["decode_dp_attention"] is True
    assert row["num_decode_gpu"] == 8


def test_renderer_rows_skip_failed_results(tmp_path):
    successful = {
        "status": "success",
        "benchmark": {
            "active_gpu_count": 4,
            "concurrency": 32,
            "effective_parallelism": "TP4",
        },
        "aggregate": {"derived_output_tput_per_gpu": 100.0},
    }
    failed = {
        "status": "failed",
        "benchmark": {
            "active_gpu_count": 4,
            "concurrency": 64,
            "effective_parallelism": "TP4",
        },
    }
    rows = renderer_rows(
        {
            "success": (tmp_path / "success.json", successful),
            "failed": (tmp_path / "failed.json", failed),
        }
    )
    assert [row["conc"] for row in rows] == [32]
