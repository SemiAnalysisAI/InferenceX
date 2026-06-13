import json

from bench_offline.summarize import discover_results, markdown


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
