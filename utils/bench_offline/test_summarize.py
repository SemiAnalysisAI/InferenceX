import json

from bench_offline.summarize import (
    discover_results,
    markdown,
    renderer_row,
    renderer_rows,
    result_row,
)


def successful_result(global_batch_size=64, hardware_profile="b300"):
    is_gb300 = hardware_profile == "gb300"
    active_gpu_count = 16 if is_gb300 else 8
    parallelism = "DEP16" if is_gb300 else "DEP8"
    hardware = "GB300 NVL16" if is_gb300 else "B300"
    return {
        "status": "success",
        "benchmark": {
            "experiment_id": f"gbs{global_batch_size}",
            "global_batch_size": global_batch_size,
            "concurrency": global_batch_size,
            "local_batch_size": global_batch_size // active_gpu_count,
            "active_gpu_count": active_gpu_count,
            "hardware": hardware,
            "hardware_profile": hardware_profile,
            "benchmark_profile": "huawei",
            "renderer_hw": (
                "gb300-nv" if is_gb300 else hardware_profile
            ),
            "physical_nodes": 4 if is_gb300 else 1,
            "is_multinode": is_gb300,
            "effective_parallelism": parallelism,
            "input_tokens": 8192,
            "generated_output_tokens": 1025,
        },
        "config": {
            "active_gpu_count": active_gpu_count,
            "tensor_parallel_size": active_gpu_count,
            "moe_expert_parallel_size": active_gpu_count,
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
            "local_batch_size": global_batch_size // active_gpu_count,
            "measured_decode_rounds": 256,
            "effective_acceptance_rate": 0.5,
            "token_yield_source": "iteration_spec_decoding_stats",
            "timing_source": "iter_latency_ms",
            "filter": {
                "retained_rounds": 254,
                "outlier_rounds": 1,
                "rounds_skipped": 1,
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
            "hardware_key": hardware_profile,
            "decode_round_tpot_ms": 19.03,
            "decode_step_tput_per_chip": 210.16,
            "hardware_to_huawei_decode_step_ratio": 400.0 / 210.16,
            "hardware_to_huawei_output_ratio": 1.9,
        },
    }


def test_summary_explains_huawei_round_units():
    row = result_row("gbs64", None, successful_result())
    rendered = markdown([row])
    assert "Decode round TPOT ms" in rendered
    assert "256 consecutive full-local-batch decode iterations" in rendered
    assert "GBS / decode_round_TPOT / 8" in rendered
    assert "8 B300 GPUs" in rendered
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


def test_gb300_renderer_row_is_multinode_dep16():
    row = renderer_row(successful_result(hardware_profile="gb300"))
    assert row is not None
    assert row["hw"] == "gb300-nv"
    assert row["is_multinode"] is True
    assert row["decode_tp"] == 16
    assert row["decode_ep"] == 16
    assert row["num_decode_gpu"] == 16
    assert row["local_batch_size"] == 4


def test_gb300_summary_uses_sixteen_gpu_formula():
    row = result_row(
        "gbs64",
        None,
        successful_result(hardware_profile="gb300"),
    )
    rendered = markdown([row])
    assert "GB300 NVL16 TRT Fixed-GBS" in rendered
    assert "GBS / decode_round_TPOT / 16" in rendered
    assert "16 GB300 NVL16 GPUs" in rendered
    assert "GB300/Huawei step" in rendered


def test_pr_max_summary_uses_decode_and_reference_denominators():
    result = successful_result(
        global_batch_size=4096,
        hardware_profile="gb300",
    )
    result["benchmark"].update(
        {
            "experiment_id": "pr-tp8-mtp1-gbs4096",
            "benchmark_profile": "pr-tp8-mtp1",
            "hardware": "GB300 NVL8",
            "active_gpu_count": 8,
            "local_batch_size": 512,
            "physical_nodes": 2,
            "effective_parallelism": "DEP8",
            "generated_output_tokens": 1024,
        }
    )
    result["config"].update(
        {
            "active_gpu_count": 8,
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
        }
    )
    result["aggregate"].update(
        {
            "local_batch_size": 512,
            "output_tput_per_gpu": 10000.0,
            "timing_source": "trt_print_iter_log_host_step_time",
            "filter": {
                "retained_rounds": 248,
                "outlier_rounds": 0,
                "rounds_skipped": 8,
            },
        }
    )
    result.pop("huawei")
    result["pr_reference"] = {
        "reference_concurrency": 4301,
        "reference_active_global_batch": 3440,
        "reference_prefill_gpu_count": 48,
        "reference_decode_gpu_count": 8,
        "reference_total_gpu_count": 56,
        "reference_output_tput_per_decode_gpu": 9686.735,
        "reference_output_tput_per_total_gpu": 1383.8193,
        "measured_output_tput_per_reference_total_gpu": 1428.571,
        "offline_to_reference_decode_gpu_ratio": 1.03234,
        "offline_to_reference_total_gpu_ratio": 1.03234,
        "reference_recipe_url": "https://example.test/recipe.yaml",
    }

    row = result_row("pr-tp8-mtp1-gbs4096", None, result)
    rendered = markdown([row])
    assert "PR-Config Offline Decode Saturation" in rendered
    assert "| pr-tp8-mtp1 | 4096 | 512 | 8 | success |" in rendered
    assert "full-batch decode iterations under overlap scheduling" in rendered
    assert "PR-fleet-normalized" in rendered

    flat = renderer_row(result)
    assert flat is not None
    assert flat["benchmark_profile"] == "pr-tp8-mtp1"
    assert flat["pr_reference_concurrency"] == 4301
    assert flat["pr_reference_active_global_batch"] == 3440
    assert flat["pr_offline_to_reference_decode_gpu_ratio"] == 1.03234
