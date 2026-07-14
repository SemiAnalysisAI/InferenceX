from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from trt_disagg_gen_only import (
    build_result,
    build_result_from_logs,
    case_metadata,
    parse_gen_iterlog,
    render_effective_config,
    timed_max_tokens,
    validate_case_against_workflow,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    REPO_ROOT
    / "benchmarks/multi_node/offline/trtllm_gen_only/benchmark/configs"
)
REQUIRED_CONFIGS = {
    "ctx1_gen1_dep32_concurrency512_mtp3.yaml",
}
UPSTREAM_RUNNER_ROOT = Path("/data/home/sa-shared/gharunners")
UPSTREAM_DATASET_ROOT = UPSTREAM_RUNNER_ROOT / "datasets/dsv4-trt-offline"
UPSTREAM_CONTAINER_IMAGE = (
    f"{UPSTREAM_RUNNER_ROOT}/squash/"
    "nvcr.io_nvidia_tensorrt-llm_release_1.3.0rc15.post1.sqsh"
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _iterlog_line(
    *,
    iteration: int,
    rank: int,
    scheduled: int,
    generation_tokens: int,
    device_ms: float,
    ctx_tokens: int = 0,
) -> str:
    return (
        f"iter = {iteration}, global_rank = {rank}, rank = {rank}, "
        f"currank_total_requests = {scheduled}/{scheduled}, "
        f"host_step_time = {device_ms + 1.0:.3f}ms, "
        f"prev_device_step_time = {device_ms:.3f}ms, "
        "timestamp = 2026-07-07 00:00:00, "
        f"num_scheduled_requests: {scheduled}, "
        "states = {'num_ctx_requests': 0, "
        f"'num_ctx_tokens': {ctx_tokens}, "
        f"'num_generation_tokens': {generation_tokens}}}"
    )


def test_offline_token_budgets_use_fixed_acceptance_lengths():
    assert timed_max_tokens(decode_steps=256, mtp=3) == 623
    assert timed_max_tokens(decode_steps=256, mtp=1) == 435


def test_tep_parser_keeps_only_exact_full_batch_generation_rows(tmp_path):
    path = tmp_path / "gen_only_0.txt"
    path.write_text(
        "\n".join(
            [
                _iterlog_line(
                    iteration=1,
                    rank=0,
                    scheduled=2,
                    generation_tokens=8,
                    device_ms=10.0,
                ),
                _iterlog_line(
                    iteration=2,
                    rank=0,
                    scheduled=1,
                    generation_tokens=4,
                    device_ms=99.0,
                ),
                _iterlog_line(
                    iteration=3,
                    rank=0,
                    scheduled=2,
                    generation_tokens=8,
                    device_ms=12.0,
                    ctx_tokens=8192,
                ),
            ]
        )
    )

    summary = parse_gen_iterlog(
        path,
        concurrency=2,
        gen_tp=4,
        mtp=3,
        attention_dp=False,
    )

    assert summary.raw_exact_batch_samples == 1
    assert summary.retained_samples == 1
    assert summary.mean_device_step_ms == pytest.approx(10.0)


def test_dep_parser_uses_per_rank_batch_and_deduplicates_iterations(tmp_path):
    path = tmp_path / "gen_only_0.txt"
    path.write_text(
        "\n".join(
            [
                _iterlog_line(
                    iteration=7,
                    rank=0,
                    scheduled=2,
                    generation_tokens=8,
                    device_ms=20.0,
                ),
                _iterlog_line(
                    iteration=7,
                    rank=0,
                    scheduled=2,
                    generation_tokens=8,
                    device_ms=22.0,
                ),
                _iterlog_line(
                    iteration=8,
                    rank=0,
                    scheduled=1,
                    generation_tokens=4,
                    device_ms=90.0,
                ),
            ]
        )
    )

    summary = parse_gen_iterlog(
        path,
        concurrency=64,
        gen_tp=32,
        mtp=3,
        attention_dp=True,
    )

    assert summary.raw_exact_batch_samples == 1
    assert summary.retained_samples == 1
    assert summary.mean_device_step_ms == pytest.approx(22.0)


def test_result_keeps_steps_primary_and_acceptance_secondary():
    result = build_result(
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        source_config="ctx1_gen1_tep4_concurrency2_mtp3.yaml",
        concurrency=2,
        ctx_gpus=4,
        gen_gpus=4,
        gen_tp=4,
        mtp=3,
        mean_device_step_ms=10.0,
        median_device_step_ms=10.0,
        p90_device_step_ms=10.0,
        p99_device_step_ms=10.0,
        raw_samples=256,
        retained_samples=250,
        decode_steps=256,
    )

    assert result["engine_mode"] == "offline"
    assert result["measurement_boundary"] == "gen_iteration"
    assert result["tpot_unit"] == "decode_step"
    assert result["total_token_throughput"] == pytest.approx(200.0)
    assert result["output_throughput"] == pytest.approx(200.0)
    assert result["decode_step_throughput_per_gen_gpu"] == pytest.approx(50.0)
    assert result["assumed_tokens_per_step"] == pytest.approx(2.44)
    assert result["token_equivalent_output_throughput"] == pytest.approx(488.0)
    assert result["token_equivalent_output_throughput_per_gen_gpu"] == pytest.approx(122.0)
    assert result["timed_max_tokens"] == 623


def test_master_config_contains_twelve_selected_source_cases(tmp_path):
    master = _load_yaml(REPO_ROOT / ".github/configs/nvidia-master.yaml")
    config = master["dsv4-fp4-gb300-trt-offline"]

    assert config["multinode"] is True
    assert config["disagg"] is True
    scenario = config["scenarios"]["fixed-seq-len"][0]
    assert (scenario["isl"], scenario["osl"]) == (8192, 256)
    rows = scenario["search-space"]
    assert len(rows) == 12
    assert {row["spec-decoding"] for row in rows} == {"offline"}

    config_files = []
    for row in rows:
        setting = row["prefill"]["additional-settings"][0]
        key, relative_path = setting.split("=", 1)
        assert key == "CONFIG_FILE"
        source_path = REPO_ROOT / relative_path
        assert source_path.is_file()
        config_files.append(source_path)
        source = _load_yaml(source_path)
        metadata = validate_case_against_workflow(
            source,
            concurrency=row["conc-list"][0],
            prefill_num_workers=row["prefill"]["num-worker"],
            prefill_tp=row["prefill"]["tp"],
            prefill_ep=row["prefill"]["ep"],
            prefill_dp_attn=row["prefill"]["dp-attn"],
            decode_num_workers=row["decode"]["num-worker"],
            decode_tp=row["decode"]["tp"],
            decode_ep=row["decode"]["ep"],
            decode_dp_attn=row["decode"]["dp-attn"],
        )
        effective, rendered_metadata = render_effective_config(
            source,
            output_path=tmp_path / source_path.name,
            partition="batch_1",
            account="benchmark",
            job_name="gb300-nv_0",
            container_image="/data/squash/trtllm.sqsh",
            container_mount="/data/:/data/,/repo:/workspace",
            model_path="/scratch/models/DeepSeek-V4-Pro",
            dataset_root="/data/datasets/dsv4-trt-offline",
            log_dir=f"/data/runs/{source_path.stem}",
            decode_steps=scenario["osl"],
        )
        assert rendered_metadata == metadata
        assert effective["benchmark"]["input_length"] == scenario["isl"]
        assert effective["benchmark"]["output_length"] == timed_max_tokens(
            scenario["osl"], metadata.mtp
        )
        assert effective["hardware"]["num_gen_servers"] == 1

    assert len(set(config_files)) == 12
    selected_names = {path.name for path in config_files}
    assert REQUIRED_CONFIGS <= selected_names
    assert {case_metadata(_load_yaml(path)).node_count for path in config_files} == {
        2,
        3,
        8,
        9,
        12,
    }


def test_case_metadata_is_json_serializable():
    source = _load_yaml(
        CONFIG_ROOT / "ctx4_gen1_dep16_concurrency4096_mtp1.yaml"
    )
    metadata = case_metadata(source)
    encoded = json.dumps(metadata.__dict__, sort_keys=True)
    assert '"mtp": 1' in encoded
    assert '"node_count": 8' in encoded


def test_source_configs_target_upstream_gb300_ci_cluster():
    source_paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(source_paths) == 12
    source_names = {path.name for path in source_paths}
    assert REQUIRED_CONFIGS <= source_names

    for source_path in source_paths:
        source = _load_yaml(source_path)
        metadata = case_metadata(source)
        slurm = source["slurm"]
        benchmark = source["benchmark"]
        environment = source["environment"]
        timed_tokens = timed_max_tokens(256, metadata.mtp)

        assert slurm["partition"] == "batch_1"
        assert slurm["account"] == "benchmark"
        assert slurm["job_time"] == "03:00:00"
        assert slurm["job_name"] == f"dsv4-offline-{source_path.stem}"
        for dead_field in (
            "use_hetjob",
            "max_segment_nodes",
            "max_hetjob_components",
        ):
            assert dead_field not in slurm
        assert benchmark["output_length"] == timed_tokens
        assert benchmark["dataset_file"] == str(
            UPSTREAM_DATASET_ROOT
            / f"DeepSeek-V4-8192-{timed_tokens}-16384-ratio-1_for_serve.json"
        )
        assert environment["container_mount"] == (
            "/data/:/data/,/scratch/:/scratch/"
        )
        assert environment["container_image"] == UPSTREAM_CONTAINER_IMAGE
        assert environment["model_path"] == "/scratch/models/DeepSeek-V4-Pro"
        assert "staging" not in environment

        serialized = source_path.read_text()
        for forbidden in (
            "/lustre",
            "/raid",
            "coreai_",
        ):
            assert forbidden not in serialized

    launcher = (REPO_ROOT / "runners/launch_gb300-nv.sh").read_text()
    assert 'SLURM_PARTITION="batch_1"' in launcher
    assert 'SLURM_ACCOUNT="benchmark"' in launcher
    assert (
        "srun --partition=$SLURM_PARTITION --exclusive --time=180"
    ) in launcher
    assert f'{UPSTREAM_RUNNER_ROOT}/squash/' in launcher

    wrapper = (
        REPO_ROOT / "benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh"
    ).read_text()
    assert '--partition "${SLURM_PARTITION:-batch_1}"' in wrapper
    assert '--account "${SLURM_ACCOUNT:-benchmark}"' in wrapper
    assert str(UPSTREAM_RUNNER_ROOT) in wrapper
    assert '--job-time "${SLURM_JOB_TIME:-03:00:00}"' in wrapper


def test_direct_submit_mounts_default_dataset_generator(tmp_path):
    submit = (
        REPO_ROOT
        / "benchmarks/multi_node/offline/trtllm_gen_only/benchmark/scripts/submit.py"
    )
    source = CONFIG_ROOT / "ctx1_gen1_tep4_concurrency2_mtp3.yaml"
    dataset_dir = (
        REPO_ROOT / "benchmarks/multi_node/offline/trtllm_gen_only/dataset"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(submit),
            "--config",
            str(source),
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
            "--wait",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    command = next(
        line for line in result.stdout.splitlines() if line.startswith("sbatch --")
    )
    assert "--time=03:00:00" in command
    assert f"{dataset_dir}:{dataset_dir}" in command
    assert f"--dataset-generator {dataset_dir / 'gen_dataset.sh'}" in command


def test_render_effective_config_changes_only_runtime_and_measurement_fields(
    tmp_path,
):
    source_path = CONFIG_ROOT / "ctx1_gen1_tep4_concurrency2_mtp3.yaml"
    source = _load_yaml(source_path)
    original = json.loads(json.dumps(source))
    output_path = tmp_path / "effective.yaml"

    effective, metadata = render_effective_config(
        source,
        output_path=output_path,
        partition="batch_1",
        account="benchmark",
        job_name="gb300-nv_0",
        container_image="/data/squash/trtllm.sqsh",
        container_mount="/data/:/data/,/scratch/:/scratch/,/repo:/workspace",
        model_path="/scratch/models/DeepSeek-V4-Pro",
        dataset_root="/data/datasets/dsv4-trt-offline",
        log_dir="/data/runs/case",
        decode_steps=256,
    )

    assert source == original
    assert _load_yaml(output_path) == effective
    assert effective["slurm"]["partition"] == "batch_1"
    assert effective["slurm"]["account"] == "benchmark"
    assert effective["slurm"]["job_time"] == "03:00:00"
    assert effective["slurm"]["job_name"] == "gb300-nv_0"
    assert effective["benchmark"]["input_length"] == 8192
    assert effective["benchmark"]["output_length"] == 623
    assert effective["benchmark"]["multi_round"] == 1
    assert effective["benchmark"]["dataset_file"].endswith(
        "DeepSeek-V4-8192-623-16384-ratio-1_for_serve.json"
    )
    assert effective["environment"]["container_image"].endswith("trtllm.sqsh")
    assert effective["environment"]["model_path"].endswith("DeepSeek-V4-Pro")
    assert effective["environment"]["log_dir"] == "/data/runs/case"
    assert metadata.mtp == 3
    assert metadata.concurrency == 2


def test_build_result_from_logs_writes_workflow_compatible_json(tmp_path):
    source = _load_yaml(
        CONFIG_ROOT / "ctx1_gen1_tep4_concurrency2_mtp3.yaml"
    )
    log_dir = tmp_path / "logs"
    iterlog = log_dir / "concurrency_2" / "gen_only_0.txt"
    iterlog.parent.mkdir(parents=True)
    iterlog.write_text(
        "\n".join(
            _iterlog_line(
                iteration=index,
                rank=0,
                scheduled=2,
                generation_tokens=8,
                device_ms=10.0,
            )
            for index in range(1, 257)
        )
    )
    output_path = tmp_path / "result.json"

    result = build_result_from_logs(
        source,
        log_dir=log_dir,
        output_path=output_path,
        source_config="ctx1_gen1_tep4_concurrency2_mtp3.yaml",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        decode_steps=256,
    )

    assert json.loads(output_path.read_text()) == result
    assert result["max_concurrency"] == 2
    assert result["num_ctx_gpu"] == 4
    assert result["num_gen_gpu"] == 4
    assert result["retained_iteration_count"] == 256
    assert result["mean_tpot_ms"] == pytest.approx(10.0)
    assert result["iteration_log_source"] == "timed_slice"
    assert result["iteration_log_file"] == "gen_only_0.txt"


def test_build_result_from_logs_falls_back_to_full_gen_worker_log(tmp_path):
    source = _load_yaml(
        CONFIG_ROOT / "ctx1_gen1_dep32_concurrency1024_mtp3.yaml"
    )
    log_dir = tmp_path / "logs"
    timed_log = log_dir / "concurrency_1024" / "gen_only_0.txt"
    timed_log.parent.mkdir(parents=True)
    timed_log.write_text(
        _iterlog_line(
            iteration=1,
            rank=0,
            scheduled=2,
            generation_tokens=8,
            device_ms=90.0,
        )
    )
    full_log = log_dir / "3_output_GEN_0.log"
    full_log.write_text(
        "\n".join(
            _iterlog_line(
                iteration=index,
                rank=0,
                scheduled=32,
                generation_tokens=128,
                device_ms=device_ms,
            )
            for index, device_ms in enumerate((20.0, 21.0, 22.0), start=1)
        )
    )

    result = build_result_from_logs(
        source,
        log_dir=log_dir,
        output_path=tmp_path / "result.json",
        source_config="ctx1_gen1_dep32_concurrency1024_mtp3.yaml",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        decode_steps=256,
    )

    assert result["raw_exact_batch_iteration_count"] == 3
    assert result["retained_iteration_count"] == 3
    assert result["mean_tpot_ms"] == pytest.approx(21.0)
    assert result["iteration_log_source"] == "full_gen_worker_fallback"
    assert result["iteration_log_file"] == "3_output_GEN_0.log"


def test_build_result_rejects_invalid_full_log_when_timed_slice_missing(
    tmp_path,
):
    source = _load_yaml(
        CONFIG_ROOT / "ctx4_gen1_dep16_concurrency4096_mtp1.yaml"
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    partial_row = _iterlog_line(
        iteration=1,
        rank=0,
        scheduled=82,
        generation_tokens=256,
        device_ms=145.0,
    )
    (log_dir / "3_output_GEN_0.log").write_text(partial_row)

    with pytest.raises(
        ValueError,
        match=(
            "No timed GEN iterlog found.*fallback also failed.*"
            "3_output_GEN_0.log"
        ),
    ):
        build_result_from_logs(
            source,
            log_dir=log_dir,
            output_path=tmp_path / "result.json",
            source_config="ctx4_gen1_dep16_concurrency4096_mtp1.yaml",
            model_id="deepseek-ai/DeepSeek-V4-Pro",
            decode_steps=256,
        )


def test_gb300_launcher_routes_multinode_offline_to_shared_wrapper():
    launcher = (REPO_ROOT / "runners/launch_gb300-nv.sh").read_text()
    wrapper_path = "benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh"
    assert '"$IS_MULTINODE" == "true"' in launcher
    assert '"$SPEC_DECODING" == "offline"' in launcher
    assert '"$FRAMEWORK" == "trt"' in launcher
    assert '"$MODEL_PREFIX" == "dsv4"' in launcher
    assert f'bash "{wrapper_path}"' in launcher

    wrapper = (REPO_ROOT / wrapper_path).read_text()
    assert "submit.py" in wrapper
    assert "--wait" in wrapper
    assert "utils/bench_offline/trt_disagg_gen_only.py" in wrapper
    assert 'python3 "${ADAPTER}" prepare' in wrapper
    assert 'python3 "${ADAPTER}" result' in wrapper

    benchmark_script = (
        REPO_ROOT
        / "benchmarks/multi_node/offline/trtllm_gen_only/benchmark/scripts/run_benchmark.sh"
    ).read_text()
    assert "Warming up one full cohort" in benchmark_script
    assert "start_line=$((start_line + 1))" in benchmark_script
