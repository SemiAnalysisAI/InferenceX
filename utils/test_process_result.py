"""Exercise process_result.py through its CLI with controlled environment and artifacts."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aggregate_power_multinode import ROLE_METRIC_KEYS, WHOLE_METRIC_KEYS
from test_aggregate_power_multinode import PRODUCER_SHA, build_package

SCRIPT_PATH = Path(__file__).parent / "process_result.py"


# =============================================================================
# Test Fixtures - Based on real benchmark output structure
# =============================================================================

@pytest.fixture
def sample_benchmark_result():
    """Sample benchmark result JSON based on real output structure."""
    return {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "max_concurrency": 64,
        "total_token_throughput": 15000.5,
        "output_throughput": 12000.0,
        "ttft_p50_ms": 150.5,
        "ttft_p99_ms": 250.3,
        "tpot_p50_ms": 25.0,
        "tpot_p99_ms": 45.0,
        "e2e_latency_p50_ms": 1500.0,
        "e2e_latency_p99_ms": 2500.0,
    }


@pytest.fixture
def base_env_vars():
    """Base environment variables for single-node setup."""
    return {
        "RUNNER_TYPE": "mi300x",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp8",
        "SPEC_DECODING": "none",
        "RESULT_FILENAME": "benchmark_result",
        "ISL": "1024",
        "OSL": "1024",
        "DISAGG": "false",
        "MODEL_PREFIX": "dsr1",
        "IMAGE": "test-image",
        "RECIPE_FINGERPRINT": "a" * 64,
    }


@pytest.fixture
def single_node_env_vars(base_env_vars):
    """Environment variables for single-node setup."""
    return {
        **base_env_vars,
        "TP": "8",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
    }


@pytest.fixture
def multinode_env_vars(base_env_vars):
    """Environment variables for multinode setup based on gb200 config."""
    return {
        **base_env_vars,
        "RUNNER_TYPE": "gb200",
        "FRAMEWORK": "dynamo-trt",
        "PRECISION": "fp4",
        "DISAGG": "true",
        "IS_MULTINODE": "true",
        "PREFILL_GPUS": "20",
        "DECODE_GPUS": "8",
        "PREFILL_NUM_WORKERS": "5",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP_ATTN": "true",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "DECODE_EP": "8",
        "DECODE_DP_ATTN": "true",
        "PREFILL_HARDWARE": "gb200",
        "DECODE_HARDWARE": "h100",
    }


def run_script(tmp_path, env, benchmark_result, result_filename="benchmark_result"):
    """Helper to run the process_result.py script."""
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(benchmark_result))

    env = env.copy()
    env["RESULT_FILENAME"] = result_filename

    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def run_script_with_broken_aggregator(
    tmp_path, env, benchmark_result, result_filename="benchmark_result"
):
    """Run process_result with the real aggregator patched to raise unexpectedly."""
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(benchmark_result))
    env = {**env, "RESULT_FILENAME": result_filename}
    wrapper = f"""
import runpy
import sys

sys.path.insert(0, {str(SCRIPT_PATH.parent)!r})
import aggregate_power

def broken_run(**kwargs):
    raise RuntimeError("forced aggregation failure")
aggregate_power.run = broken_run
runpy.run_path({str(SCRIPT_PATH)!r}, run_name="__main__")
"""
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


# =============================================================================
# Test script execution via subprocess
# =============================================================================

class TestProcessResultScript:
    """Tests for process_result.py script execution."""

    def test_single_node_processing(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test single-node result processing."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)

        # Verify base fields
        assert output_data["hw"] == "mi300x"
        assert output_data["framework"] == "sglang"
        assert output_data["precision"] == "fp8"
        assert output_data["spec_decoding"] == "none"
        assert output_data["model"] == "deepseek-ai/DeepSeek-R1-0528"
        assert output_data["conc"] == 64
        assert output_data["isl"] == 1024
        assert output_data["osl"] == 1024
        assert output_data["disagg"] is False
        assert output_data["recipe_fingerprint"] == "a" * 64

        # Verify single-node specific fields
        assert output_data["is_multinode"] is False
        assert output_data["tp"] == 8
        assert output_data["ep"] == 1
        assert output_data["dp_attention"] == "false"

        # Verify throughput calculations (divided by tp=8)
        assert output_data["tput_per_gpu"] == pytest.approx(1875.0625)
        assert output_data["output_tput_per_gpu"] == pytest.approx(1500.0)
        assert output_data["input_tput_per_gpu"] == pytest.approx(375.0625)

        # Verify latency conversions (ms to seconds)
        assert output_data["ttft_p50"] == pytest.approx(0.1505)
        assert output_data["ttft_p99"] == pytest.approx(0.2503)
        assert output_data["e2e_latency_p50"] == pytest.approx(1.5)
        assert output_data["e2e_latency_p99"] == pytest.approx(2.5)

        # Verify interactivity calculations (1000 / tpot_ms)
        assert output_data["intvty_p50"] == pytest.approx(40.0)
        assert output_data["intvty_p99"] == pytest.approx(22.222222)

        # Verify output file created
        output_file = tmp_path / "agg_benchmark_result.json"
        assert output_file.exists()

    def test_multinode_processing(self, tmp_path, sample_benchmark_result, multinode_env_vars):
        """Test multinode result processing."""
        result = run_script(tmp_path, multinode_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)

        # Verify base fields
        assert output_data["hw"] == "gb200"
        assert output_data["framework"] == "dynamo-trt"
        assert output_data["precision"] == "fp4"
        assert output_data["disagg"] is True

        # Verify multinode specific fields
        assert output_data["is_multinode"] is True
        assert output_data["prefill_tp"] == 4
        assert output_data["prefill_ep"] == 4
        assert output_data["prefill_dp_attention"] == "true"
        assert output_data["prefill_num_workers"] == 5
        assert output_data["decode_tp"] == 8
        assert output_data["decode_ep"] == 8
        assert output_data["decode_dp_attention"] == "true"
        assert output_data["decode_num_workers"] == 1
        assert output_data["num_prefill_gpu"] == 20
        assert output_data["num_decode_gpu"] == 8
        assert output_data["prefill_hw"] == "gb200"
        assert output_data["decode_hw"] == "h100"

        # Verify throughput calculations
        assert output_data["tput_per_gpu"] == pytest.approx(535.732143)  # 28 GPUs total
        assert output_data["output_tput_per_gpu"] == pytest.approx(1500.0)  # 8 decode GPUs
        assert output_data["input_tput_per_gpu"] == pytest.approx(150.025)  # 20 prefill GPUs

    def test_component_metadata_is_emitted_when_present(
        self, tmp_path, sample_benchmark_result, multinode_env_vars
    ):
        env = {
            **multinode_env_vars,
            "ROUTER_METADATA": json.dumps({"name": "vllm-router", "version": "0.1.14"}),
            "KV_P2P_TRANSFER": "mooncake",
        }

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        output_data = json.loads(result.stdout)
        assert output_data["router"] == {"name": "vllm-router", "version": "0.1.14"}
        assert output_data["kv_p2p_transfer"] == "mooncake"

    @pytest.mark.parametrize("metadata", [
        {"name": "vllm-router"},
        {"name": "vllm-router", "version": "0.1.14", "mode": "round-robin"},
    ])
    def test_component_metadata_rejects_partial_or_extra_fields(
        self, tmp_path, sample_benchmark_result, single_node_env_vars, metadata
    ):
        env = {**single_node_env_vars, "ROUTER_METADATA": json.dumps(metadata)}

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "must contain exactly 'name' and 'version'" in result.stderr

    def test_homogeneous_multinode_omits_hardware_fields(
        self, tmp_path, sample_benchmark_result, multinode_env_vars
    ):
        """Absent hardware metadata should preserve homogeneous result output."""
        multinode_env_vars.pop("PREFILL_HARDWARE")
        multinode_env_vars.pop("DECODE_HARDWARE")

        result = run_script(tmp_path, multinode_env_vars, sample_benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        output_data = json.loads(result.stdout)
        assert "prefill_hw" not in output_data
        assert "decode_hw" not in output_data

    @pytest.mark.parametrize("missing_var", ["PREFILL_HARDWARE", "DECODE_HARDWARE"])
    def test_partial_hardware_metadata_fails(
        self, tmp_path, sample_benchmark_result, multinode_env_vars, missing_var
    ):
        """Prefill and decode hardware must always be provided together."""
        multinode_env_vars.pop(missing_var)

        result = run_script(tmp_path, multinode_env_vars, sample_benchmark_result)

        assert result.returncode != 0
        assert "PREFILL_HARDWARE and DECODE_HARDWARE" in result.stderr

    def test_missing_base_env_vars(self, tmp_path, sample_benchmark_result):
        """Test that missing base env vars causes failure."""
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text(json.dumps(sample_benchmark_result))

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=tmp_path,
            env={"PATH": "/usr/bin", "RESULT_FILENAME": "benchmark_result"},
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_missing_single_node_env_vars(self, tmp_path, sample_benchmark_result, base_env_vars):
        """Test that missing single-node env vars causes failure."""
        # base_env_vars doesn't have TP, EP_SIZE, DP_ATTENTION
        result = run_script(tmp_path, base_env_vars, sample_benchmark_result)

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_missing_multinode_env_vars(self, tmp_path, sample_benchmark_result, base_env_vars):
        """Test that missing multinode env vars causes failure."""
        env = base_env_vars.copy()
        env["IS_MULTINODE"] = "true"
        env["DISAGG"] = "true"
        # Missing multinode-specific vars

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_disagg_without_multinode_fails(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that disagg=true without multinode raises error."""
        env = single_node_env_vars.copy()
        env["DISAGG"] = "true"  # Disagg without multinode

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "Disaggregated mode requires multinode setup" in result.stderr

    def test_missing_result_file(self, tmp_path, single_node_env_vars):
        """Test that missing result file causes failure."""
        env = single_node_env_vars.copy()
        env["RESULT_FILENAME"] = "nonexistent"

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


# =============================================================================
# Test latency and throughput calculations
# =============================================================================

class TestCalculations:
    """Tests for throughput and latency calculations."""

    def test_latency_ms_to_seconds_conversion(self, tmp_path, single_node_env_vars):
        """Test that _ms fields are converted to seconds."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 1000.0,
            "output_throughput": 800.0,
            "custom_metric_ms": 500.0,  # Should become custom_metric = 0.5
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["custom_metric"] == pytest.approx(0.5)

    def test_tpot_to_interactivity_conversion(self, tmp_path, single_node_env_vars):
        """Test that tpot fields are converted to interactivity."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 1000.0,
            "output_throughput": 800.0,
            "tpot_p50_ms": 20.0,  # Should become intvty_p50 = 50
            "tpot_p99_ms": 50.0,  # Should become intvty_p99 = 20
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["intvty_p50"] == pytest.approx(50.0)
        assert output_data["intvty_p99"] == pytest.approx(20.0)

    def test_throughput_per_gpu_single_node(self, tmp_path, single_node_env_vars):
        """PP and PCP expand the GPU denominator while DCP remains metadata."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 8000.0,
            "output_throughput": 6000.0,
        }

        env = single_node_env_vars.copy()
        env.update({"TP": "4", "PP_SIZE": "2", "DCP_SIZE": "2", "PCP_SIZE": "2"})

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["pp"] == 2
        assert output_data["dcp_size"] == 2
        assert output_data["pcp_size"] == 2
        assert output_data["tput_per_gpu"] == pytest.approx(500.0)
        assert output_data["output_tput_per_gpu"] == pytest.approx(375.0)
        assert output_data["input_tput_per_gpu"] == pytest.approx(125.0)

    def test_throughput_per_gpu_multinode(self, tmp_path, multinode_env_vars):
        """Test throughput per GPU calculation for multinode."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 64,
            "total_token_throughput": 28000.0,  # Will be divided by total GPUs
            "output_throughput": 16000.0,  # Will be divided by decode GPUs
        }

        env = multinode_env_vars.copy()
        env["PREFILL_GPUS"] = "20"
        env["DECODE_GPUS"] = "8"
        env.update({
            "PREFILL_PP_SIZE": "2",
            "PREFILL_DCP_SIZE": "2",
            "PREFILL_PCP_SIZE": "2",
            "DECODE_PP_SIZE": "2",
            "DECODE_DCP_SIZE": "4",
            "DECODE_PCP_SIZE": "1",
        })

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert (
            output_data["prefill_pp"],
            output_data["prefill_dcp_size"],
            output_data["prefill_pcp_size"],
        ) == (2, 2, 2)
        assert (
            output_data["decode_pp"],
            output_data["decode_dcp_size"],
            output_data["decode_pcp_size"],
        ) == (2, 4, 1)
        assert output_data["tput_per_gpu"] == pytest.approx(1000.0)  # 28000 / 28
        assert output_data["output_tput_per_gpu"] == pytest.approx(2000.0)  # 16000 / 8
        assert output_data["input_tput_per_gpu"] == pytest.approx(600.0)  # (28000 - 16000) / 20

    def test_multinode_aggregate_decode_fields_zero(self, tmp_path, multinode_env_vars):
        """Aggregate multinode results should report zero decode TP/EP when no decode GPUs exist."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 1,
            "total_token_throughput": 8000.0,
            "output_throughput": 6000.0,
        }

        env = multinode_env_vars.copy()
        env["PREFILL_GPUS"] = "8"
        env["DECODE_GPUS"] = "0"
        env["PREFILL_NUM_WORKERS"] = "1"
        env["PREFILL_TP"] = "8"
        env["PREFILL_EP"] = "1"
        env["PREFILL_DP_ATTN"] = "false"
        env["DECODE_NUM_WORKERS"] = "0"
        env["DECODE_TP"] = "8"
        env["DECODE_EP"] = "1"
        env["DECODE_DP_ATTN"] = "false"

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["decode_tp"] == 0
        assert output_data["decode_ep"] == 0
        assert output_data["decode_num_workers"] == 0
        assert output_data["num_decode_gpu"] == 0
        assert output_data["num_prefill_gpu"] == 8
        assert output_data["tput_per_gpu"] == pytest.approx(1000.0)
        assert output_data["output_tput_per_gpu"] == pytest.approx(750.0)
        assert output_data["input_tput_per_gpu"] == pytest.approx(250.0)

    def test_multinode_zero_total_gpus_fails(self, tmp_path, sample_benchmark_result, multinode_env_vars):
        """Invalid multinode metadata should fail before throughput division."""
        env = multinode_env_vars.copy()
        env["PREFILL_GPUS"] = "0"
        env["DECODE_GPUS"] = "0"

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "Multinode results require at least one GPU" in result.stderr


# =============================================================================
# Test output file generation
# =============================================================================

class TestOutputFile:
    """Tests for output file generation."""

    def test_output_file_created(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that aggregated output file is created."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = tmp_path / "agg_benchmark_result.json"
        assert output_file.exists()

        # Verify content matches stdout
        with open(output_file) as f:
            file_content = json.load(f)

        stdout_content = json.loads(result.stdout)
        assert file_content == stdout_content

    def test_output_file_has_correct_prefix(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that output file has 'agg_' prefix."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result, "my_custom_result")
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = tmp_path / "agg_my_custom_result.json"
        assert output_file.exists()


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_boolean_disagg_parsing_false(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that DISAGG env var is parsed as boolean correctly for false values."""
        for disagg_value in ["false", "False", "FALSE"]:
            env = single_node_env_vars.copy()
            env["DISAGG"] = disagg_value

            result = run_script(tmp_path, env, sample_benchmark_result)
            assert result.returncode == 0, f"Script failed for DISAGG={disagg_value}: {result.stderr}"

            output_data = json.loads(result.stdout)
            assert output_data["disagg"] is False

    def test_boolean_disagg_parsing_true_requires_multinode(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that DISAGG=true without multinode fails."""
        for disagg_value in ["true", "True", "TRUE"]:
            env = single_node_env_vars.copy()
            env["DISAGG"] = disagg_value

            result = run_script(tmp_path, env, sample_benchmark_result)
            assert result.returncode != 0


    def test_integer_conversion(self, tmp_path, single_node_env_vars):
        """Test that numeric env vars are converted to integers."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 32,
            "total_token_throughput": 5000.0,
            "output_throughput": 4000.0,
        }

        env = single_node_env_vars.copy()
        env["ISL"] = "8192"
        env["OSL"] = "1024"

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["isl"] == 8192
        assert output_data["osl"] == 1024
        assert isinstance(output_data["isl"], int)
        assert isinstance(output_data["osl"], int)

    def test_conc_from_benchmark_result(self, tmp_path, single_node_env_vars):
        """Test that conc is read from benchmark result max_concurrency."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 128,
            "total_token_throughput": 5000.0,
            "output_throughput": 4000.0,
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["conc"] == 128


# =============================================================================
# Integration: power aggregation patches the agg JSON
# =============================================================================

class TestPowerAggregationIntegration:
    """End-to-end wiring: process_result.py invokes aggregate_power.py and
    patches the validated whole-deployment power contract into the agg JSON.

    Exercises the env-var path resolution (GPU_METRICS_CSV), the subprocess
    boundary, topology validation, and best-effort/strict modes.
    """

    @staticmethod
    def _write_nvidia_csv(path, start_unix, end_unix, watts_per_gpu=500.0, num_gpus=8):
        """Stage a 1Hz nvidia-smi-style CSV bracketing the bench window with
        warmup/eval phases that should be filtered out by the aggregator."""
        from datetime import datetime

        def ts(t):
            return datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")

        lines = ["timestamp, index, power.draw [W], temperature.gpu"]
        # 5s warmup at 100W (before start) — must be excluded.
        for s in range(5):
            for g in range(num_gpus):
                lines.append(f"{ts(start_unix - 5 + s)}, {g}, 100.00 W, 50")
        # Bench window samples at the requested wattage.
        duration_s = int(end_unix - start_unix)
        for s in range(duration_s + 1):
            for g in range(num_gpus):
                lines.append(f"{ts(start_unix + s)}, {g}, {watts_per_gpu:.2f} W, 75")
        # 5s eval at 200W (after end) — must be excluded.
        for s in range(5):
            for g in range(num_gpus):
                lines.append(f"{ts(end_unix + 1 + s)}, {g}, 200.00 W, 65")
        path.write_text("\n".join(lines) + "\n")

    def test_agg_json_gets_patched_with_power_and_joules(self, tmp_path, single_node_env_vars):
        """The full pipeline: process_result.py + aggregate_power.py."""
        start, end = 1_700_000_100.0, 1_700_000_160.0  # 60s bench window
        csv_path = tmp_path / "gpu_metrics.csv"
        self._write_nvidia_csv(csv_path, start, end, watts_per_gpu=600.0, num_gpus=8)

        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 64,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            # Fields read by aggregate_power.py.
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": 60.0,
            "completed": 30,
            "total_input_tokens": 240_000,
            "total_output_tokens": 30_000,
        }
        env = {**single_node_env_vars, "GPU_METRICS_CSV": str(csv_path)}

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        agg_path = tmp_path / "agg_benchmark_result.json"
        assert agg_path.is_file()
        patched = json.loads(agg_path.read_text())

        # Pre-existing fields still present.
        assert patched["hw"] == "mi300x"
        assert patched["tp"] == 8
        assert patched["conc"] == 64
        # New power fields.
        assert patched["power_valid"] == 1
        assert patched["avg_power_w"] == pytest.approx(600.0, abs=0.5)
        assert patched["avg_total_gpu_power_w"] == pytest.approx(4_800.0, abs=0.5)
        assert patched["total_gpu_energy_j"] == pytest.approx(288_000.0, abs=0.5)
        assert patched["joules_per_successful_query"] == pytest.approx(9_600.0, abs=0.05)
        assert patched["joules_per_input_token"] == pytest.approx(1.2, abs=0.01)
        # 600W × 8 GPUs × 60s / 30_000 tokens = 9.6 J/tok
        assert patched["joules_per_output_token"] == pytest.approx(9.6, abs=0.05)
        assert (tmp_path / "power_validation_benchmark_result.json").is_file()

    def test_missing_csv_does_not_break_process_result(self, tmp_path, single_node_env_vars):
        """Without GPU_METRICS_CSV (or with a missing file), process_result.py
        still succeeds and writes the agg JSON — just without the power fields.
        This is the production case for runs that ship without monitoring."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 64,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            "benchmark_start_time_unix": 1_700_000_100.0,
            "benchmark_end_time_unix": 1_700_000_110.0,
            "duration": 10.0,
            "completed": 4,
            "total_input_tokens": 32_768,
            "total_output_tokens": 4_096,
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        agg_path = tmp_path / "agg_benchmark_result.json"
        patched = json.loads(agg_path.read_text())
        assert "avg_power_w" not in patched
        assert "joules_per_output_token" not in patched
        assert patched["power_valid"] == 0
        assert "power_invalid_reasons" not in patched

        validation = json.loads(
            (tmp_path / "power_validation_benchmark_result.json").read_text()
        )
        assert validation["reasons"] == ["telemetry_file_missing"]

    def test_missing_bench_timestamps_does_not_patch(self, tmp_path, single_node_env_vars):
        """A CSV is present but the bench JSON predates the timestamp fields
        (legacy benchmark_serving.py). Aggregator should skip silently."""
        start, end = 1_700_000_100.0, 1_700_000_160.0
        csv_path = tmp_path / "gpu_metrics.csv"
        self._write_nvidia_csv(csv_path, start, end, watts_per_gpu=600.0, num_gpus=1)

        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 64,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            # NOTE: deliberately missing benchmark_start_time_unix/end/total_output_tokens.
        }
        env = {**single_node_env_vars, "GPU_METRICS_CSV": str(csv_path)}

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        agg_path = tmp_path / "agg_benchmark_result.json"
        patched = json.loads(agg_path.read_text())
        assert "avg_power_w" not in patched
        assert "joules_per_output_token" not in patched
        assert patched["power_valid"] == 0
        assert "power_invalid_reasons" not in patched

    def test_expected_gpu_count_mismatch_is_invalid(self, tmp_path, single_node_env_vars):
        """TP/PP/PCP topology is checked against the observed device IDs."""
        start, end = 1_700_000_100.0, 1_700_000_110.0
        csv_path = tmp_path / "gpu_metrics.csv"
        self._write_nvidia_csv(csv_path, start, end, watts_per_gpu=600.0, num_gpus=4)
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 4,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": 10.0,
            "completed": 4,
            "total_input_tokens": 32_768,
            "total_output_tokens": 4_096,
        }
        env = {**single_node_env_vars, "GPU_METRICS_CSV": str(csv_path)}

        result = run_script(tmp_path, env, benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        patched = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert patched["power_valid"] == 0
        assert "power_invalid_reasons" not in patched
        assert "total_gpu_energy_j" not in patched

    def test_require_power_propagates_validation_failure(
        self, tmp_path, single_node_env_vars
    ):
        """Study/CI mode must fail after preserving validation artifacts."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 4,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            "benchmark_start_time_unix": 1_700_000_100.0,
            "benchmark_end_time_unix": 1_700_000_110.0,
            "duration": 10.0,
            "completed": 4,
            "total_input_tokens": 32_768,
            "total_output_tokens": 4_096,
        }
        env = {**single_node_env_vars, "REQUIRE_POWER": "1"}

        result = run_script(tmp_path, env, benchmark_result)

        assert result.returncode != 0
        assert "Power validation failed" in result.stderr
        validation = json.loads(
            (tmp_path / "power_validation_benchmark_result.json").read_text()
        )
        assert validation["reasons"] == ["telemetry_file_missing"]

    def test_require_power_accepts_valid_single_node_measurement(
        self, tmp_path, single_node_env_vars
    ):
        """The strict H100/H200 canary path also has a protected success case."""
        start, end = 1_700_000_100.0, 1_700_000_110.0
        csv_path = tmp_path / "gpu_metrics.csv"
        self._write_nvidia_csv(
            csv_path,
            start,
            end,
            watts_per_gpu=500.0,
            num_gpus=8,
        )
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 4,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": 10.0,
            "completed": 4,
            "total_input_tokens": 32_768,
            "total_output_tokens": 4_096,
        }
        env = {
            **single_node_env_vars,
            "GPU_METRICS_CSV": str(csv_path),
            "REQUIRE_POWER": "1",
        }

        result = run_script(tmp_path, env, benchmark_result)

        assert result.returncode == 0, result.stderr
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_metric_schema_version"] == 2
        assert agg["power_valid"] == 1
        assert agg["total_gpu_energy_j"] == pytest.approx(40_000.0)
        validation = json.loads(
            (tmp_path / "power_validation_benchmark_result.json").read_text()
        )
        assert validation["power_valid"] is True
        assert validation["reasons"] == []

    @pytest.mark.parametrize(
        ("require_power", "expected_returncode"),
        [(False, 0), (True, 1)],
    )
    def test_internal_aggregation_error_is_always_auditable(
        self,
        tmp_path,
        single_node_env_vars,
        require_power,
        expected_returncode,
    ):
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 4,
            "total_token_throughput": 1000.0,
            "output_throughput": 500.0,
            "benchmark_start_time_unix": 1_700_000_100.0,
            "benchmark_end_time_unix": 1_700_000_110.0,
            "duration": 10.0,
            "completed": 4,
            "total_input_tokens": 32_768,
            "total_output_tokens": 4_096,
        }
        env = single_node_env_vars.copy()
        if require_power:
            env["REQUIRE_POWER"] = "1"

        result = run_script_with_broken_aggregator(tmp_path, env, benchmark_result)

        assert result.returncode == expected_returncode
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_metric_schema_version"] == 2
        assert agg["power_valid"] == 0
        assert "power_invalid_reasons" not in agg
        validation = json.loads(
            (tmp_path / "power_validation_benchmark_result.json").read_text()
        )
        assert validation["power_valid"] is False
        assert validation["reasons"] == ["aggregation_internal_error"]
        assert validation["internal_error"]["type"] == "RuntimeError"

    def test_stop_gpu_monitor_appends_final_nvidia_sample(self, tmp_path):
        """Stopping between 1 Hz ticks still records one post-benchmark sample."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        args_log = tmp_path / "nvidia_args.txt"
        fake_nvidia_smi = fake_bin / "nvidia-smi"
        fake_nvidia_smi.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$*\" > {str(args_log)!r}\n"
            "printf '%s\\n' "
            "'2026/07/23 12:00:11.000, 0, 500.00 W, 65, 1000, 1000, 90 %, 10 %'\n"
        )
        fake_nvidia_smi.chmod(0o755)
        metrics = tmp_path / "gpu_metrics.csv"
        metrics.write_text(
            "timestamp, index, power.draw [W], temperature.gpu, "
            "clocks.current.sm [MHz], clocks.current.memory [MHz], "
            "utilization.gpu [%], utilization.memory [%]\n"
        )
        benchmark_lib = Path(__file__).parents[1] / "benchmarks/benchmark_lib.sh"
        script = f"""
source {str(benchmark_lib)!r}
kill() {{ return 0; }}
wait() {{ return 0; }}
GPU_MONITOR_PID=999
GPU_MONITOR_VENDOR=nvidia
GPU_METRICS_CSV={str(metrics)!r}
stop_gpu_monitor
"""
        env = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "--format=csv,noheader" in args_log.read_text()
        assert "2026/07/23 12:00:11.000, 0, 500.00 W" in metrics.read_text()

    def test_stop_gpu_monitor_drops_truncated_row_before_final_sample(self, tmp_path):
        """A killed monitor cannot concatenate its partial row with the final sample."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_nvidia_smi = fake_bin / "nvidia-smi"
        final_sample = (
            "2026/07/23 12:00:11.000, 0, 500.00 W, "
            "65, 1000, 1000, 90 %, 10 %"
        )
        fake_nvidia_smi.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' {final_sample!r}\n"
        )
        fake_nvidia_smi.chmod(0o755)

        header = (
            "timestamp, index, power.draw [W], temperature.gpu, "
            "clocks.current.sm [MHz], clocks.current.memory [MHz], "
            "utilization.gpu [%], utilization.memory [%]"
        )
        complete_sample = (
            "2026/07/23 12:00:09.000, 0, 490.00 W, "
            "64, 990, 990, 89 %, 9 %"
        )
        truncated_sample = "2026/07/23 12:00:10.000, 0, 52"
        metrics = tmp_path / "gpu_metrics.csv"
        metrics.write_text(
            f"{header}\n{complete_sample}\n{truncated_sample}"
        )

        benchmark_lib = Path(__file__).parents[1] / "benchmarks/benchmark_lib.sh"
        script = f"""
source {str(benchmark_lib)!r}
kill() {{ return 0; }}
wait() {{ return 0; }}
GPU_MONITOR_PID=999
GPU_MONITOR_VENDOR=nvidia
GPU_METRICS_CSV={str(metrics)!r}
stop_gpu_monitor
"""
        env = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert metrics.read_text().splitlines() == [
            header,
            complete_sample,
            final_sample,
        ]

    def test_stop_gpu_monitor_amd_covers_stop_request_and_snapshots_energy(self, tmp_path):
        """AMD stop returns once telemetry covers the stop entry, then snapshots energy.

        A usable tick stamped past the stop request satisfies the coverage
        poll on its first pass: the legacy fixed tail sleep never runs, the
        stream is not mutated, and the end-side accumulator snapshot is
        written."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        args_log = tmp_path / "amd_args.txt"
        fake_amd_smi = fake_bin / "amd-smi"
        fake_amd_smi.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$*\" >> {str(args_log)!r}\n"
            "printf 'gpu,total_energy_consumption\\n0,178319501.7\\n'\n"
        )
        fake_amd_smi.chmod(0o755)
        sleep_log = tmp_path / "sleep_args.txt"
        covered_tick = int(time.time()) + 30
        contents = f"timestamp,gpu,socket_power\n{covered_tick},0,238\n"
        metrics = tmp_path / "gpu_metrics.csv"
        metrics.write_text(contents)
        benchmark_lib = Path(__file__).parents[1] / "benchmarks/benchmark_lib.sh"
        script = f"""
source {str(benchmark_lib)!r}
kill() {{ return 0; }}
wait() {{ return 0; }}
sleep() {{ printf '%s\\n' "$1" >> {str(sleep_log)!r}; }}
GPU_MONITOR_PID=999
GPU_MONITOR_VENDOR=amd
GPU_MONITOR_INTERVAL=3
GPU_METRICS_CSV={str(metrics)!r}
stop_gpu_monitor
"""
        env = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not sleep_log.exists()
        assert "never covered the stop request" not in result.stderr
        assert metrics.read_text() == contents
        assert "metric -E --csv" in args_log.read_text()
        energy_end = tmp_path / "gpu_metrics_energy_end.csv"
        assert energy_end.read_text().startswith("gpu,total_energy_consumption")

    def test_stop_gpu_monitor_amd_drops_truncated_row_without_append(self, tmp_path):
        """The AMD path repairs a partial trailing row but appends no sample."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_amd_smi = fake_bin / "amd-smi"
        fake_amd_smi.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'gpu,total_energy_consumption\\n0,178319501.7\\n'\n"
        )
        fake_amd_smi.chmod(0o755)
        header = "timestamp,gpu,socket_power"
        complete_sample = "1785881113,0,238"
        metrics = tmp_path / "gpu_metrics.csv"
        metrics.write_text(f"{header}\n{complete_sample}\n1785881114,0,2")
        benchmark_lib = Path(__file__).parents[1] / "benchmarks/benchmark_lib.sh"
        script = f"""
source {str(benchmark_lib)!r}
kill() {{ return 0; }}
wait() {{ return 0; }}
sleep() {{ return 0; }}
GPU_MONITOR_PID=999
GPU_MONITOR_VENDOR=amd
GPU_METRICS_CSV={str(metrics)!r}
AMD_MONITOR_STOP_TIMEOUT_S=0
stop_gpu_monitor
"""
        env = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert metrics.read_text().splitlines() == [header, complete_sample]
        assert (tmp_path / "gpu_metrics_energy_end.csv").exists()

    def test_start_stop_gpu_monitor_amd_lifecycle(self, tmp_path):
        """Watch rows survive the kill and both boundary snapshots are written."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_amd_smi = fake_bin / "amd-smi"
        fake_amd_smi.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *" -w "* ]]; then\n'
            "    echo \"'CTRL' + 'C' to stop watching output:\"\n"
            "    echo 'timestamp,gpu,socket_power,power_management'\n"
            "    for _ in $(seq 1 30); do\n"
            "        echo \"$(date +%s),0,238,ENABLED\"\n"
            "        echo 'timestamp,gpu,socket_power,power_management'\n"
            "        sleep 0.2\n"
            "    done\n"
            'elif [[ "$*" == *"metric -E --csv"* ]]; then\n'
            "    printf 'gpu,total_energy_consumption\\n0,178319501.7\\n'\n"
            'elif [[ "$1" == "static" ]]; then\n'
            "    printf '{\"gpu_data\": []}\\n'\n"
            "fi\n"
        )
        fake_amd_smi.chmod(0o755)
        metrics = tmp_path / "gpu_metrics.csv"
        benchmark_lib = Path(__file__).parents[1] / "benchmarks/benchmark_lib.sh"
        script = f"""
source {str(benchmark_lib)!r}
start_gpu_monitor --output {str(metrics)!r} --interval 1
sleep 0.8
stop_gpu_monitor
"""
        env = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        lines = metrics.read_text().splitlines()
        assert lines[0] == "timestamp,gpu,socket_power,power_management"
        assert sum(1 for line in lines if line.startswith("timestamp,")) == 1
        assert "CTRL" not in metrics.read_text()
        data_rows = [line for line in lines[1:] if line]
        assert len(data_rows) >= 2
        assert all(row.split(",")[2] == "238" for row in data_rows)
        energy_start = tmp_path / "gpu_metrics_energy_start.csv"
        assert energy_start.read_text().startswith("gpu,total_energy_consumption")
        assert (tmp_path / "gpu_metrics_energy_end.csv").exists()
        identity = json.loads((tmp_path / "gpu_metrics_identity.json").read_text())
        assert identity == {"gpu_data": []}


# =============================================================================
# Integration: multinode power aggregation patches the agg JSON
# =============================================================================


class TestMultinodePower:
    """End-to-end wiring: process_result.py invokes aggregate_power_multinode.py
    against the srt-slurm artifact package staged under LOGS/.

    The consumer binds the processed copy to LOGS/<result_path> by canonical
    sha256, so the benchmark result passed to run_script must equal the
    package's original result byte-for-byte after JSON canonicalization.
    """

    BENCH_EXTRA = {"total_token_throughput": 15000.5, "output_throughput": 12000.0}

    @pytest.fixture
    def power_env(self, multinode_env_vars):
        return {
            **multinode_env_vars,
            "PREFILL_GPUS": "2",
            "DECODE_GPUS": "2",
            "POWER_PRODUCER_SHA": PRODUCER_SHA,
        }

    def _build(self, tmp_path, **kwargs):
        pkg = build_package(tmp_path, bench_extra=self.BENCH_EXTRA, **kwargs)
        return json.loads(pkg.original_result.read_text())

    def _bench_without_package(self):
        return {
            "model_id": "test-model",
            "max_concurrency": 4,
            **self.BENCH_EXTRA,
        }

    def test_valid_package_patches_role_energy(self, tmp_path, power_env):
        benchmark_result = self._build(tmp_path)

        result = run_script(tmp_path, power_env, benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_metric_schema_version"] == 2
        assert agg["power_valid"] == 1
        assert agg["prefill_gpu_energy_j"] == 48000.0
        assert agg["decode_gpu_energy_j"] == 36000.0
        assert agg["prefill_avg_power_w"] == 400.0
        assert agg["decode_avg_power_w"] == 300.0
        assert (tmp_path / "power_validation_benchmark_result.json").is_file()

    def test_missing_package_is_best_effort(self, tmp_path, power_env):
        result = run_script(tmp_path, power_env, self._bench_without_package())

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_metric_schema_version"] == 2
        assert agg["power_valid"] == 0
        for key in WHOLE_METRIC_KEYS + ROLE_METRIC_KEYS:
            assert key not in agg
        assert (tmp_path / "power_validation_benchmark_result.json").is_file()

    def test_missing_package_fails_in_strict_mode(self, tmp_path, power_env):
        env = {**power_env, "REQUIRE_POWER": "1"}

        result = run_script(tmp_path, env, self._bench_without_package())

        assert result.returncode != 0
        assert (tmp_path / "power_validation_benchmark_result.json").is_file()

    def test_invalid_package_withholds_metrics(self, tmp_path, power_env):
        benchmark_result = self._build(tmp_path, publication_valid=False)

        result = run_script(tmp_path, power_env, benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_valid"] == 0
        for key in WHOLE_METRIC_KEYS + ROLE_METRIC_KEYS:
            assert key not in agg
        validation = json.loads(
            (tmp_path / "power_validation_benchmark_result.json").read_text()
        )
        assert "producer_verdict_mismatch" in validation["reasons"]

    def test_strict_mode_passes_on_valid_package(self, tmp_path, power_env):
        benchmark_result = self._build(tmp_path)
        env = {**power_env, "REQUIRE_POWER": "1"}

        result = run_script(tmp_path, env, benchmark_result)

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        agg = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
        assert agg["power_valid"] == 1
