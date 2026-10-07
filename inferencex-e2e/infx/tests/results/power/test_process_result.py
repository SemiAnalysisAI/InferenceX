"""Exercise the fixed-sequence module CLI with controlled environment and artifacts."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.results.power.multinode import ROLE_METRIC_KEYS, WHOLE_METRIC_KEYS
from test_aggregate_power_multinode import PRODUCER_SHA, build_package

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_COMMAND = [sys.executable, "-m", "infx.results.fixed_sequence"]


@pytest.mark.parametrize('fingerprint', ['', 'a' * 64, 'a' * 16 + 'b' * 48])
def test_long_multinode_names_survive_result_and_power_processing(
    tmp_path, multinode_env_vars, sample_benchmark_result, fingerprint
):
    from infx.results.result_filename import point_filename, result_stem

    base = ('example_8k1k_fp4_dynamo-sglang_prefill-tp4-pp1-dcp1-pcp1-ep1-dpfalse-nw1_'
            'decode-tp4-pp1-dcp1-pcp1-ep1-dpfalse-nw1_disagg-true_spec-none_'
            'conc1x4x8x16x32x64x256_cluster-runner_00')
    stem = result_stem(base, fingerprint)
    name = point_filename(stem, 'sa-bench_isl_8192_osl_1024', '16', '8', '4', '4')
    assert len(('power_validation_' + name + '.tmp').encode()) <= 255
    env = {**multinode_env_vars, 'RECIPE_FINGERPRINT': fingerprint}
    result = run_script(tmp_path, env, sample_benchmark_result, name.removesuffix('.json'))
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / ('agg_' + name)).read_text())
    assert aggregate['recipe_fingerprint'] == fingerprint
    assert aggregate['model'] == 'deepseek-ai/DeepSeek-R1-0528'
    assert (tmp_path / ('power_validation_' + name)).is_file()
    # Different full fingerprints must remain distinct even with the same first 16 characters.
    assert result_stem(base, 'a' * 64) != result_stem(base, 'a' * 16 + 'b' * 48)


def test_result_builder_uses_explicit_readonly_inputs(single_node_env_vars, monkeypatch, tmp_path):
    from types import MappingProxyType
    from infx.results.fixed_sequence import build_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TP", "invalid ambient value")
    env = {**single_node_env_vars, "TP": "2"}
    del env["RESULT_FILENAME"]
    benchmark = {
        "model_id": "fixture", "max_concurrency": 3,
        "total_token_throughput": 12, "output_throughput": 8,
        "ttft_p50_ms": 250, "tpot_p50_ms": 20,
    }
    result = build_result(MappingProxyType(benchmark), MappingProxyType(env))
    assert result["tput_per_gpu"] == 6
    assert result["input_tput_per_gpu"] == 2
    assert result["output_tput_per_gpu"] == 4
    assert result["ttft_p50"] == 0.25
    assert result["intvty_p50"] == 50

    second = build_result(benchmark, {**env, "TP": "4"})
    assert second["tput_per_gpu"] == 3
    assert result["tput_per_gpu"] == 6
    assert env["TP"] == "2"
    assert benchmark["ttft_p50_ms"] == 250
    assert list(tmp_path.iterdir()) == []


def test_multinode_explicit_gpu_counts_control_decode_fields_and_denominators(
    multinode_env_vars, sample_benchmark_result,
):
    from infx.results.fixed_sequence import build_result

    # Worker dimensions describe 48 prefill and 180 decode GPUs. The supplied
    # allocation counts, 20 and 0, remain authoritative for this collector.
    env = {**multinode_env_vars, "PREFILL_NUM_WORKERS": "2", "PREFILL_TP": "3",
           "PREFILL_PP_SIZE": "2", "PREFILL_PCP_SIZE": "4", "DECODE_NUM_WORKERS": "3",
           "DECODE_TP": "6", "DECODE_EP": "5", "DECODE_PP_SIZE": "2",
           "DECODE_DCP_SIZE": "3", "DECODE_PCP_SIZE": "5", "DECODE_GPUS": "0"}
    benchmark = {**sample_benchmark_result, "total_token_throughput": 600,
                 "output_throughput": 400}
    result = build_result(benchmark, env)
    assert [result[key] for key in ("decode_tp", "decode_ep", "decode_pp",
                                   "decode_dcp_size", "decode_pcp_size")] == [0, 0, 1, 1, 1]
    assert result["decode_num_workers"] == 3
    assert result["num_prefill_gpu"] == 20
    assert result["num_decode_gpu"] == 0
    assert result["tput_per_gpu"] == 30
    assert result["input_tput_per_gpu"] == 10
    assert result["output_tput_per_gpu"] == 20

    result = build_result(benchmark, {**env, "DECODE_GPUS": "4"})
    assert [result[key] for key in ("decode_tp", "decode_ep", "decode_pp",
                                   "decode_dcp_size", "decode_pcp_size")] == [6, 5, 2, 3, 5]
    assert result["tput_per_gpu"] == 25
    assert result["output_tput_per_gpu"] == 100


@pytest.mark.parametrize("overrides,message", [
    ({"DECODE_HARDWARE": "", "PREFILL_TP": "invalid"},
     "PREFILL_HARDWARE and DECODE_HARDWARE must be specified together."),
    ({"PREFILL_PP_SIZE": "0", "DECODE_GPUS": "-20"},
     "Multinode PP, DCP, and PCP sizes must be positive integers."),
    ({"DECODE_PP_SIZE": "0", "DECODE_GPUS": "0"},
     "Multinode PP, DCP, and PCP sizes must be positive integers."),
])
def test_multinode_topology_preserves_validation_order(
    multinode_env_vars, sample_benchmark_result, overrides, message,
):
    from infx.results.fixed_sequence import build_result

    with pytest.raises(ValueError) as error:
        build_result(sample_benchmark_result, {**multinode_env_vars, **overrides})
    assert str(error.value) == message


@pytest.mark.parametrize("name", ["PP_SIZE", "DCP_SIZE", "PCP_SIZE"])
def test_fixed_topology_rejects_empty_parallelism(
    single_node_env_vars, sample_benchmark_result, name,
):
    from infx.results.fixed_sequence import build_result

    with pytest.raises(ValueError, match="invalid literal for int"):
        build_result(sample_benchmark_result, {**single_node_env_vars, name: ""})



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
    """Helper to run the infx.results.fixed_sequence script."""
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(benchmark_result))

    env = env.copy()
    env["RESULT_FILENAME"] = result_filename
    env["PYTHONPATH"] = str(REPO_ROOT)

    return subprocess.run(
        MODULE_COMMAND,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def run_script_with_broken_aggregator(
    tmp_path, env, benchmark_result, result_filename="benchmark_result", *, fail_import=False,
):
    """Run process_result with the real multinode aggregator failing unexpectedly."""
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(benchmark_result))
    env = {**env, "RESULT_FILENAME": result_filename}
    wrapper = f"""
import runpy
import sys
import json
import builtins
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT)!r})

def broken_run(*args, **kwargs):
    path = Path(kwargs['agg_result'] if 'agg_result' in kwargs else args[2])
    data = json.loads(path.read_text())
    data.update(prefill_gpu_energy_j=99, total_gpu_energy_j=99)
    path.write_text(json.dumps(data))
    raise RuntimeError("forced aggregation failure")
if {fail_import!r}:
    original_import = builtins.__import__
    def failing_import(name, *args, **kwargs):
        if name.endswith('power.multinode'):
            raise ImportError("forced import failure")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = failing_import
else:
    from infx.results.power import multinode as aggregate_power_multinode
    aggregate_power_multinode.run = broken_run
runpy.run_module("infx.results.fixed_sequence", run_name="__main__")
"""
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )



class TestProcessResultScript:
    """Tests for infx.results.fixed_sequence script execution."""

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

    @pytest.mark.parametrize("raw", ["", "null"])
    def test_null_component_metadata_is_omitted(
        self, tmp_path, sample_benchmark_result, single_node_env_vars, raw
    ):
        result = run_script(
            tmp_path, {**single_node_env_vars, "ROUTER_METADATA": raw},
            sample_benchmark_result,
        )
        assert result.returncode == 0, result.stderr
        assert "router" not in json.loads(result.stdout)

    @pytest.mark.parametrize(("raw", "message"), [
        ("{", "must contain valid JSON"),
        ("[]", "must contain exactly 'name' and 'version'"),
        ('{"name":"router","version":""}', "name and version must be non-empty strings"),
        ('{"name":42,"version":"1"}', "name and version must be non-empty strings"),
    ])
    def test_malformed_component_metadata_fails_before_writing(
        self, tmp_path, sample_benchmark_result, single_node_env_vars, raw, message
    ):
        result = run_script(
            tmp_path, {**single_node_env_vars, "ROUTER_METADATA": raw},
            sample_benchmark_result,
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert f"ValueError: ROUTER_METADATA {message}" in result.stderr
        assert not (tmp_path / "agg_benchmark_result.json").exists()

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
            MODULE_COMMAND,
            cwd=tmp_path,
            env={"PATH": "/usr/bin", "RESULT_FILENAME": "benchmark_result", "PYTHONPATH": str(REPO_ROOT)},
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
        env["PYTHONPATH"] = str(REPO_ROOT)

        result = subprocess.run(
            MODULE_COMMAND,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0



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



class TestEdgeCases:
    """Tests for edge cases and special scenarios."""


    def test_boolean_disagg_parsing_true_requires_multinode(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that DISAGG=true without multinode fails."""
        for disagg_value in ["true", "True", "TRUE"]:
            env = single_node_env_vars.copy()
            env["DISAGG"] = disagg_value

            result = run_script(tmp_path, env, sample_benchmark_result)
            assert result.returncode != 0



@pytest.mark.parametrize("workflow_name,step_name", [
    ("benchmark-tmpl.yml", "Process result"),
    ("profile.yml", "Process result (json -> agg)"),
])
def test_workflow_uses_result_python_with_unsupported_ambient_python(
    tmp_path, single_node_env_vars, workflow_name, step_name
):
    import yaml

    workflow = yaml.safe_load((REPO_ROOT.parent / ".github/workflows" / workflow_name).read_text())
    step = next(
        s for job in workflow["jobs"].values() for s in job.get("steps", [])
        if s.get("name") == step_name
    )
    (tmp_path / ".result-tooling").symlink_to(REPO_ROOT.parent, target_is_directory=True)
    (tmp_path / "infx").mkdir()
    (tmp_path / "infx/__init__.py").write_text("raise RuntimeError('measured package imported')\n")
    (tmp_path / "bin").mkdir()
    ambient_python = tmp_path / "bin/python3"
    ambient_python.write_text("#!/bin/sh\nexit 73\n")
    ambient_python.chmod(0o755)
    (tmp_path / "benchmark_result.json").write_text(json.dumps({
        "model_id": "fixture",
        "max_concurrency": 8,
        "total_token_throughput": 1000,
        "output_throughput": 500,
    }))
    env = {
        **os.environ,
        **single_node_env_vars,
        "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "INFERENCEX_RESULTS_PYTHON": sys.executable,
        "PYTHONPATH": step["env"]["PYTHONPATH"].replace("${{ github.workspace }}", str(tmp_path)),
    }
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
    # 1000 tokens/s over the fixture's TP=8.
    assert aggregate["tput_per_gpu"] == pytest.approx(125.0)


@pytest.mark.parametrize("require_power", ["", "1"])
def test_single_node_result_publishes_no_power(tmp_path, single_node_env_vars, require_power):
    """No single-node lane collects power, so even a strict run publishes none."""
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
    env = {**single_node_env_vars, "REQUIRE_POWER": require_power}

    result = run_script(tmp_path, env, benchmark_result)

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "agg_benchmark_result.json").read_text())
    power_fields = {
        "power_valid", "power_metric_schema_version", "power_invalid_reasons", "power_audit",
        *WHOLE_METRIC_KEYS,
    }
    assert power_fields.isdisjoint(aggregate)
    assert not list(tmp_path.glob("power_validation_*"))


class TestMultinodePower:
    """End-to-end wiring: infx.results.fixed_sequence invokes aggregate_power_multinode.py
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

    @pytest.mark.parametrize("require_power", ["", "yes"])
    @pytest.mark.parametrize("fail_import", [False, True])
    def test_internal_error_preserves_validation(
        self, tmp_path, multinode_env_vars, sample_benchmark_result,
        require_power, fail_import,
    ):
        result = run_script_with_broken_aggregator(
            tmp_path, {**multinode_env_vars, "REQUIRE_POWER": require_power},
            {**sample_benchmark_result, "prefill_gpu_energy_j_ms": 99000},
            fail_import=fail_import,
        )
        assert result.returncode == (1 if require_power else 0)
        agg = json.loads(result.stdout)
        assert agg["power_valid"] == 0
        assert agg["power_invalid_reasons"] == ["aggregation_internal_error"]
        assert "prefill_gpu_energy_j" not in agg
        assert "total_gpu_energy_j" not in agg
        validation = json.loads((tmp_path / "power_validation_benchmark_result.json").read_text())
        assert validation["reasons"] == ["aggregation_internal_error"]
        assert validation["internal_error"] == {
            "type": "ImportError" if fail_import else "RuntimeError",
            "message": "forced import failure" if fail_import else "forced aggregation failure",
        }


@pytest.mark.parametrize('completed,status', [(100, 'passed'), (95, 'passed'), (94, 'failed')])
def test_request_outcome_preserves_existing_failure_threshold(completed, status):
    from infx.bench_serving.benchmark_outcome import benchmark_outcome

    outcome = benchmark_outcome(100, completed)
    assert outcome['status'] == status
    assert outcome['failed'] == 100 - completed
    assert outcome['max_failure_rate'] == 0.05


def test_request_outcome_cannot_disagree_with_raw_counts(single_node_env_vars, sample_benchmark_result):
    from infx.results.fixed_sequence import build_result

    raw = {**sample_benchmark_result, 'completed': 94,
           'benchmark_outcome': {'status': 'passed', 'requested': 100, 'completed': 100,
                                 'failed': 0, 'max_failure_rate': 0.05}}
    with pytest.raises(ValueError, match='request counts and gate'):
        build_result(raw, single_node_env_vars)


@pytest.mark.parametrize("provenance", [
    {},
    # Fork-era AMD packages carry the retired power_profile key beside the AMD metric.
    {"power_profile": "amd-device-metrics", "source_metric": "gpu_power_usage",
     "power_scope": "gpu_device_power_as_reported_by_amd_device_metrics_exporter"},
], ids=["dcgm", "fork-amd"])
@pytest.mark.parametrize('multinode', [True, False])
def test_native_aggregate_role_through_result_processor(
    tmp_path, multinode_env_vars, single_node_env_vars, multinode, provenance,
):
    pkg = build_package(tmp_path, bench_extra=TestMultinodePower.BENCH_EXTRA)
    manifest_path = pkg.power_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(provenance)
    for device in manifest['expected_devices']:
        for assignment in device['assignments']:
            assignment.update(worker_role='agg', het_group=None)
    manifest_path.write_text(json.dumps(manifest))
    env = {**(multinode_env_vars if multinode else single_node_env_vars),
           'DISAGG': 'false', 'PREFILL_GPUS': '0',
           'DECODE_GPUS': '0', 'AGGREGATE_GPUS': '4', 'POWER_PRODUCER_SHA': PRODUCER_SHA,
           'TP': '4', 'GPU_COUNT': '4', 'POWER_ARTIFACT_DIR': str(pkg.power_dir),
           'REQUIRE_POWER': '1'}
    result = run_script(tmp_path, env, json.loads(pkg.original_result.read_text()))
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / 'agg_benchmark_result.json').read_text())
    assert aggregate['power_valid'] == 1
    assert aggregate['is_multinode'] is multinode
    assert aggregate['num_aggregate_gpu' if multinode else 'tp'] == 4
    assert aggregate['power_audit']['expected_gpu_count'] == 4
    assert aggregate['avg_power_w'] == 350
    assert aggregate['total_gpu_energy_j'] == 84_000
    assert aggregate['power_audit']['producer_sha'] == PRODUCER_SHA
    assert set(ROLE_METRIC_KEYS).isdisjoint(aggregate)

    manifest['source_metric'] = ''
    manifest_path.write_text(json.dumps(manifest))
    result = run_script(tmp_path, env, json.loads(pkg.original_result.read_text()))
    assert result.returncode == 1
    invalid = json.loads((tmp_path / 'agg_benchmark_result.json').read_text())
    assert invalid['power_valid'] == 0
    assert 'avg_power_w' not in invalid


@pytest.mark.parametrize('conc_token,rate_suffix', [
    ('c', ''), ('conc', ''),
    ('concurrency_', '_req_rate_1'), ('concurrency_', '_req_rate_inf'),
])
@pytest.mark.parametrize('expected_concs,missing', [('4 8 16', [8]), ('4 16', [])])
def test_multinode_batch_preserves_points_and_checks_completeness(
    tmp_path, multinode_env_vars, sample_benchmark_result, conc_token, rate_suffix,
    expected_concs, missing,
):
    import yaml

    workflow = yaml.safe_load((REPO_ROOT.parent / ".github/workflows/benchmark-multinode-tmpl.yml").read_text())
    step = next(s for job in workflow["jobs"].values() for s in job.get("steps", [])
                if s.get("name") == "Process result")
    (tmp_path / ".result-tooling").symlink_to(REPO_ROOT.parent, target_is_directory=True)
    for conc in (4, 16):
        (tmp_path / f'run_recipe_{conc_token}{conc}{rate_suffix}_gpus_4_ctx_2_gen_2.json').write_text(
            json.dumps({**sample_benchmark_result, 'max_concurrency': conc}))
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run',
           'CONC_LIST': expected_concs, 'REQUIRE_POWER': '0'}
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', step['run']], cwd=tmp_path,
                            env={**env, 'INFERENCEX_RESULTS_PYTHON': sys.executable,
                                 'PYTHONPATH': step['env']['PYTHONPATH'].replace(
                                     '${{ github.workspace }}', str(tmp_path))},
                            capture_output=True, text=True)
    assert result.returncode == int(bool(missing)), result.stderr
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['missing_concurrencies'] == missing
    assert receipt['unexpected_concurrencies'] == []
    assert len(receipt['points']) == 2
    for conc in (4, 16):
        stem = f'run_recipe_{conc_token}{conc}{rate_suffix}_gpus_4_ctx_2_gen_2'
        aggregate = json.loads((tmp_path / f'agg_{stem}.json').read_text())
        assert aggregate['power_valid'] == 0
        assert aggregate['power_invalid_reasons']
        assert (tmp_path / f'power_validation_{stem}.json').is_file()


@pytest.mark.parametrize('role_suffix', ['', '_ctx_4_gen_0'])
def test_multinode_batch_normalizes_legacy_zero_decode_aggregate(
    tmp_path, multinode_env_vars, role_suffix,
):
    pkg = build_package(tmp_path, bench_extra=TestMultinodePower.BENCH_EXTRA)
    manifest_path = pkg.power_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for device in manifest['expected_devices']:
        for assignment in device['assignments']:
            assignment.update(worker_role='agg', het_group=None)
    manifest_path.write_text(json.dumps(manifest))
    stem = f'run_recipe_c4_gpus_4{role_suffix}'
    (tmp_path / f'{stem}.json').write_text(pkg.original_result.read_text())
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run',
           'CONC_LIST': '4', 'DECODE_NUM_WORKERS': '0', 'REQUIRE_POWER': '1',
           'POWER_PRODUCER_SHA': PRODUCER_SHA}
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path,
                            env={**env, 'PYTHONPATH': str(REPO_ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / f'agg_{stem}.json').read_text())
    assert aggregate['disagg'] is False
    assert aggregate['num_aggregate_gpu'] == 4
    assert aggregate['num_decode_gpu'] == 0
    assert aggregate['decode_num_workers'] == 0
    assert aggregate['power_valid'] == 1
    assert aggregate['avg_power_w'] == 350
    assert set(ROLE_METRIC_KEYS).isdisjoint(aggregate)


@pytest.mark.parametrize('decode_workers,role_suffix', [
    ('0', '_ctx_2_gen_2'), ('0', '_ctx_3_gen_0'), ('1', ''),
])
def test_multinode_batch_rejects_misdeclared_aggregate_role_counts(
    tmp_path, multinode_env_vars, sample_benchmark_result, decode_workers, role_suffix,
):
    stem = f'run_recipe_c4_gpus_4{role_suffix}'
    (tmp_path / f'{stem}.json').write_text(
        json.dumps({**sample_benchmark_result, 'max_concurrency': 4}))
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run',
           'CONC_LIST': '4', 'DECODE_NUM_WORKERS': decode_workers}
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path,
                            env={**env, 'PYTHONPATH': str(REPO_ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['points'][0]['exit_code'] == 1
    assert not (tmp_path / f'agg_{stem}.json').exists()


def test_zero_successful_requests_preserves_diagnostic_result(tmp_path, single_node_env_vars, sample_benchmark_result):
    raw = {**sample_benchmark_result, 'tpot_p50_ms': 0, 'tpot_p99_ms': float('nan'),
           'completed': 0, 'num_prompts': 4,
           'benchmark_outcome': {'status': 'failed', 'requested': 4, 'completed': 0,
                                 'failed': 4, 'max_failure_rate': 0.05}}
    result = run_script(tmp_path, single_node_env_vars, raw)
    assert result.returncode == 1, result.stderr
    aggregate = json.loads((tmp_path / 'agg_benchmark_result.json').read_text())
    assert aggregate['benchmark_outcome']['failed'] == 4
    assert 'intvty_p50' not in aggregate
    assert 'tpot_p99' not in aggregate


@pytest.mark.parametrize('extra_conc,error', [(4, 'Duplicate concurrency'), (8, None)])
def test_multinode_batch_rejects_extra_results_without_losing_other_points(
    tmp_path, multinode_env_vars, sample_benchmark_result, extra_conc, error,
):
    for label, conc in [('a', 4), ('b', extra_conc), ('c', 16)]:
        (tmp_path / f'run_{label}_conc{conc}_gpus_4_ctx_2_gen_2.json').write_text(
            json.dumps({**sample_benchmark_result, 'max_concurrency': conc}))
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run', 'CONC_LIST': '4 16'}
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path,
                            env={**env, 'PYTHONPATH': str(REPO_ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['missing_concurrencies'] == []
    assert receipt['unexpected_concurrencies'] == ([] if error else [8])
    if error:
        assert error in receipt['points'][1]['error']
    assert (tmp_path / 'agg_run_a_conc4_gpus_4_ctx_2_gen_2.json').is_file()
    assert (tmp_path / 'agg_run_c_conc16_gpus_4_ctx_2_gen_2.json').is_file()


def test_multinode_empty_sweep_records_every_missing_point(tmp_path, multinode_env_vars):
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run', 'CONC_LIST': '4 8'}
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path,
                            env={**env, 'PYTHONPATH': str(REPO_ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['missing_concurrencies'] == [4, 8]
    assert receipt['points'] == []


def test_public_power_audit_bounds_text_and_device_identifiers():
    from infx.results.power.audit import audit_summary

    summary = audit_summary({
        'producer': {'producer_git_commit': 'x' * 129, 'exporter_image_sha256': 'a' * 64},
        'observed_gpu_ids': ['gpu0', 'gpu0', 'x' * 129] + [f'gpu{i}' for i in range(1, 1025)],
        'reasons': ['window_missing', '<raw log>'] + [f'reason_{i}' for i in range(40)],
    }, 'power_validation_run.json')
    assert len(summary['power_invalid_reasons']) == 32
    assert '<raw log>' not in summary['power_invalid_reasons']
    audit = summary['power_audit']
    assert 'producer_sha' not in audit
    assert audit['exporter_image_sha256'] == 'a' * 64
    assert len(audit['observed_gpu_ids']) == 1024
    assert audit['observed_gpu_ids'][:2] == ['gpu0', 'gpu1']


@pytest.mark.parametrize('sidecar', ['run_recipe_conc4_gpus_4_ctx_2_gen_2.pytorch.json'])
@pytest.mark.parametrize('point_state', ['valid', 'missing', 'malformed'])
def test_multinode_batch_retains_sidecars_without_counting_them_as_points(
    tmp_path, multinode_env_vars, sample_benchmark_result, sidecar, point_state,
):
    (tmp_path / sidecar).write_text('{"diagnostic": true}')
    if point_state != 'missing':
        (tmp_path / 'run_recipe_conc4_gpus_4_ctx_2_gen_2.json').write_text(
            json.dumps({**sample_benchmark_result, 'max_concurrency': 4})
            if point_state == 'valid' else '{broken')
    env = {**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run', 'CONC_LIST': '4',
           'PYTHONPATH': str(REPO_ROOT)}
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == int(point_state != 'valid'), result.stderr
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['ignored_sidecars'] == [sidecar]
    assert receipt['missing_concurrencies'] == ([] if point_state == 'valid' else [4])
    assert (tmp_path / sidecar).read_text() == '{"diagnostic": true}'


def test_multinode_batch_rejects_unknown_point_filename(
    tmp_path, multinode_env_vars, sample_benchmark_result,
):
    for name in ['run_conc4_gpus_4_ctx_2_gen_2.json', 'run_conc16_gpus_bad.json']:
        (tmp_path / name).write_text(json.dumps({**sample_benchmark_result, 'max_concurrency': 4}))
    result = subprocess.run([*MODULE_COMMAND, '--all'], cwd=tmp_path,
                            env={**os.environ, **multinode_env_vars, 'RESULT_FILENAME': 'run',
                                 'CONC_LIST': '4 16', 'PYTHONPATH': str(REPO_ROOT)},
                            capture_output=True, text=True)
    assert result.returncode == 1
    receipt = json.loads((tmp_path / 'result_processing_run.json').read_text())
    assert receipt['missing_concurrencies'] == [16]
    assert any('filename lacks' in point.get('error', '') for point in receipt['points'])


@pytest.mark.parametrize("collector", ["shared", "h200-dcgm"])
@pytest.mark.parametrize("result_python", [None, "", sys.executable])
def test_agentic_collector_preserves_archive_when_result_python_is_missing(
    tmp_path: Path, monkeypatch, capfd, result_python: str | None, collector: str
) -> None:
    import tarfile

    from infx.launch.artifacts import (
        bundle_server_logs,
        collect_agentic_power_results,
        validate_agentic_power,
    )
    from infx.launch.backends.base import JobState, JobStatus

    pkg = build_package(tmp_path)
    result_dir = pkg.logs_root / "agentic/conc_4"
    result_dir.mkdir(parents=True)
    stem = "agentic_power_concurrency_4"
    pkg.original_result.replace(result_dir / f"{stem}.json")
    old_window = pkg.windows_dir / "my_result.json"
    window = json.loads(old_window.read_text())
    window.update(benchmark_type="custom", result_path=f"agentic/conc_4/{stem}.json")
    old_window.unlink()
    (pkg.windows_dir / f"{stem}.json").write_text(json.dumps(window))
    manifest_path = pkg.power_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["expected_windows"] = [{"benchmark_type": "custom", "concurrency": 4}]
    manifest["window_validations"][0].update(
        benchmark_type="custom", window_file=f"windows/{stem}.json"
    )
    manifest_path.write_text(json.dumps(manifest))
    source, workspace, bin_dir = [tmp_path / name for name in ("source", "workspace", "bin")]
    for directory in (source, workspace, bin_dir):
        directory.mkdir()
    raw_result = {
        "hw": "h200",
        "conc": 4,
        "disagg": True,
        "num_prefill_gpu": 2,
        "num_decode_gpu": 2,
    }
    (source / "point_conc4.json").write_text(json.dumps(raw_result))
    for name in ("python", "python3"):
        ambient_python = bin_dir / name
        ambient_python.write_text("#!/bin/sh\nexit 73\n")
        ambient_python.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    if collector == "shared":
        completed = JobStatus(JobState.SUCCEEDED, "COMPLETED|0:0", 0)
        rc = collect_agentic_power_results(
            completed, "12345", pkg.logs_root, source, workspace, "point", PRODUCER_SHA, [4],
            results_python=result_python,
        )  # fmt: skip
        archive_name = "server-logs.tar.gz"
    else:
        (workspace / "point_conc4.json").write_text(json.dumps(raw_result))
        rc = validate_agentic_power(
            pkg.logs_root, workspace, "point", PRODUCER_SHA, [4],
            results_python=result_python, require_power=True,
        )  # fmt: skip
        archive_name = "multinode_server_logs.tar.gz"
    bundle_server_logs(pkg.logs_root, workspace / archive_name)
    assert rc == (0 if result_python else 1)
    with tarfile.open(workspace / archive_name) as archive:
        if collector == "shared":
            assert "./power/native-job-status.txt" in archive.getnames()
        assert f"./agentic/conc_4/{stem}.json" in archive.getnames()
    aggregate = json.loads((workspace / "point_conc4.json").read_text())
    if result_python:
        assert aggregate["power_valid"] == 1
        assert aggregate["total_gpu_energy_j"] == pytest.approx(84_000)
    else:
        assert "INFERENCEX_RESULTS_PYTHON" in capfd.readouterr().err
        assert aggregate == raw_result
