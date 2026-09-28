"""Exercise generated K3 role flags and connector JSON, without a GPU."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "benchmarks/multi_node/amd_utils/server_vllm.sh"
CONFIG = yaml.safe_load((ROOT / "configs/amd-master.yaml").read_text())
RECIPE = CONFIG["kimik3-fp4-mi355x-vllm-disagg-agentic"]
ARMS = RECIPE["scenarios"]["agentic-coding"][0]["search-space"]


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: str(a["conc-list"][0]))
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_effective_role_flags_and_transport(arm, role):
    source = SERVER.read_text(encoding="utf-8-sig")
    start = source.index("apply_vllm_dp_config() {")
    end = source.index('echo "PREFILL_SERVER_CONFIG (after TP/EP/DP)')
    config_code = source[start:end]
    connector = source[source.index("build_simple_kv_transfer_config_json() {") :]
    connector = connector[: connector.index("\nbuild_kv_transfer_config_json() {")]
    models = yaml.safe_load(
        (ROOT / "benchmarks/multi_node/amd_utils/models_vllm.yaml").read_text()
    )["Kimi-K3"]
    settings = {
        key: value
        for worker in (arm["prefill"], arm["decode"])
        for key, value in (item.split("=", 1) for item in worker["additional-settings"])
    }
    assert not any(key.startswith(("VLLM_K3_FORK", "LMCACHE_")) for key in settings)
    env = {
        **os.environ,
        **settings,
        "MODEL_NAME": "Kimi-K3",
        "SPEC_DECODING": "mtp",
        "SPEC_ATTN_BACKEND": "ROCM_AITER_MLA",
        "SPEC_KV_CACHE_DTYPE": "fp8",
        "SPEC_NUM_TOKENS": "4",
        "SPEC_MODEL": "/models/Inferact-Kimi-K3-DSpark",
        "SPEC_REJECTION_SAMPLE_METHOD": "synthetic",
        "SPEC_SYNTHETIC_ACCEPTANCE_LENGTH": "3.36",
        "PREFILL_TP_SIZE": "8",
        "DECODE_TP_SIZE": "8",
        "PREFILL_DCP_SIZE": str(arm["prefill"]["dcp-size"]),
        "DECODE_DCP_SIZE": str(arm["decode"]["dcp-size"]),
        "NODE0_ADDR": "10.0.0.1",
        "MORIIO_HOST_IP": "10.0.0.2",
        "TOTAL_CPU_DRAM_GB": "1799",
        "KV_OFFLOADING": "dram",
    }
    initial = {
        "PREFILL_SERVER_CONFIG": models["prefill_flags"],
        "DECODE_SERVER_CONFIG": models["decode_flags"],
        "PREFILL_MODEL_ENVS": models["prefill_env"],
        "DECODE_MODEL_ENVS": models["decode_env"],
    }
    script = "\n".join(f"{key}={shlex.quote(val)}" for key, val in initial.items())
    script += "\n" + config_code + "\n" + connector
    script += f'\nprintf "FLAGS=%s\\n" "${{{role.upper()}_SERVER_CONFIG}}"\n'
    script += f'printf "ENV=%s\\n" "${{{role.upper()}_MODEL_ENVS}}"\n'
    script += f"build_simple_kv_transfer_config_json kv_{'producer' if role == 'prefill' else 'consumer'}\n"
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    flags = shlex.split(
        next(s[6:] for s in result.stdout.splitlines() if s.startswith("FLAGS="))
    )
    value = lambda flag: flags[flags.index(flag) + 1]
    assert value("--gpu-memory-utilization") == "0.90"
    assert value("--mamba-ssm-cache-dtype") == settings["MAMBA_SSM_CACHE_DTYPE"]
    assert value("--max-num-seqs") == settings[f"SPEC_{role.upper()}_MAX_NUM_SEQS"]
    assert value("--max-num-batched-tokens") == ("8192" if role == "prefill" else "512")
    graph = json.loads(value("--compilation-config"))
    assert graph["cudagraph_mode"] == settings[f"SPEC_{role.upper()}_CUDAGRAPH_MODE"]
    assert graph["max_cudagraph_capture_size"] == int(
        settings[f"SPEC_{role.upper()}_MAX_CUDAGRAPH_CAPTURE_SIZE"]
    )
    role_env = next(s[4:] for s in result.stdout.splitlines() if s.startswith("ENV="))
    final_env = dict(item.split("=", 1) for item in shlex.split(role_env))
    assert final_env["VLLM_USE_BREAKABLE_CUDAGRAPH"] == str(
        int(graph["cudagraph_mode"] == "PIECEWISE")
    )
    assert final_env["TORCH_NCCL_BLOCKING_WAIT"] == "0"
    kv = json.loads(result.stdout.splitlines()[-1])
    assert kv["kv_load_failure_policy"] == "fail"
    if role == "prefill":
        kv, cpu = kv["kv_connector_extra_config"]["connectors"]
        assert cpu["kv_connector_extra_config"] == {
            "cpu_bytes_to_use": 1799000000000,
            "lazy_offload": False,
        }
    assert kv["kv_connector_extra_config"]["qp_per_transfer"] == 8
    assert kv["kv_connector_extra_config"]["host_ip"] == "10.0.0.2"


def test_targeted_points_use_original_runner_and_pinned_image():
    assert RECIPE["runner"] == "cluster:mi355x-amds"
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", RECIPE["image"])
    identities = {
        (a["prefill"]["num-worker"], a["decode"]["num-worker"], tuple(a["conc-list"]))
        for a in ARMS
    }
    assert identities == {(1, 1, (48,)), (1, 2, (24,)), (1, 1, (10,))}


@pytest.mark.parametrize("qps", ["0", "-1", "bad", "1.5"])
def test_invalid_qp_refuses_before_launch(qps):
    code = SERVER.read_text().split("build_simple_kv_transfer_config_json() {", 1)[1]
    code = (
        "build_simple_kv_transfer_config_json() {"
        + code.split("\nbuild_kv_transfer_config_json() {", 1)[0]
    )
    result = subprocess.run(
        ["bash", "-c", code + "\nbuild_simple_kv_transfer_config_json kv_consumer"],
        env={**os.environ, "MORI_QP_PER_TRANSFER": qps},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
