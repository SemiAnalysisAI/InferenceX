"""Execute K3 config building and final role exports without GPU/network setup."""

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
MODELS = ROOT / "benchmarks/multi_node/amd_utils/models_vllm.yaml"


@pytest.mark.parametrize(
    "prefill_mode,expected",
    [
        ("PIECEWISE", "1"),
        ("FULL_AND_PIECEWISE", "1"),
        ("NONE", "0"),
    ],
)
@pytest.mark.parametrize("role", ["PREFILL", "DECODE"])
def test_final_k3_role_environment_agrees_with_graph_mode(prefill_mode, expected, role):
    source = SERVER.read_text()
    marker = (
        'if [[ "${MODEL_NAME:-}" == "Kimi-K3" && "${SPEC_DECODING:-}" == "mtp" ]]; then'
    )
    config_block = (
        marker
        + source.split(marker, 1)[1].split(
            'echo "PREFILL_SERVER_CONFIG (after TP/EP/DP)', 1
        )[0]
    )
    setup = re.search(
        r"^setup_vllm_env\(\) \{.*?^\}", source, re.MULTILINE | re.DOTALL
    ).group()
    exports = re.findall(
        rf"^    for env_pair in \$\{{{role}_MODEL_ENVS\}}; do.*?^    done",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert exports
    model = yaml.safe_load(MODELS.read_text())["Kimi-K3"]
    initial = {
        "PREFILL_SERVER_CONFIG": model["prefill_flags"],
        "DECODE_SERVER_CONFIG": model["decode_flags"],
        "MODEL_ENVS": model["env"],
        "PREFILL_MODEL_ENVS": model["prefill_env"],
        "DECODE_MODEL_ENVS": model["decode_env"],
    }
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("SPEC_")
    }
    env.update(
        {
            "MODEL_NAME": "Kimi-K3",
            "SPEC_DECODING": "mtp",
            "SPEC_PREFILL_CUDAGRAPH_MODE": prefill_mode,
            "SPEC_DECODE_CUDAGRAPH_MODE": "FULL_DECODE_ONLY",
            "SPEC_PREFILL_MAX_CUDAGRAPH_CAPTURE_SIZE": "128",
            "SPEC_DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE": "80",
            # The workflow merges P/D additional-settings; the D value wins.
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            "rdma_ip": "127.0.0.1",
        }
    )
    for export in exports:
        script = "\n".join(
            f"{key}={shlex.quote(value)}" for key, value in initial.items()
        )
        script += "\n" + config_block + "\n" + setup + "\nsetup_vllm_env\n" + export
        script += '\nprintf "FINAL_BREAKABLE=%s\\n" "$VLLM_USE_BREAKABLE_CUDAGRAPH"\n'
        script += f'printf "FINAL_CONFIG=%s\\n" "${{{role}_SERVER_CONFIG}}"\n'
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        values = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("FINAL_")
        )
        tokens = shlex.split(values["FINAL_CONFIG"])
        graph = json.loads(tokens[tokens.index("--compilation-config") + 1])
        assert graph["cudagraph_mode"] == (
            prefill_mode if role == "PREFILL" else "FULL_DECODE_ONLY"
        )
        assert values["FINAL_BREAKABLE"] == (expected if role == "PREFILL" else "0")
