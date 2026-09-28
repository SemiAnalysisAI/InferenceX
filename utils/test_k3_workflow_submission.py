"""Replay FP32 workflow exports through the real launcher/recipe/submit chain.

Only scheduler, wait and privileged cleanup commands are mocked. No GPU job,
container, credential, or shared path is used by this test.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KEY = "kimik3-fp4-mi355x-vllm-disagg-agentic"


@pytest.fixture(scope="module")
def matrix():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "utils/matrix_logic/generate_sweep_configs.py"),
            "test-config",
            "--config-files",
            "configs/amd-master.yaml",
            "--runner-config",
            "configs/runners.yaml",
            "--config-keys",
            KEY,
            "--conc",
            "24",
            "10",
            "--no-evals",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    rows = json.loads(result.stdout)
    assert len(rows) == 2
    assert {tuple(row["conc"]) for row in rows} == {(10,), (24,)}
    return rows


def matrix_value(expression, row):
    """Resolve the small, explicit expression subset used by these inputs."""
    if not isinstance(expression, str) or not expression.startswith("${{"):
        return expression
    expr = expression[3:-2].strip()
    expr = re.sub(r"\['([\w-]+)'\]", r".\1", expr)
    if expr.startswith("toJson(") and expr.endswith(")"):
        return json.dumps(matrix_value("${{ " + expr[7:-1] + " }}", row))
    assert re.fullmatch(r"matrix\.config(?:\.[\w-]+)+", expr), expr
    value = row
    for key in expr.split(".")[2:]:
        value = value[key]
    return value


def workflow_environment(row):
    workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-tests.yml").read_text())
    template = yaml.safe_load(
        (ROOT / ".github/workflows/benchmark-multinode-tmpl.yml").read_text()
    )
    call = workflow["jobs"]["test-sweep-multi-node-agentic"]["with"]
    # PyYAML's YAML1.1 loader treats the workflow's `on` key as True.
    defaults = template[True]["workflow_call"]["inputs"]
    keys = [
        "EXP_NAME",
        "IMAGE",
        "MODEL_PREFIX",
        "MODEL",
        "FRAMEWORK",
        "PRECISION",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "SPEC_DECODING",
        "KV_OFFLOADING",
        "KV_OFFLOAD_BACKEND",
        "PREFILL_NUM_WORKERS",
        "PREFILL_TP",
        "PREFILL_PP_SIZE",
        "PREFILL_DCP_SIZE",
        "PREFILL_PCP_SIZE",
        "PREFILL_EP",
        "PREFILL_DP_ATTN",
        "DECODE_NUM_WORKERS",
        "DECODE_TP",
        "DECODE_PP_SIZE",
        "DECODE_DCP_SIZE",
        "DECODE_PCP_SIZE",
        "DECODE_EP",
        "DECODE_DP_ATTN",
        "TOTAL_CPU_DRAM_GB",
    ]
    env = {}
    for name in keys:
        match = re.fullmatch(r"\$\{\{ inputs\.([\w-]+) \}\}", template["env"][name])
        assert match, (name, template["env"][name])
        key = match[1]
        value = (
            matrix_value(call[key], row)
            if key in call
            else defaults[key].get("default", "")
        )
        env[name] = str(value).lower() if isinstance(value, bool) else str(value)
    assert (
        template["env"]["CONC_LIST"] == "${{ join(fromJson(inputs.conc-list), ' ') }}"
    )
    env["CONC_LIST"] = " ".join(
        map(str, json.loads(matrix_value(call["conc-list"], row)))
    )
    assert call["scenario-type"] == "agentic-coding"
    assert (
        template["env"]["IS_AGENTIC"]
        == "${{ inputs.scenario-type == 'agentic-coding' && '1' || '0' }}"
    )
    assert (
        template["env"]["SCENARIO_SUBDIR"]
        == "${{ inputs.scenario-type == 'agentic-coding' && 'agentic/' || 'fixed_seq_len/' }}"
    )
    assert (
        call["duration"]
        == "${{ inputs.agentx-fast && '1200' || (inputs.duration-override != '' && inputs.duration-override || matrix.config.duration) }}"
    )
    env.update(
        IS_AGENTIC="1",
        SCENARIO_SUBDIR="agentic/",
        DURATION="3600",
        AIPERF_EXPERIMENTAL_FAST="0",
        RANDOM_RANGE_RATIO=str(template["env"]["RANDOM_RANGE_RATIO"]),
    )
    step = next(
        s
        for s in template["jobs"]["benchmark"]["steps"]
        if s.get("name") == "Launch multi-node job script"
    )
    end = "bash ./runners/launch_${RUNNER_NAME%%_*}.sh"
    script = step["run"].split(end, 1)[0] + end + "\n"
    for role in ("prefill", "decode"):
        expression = (
            "${{ join(fromJson(inputs." + role + "-additional-settings), ' ') }}"
        )
        assert expression in script
        settings = json.loads(matrix_value(call[role + "-additional-settings"], row))
        script = script.replace(expression, " ".join(settings))
    assert "${{" not in script
    return env, script


@pytest.mark.parametrize("conc", [10, 24])
@pytest.mark.parametrize(
    "omitted", [None, "PREFILL_NODES", "DECODE_NODES", "TOTAL_CPU_DRAM_GB"]
)
def test_real_submission_contract(matrix, tmp_path, conc, omitted):
    row = next(row for row in matrix if row["conc"] == [conc])
    env, script = workflow_environment(row)
    # Negative controls recreate the actual regression without mutating sources.
    if omitted:
        script = re.sub(rf"\b{omitted}=\S+", "", script)
    for rel in (
        "runners/launch_mi355x-amds.sh",
        "benchmarks/benchmark_lib.sh",
        "benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh",
        "benchmarks/multi_node/amd_utils/submit.sh",
        "benchmarks/multi_node/amd_utils/node_excludes.yaml",
    ):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    capture = tmp_path / "submission.json"
    mocks = tmp_path / "scheduler-mocks.sh"
    mocks.write_text("""
sudo() { :; }
sleep() { :; }
tail() { :; }
squeue() { :; }
scontrol() { :; }
scancel() { :; }
sacct() { printf '9321|COMPLETED|0:0\\n'; }
sbatch() {
    python3 - "$@" <<'PY'
import json, os, sys
from pathlib import Path
keys = "PREFILL_NODES DECODE_NODES NUM_NODES PREFILL_TP_SIZE DECODE_TP_SIZE xP yD TOTAL_CPU_DRAM_GB SLURM_PARTITION MODEL_PATH IBDEVICES SPEC_MODEL MAMBA_SSM_CACHE_DTYPE MORI_IB_MAX_RD_ATOMIC MORI_QP_PER_TRANSFER GPU_MEMORY_UTILIZATION DURATION AIPERF_EXPERIMENTAL_FAST AIPERF_FAILED_REQUEST_THRESHOLD AIPERF_LIVE_FAILED_REQUEST_THRESHOLD EXPECTED_KV_CACHE_GROUP_SIZES PREFILL_DCP_SIZE DECODE_DCP_SIZE".split()
Path(os.environ["TEST_SUBMISSION"]).write_text(json.dumps({"args": sys.argv[1:], "env": {key: os.environ.get(key) for key in keys}}))
(Path(os.environ["BENCHMARK_LOGS_DIR"]) / "slurm_job-9321.out").touch()
PY
    printf '9321\\n'
}
""")
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", script],
        cwd=tmp_path,
        env={
            **env,
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "USER": "k3-contract-test",
            "GITHUB_WORKSPACE": str(tmp_path),
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "RUNNER_NAME": "mi355x-amds_00",
            "RESULT_FILENAME_BASE": "contract",
            "BASH_ENV": str(mocks),
            "KEEP_LOGS": "1",
            "TEST_SUBMISSION": str(capture),
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if omitted in ("PREFILL_NODES", "DECODE_NODES"):
        assert result.returncode != 0
        assert not capture.exists(), result.stdout
        assert "required environment variables are not set" in result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    submitted = json.loads(capture.read_text())
    actual = submitted["env"]
    expected = {
        "PREFILL_NODES": "1",
        "DECODE_NODES": "2" if conc == 24 else "1",
        "NUM_NODES": str(row["node-count"]),
        "PREFILL_TP_SIZE": "8",
        "DECODE_TP_SIZE": "8",
        "xP": "1",
        "yD": "2" if conc == 24 else "1",
        "TOTAL_CPU_DRAM_GB": "600" if omitted else str(row["total-cpu-dram-gb"]),
        "SLURM_PARTITION": "compute",
        "MODEL_PATH": "/it-share/data",
        "IBDEVICES": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        "SPEC_MODEL": "/models/Inferact-Kimi-K3-DSpark",
        "MAMBA_SSM_CACHE_DTYPE": "float32",
        "MORI_IB_MAX_RD_ATOMIC": "1",
        "MORI_QP_PER_TRANSFER": "8",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "DURATION": "3600",
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "AIPERF_FAILED_REQUEST_THRESHOLD": "0.01",
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": "0.01",
        "EXPECTED_KV_CACHE_GROUP_SIZES": "1536,1536,1536,"
        + ("12288" if conc == 24 else "1536"),
        "PREFILL_DCP_SIZE": "8" if conc == 24 else "1",
        "DECODE_DCP_SIZE": "8" if conc == 24 else "1",
    }
    assert actual == expected
    assert row["total-cpu-dram-gb"] == 1799
    args = submitted["args"]
    assert args[args.index("-N") + 1] == str(row["node-count"])
    assert args[args.index("-n") + 1] == str(row["node-count"])
    assert "--parsable" in args and "--exclusive" in args
    assert "[slurm-result] job=9321 state=COMPLETED exit=0:0" in result.stdout
