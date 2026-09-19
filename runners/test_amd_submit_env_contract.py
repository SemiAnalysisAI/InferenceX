"""Guard the env contract between the generated matrix and AMD submit.sh.

Only agentic search-space entries declare kv-offloading, so a fixed-sequence
multinode row reaches the workflow without one and exports KV_OFFLOADING
empty. submit.sh still has to submit that job while rejecting genuinely
missing configuration.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / 'benchmarks/multi_node/amd_utils/submit.sh'

# Everything submit.sh requires apart from KV_OFFLOADING, which the fixed-
# sequence rows omit. Values are placeholders; no argument reaches sbatch.
BASE_ENV = {
    'PATH': '/usr/bin:/bin',
    'SLURM_ACCOUNT': 'acct', 'SLURM_PARTITION': 'part', 'TIME_LIMIT': '01:00:00',
    'MODEL_PATH': '/models', 'MODEL_NAME': 'model', 'CONTAINER_IMAGE': 'image',
    'RUNNER_NAME': 'runner', 'FRAMEWORK': 'sglang-disagg', 'GPUS_PER_NODE': '8',
    'PREFILL_EP': '1', 'PREFILL_DP_ATTN': 'false', 'PREFILL_NUM_WORKERS': '1',
    'PREFILL_PP_SIZE': '1', 'PREFILL_DCP_SIZE': '1', 'PREFILL_PCP_SIZE': '1',
    'DECODE_EP': '1', 'DECODE_DP_ATTN': 'false', 'DECODE_NUM_WORKERS': '2',
    'DECODE_PP_SIZE': '1', 'DECODE_DCP_SIZE': '1', 'DECODE_PCP_SIZE': '1',
    'DECODE_MTP_SIZE': '2', 'BENCH_NUM_PROMPTS_MULTIPLIER': '1', 'DRY_RUN': '0',
    'RUN_EVAL': 'false', 'EVAL_ONLY': 'false', 'EVAL_FRAMEWORK': 'none',
    'IS_MULTINODE': 'true', 'SWEBENCH_USE_MODAL': 'false',
    'BENCHMARK_LOGS_DIR': '/tmp/logs', 'KEEP_CONTAINERS': 'false',
    'ROUTER_TYPE': 'sglang-router', 'ROUTER_PORT': '8000',
    'PROXY_PING_PORT': '8001', 'HEADNODE_PORT': '8002', 'SERVER_PORT': '8003',
    'IS_AGENTIC': 'false',
}


def _run(env: dict[str, str]) -> subprocess.CompletedProcess:
    # No positional arguments, so submit.sh stops at its usage check well
    # before sbatch. Only the env validation above it is under test.
    return subprocess.run(['bash', str(SUBMIT)], env=env, capture_output=True, text=True)


def test_absent_kv_offloading_does_not_block_submission():
    result = _run(dict(BASE_ENV))
    output = result.stdout + result.stderr

    assert 'KV_OFFLOADING' not in output
    assert 'Usage:' in output


def test_a_genuinely_missing_variable_still_fails():
    env = dict(BASE_ENV)
    del env['MODEL_NAME']

    result = _run(env)
    output = result.stdout + result.stderr

    assert 'MODEL_NAME' in output
    assert 'Usage:' not in output
