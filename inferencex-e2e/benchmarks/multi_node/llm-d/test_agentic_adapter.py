import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def replay_env() -> dict[str, str]:
    """Explicit caller inputs for the shared replay command builder."""
    return {
        "AIPERF_FAILED_REQUEST_THRESHOLD": "0.10",
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": "0.10",
        "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300",
        "AGENTIC_WARMUP_GRACE_PERIOD": "1800",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AIPERF_UNSAFE_OVERRIDE": "false",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
        "AIPERF_WARMUP_REQUESTS_PER_LANE": "10",
    }


@pytest.mark.parametrize("metrics_body", ["vllm:num_requests_running 0\n", "envoy_http_requests_total 0\n"])
def test_llmd_agentic_adapter_uses_discovered_worker_metrics(
    tmp_path: Path, metrics_body: str, replay_env: dict[str, str]
) -> None:
    """Check endpoint selection, preflight failure, and the real AIPerf CLI builder."""
    client = tmp_path / "benchmarks/srt_agentic.sh"
    client.parent.mkdir(parents=True)
    client.write_text('''source "$REAL_BENCHMARK_LIB"
build_replay_cmd "$RESULT_DIR"
export REPLAY_CMD
python3 - <<'PY'
import json, os
keys = ["AIPERF_METRIC_URLS", "AIPERF_SERVER_METRICS_URLS", "REPLAY_CMD"]
print(json.dumps({key: os.environ[key] for key in keys}))
PY
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        '#!/usr/bin/env python3\n'
        'import os, sys\nfrom pathlib import Path\n'
        'args = sys.argv[1:]\n'
        'url = next((a for a in args if a.startswith("http://")), "")\n'
        'if "--write-out" in args:\n'
        '    print("404", end="")\n'
        'else:\n'
        '    out_path = args[args.index("--output") + 1]\n'
        '    if out_path != "/dev/null":\n'
        '        Path(out_path).write_text(os.environ["METRICS_BODY"])\n'
        '    with open(os.environ["METRICS_REQUESTS"], "a") as f:\n'
        '        f.write(url + "\\n")\n'
    )
    curl.chmod(0o755)
    endpoints = tmp_path / "endpoints.yaml"
    endpoints.write_text(yaml.safe_dump({"endpoints": [
        {"address": "10.0.0.1", "port": "8200", "name": "vllm-node-0",
         "labels": {"llm-d.ai/role": "combined"}},
        {"address": "10.0.0.2", "port": "8201", "name": "vllm-node-1",
         "labels": {"llm-d.ai/role": "combined"}},
        {"address": "10.0.0.3", "port": "8202", "name": "vllm-node-2",
         "labels": {"llm-d.ai/role": "combined"}},
    ]}))
    requests = tmp_path / "metrics-requests.txt"
    env = dict(os.environ, **replay_env, INFMAX_CONTAINER_WORKSPACE=str(tmp_path),
               REAL_BENCHMARK_LIB=str(REPO_ROOT / "benchmarks/benchmark_lib.sh"),
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               METRICS_BODY=metrics_body, METRICS_REQUESTS=str(requests),
               LLMD_ENDPOINTS_FILE=str(endpoints), MODEL_NAME="test-model", MODEL_PREFIX="dsv4",
               FRAMEWORK="llmd-vllm", DURATION="3600", IS_AGENTIC="1", KV_OFFLOADING="none",
               ENVOY_PORT="8080", VLLM_PORT="8200", SIDECAR_PORT="8000",
               BENCHMARK_LOGS_DIR=str(tmp_path / "logs"),
               BENCH_MAX_CONCURRENCY="64", DECODE_NODES="0")
    result = subprocess.run(["bash", str(REPO_ROOT / "benchmarks/multi_node/llm-d/agentic.sh")],
                            env=env, text=True, capture_output=True)
    if metrics_body.startswith("envoy_"):
        assert result.returncode != 0
        assert "no vLLM metrics exposed" in result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = json.loads(result.stdout.splitlines()[-1])
    expected_urls = [
        "http://10.0.0.1:8200/metrics",
        "http://10.0.0.2:8201/metrics",
        "http://10.0.0.3:8202/metrics",
    ]
    assert requests.read_text().splitlines() == expected_urls
    assert recorded["AIPERF_METRIC_URLS"].split(",") == expected_urls
    assert recorded["AIPERF_SERVER_METRICS_URLS"].split(",") == expected_urls
    assert "--url http://localhost:8080 " in recorded["REPLAY_CMD"]
    assert "--server-metrics " + " ".join(expected_urls) + " " in recorded["REPLAY_CMD"]


def test_llmd_agentic_adapter_maps_decode_sidecar_ports_to_vllm_metrics(
    tmp_path: Path, replay_env: dict[str, str]
) -> None:
    """Disagg decode endpoints list sidecar ports; metrics scrape vLLM DP ranks."""
    client = tmp_path / "benchmarks/srt_agentic.sh"
    client.parent.mkdir(parents=True)
    client.write_text('''source "$REAL_BENCHMARK_LIB"
build_replay_cmd "$RESULT_DIR"
export REPLAY_CMD
python3 - <<'PY'
import json, os
keys = ["AIPERF_METRIC_URLS", "AIPERF_SERVER_METRICS_URLS", "REPLAY_CMD"]
print(json.dumps({key: os.environ[key] for key in keys}))
PY
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(
        '#!/usr/bin/env python3\n'
        'import os, sys\nfrom pathlib import Path\n'
        'args = sys.argv[1:]\n'
        'url = next((a for a in args if a.startswith("http://")), "")\n'
        'if "--write-out" in args:\n'
        '    print("404", end="")\n'
        'else:\n'
        '    out_path = args[args.index("--output") + 1]\n'
        '    if out_path != "/dev/null":\n'
        '        Path(out_path).write_text(os.environ["METRICS_BODY"])\n'
        '    with open(os.environ["METRICS_REQUESTS"], "a") as f:\n'
        '        f.write(url + "\\n")\n'
    )
    curl.chmod(0o755)
    endpoints = tmp_path / "endpoints.yaml"
    endpoints.write_text(yaml.safe_dump({"endpoints": [
        {"address": "10.0.0.10", "port": "8200", "name": "prefill-0",
         "labels": {"llm-d.ai/role": "prefill"}},
        {"address": "10.0.0.10", "port": "8201", "name": "prefill-1",
         "labels": {"llm-d.ai/role": "prefill"}},
        {"address": "10.0.0.20", "port": "8000", "name": "decode-0",
         "labels": {"llm-d.ai/role": "decode"}},
        {"address": "10.0.0.20", "port": "8001", "name": "decode-1",
         "labels": {"llm-d.ai/role": "decode"}},
    ]}))
    requests = tmp_path / "metrics-requests.txt"
    env = dict(os.environ, **replay_env, INFMAX_CONTAINER_WORKSPACE=str(tmp_path),
               REAL_BENCHMARK_LIB=str(REPO_ROOT / "benchmarks/benchmark_lib.sh"),
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               METRICS_BODY="vllm:num_requests_running 0\n", METRICS_REQUESTS=str(requests),
               LLMD_ENDPOINTS_FILE=str(endpoints), MODEL_NAME="test-model", MODEL_PREFIX="dsv4",
               FRAMEWORK="llmd-vllm", DURATION="3600", IS_AGENTIC="1", KV_OFFLOADING="none",
               ENVOY_PORT="8080", VLLM_PORT="8200", SIDECAR_PORT="8000",
               BENCHMARK_LOGS_DIR=str(tmp_path / "logs"),
               BENCH_MAX_CONCURRENCY="64", DECODE_NODES="2")
    result = subprocess.run(["bash", str(REPO_ROOT / "benchmarks/multi_node/llm-d/agentic.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = json.loads(result.stdout.splitlines()[-1])
    expected_urls = [
        "http://10.0.0.10:8200/metrics",
        "http://10.0.0.10:8201/metrics",
        "http://10.0.0.20:8200/metrics",
        "http://10.0.0.20:8201/metrics",
    ]
    assert requests.read_text().splitlines() == expected_urls
    assert recorded["AIPERF_METRIC_URLS"].split(",") == expected_urls
