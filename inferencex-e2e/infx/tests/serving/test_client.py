import json
from pathlib import Path

from infx.serving.client import agentx_argv, label_result, online

ENV = {
    "PORT": "21000", "MODEL": "Qwen/Qwen3.8-27B", "SERVED_MODEL_NAME": "", "CONC": "8",
    "RESULT_DIR": "/workspace/results", "SERVING_SEARCH": "0",
    "AIPERF_GPU_TELEMETRY_URL": "http://localhost:21001/metrics",
    "AIPERF_SERVER_METRICS_URLS": "http://localhost:21000/metrics,http://localhost:21002/metrics",
}


def test_fixed_concurrency_argv():
    argv = agentx_argv("/venv/bin/agentx", Path("/workspace/results/scenario.yaml"), ENV)
    assert argv[:4] == ["/venv/bin/agentx", "profile", "--config", "/workspace/results/scenario.yaml"]
    assert argv[argv.index("--url") + 1] == "http://localhost:21000"
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.8-27B"
    assert argv[argv.index("--concurrency") + 1] == "8"
    assert argv[argv.index("--gpu-telemetry") + 1] == "http://localhost:21001/metrics"
    metrics = argv.index("--server-metrics")
    assert argv[metrics + 1: metrics + 3] == ["http://localhost:21000/metrics", "http://localhost:21002/metrics"]
    assert argv[-2:] == ["--output-artifact-dir", "/workspace/results/aiperf_artifacts"]


def test_search_lets_aiperf_pick_the_concurrency_and_served_name_wins():
    env = {**ENV, "SERVING_SEARCH": "1", "SERVED_MODEL_NAME": "qwen", "AIPERF_GPU_TELEMETRY_URL": ""}
    argv = agentx_argv("agentx", Path("s.yaml"), env)
    assert "--concurrency" not in argv and "--no-gpu-telemetry" in argv
    assert argv[argv.index("--model") + 1] == "qwen"
    assert argv[argv.index("--tokenizer") + 1] == "Qwen/Qwen3.8-27B"


def test_label_result_files_the_aggregate_under_the_legacy_label(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"conc": 8}))
    label_result(path, "agentx-cc-062126-smoke")
    assert json.loads(path.read_text()) == {"conc": 8, "scenario": "agentx-cc-062126-smoke"}
    label_result(tmp_path / "missing.json", "x")  # a failed aggregation leaves nothing to label


def test_online_drops_the_engine_offline_switches():
    assert online({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "A": "b"}) == {"A": "b"}
