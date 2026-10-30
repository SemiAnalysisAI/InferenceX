import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from infx.serving.client import (
    ProbeError,
    agentx_argv,
    datasets_cache,
    label_result,
    online,
    probe,
    redacted,
    with_requests,
    with_token_source,
)

ENV = {
    "PORT": "21000",
    "MODEL": "Qwen/Qwen3.8-27B",
    "SERVED_MODEL_NAME": "",
    "CONC": "8",
    "RESULT_DIR": "/workspace/results",
    "SERVING_SEARCH": "0",
    "AIPERF_GPU_TELEMETRY_URL": "http://localhost:21001/metrics",
    "AIPERF_SERVER_METRICS_URLS": "http://localhost:21000/metrics,http://localhost:21002/metrics",
}


def test_fixed_concurrency_argv():
    argv = agentx_argv("/venv/bin/agentx", Path("/workspace/results/scenario.yaml"), ENV)
    assert argv[:4] == [
        "/venv/bin/agentx",
        "profile",
        "--config",
        "/workspace/results/scenario.yaml",
    ]
    assert argv[argv.index("--url") + 1] == "http://localhost:21000"
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.8-27B"
    assert argv[argv.index("--concurrency") + 1] == "8"
    assert argv[argv.index("--gpu-telemetry") + 1] == "http://localhost:21001/metrics"
    metrics = argv.index("--server-metrics")
    assert argv[metrics + 1 : metrics + 3] == [
        "http://localhost:21000/metrics",
        "http://localhost:21002/metrics",
    ]
    assert argv[-2:] == ["--output-artifact-dir", "/workspace/results/aiperf_artifacts"]


def test_search_lets_aiperf_pick_the_concurrency_and_served_name_wins():
    env = {
        **ENV,
        "SERVING_SEARCH": "1",
        "SERVED_MODEL_NAME": "qwen",
        "AIPERF_GPU_TELEMETRY_URL": "",
    }
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


REMOTE_ENV = {
    "MODEL": "gpt-5",
    "CONC": "4",
    "RESULT_DIR": "/workspace/results",
    "SERVING_SEARCH": "0",
    "SERVING_REMOTE_URL": "https://api.example.com",
    "SERVING_TOKENIZER": "builtin",
    "SERVING_API_KEY": "sk-secret-123",
}


def test_remote_argv_targets_the_endpoint_with_its_key_and_tokenizer():
    argv = agentx_argv("agentx", Path("s.yaml"), REMOTE_ENV)
    assert argv[argv.index("--url") + 1] == "https://api.example.com"
    assert argv[argv.index("--tokenizer") + 1] == "builtin"
    assert argv[argv.index("--api-key") + 1] == "sk-secret-123"
    assert "--no-gpu-telemetry" in argv and "--server-metrics" not in argv


def test_engine_argv_sends_no_key():
    assert "--api-key" not in agentx_argv("agentx", Path("s.yaml"), ENV)


def test_redacted_hides_the_key():
    argv = agentx_argv("agentx", Path("s.yaml"), REMOTE_ENV)
    assert "sk-secret-123" not in " ".join(redacted(argv))
    assert redacted(argv)[argv.index("--api-key") + 1] == "***"


def test_basic_auth_key_becomes_an_authorization_header():
    env = {**REMOTE_ENV, "SERVING_API_KEY": "Basic dXNlcjpwYXNz"}
    argv = agentx_argv("agentx", Path("s.yaml"), env)
    assert argv[argv.index("--header") + 1] == "Authorization:Basic dXNlcjpwYXNz"
    assert "--api-key" not in argv
    assert "Basic dXNlcjpwYXNz" not in " ".join(redacted(argv))


def test_probe_sends_a_basic_key_as_the_raw_authorization():
    url, seen, server = serve(200, WITH_USAGE)
    try:
        assert probe(url, "m", "Basic dXNlcjpwYXNz") == "server"
    finally:
        server.shutdown()
    assert seen[0][1]["Authorization"] == "Basic dXNlcjpwYXNz"


def _serve_tls(status: int, body: bytes):
    """A one-route stub endpoint behind a self-signed certificate, like a -k-only gateway."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append(dict(self.headers))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    fixtures = Path(__file__).parent / "fixtures"
    context.load_cert_chain(fixtures / "self-signed.pem", fixtures / "self-signed.key")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"https://127.0.0.1:{server.server_port}", seen, server


def test_probe_skips_tls_verification_when_the_endpoint_asks_for_it():
    url, seen, server = _serve_tls(200, WITH_USAGE)
    try:
        assert probe(url, "m", "", verify=False) == "server"
        assert seen, "the unverified probe reached the self-signed endpoint"
    finally:
        server.shutdown()
    with pytest.raises(ProbeError, match="unreachable"):
        probe(url, "m", "", verify=True)


def test_client_token_source_turns_server_counting_off():
    text = yaml.safe_dump(
        {"benchmark": {"endpoint": {"type": "chat", "useServerTokenCount": True}}}
    )
    assert with_token_source(text, "server") == text
    assert yaml.safe_load(with_token_source(text, "client"))["benchmark"]["endpoint"] == {
        "type": "chat",
        "useServerTokenCount": False,
    }


def test_label_result_records_the_token_count_source(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"conc": 4}))
    label_result(path, "agentx-cc-062126-smoke", token_count_source="client")
    assert json.loads(path.read_text()) == {
        "conc": 4,
        "scenario": "agentx-cc-062126-smoke",
        "token_count_source": "client",
    }


def serve(status: int, body: bytes):
    """A one-route stub endpoint on a free port; returns (base url, received requests, server)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(length))))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", seen, server


WITH_USAGE = (
    b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1}}\n\n'
    b"data: [DONE]\n\n"
)
WITHOUT_USAGE = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'


def test_probe_reads_server_usage_and_sends_the_harness_request_shape():
    url, seen, server = serve(200, WITH_USAGE)
    try:
        assert probe(url, "gpt-5", "sk-1") == "server"
    finally:
        server.shutdown()
    [(path, headers, body)] = seen
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-1"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["ignore_eos"] is True and body["model"] == "gpt-5"


def test_probe_without_usage_falls_back_to_client_counting():
    url, _, server = serve(200, WITHOUT_USAGE)
    try:
        assert probe(url, "gpt-5", "") == "client"
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "status, body, secret, message",
    [
        (
            401,
            b'{"error":"bad key"}',
            "OPENAI_API_KEY",
            "HTTP 401: key rejected; check GitHub secret OPENAI_API_KEY",
        ),
        (401, b"", "", "HTTP 401: the endpoint wants a key; set endpoint.api-key-secret"),
        (404, b"no such model", "", "HTTP 404: check endpoint.url and endpoint.model"),
        (
            400,
            b'{"error":{"message":"Unrecognized request argument supplied: ignore_eos"}}',
            "",
            "HTTP 400: Unrecognized request argument supplied: ignore_eos",
        ),
    ],
)
def test_probe_failures_name_the_fix(status, body, secret, message):
    url, _, server = serve(status, body)
    try:
        with pytest.raises(ProbeError) as caught:
            probe(url, "gpt-5", "sk-1" if secret else "", secret)
    finally:
        server.shutdown()
    assert message in str(caught.value)
    assert "sk-1" not in str(caught.value)


def test_datasets_arrow_cache_lives_beside_the_node_hub_cache():
    assert datasets_cache({"HF_HUB_CACHE": "/mnt/hf_hub_cache/"}) == {
        "HF_DATASETS_CACHE": "/mnt/hf_hub_cache/.datasets-arrow"
    }
    assert (
        datasets_cache({"HF_HUB_CACHE": "/mnt/hf", "HF_DATASETS_CACHE": "/x"}) == {}
    )  # caller's choice wins
    assert datasets_cache({}) == {}  # no node cache mounted: keep the library default


def test_with_requests_sets_the_profiling_phase_only():
    scenario = yaml.safe_dump({"benchmark": {"phases": [
        {"name": "warmup", "kind": "warmup", "requests": 4},
        {"name": "profiling", "kind": "profiling", "requests": 1},
    ]}})
    phases = yaml.safe_load(with_requests(scenario, 777))["benchmark"]["phases"]
    assert [phase["requests"] for phase in phases] == [4, 777]


def test_guard_scans_the_release_and_prunes_the_result_dir(tmp_path, monkeypatch):
    from infx.serving import client

    release = tmp_path / "greenspec.jsonl"
    release.write_text(json.dumps({"question_id": "m/secret", "category": "llm_task"}) + "\n")
    monkeypatch.setattr("infx.serving.datasets.fetch", lambda name: release)
    monkeypatch.setattr(client, "SELECTION_DIR", str(tmp_path / "selection"))
    (tmp_path / "selection").mkdir()
    result_dir, output = tmp_path / "results", tmp_path / "out"
    result_dir.mkdir(), output.mkdir()
    (result_dir / "server.log").write_text("engine output")
    (result_dir / "scenario.yaml").write_text("benchmark: {}")
    aggregate = output / "agg.json"
    aggregate.write_text("{}")
    assert client.guard({"SERVING_DATASET": "greenspec-v1.1.1"}, result_dir, aggregate, output) == 0
    assert not (tmp_path / "selection").exists() and not (result_dir / "server.log").exists()
    assert (output / client.PRIVATE_LOGS / "server.log").read_text() == "engine output"
    aggregate.write_text('{"note": "m/secret"}')
    assert client.guard({"SERVING_DATASET": "greenspec-v1.1.1"}, result_dir, aggregate, output) == 1
