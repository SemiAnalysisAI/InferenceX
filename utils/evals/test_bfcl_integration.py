"""Optional stock-package BFCL integration; local HTTP fixtures, no model calls."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

bfcl = pytest.importorskip("bfcl_eval", reason="Install pinned BFCL for integration tests")
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("suite", ["bfcl_smoke", "bfcl_responses_smoke"])
@pytest.mark.parametrize("reject_store", [False, True])
def test_stock_bfcl_generation_scoring_and_error_reports(tmp_path: Path, reject_store: bool, suite: str):
    """Run the real CLI, corpus and scorer; distinguish a quality miss from HTTP failure."""
    assert importlib.metadata.version("bfcl-eval") == "2026.3.23"
    package_root = Path(bfcl.__file__).parent
    sources = [
        package_root / "model_handler/api_inference/openai_completion.py",
        package_root / "model_handler/api_inference/openai_response.py",
        package_root / "constants/default_prompts.py",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if reject_store:
                status = 422
                body = {"error": {"message": "Extra inputs are not permitted: store",
                                  "type": "extra_forbidden"}}
            else:
                status = 200
                # Deliberately answer without tools: three quality misses and
                # one correct irrelevance answer, all four infrastructure successes.
                body = {
                    "id": "chatcmpl-fixture", "object": "chat.completion", "created": 1,
                    "model": payload["model"],
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "No tool call."
                    }, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            if not reject_store and suite == "bfcl_responses_smoke":
                body = {
                    "id": "resp_fixture", "object": "response", "created_at": 1,
                    "status": "completed", "model": payload["model"],
                    "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                                "status": "completed", "content": [{"type": "output_text",
                                "text": "No tool call.", "annotations": []}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "output"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "infx.evals.bfcl_adapter",
             "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
             "--api-key", "EMPTY", "--model", "inferencex-fixture", "--suite", suite,
             "--output-dir", str(output), "--bfcl-project-root", str(tmp_path / "bfcl")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == int(reject_store), result.stdout + result.stderr
    native = json.loads((output / "bfcl_report.json").read_text())
    compatibility = json.loads((output / "results_bfcl.json").read_text())
    assert native["task"] == suite
    assert native["bfcl"]["source"]["api_format"] == (
        "responses" if suite == "bfcl_responses_smoke" else "chat-completions"
    )
    assert len(requests) == 4
    route = "/v1/responses" if suite == "bfcl_responses_smoke" else "/v1/chat/completions"
    assert all(path == route for path, _ in requests)
    # Never silently strip the unsupported field to hide a backend incompatibility.
    assert all(payload["store"] is False for _, payload in requests)
    assert native["completed"] is not reject_store
    if reject_store:
        assert native["integration_error"]
        assert compatibility["n-samples"][suite]["effective"] == 0
    else:
        assert "integration_error" not in native
        assert compatibility["n-samples"][suite]["effective"] == 4
        assert compatibility["results"][suite]["acc,none"] == 0.25
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
