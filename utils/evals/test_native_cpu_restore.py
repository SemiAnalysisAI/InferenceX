"""Exercise the experimental restore check over real local HTTP fixtures."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROBE = (
    Path(__file__).resolve().parents[2]
    / "experimental/bfcl/verify_native_cpu_restore.py"
)


@pytest.mark.parametrize(
    "mode,error",
    [
        ("success", None),
        ("changed_output", "greedy output changed"),
        ("no_external_hits", "no native external-cache hits"),
        ("missing_cached_usage", "did not report native cached tokens"),
        ("reset_failed", "reset did not succeed"),
        ("http_error", "HTTP Error 500"),
    ],
)
def test_restore_cli_requires_native_evidence(
    tmp_path: Path, mode: str, error: str | None
) -> None:
    """A green HTTP request alone cannot satisfy restore correctness."""
    requests = []
    completions = []

    class Endpoint(BaseHTTPRequestHandler):
        def respond(self, body, *, status=200, content_type="application/json"):
            raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(self.path)
            if self.path.startswith("/reset_prefix_cache?"):
                self.respond({"success": mode != "reset_failed"})
                return
            assert self.path == "/v1/completions"
            completions.append(payload)
            if mode == "http_error":
                self.respond({"error": "fixture failure"}, status=500)
                return
            restored = len(completions) == 2
            usage = {"prompt_tokens": 8192, "completion_tokens": 32}
            if not (restored and mode == "missing_cached_usage"):
                usage["prompt_tokens_details"] = {
                    "cached_tokens": 7680 if restored else 0
                }
            self.respond(
                {
                    "choices": [
                        {
                            "text": "wrong"
                            if restored and mode == "changed_output"
                            else "alpha beta",
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            )

        def do_GET(self):
            assert self.path == "/metrics"
            hits = 7680 if len(completions) == 2 and mode != "no_external_hits" else 0
            self.respond(
                "# TYPE vllm:external_prefix_cache_hits_total counter\n"
                f'vllm:external_prefix_cache_hits_total{{model_name="fixture",engine="0"}} {hits}\n',
                content_type="text/plain",
            )

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    output = tmp_path / "native_cpu_restore_report.json"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(PROBE),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "fixture",
                "--output",
                str(output),
                "--settle-seconds",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
    report = json.loads(output.read_text())
    assert report["passed"] is (error is None)
    assert result.returncode == (0 if error is None else 1)
    if error:
        assert error in report["error"]["message"]
    else:
        assert completions[0] == completions[1]
        assert requests == [
            "/reset_prefix_cache?reset_external=true",
            "/v1/completions",
            "/reset_prefix_cache?reset_external=false",
            "/v1/completions",
        ]
        assert report["external_hit_tokens"]["delta"] == 7680
        assert (
            report["cold_response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
            == 0
        )
        assert (
            report["restored_response"]["usage"]["prompt_tokens_details"][
                "cached_tokens"
            ]
            == 7680
        )


def test_restore_cli_rejects_non_loopback_servers(tmp_path: Path) -> None:
    """The cache-reset diagnostic must not target a remote/shared endpoint."""
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--base-url",
            "http://example.com",
            "--model",
            "fixture",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert "loopback" in report["error"]["message"]
