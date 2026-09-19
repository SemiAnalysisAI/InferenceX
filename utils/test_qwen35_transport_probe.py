import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[1]
    / "experimental/qwen35_cross_tp/transport_probe.py"
)
spec = importlib.util.spec_from_file_location("transport_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_prompt_budget_preserves_unique_answer_and_bounds():
    messages, answer, count = probe.bounded_prompt(
        7, 180, lambda messages: len(messages[0]["content"])
    )
    assert answer == "710007"
    assert "retrieval code is 710007" in messages[0]["content"]
    assert count <= 180
    assert count + len("This is neutral padding text for a long document.\n") > 180
    with pytest.raises(ValueError, match="smaller"):
        probe.bounded_prompt(0, 1, lambda messages: len(messages[0]["content"]))


@pytest.mark.parametrize(
    "answer,done,expected",
    [
        ("710002", True, "pass"),
        ("999999", True, "needle_answer_mismatch"),
        ("710002", False, "transport_or_runtime_error"),
    ],
)
def test_real_http_stream_distinguishes_answers_from_transport(answer, done, expected):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {
                "id": "req-test",
                "choices": [{"delta": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 170, "completion_tokens": 3},
            }
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            if done:
                self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            row = probe.request_one(
                f"http://127.0.0.1:{server.server_port}",
                "test-model",
                (2, 180, [{"role": "user", "content": "read code"}], "710002", 170),
                threading.Barrier(1),
                2,
            )
        finally:
            server.shutdown()
            thread.join()
    assert row["classification"] == expected
    assert row["transport_ok"] is done
    assert received[0]["stream"] is True
    assert received[0]["max_tokens"] == 32
    assert row["response_ids"] == ["req-test"]
