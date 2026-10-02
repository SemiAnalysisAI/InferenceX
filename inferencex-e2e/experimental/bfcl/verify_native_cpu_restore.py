"""Check one CPU-cache restore through an isolated vLLM server's native APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

HIT_COUNTER = "vllm:external_prefix_cache_hits_total"


class Client:
    """Use only the loopback server owned by the diagnostic job."""

    def __init__(self, base_url: str) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path not in {"", "/"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base URL must be a loopback HTTP server root")
        self.base_url = base_url.rstrip("/")

    def request(self, path: str, payload: dict[str, Any] | None = None) -> str:
        """Read a native response with a bounded transport timeout."""
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(
            request, timeout=180 if path == "/v1/completions" else 15
        ) as response:
            return response.read().decode()

    def reset(self, *, external: bool, settle_seconds: float) -> None:
        """Wait for in-flight transfers, then require a successful native reset."""
        deadline = time.monotonic() + settle_seconds
        path = f"/reset_prefix_cache?reset_external={str(external).lower()}"
        while True:
            result = json.loads(self.request(path, {}))
            if result.get("success") is True:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"native prefix-cache reset did not succeed: {result}"
                )
            time.sleep(0.5)

    def external_hits(self) -> float:
        """Require the native external-cache token counter, across engine labels."""
        pattern = re.compile(
            rf"^{re.escape(HIT_COUNTER)}(?:\{{.*\}})?\s+(\S+)(?:\s+\S+)?$"
        )
        values = [
            float(match.group(1))
            for line in self.request("/metrics").splitlines()
            if (match := pattern.fullmatch(line))
        ]
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            raise RuntimeError(f"missing or invalid native metric: {HIT_COUNTER}")
        return sum(values)


def verify(
    client: Client, model: str, report: dict[str, Any], settle_seconds: float
) -> None:
    """Compare cold and restored greedy output, requiring actual external hits."""
    prompt = (
        "Native CPU cache restore diagnostic.\n"
        + "".join(
            f"Record {i:04d}: alpha beta gamma delta epsilon.\n" for i in range(768)
        )
        + "Continue the next record:\nRecord 0768:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 32,
        "stream": False,
    }
    report["request"] = {
        **{key: value for key, value in payload.items() if key != "prompt"},
        "prompt_recipe": "numbered-records-v1",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    client.reset(external=True, settle_seconds=settle_seconds)
    cold = json.loads(client.request("/v1/completions", payload))
    report["cold_response"] = cold
    if cold.get("usage", {}).get("prompt_tokens", 0) < 6144:
        raise RuntimeError(
            "diagnostic prompt did not span at least four 1536-token blocks"
        )
    if cold.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens") != 0:
        raise RuntimeError("cold request did not report zero cached tokens")

    # This clears GPU prefix state only. The CPU connector's cache is retained.
    client.reset(external=False, settle_seconds=settle_seconds)
    before = client.external_hits()
    restored = json.loads(client.request("/v1/completions", payload))
    report["restored_response"] = restored
    cached_tokens = (
        restored.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
    )
    if (
        not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or cached_tokens <= 0
    ):
        raise RuntimeError("restored request did not report native cached tokens")
    deadline = time.monotonic() + settle_seconds
    while True:
        after = client.external_hits()
        report["external_hit_tokens"] = {
            "before": before,
            "after": after,
            "delta": after - before,
        }
        if after > before:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "repeated request produced no native external-cache hits"
            )
        time.sleep(0.5)

    cold_choices, restored_choices = (
        cold.get("choices", []),
        restored.get("choices", []),
    )
    if len(cold_choices) != 1 or len(restored_choices) != 1:
        raise RuntimeError("expected exactly one completion for each request")
    cold_text = cold_choices[0].get("text")
    restored_text = restored_choices[0].get("text")
    if not isinstance(cold_text, str) or not cold_text:
        raise RuntimeError("cold completion was empty or malformed")
    if restored_text != cold_text:
        raise RuntimeError("greedy output changed after the native CPU-cache restore")
    if cold_choices[0].get("finish_reason") != restored_choices[0].get("finish_reason"):
        raise RuntimeError(
            "completion termination changed after the native CPU-cache restore"
        )
    report["passed"] = True


def main() -> int:
    """Preserve native evidence on success and failure; never alter BFCL scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=45)
    args = parser.parse_args()
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 60:
        parser.error("settle-seconds must be between 0 and 60")
    report: dict[str, Any] = {
        "verifier": "vllm-native-cpu-restore-v1",
        "scope": "one isolated prefix, GPU cache cleared, CPU cache retained",
        "endpoint": args.base_url,
        "passed": False,
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Leave an explicit incomplete artifact even if the outer deadline kills us.
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        verify(Client(args.base_url), args.model, report, args.settle_seconds)
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 - CLI preserves evidence and fails closed.
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "error": report.get("error"),
                "report": str(args.output),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
