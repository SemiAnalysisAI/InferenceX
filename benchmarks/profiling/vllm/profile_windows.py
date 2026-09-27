"""Open torch profiler windows on every vLLM server during an AgentX replay.

Usage: profile_windows.py WINDOWS_JSON LOG_PATH WARMUP_REQUESTS SERVER_URL...

WINDOWS_JSON lists [delay_seconds, iterations] pairs. Delays count from the
end of the client's warmup: the moment the servers have completed
WARMUP_REQUESTS requests (vllm:request_success_total), or, if the servers
never report it, from this script's start. Each window POSTs /start_profile
to every server; vLLM stops on its own after its configured max_iterations,
and a /stop_profile after a grace period closes a window that did not fill
(for example when the servers went idle).
"""

import json
import sys
import time
import urllib.request

SUCCESS_METRIC = "vllm:request_success_total"
WARMUP_TIMEOUT_S = 3600.0


def post(url: str) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return f"{response.status}"
    except Exception as error:  # a failed window must not fail the benchmark
        return f"error: {error}"


def completed_requests(servers: list[str]) -> float | None:
    """Sum of the success counter over every server, or None if none reports it."""
    total, seen = 0.0, False
    for server in servers:
        try:
            with urllib.request.urlopen(f"{server}/metrics", timeout=10) as response:
                text = response.read().decode()
        except Exception:
            continue
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].split("{", 1)[0] == SUCCESS_METRIC:
                total += float(fields[1])
                seen = True
    return total if seen else None


def main() -> None:
    windows = json.loads(sys.argv[1])
    log_path = sys.argv[2]
    warmup_requests = int(sys.argv[3])
    servers = [url.rstrip("/") for url in sys.argv[4:]]
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows, warmup_requests=warmup_requests)
        start = time.monotonic()
        deadline = start + WARMUP_TIMEOUT_S
        done = completed_requests(servers)
        while time.monotonic() < deadline:
            done = completed_requests(servers)
            if done is not None and done >= warmup_requests:
                break
            if done is None and time.monotonic() - start > 120:
                break  # no counter to watch: fall back to wall time
            time.sleep(5)
        start = time.monotonic()
        note(event="measurement_start", completed_requests=done,
             basis="warmup" if done is not None and done >= warmup_requests else "wall")
        for index, (delay, iterations) in enumerate(windows):
            time.sleep(max(0.0, start + float(delay) - time.monotonic()))
            for server in servers:
                note(event="start", window=index, server=server, iterations=iterations,
                     status=post(f"{server}/start_profile"))
            # Generous grace: a step is well under a second at every agentic point.
            time.sleep(max(120.0, 4.0 * float(iterations)))
            for server in servers:
                note(event="stop", window=index, server=server,
                     status=post(f"{server}/stop_profile"))


if __name__ == "__main__":
    main()
