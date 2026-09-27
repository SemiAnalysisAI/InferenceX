"""Open torch profiler windows on every vLLM server during an AgentX replay.

Usage: profile_windows.py WINDOWS_JSON LOG_PATH AIPERF_LOG WARMUP_REQUESTS SERVER_URL...

WINDOWS_JSON lists [delay_seconds, iterations] pairs. Delays count from the
start of the client's measured phase: aiperf logging a non-warmup phase start
in AIPERF_LOG, else the servers having completed WARMUP_REQUESTS requests
(vllm:request_success_total), else, when neither signal exists, from this
script's start. Each window POSTs /start_profile to every server; vLLM stops
on its own after its configured max_iterations, and a /stop_profile after a
grace period closes a window that did not fill.
"""

import json
import os
import re
import sys
import time
import urllib.request

SUCCESS_METRIC = "vllm:request_success_total"
PHASE_START = re.compile(r"Phase (\S+) \((\S+)\) started")
POLL_S = 5.0
SIGNAL_WAIT_S = 900.0  # give up on both signals if neither has appeared by then
MEASURED_WAIT_S = 7200.0


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


def measured_phase_started(aiperf_log: str) -> str | None:
    try:
        with open(aiperf_log) as f:
            for line in f:
                m = PHASE_START.search(line)
                if m and m.group(2) != "warmup":
                    return m.group(2)
    except OSError:
        return None
    return None


def wait_for_measurement(aiperf_log: str, warmup_requests: int, servers: list[str]) -> dict:
    start = time.monotonic()
    while time.monotonic() - start < MEASURED_WAIT_S:
        phase = measured_phase_started(aiperf_log)
        if phase:
            return {"basis": "aiperf", "phase": phase}
        done = completed_requests(servers)
        if done is not None and done >= warmup_requests:
            return {"basis": "metric", "completed_requests": done}
        if (done is None and not os.path.exists(aiperf_log)
                and time.monotonic() - start > SIGNAL_WAIT_S):
            break
        time.sleep(POLL_S)
    return {"basis": "wall"}


def main() -> None:
    windows = json.loads(sys.argv[1])
    log_path, aiperf_log = sys.argv[2], sys.argv[3]
    warmup_requests = int(sys.argv[4])
    servers = [url.rstrip("/") for url in sys.argv[5:]]
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows, aiperf_log=aiperf_log,
             warmup_requests=warmup_requests)
        note(event="measurement_start", **wait_for_measurement(aiperf_log, warmup_requests, servers))
        start = time.monotonic()
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
