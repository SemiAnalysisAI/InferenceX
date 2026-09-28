"""Open torch profiler windows on every vLLM server during an AgentX replay.

Usage: profile_windows.py WINDOWS_JSON LOG_PATH AIPERF_LOG SERVER_URL...

WINDOWS_JSON lists [anchor, delay_seconds, iterations] windows. The anchor is
an aiperf phase ("warmup" or "profiling"): a window opens delay_seconds after
AIPERF_LOG records that phase's start. AgentX warmup is the lanes' long first
turns (prefill-heavy); the measured phase follows once they drain (decode-
dominated). If aiperf never logs a phase, windows fall back to this script's
start. Each window POSTs /start_profile to every server; vLLM stops on its
own after its configured max_iterations, and a /stop_profile after a grace
period closes a window that did not fill.
"""

import json
import re
import sys
import time
import urllib.request

PHASE_START = re.compile(r"Phase (\S+) \((\S+)\) started")
POLL_S = 5.0
NO_LOG_FALLBACK_S = 900.0  # aiperf logs its first phase within minutes of starting
PHASE_TIMEOUT_S = 7200.0


def post(url: str) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return f"{response.status}"
    except Exception as error:  # a failed window must not fail the benchmark
        return f"error: {error}"


_seen: dict[str, dict[str, float]] = {}


def phase_starts(aiperf_log: str) -> dict[str, float]:
    """aiperf phase name -> wall time this script first saw its start logged."""
    seen = _seen.setdefault(aiperf_log, {})
    try:
        with open(aiperf_log) as f:
            for line in f:
                m = PHASE_START.search(line)
                if m:
                    seen.setdefault(m.group(2), time.time())
    except OSError:
        pass
    return seen


def wait_for_phase(aiperf_log: str, phase: str, started: float) -> tuple[float, str]:
    """Wall time of `phase`'s start (to within one poll), else a fallback and why."""
    while True:
        starts = phase_starts(aiperf_log)
        if phase in starts:
            return starts[phase], "aiperf"
        elapsed = time.time() - started
        if not starts and elapsed > NO_LOG_FALLBACK_S:
            return started, "no aiperf phases logged"
        if elapsed > PHASE_TIMEOUT_S:
            return time.time(), "phase never started"
        time.sleep(POLL_S)


def main() -> None:
    windows = json.loads(sys.argv[1])
    log_path, aiperf_log = sys.argv[2], sys.argv[3]
    servers = [url.rstrip("/") for url in sys.argv[4:]]
    started = time.time()
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows, aiperf_log=aiperf_log)
        for index, (anchor, delay, iterations) in enumerate(windows):
            anchor_t, basis = wait_for_phase(aiperf_log, anchor, started)
            note(event="anchor", window=index, anchor=anchor, basis=basis, anchor_unix=anchor_t)
            time.sleep(max(0.0, anchor_t + float(delay) - time.time()))
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
