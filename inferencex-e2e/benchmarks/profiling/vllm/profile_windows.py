"""Open torch profiler windows on every vLLM server during an AgentX replay.

Usage: profile_windows.py WINDOWS_JSON LOG_PATH AIPERF_LOG STEPS_DIR SERVER_URL...

WINDOWS_JSON lists [anchor, delay_seconds, iterations] windows. An anchor
"warmup" or "profiling" is an aiperf phase: the window opens delay_seconds
after AIPERF_LOG records that phase's start. AgentX warmup is the lanes' long
first turns (prefill-heavy). The anchor "decode" is steady decode: from the
warmup start, the window opens delay_seconds after the engines' step logs in
STEPS_DIR show at least DECODE_SHARE of recent steps replaying FULL CUDA
graphs. Each new turn re-prefills, so how long that takes depends on
concurrency; if it has not happened DECODE_DEADLINE_S into the measured phase,
the window opens then. If aiperf never logs a phase, windows fall back to this
script's start. Each window POSTs /start_profile to every server; vLLM stops on
its own after its configured max_iterations, and a /stop_profile after a grace
period closes a window that did not fill.

The measured phase is capped at INFX_PROFILE_DURATION seconds, long enough for
steady decode at any concurrency. Once the last window has closed and the
measured phase has results to export, this script ends the replay early with
SIGINT, which aiperf treats as a user cancel: it stops issuing requests,
exports what it measured and exits zero.
"""

import glob
import json
import os
import re
import signal
import sys
import time
import urllib.request

PHASE_START = re.compile(r"Phase (\S+) \((\S+)\) started")
POLL_S = 5.0
NO_LOG_FALLBACK_S = 900.0  # aiperf logs its first phase within minutes of starting
PHASE_TIMEOUT_S = 7200.0
DECODE_SHARE = 0.9
DECODE_LOOKBACK_S = 20.0
DECODE_MIN_STEPS = 50
MEASURED_S = float(os.environ.get("INFX_PROFILE_DURATION") or 600)
# Latest a decode window opens in the measured phase, leaving it room to close.
DECODE_DEADLINE_S = max(0.0, MEASURED_S - 150.0)
# Measured replay before an early stop: aiperf fails a run with no results.
MIN_MEASURED_S = 60.0


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


_offsets: dict[str, int] = {}
_steps: list[tuple[float, bool]] = []


def read_steps(steps_dir: str) -> list[tuple[float, bool]]:
    """(step start unix time, replayed a FULL graph) for every engine step logged so far."""
    for path in glob.glob(os.path.join(steps_dir, "*.jsonl")):
        try:
            with open(path) as f:
                f.seek(_offsets.get(path, 0))
                while line := f.readline():
                    if not line.endswith("\n"):
                        break  # the engine is mid-write; reread it next poll
                    _offsets[path] = f.tell()
                    record = json.loads(line)
                    if "step" in record and record.get("dispatch"):
                        _steps.append((record["t0_ns"] / 1e9,
                                       record["dispatch"][0].get("cg_mode") == "FULL"))
        except (OSError, ValueError):
            continue
    return _steps


def steady_decode(steps_dir: str) -> bool:
    now = time.time()
    recent = [full for t, full in read_steps(steps_dir) if t >= now - DECODE_LOOKBACK_S]
    return len(recent) >= DECODE_MIN_STEPS and sum(recent) >= DECODE_SHARE * len(recent)


def wait_for_decode(aiperf_log: str, steps_dir: str, started: float) -> tuple[float, str]:
    """Wall time steady decode was first seen after warmup began, else the measured-phase deadline."""
    wait_for_phase(aiperf_log, "warmup", started)
    while True:
        if steady_decode(steps_dir):
            return time.time(), "steady decode"
        measured = phase_starts(aiperf_log).get("profiling")
        if measured is not None and time.time() > measured + DECODE_DEADLINE_S:
            return time.time(), "measured-phase deadline"
        if time.time() - started > PHASE_TIMEOUT_S:
            return time.time(), "decode never steady"
        time.sleep(POLL_S)


def aiperf_pids() -> list[int]:
    """The aiperf CLI processes running a benchmark (not wrappers naming it in their argv)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        # The program itself, or the script its interpreter runs: [.../bin/aiperf, profile]
        # or [python, .../bin/aiperf, profile]. A wrapper names it later in its argv.
        if any(argv[i].endswith("/bin/aiperf") and argv[i + 1:i + 2] == ["profile"] for i in (0, 1)
               if i < len(argv)):
            pids.append(int(entry))
    return pids


def stop_replay(aiperf_log: str, started: float, note) -> None:
    """End the measured phase once it has results; aiperf exports them and exits zero."""
    measured, basis = wait_for_phase(aiperf_log, "profiling", started)
    if basis != "aiperf":
        note(event="replay_stop", status=f"skipped: {basis}")
        return
    time.sleep(max(0.0, measured + MIN_MEASURED_S - time.time()))
    pids = aiperf_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            pass
    note(event="replay_stop", pids=pids)


def main() -> None:
    windows = json.loads(sys.argv[1])
    log_path, aiperf_log, steps_dir = sys.argv[2], sys.argv[3], sys.argv[4]
    servers = [url.rstrip("/") for url in sys.argv[5:]]
    started = time.time()
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows, aiperf_log=aiperf_log)
        for index, (anchor, delay, iterations) in enumerate(windows):
            if anchor == "decode":
                anchor_t, basis = wait_for_decode(aiperf_log, steps_dir, started)
            else:
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
        stop_replay(aiperf_log, started, note)


if __name__ == "__main__":
    main()
