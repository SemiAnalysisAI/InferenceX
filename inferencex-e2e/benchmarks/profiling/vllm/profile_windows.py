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

Each window's GPU clocks are sampled through NVML (clock_sampler.py) from
just before it opens until every active engine has logged its iterations.

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

from clock_sampler import ClockSampler

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
CLOCK_WINDOW_MAX_S = 120.0  # a window's steps take seconds; stop sampling regardless
# The GPU finishes a window's last step after its engine logs it (launches are
# asynchronous): keep sampling for twice the window's longest step past that.
CLOCK_TAIL_S = 0.5
CLOCK_TAIL_MAX_S = 10.0


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


class StepCounter:
    """Engine steps (scheduled or dummy) each rank's step log records after a time."""

    def __init__(self, steps_dir: str):
        self.steps_dir = steps_dir
        self.offsets: dict[str, int] = {}
        self.times: dict[str, list[tuple[int, int]]] = {}  # (t0_ns, t1_ns) per step

    def counts_since(self, t_ns: int) -> dict[str, int]:
        for path in glob.glob(os.path.join(self.steps_dir, "*.jsonl")):
            try:
                with open(path) as f:
                    f.seek(self.offsets.get(path, 0))
                    while line := f.readline():
                        if not line.endswith("\n"):
                            break
                        self.offsets[path] = f.tell()
                        record = json.loads(line)
                        if "t0_ns" in record:
                            self.times.setdefault(path, []).append(
                                (record["t0_ns"], record.get("t1_ns", record["t0_ns"])))
            except (OSError, ValueError):
                continue
        return {path: sum(t0 >= t_ns for t0, _ in times) for path, times in self.times.items()}

    def longest_since(self, t_ns: int, steps: int) -> float:
        """Seconds of the longest of each rank's first `steps` steps after t_ns."""
        longest = 0
        for times in self.times.values():
            window = [t1 - t0 for t0, t1 in times if t0 >= t_ns][:steps]
            longest = max([longest, *window])
        return longest / 1e9


def wait_for_window_steps(counter: StepCounter, t_ns: int, iterations: int) -> None:
    """Until every rank that stepped since t_ns has stepped `iterations` times."""
    deadline = time.time() + CLOCK_WINDOW_MAX_S
    while time.time() < deadline:
        active = [n for n in counter.counts_since(t_ns).values() if n]
        if active and min(active) >= iterations:
            return
        time.sleep(0.2)


def aiperf_pids() -> list[int]:
    """The aiperf SystemController, which titles itself "aiperf system_controller"."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                title = f.read().replace(b"\0", b" ").decode(errors="replace").split()
        except OSError:
            continue
        if title[:2] == ["aiperf", "system_controller"]:
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
    clocks = ClockSampler(os.path.join(os.path.dirname(steps_dir), "clocks"))
    counter = StepCounter(steps_dir)
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows, aiperf_log=aiperf_log,
             clocks_error=clocks.error)
        for index, (anchor, delay, iterations) in enumerate(windows):
            if anchor == "decode":
                anchor_t, basis = wait_for_decode(aiperf_log, steps_dir, started)
            else:
                anchor_t, basis = wait_for_phase(aiperf_log, anchor, started)
            note(event="anchor", window=index, anchor=anchor, basis=basis, anchor_unix=anchor_t)
            time.sleep(max(0.0, anchor_t + float(delay) - time.time()))
            window_t0 = time.time()
            clocks.start(index)
            for server in servers:
                note(event="start", window=index, server=server, iterations=iterations,
                     status=post(f"{server}/start_profile"))
            wait_for_window_steps(counter, int(window_t0 * 1e9), int(iterations))
            tail = 2 * counter.longest_since(int(window_t0 * 1e9), int(iterations))
            time.sleep(CLOCK_TAIL_S + min(tail, CLOCK_TAIL_MAX_S))
            clocks.stop()
            note(event="clocks", window=index, polls=clocks.polls,
                 seconds=round(time.time() - window_t0, 3))
            # Generous grace: a step is well under a second at every agentic point.
            time.sleep(max(0.0, window_t0 + max(120.0, 4.0 * float(iterations)) - time.time()))
            for server in servers:
                note(event="stop", window=index, server=server,
                     status=post(f"{server}/stop_profile"))
        stop_replay(aiperf_log, started, note)


if __name__ == "__main__":
    main()
