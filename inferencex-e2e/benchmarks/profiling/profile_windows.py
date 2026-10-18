"""Open torch profiler windows on every engine during an AgentX replay."""

import glob
import json
import os
import re
import signal
import sys
import time
import urllib.request

from clock_sampler import ClockSampler
import rate_probe

PHASE_START = re.compile(r"Phase (\S+) \((\S+)\) started")
POLL_S = 5.0
NO_LOG_FALLBACK_S = 900.0
PHASE_TIMEOUT_S = 7200.0
DECODE_SHARE = 0.9
DECODE_LOOKBACK_S = 20.0
DECODE_MIN_STEPS = 50
MEASURED_S = float(os.environ.get("INFX_PROFILE_DURATION") or 600)
DECODE_DEADLINE_S = max(0.0, MEASURED_S - 150.0)
MIN_MEASURED_S = 60.0
EXPORT_WAIT_S = 900.0
CLOCK_WINDOW_MAX_S = 120.0
CLOCK_TAIL_S = 0.5
CLOCK_TAIL_MAX_S = 10.0
PROFILER_START_WAIT_S = 60.0
WINDOW_REACH_MAX_S = 900.0
START_LEAD_S = 1.0
START_LEAD_MIN_STEPS = 4


def post(url: str) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return f"{response.status}"
    except Exception as error:
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


def step_graph(record: dict) -> str | None:
    """The CUDA graph mode a logged step ran in."""
    if record.get("graph"):
        return record["graph"]
    dispatch = record.get("dispatch")
    return dispatch[-1].get("cg_mode") if dispatch else None

_steps: list[tuple[float, bool]] = []


def read_steps(steps_dir: str) -> list[tuple[float, bool]]:
    """(step start unix time, replayed a FULL graph) for every engine step logged so far."""
    for path in glob.glob(os.path.join(steps_dir, "*.jsonl")):
        try:
            with open(path) as f:
                f.seek(_offsets.get(path, 0))
                while line := f.readline():
                    if not line.endswith("\n"):
                        break
                    _offsets[path] = f.tell()
                    record = json.loads(line)
                    if "step" in record:
                        _steps.append((record["t0_ns"] / 1e9, step_graph(record) == "FULL"))
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
        self.times: dict[str, list[tuple[int, int]]] = {}
        self.latest: dict[str, int] = {}

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
                        k = record.get("step", record.get("dummy"))
                        if k is not None:
                            rank = os.path.basename(path)[:-len(".jsonl")]
                            self.latest[rank] = max(self.latest.get(rank, -1), int(k))
                        if "t0_ns" in record:
                            self.times.setdefault(path, []).append(
                                (record["t0_ns"], record.get("t1_ns", record["t0_ns"])))
            except (OSError, ValueError):
                continue
        return {path: sum(t0 >= t_ns for t0, _ in times) for path, times in self.times.items()}

    def start_steps(self) -> dict[str, int]:
        """rank -> the step a window requested now starts at (the same on every rank)."""
        self.counts_since(time.time_ns())
        if not self.latest:
            return {}
        recent = sorted(b - a for times in self.times.values() for a, b in times[-64:])
        step_s = max(recent[len(recent) // 2] / 1e9, 1e-3) if recent else 1.0
        start = max(self.latest.values()) + max(START_LEAD_MIN_STEPS, int(START_LEAD_S / step_s) + 1)
        return {rank: start for rank in self.latest}

    def longest_between(self, t0_ns: int, t1_ns: int) -> float:
        """Seconds of the longest step any rank logged starting between t0_ns and t1_ns."""
        self.counts_since(t0_ns)
        return max((b - a for times in self.times.values() for a, b in times
                    if t0_ns <= a <= t1_ns), default=0) / 1e9


def profiler_events(prof_dir: str, t_ns: int) -> dict[str, list[tuple[str, int]]]:
    """rank -> the engine's profiler start/stop events after t_ns."""
    events = {}
    for path in glob.glob(os.path.join(prof_dir, "profiler", "*.jsonl")):
        try:
            with open(path) as f:
                rows = [json.loads(line) for line in f if line.endswith("\n")]
        except (OSError, ValueError):
            continue
        events[os.path.basename(path)] = [(r["event"], r["t_ns"]) for r in rows if r["t_ns"] >= t_ns]
    return events


def wait_for_profilers(prof_dir: str, t_ns: int, due: float | None = None) -> int | None:
    """Wall time (ns) the last engine that started profiling after t_ns stopped, else None when none started within PROFILER_START_WAIT_S of `due` (default t_ns)."""
    deadline = time.time() + CLOCK_WINDOW_MAX_S
    due = t_ns / 1e9 if due is None else due
    while time.time() < deadline:
        events = profiler_events(prof_dir, t_ns)
        started = {rank: ev for rank, ev in events.items() if any(e == "start" for e, _ in ev)}
        if not started and time.time() - due > PROFILER_START_WAIT_S:
            return None
        stops = [max(t for e, t in ev if e == "stop") for ev in started.values()
                 if any(e == "stop" for e, _ in ev)]
        if started and len(stops) == len(started):
            return max(stops)
        time.sleep(0.2)
    return None


def wait_for_exports(prof_dir: str, t_ns: int) -> list[str]:
    """Wait until every rank that profiled after t_ns has written its last trace; the ranks still exporting at the deadline."""
    deadline = time.time() + EXPORT_WAIT_S
    while True:
        pending = [rank for rank, ev in profiler_events(prof_dir, t_ns).items()
                   if any(e == "stop" for e, _ in ev)
                   and max(t for e, t in ev if e == "stop") > max((t for e, t in ev if e == "exported"), default=0)]
        if not pending or time.time() > deadline:
            return pending
        time.sleep(1.0)


def wait_for_window_steps(counter: StepCounter, t_ns: int, iterations: int) -> None:
    """Until every rank that stepped since t_ns has stepped `iterations` times."""
    deadline = time.time() + CLOCK_WINDOW_MAX_S
    while time.time() < deadline:
        active = [n for n in counter.counts_since(t_ns).values() if n]
        if active and min(active) >= iterations:
            return
        time.sleep(0.2)


class HttpControl:
    """Engines with an on-demand profiler endpoint (vLLM)."""

    def __init__(self, servers: list[str], note):
        self.servers, self.note = servers, note

    def wait_until_due(self, index: int) -> None:
        """The window starts on the request."""

    def open(self, index: int, iterations: int) -> None:
        for server in self.servers:
            self.note(event="start", window=index, server=server, iterations=iterations,
                      status=post(f"{server}/start_profile"))

    def close(self, index: int) -> None:
        for server in self.servers:
            self.note(event="stop", window=index, server=server, status=post(f"{server}/stop_profile"))


class FileControl:
    """Engines whose profiling hook reads window requests from control/windows.jsonl (SGLang, TRT-LLM)."""

    def __init__(self, prof_dir: str, counter: StepCounter, note):
        self.path = os.path.join(prof_dir, "control", "windows.jsonl")
        self.counter, self.note = counter, note
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _append(self, request: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(request) + "\n")

    def open(self, index: int, iterations: int) -> None:
        start_at = self.counter.start_steps()
        self.start_at = start_at
        self._append({"window": index, "steps": iterations, "start_at": start_at})
        self.note(event="start", window=index, iterations=iterations, start_at=start_at)

    def wait_until_due(self, index: int) -> None:
        """Until every rank is within two steps of the window's start step (bounded)."""
        due = {rank: step - 2 for rank, step in getattr(self, "start_at", {}).items()}
        deadline = time.time() + WINDOW_REACH_MAX_S
        while due and time.time() < deadline:
            self.counter.counts_since(time.time_ns())
            if all(self.counter.latest.get(rank, -1) >= step for rank, step in due.items()):
                return
            time.sleep(0.2)
        if due:
            self.note(event="window_not_reached", window=index, latest=dict(self.counter.latest))

    def close(self, index: int) -> None:
        self._append({"stop": index})
        self.note(event="stop", window=index)


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

        mode = os.environ.get("INFX_PROF_CONTROL", "http")
        control = (FileControl(os.path.dirname(steps_dir), counter, note) if mode == "file"
                   else HttpControl(servers, note))
        note(event="servers", servers=servers, windows=windows, aiperf_log=aiperf_log, control=mode,
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
            control.open(index, int(iterations))
            control.wait_until_due(index)
            window_ns = int(window_t0 * 1e9)
            stopped_ns = wait_for_profilers(os.path.dirname(steps_dir), window_ns, due=time.time())
            if stopped_ns is None:
                wait_for_window_steps(counter, window_ns, int(iterations))
                stopped_ns = time.time_ns()
            tail = 2 * counter.longest_between(window_ns, stopped_ns)
            time.sleep(max(0.0, stopped_ns / 1e9 + CLOCK_TAIL_S + min(tail, CLOCK_TAIL_MAX_S)
                           - time.time()))
            clocks.stop()
            note(event="clocks", window=index, polls=clocks.polls, telemetry_rows=clocks.telemetry_rows,
                 telemetry_errors=clocks.telemetry.errors[:5] if clocks.telemetry else None,
                 seconds=round(time.time() - window_t0, 3))
            time.sleep(max(0.0, window_t0 + max(120.0, 4.0 * float(iterations)) - time.time()))
            control.close(index)
        if clocks.nvml is not None and os.environ.get("INFX_PROF_RATE_PROBE", "1") == "1":
            try:
                result = rate_probe.probe(clocks.nvml, os.path.join(clocks.out_dir, "rates.json"))
                note(event="rate_probe", seconds=result["seconds"])
            except Exception as e:
                note(event="rate_probe", error=repr(e))
        if mode == "file":
            pending = wait_for_exports(os.path.dirname(steps_dir), int(started * 1e9))
            note(event="exports", pending=pending, t_unix_done=time.time())
        stop_replay(aiperf_log, started, note)


if __name__ == "__main__":
    main()
