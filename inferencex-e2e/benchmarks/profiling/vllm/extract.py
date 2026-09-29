"""Attribute every device kernel of a profiled vLLM step to the op that launched it.

Usage: extract.py PROFILE_DIR OUT_DIR

PROFILE_DIR is an unpacked profile artifact (infx_profile/): torch/ holds the
engine's per-rank torch traces, capture/ the CUDA graph capture trace,
steps/ the per-step batch log, copies/ the CPU KV-offload copy log, clocks/
the window client's NVML clock samples and env/ the per-rank environment.

A kernel launched eagerly joins its CPU launch through the CUDA correlation
id. A kernel replayed from CUDA graph n carries a ``graph node id``; ordered
by node id, graph n's nodes pair with the launches recorded inside
``infx_graph_capture#n``, which carry the op, shapes and stack of the
capture. Capture ordinals follow capture order, which every rank shares.
A CPU KV-offload memcpy has no CPU launch (a driver call Kineto does not
record, from the connector's copy thread); per direction, the memcpys pair in
order with the logged copies issued before them, least total issue lag first.
Each kernel's clocks are the NVML samples of its rank's GPU over its lifetime,
the state at its start being the last sample before it.

Outputs, per replay trace: kernels.jsonl.gz (one row per device activity)
and steps.jsonl (per step: batch composition and device time), plus
report.json with the join checks.
"""

import bisect
import collections
import glob
import gzip
import json
import os
import re
import sys

MARK = re.compile(r"^infx_(step|dummy|graph_replay|graph_capture|piece)#(\d+)$")
MODULE_MARK = "infx_mod#"
LAUNCHER_MARK = "infx_py#"
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
CONTEXT_CATS = {"cpu_op", "user_annotation", "python_function"}
NODE_ID_MASK = (1 << 32) - 1


KEPT_ARGS = {
    "correlation", "External id", "Input Dims", "Input type", "Concrete Inputs",
    "kernel_file", "graph node id", "stream", "device", "grid", "block", "bytes",
}


class Event:
    """The fields of one complete ("X") trace event the joins use."""

    __slots__ = ("cat", "name", "ts", "dur", "pid", "tid", "args")

    def __init__(self, raw, names):
        self.cat = raw.get("cat")
        self.name = names.setdefault(raw.get("name", ""), raw.get("name", ""))
        self.ts = raw["ts"]
        self.dur = raw.get("dur", 0)
        self.pid = raw.get("pid")
        self.tid = raw.get("tid")
        args = raw.get("args")
        # Python frames carry nothing the joins read; everything else keeps its ids.
        self.args = ({k: v for k, v in args.items() if k in KEPT_ARGS}
                     if args and self.cat != "python_function" else {})

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        value = getattr(self, key, None)
        return default if value is None else value


def iter_trace_events(path, chunk_size=1 << 22):
    """Stream the traceEvents array one event at a time; traces reach several GB."""
    decoder = json.JSONDecoder()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        buf, pos = "", 0
        while True:  # find the array
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk
            start = buf.find('"traceEvents"')
            if start >= 0:
                bracket = buf.find("[", start)
                if bracket >= 0:
                    pos = bracket + 1
                    break
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                event, end = decoder.raw_decode(buf, pos)
            except ValueError as error:
                # An event never spans more than a chunk or two; a growing
                # undecodable buffer is corrupt input, not a partial read.
                if len(buf) - pos > 8 * chunk_size:
                    raise ValueError(f"{path}: undecodable trace near offset {pos}: {error}") from error
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(f"{path}: trace ends inside an event: {error}") from error
                buf, pos = buf[pos:] + chunk, 0
                continue
            yield event
            pos = end
            if pos > chunk_size:
                buf, pos = buf[pos:], 0


def load_events(path):
    names = {}
    return [Event(raw, names) for raw in iter_trace_events(path) if raw.get("ph") == "X"]


def parse_signature(sig):
    """[[shape, dtype], ...] from '7x7168:bfloat16;1x2:int64'."""
    if not sig:
        return []
    out = []
    for item in sig.split(";"):
        dims, _, dtype = item.rpartition(":")
        out.append([[int(d) for d in dims.split("x")] if dims else [], dtype])
    return out


def parse_launcher(text):
    """[launcher, [vLLM caller frames]] from '<launcher>#<frame>|<frame>'."""
    label, _, callers = text.partition("#")
    return [label, [c for c in callers.split("|") if c]]


def launch_kind(name):
    """kernel / memcpy / memset for APIs that put work on a stream, else None."""
    if "Graph" in name:
        return None
    if "LaunchKernel" in name or "LaunchCooperativeKernel" in name:
        return "kernel"
    if "Memcpy" in name:
        return "memcpy"
    if "Memset" in name:
        return "memset"
    return None


class Trace:
    """One Kineto trace with each CPU launch resolved to its enclosing context."""

    def __init__(self, path):
        self.path = path
        self.events = load_events(path)
        self._index()

    def _index(self):
        by_thread = collections.defaultdict(list)
        self.launches = {}  # correlation -> launch event index
        self.device = []
        for i, e in enumerate(self.events):
            cat = e.get("cat")
            if cat in CONTEXT_CATS:
                by_thread[(e["pid"], e["tid"])].append(i)
            elif cat in LAUNCH_CATS:
                by_thread[(e["pid"], e["tid"])].append(i)
                corr = e.get("args", {}).get("correlation")
                if corr is not None:
                    self.launches[corr] = i
            elif cat in DEVICE_CATS:
                self.device.append(i)
        # Sweep each thread's intervals; every launch gets its enclosing stack.
        self.stack_of = {}
        for items in by_thread.values():
            items.sort(key=lambda i: (self.events[i]["ts"], -self.events[i].get("dur", 0)))
            stack = []
            for i in items:
                e = self.events[i]
                ts = e["ts"]
                while stack and self.events[stack[-1]]["ts"] + self.events[stack[-1]].get("dur", 0) <= ts:
                    stack.pop()
                if e["cat"] in LAUNCH_CATS:
                    self.stack_of[i] = tuple(stack)
                else:
                    stack.append(i)

    def context(self, launch):
        """Markers, op chain, innermost op args and module path of a launch."""
        marks = {}
        ops = []
        annotations = []
        modules = []  # qualified names from infx_mod markers
        stack_modules = []  # class-instance names from Python stacks (capture only)
        frame = None
        module_stack = []  # [qualified name, input signature], outermost first
        launcher = None  # innermost [launcher, vLLM caller frames]
        for i in self.stack_of.get(launch, ()):
            e = self.events[i]
            name = e["name"]
            m = MARK.match(name)
            if m:
                marks[m.group(1)] = int(m.group(2))
            elif name.startswith(MODULE_MARK):
                qualname, _, sig = name[len(MODULE_MARK):].partition("#")
                modules.append(qualname)
                module_stack.append([qualname, parse_signature(sig)])
            elif name.startswith(LAUNCHER_MARK):
                launcher = parse_launcher(name[len(LAUNCHER_MARK):])
            elif e["cat"] == "cpu_op":
                ops.append(i)
            elif e["cat"] == "user_annotation":
                annotations.append(name)  # e.g. vLLM's per-step execute_context_* scope
            elif e["cat"] == "python_function":
                if name.startswith("nn.Module: "):
                    stack_modules.append(name[len("nn.Module: "):])
                else:
                    frame = name
        inner = self.events[ops[-1]] if ops else None
        args = inner.get("args", {}) if inner else {}
        return {
            "marks": marks,
            "op": inner["name"] if inner else None,
            "op_chain": [self.events[i]["name"] for i in ops],
            "annotations": annotations,
            "input_dims": args.get("Input Dims"),
            "input_types": args.get("Input type"),
            "concrete_inputs": args.get("Concrete Inputs"),
            "kernel_file": args.get("kernel_file"),
            "module_path": modules or stack_modules,
            "module_stack": module_stack,
            "launcher": launcher[0] if launcher else None,
            "launcher_callers": launcher[1] if launcher else None,
            "py_frame": frame,
            "launch_api": self.events[launch]["name"],
        }

    def process_ids(self):
        return {e["pid"] for e in self.events if e.get("cat") in CONTEXT_CATS}


def capture_launches(trace):
    """graph ordinal -> ordered contexts of every stream launch captured into it."""
    graphs = collections.defaultdict(list)
    order = sorted(
        (i for i in trace.stack_of if launch_kind(trace.events[i]["name"])),
        key=lambda i: trace.events[i]["ts"],
    )
    for i in order:
        ctx = trace.context(i)
        gid = ctx["marks"].get("graph_capture")
        if gid is not None:
            ctx["kind"] = launch_kind(trace.events[i]["name"])
            graphs[gid].append(ctx)
    return graphs


def eager_kernel_names(trace):
    """(op, input dims) -> kernel names seen from eager launches, for name checks."""
    names = collections.defaultdict(set)
    for i in trace.device:
        e = trace.events[i]
        launch = trace.launches.get(e.get("args", {}).get("correlation"))
        if launch is None or launch_kind(trace.events[launch]["name"]) is None:
            continue
        ctx = trace.context(launch)
        if "graph_capture" in ctx["marks"]:
            continue
        names[ctx["op"]].add(e["name"])
    return names


def rank_of_pid(profile_dir):
    ranks = {}
    for path in glob.glob(os.path.join(profile_dir, "env", "*.json")):
        with open(path) as f:
            info = json.load(f)
        ranks[info["pid"]] = info["rank"]
    return ranks


def step_key(record):
    """("step", k) for scheduled steps, ("dummy", k) for idle-rank dummy forwards."""
    if "step" in record:
        return ("step", record["step"])
    if "dummy" in record:
        return ("dummy", record["dummy"])
    return None


def load_steps(profile_dir):
    steps = {}
    for path in glob.glob(os.path.join(profile_dir, "steps", "*.jsonl")):
        rank = os.path.basename(path)[: -len(".jsonl")]
        with open(path) as f:
            steps[rank] = {step_key(r): r for r in map(json.loads, f)}
    return steps


def load_copies(profile_dir, ranks):
    """rank -> logged CPU KV-offload copies (one memcpy each), in issue order.

    A log written before the rank's process groups existed is named pid<pid>.
    """
    copies = collections.defaultdict(list)
    for path in glob.glob(os.path.join(profile_dir, "copies", "*.jsonl")):
        tag = os.path.basename(path)[: -len(".jsonl")]
        if tag.startswith("pid") and tag[3:].isdigit():
            tag = ranks.get(int(tag[3:]), tag)
        with open(path) as f:
            copies[tag] += (r for r in map(json.loads, f) if r.get("blocks"))
    for logged in copies.values():
        logged.sort(key=lambda r: r["t_ns"])
    return copies


def unix_to_trace_us(trace, step_log):
    """Offset from unix microseconds to the trace clock, from the step markers."""
    deltas = []
    for e in trace.events:
        m = MARK.match(e["name"]) if e.get("cat") in CONTEXT_CATS else None
        key = (m.group(1), int(m.group(2))) if m else None
        if key in step_log and "t0_ns" in step_log[key]:
            deltas.append(e["ts"] - step_log[key]["t0_ns"] / 1e3)
    deltas.sort()
    return deltas[len(deltas) // 2] if deltas else None


CLOCK_NAMES = ("graphics_mhz", "sm_mhz", "mem_mhz", "video_mhz")


def normalize_uuid(uuid):
    text = str(uuid).lower()
    for prefix in ("gpu-", "mig-"):
        text = text.removeprefix(prefix)
    return text


def load_clocks(profile_dir):
    """rank -> (sample unix ns, [(graphics, sm, mem, video MHz, event reasons)]), time-ordered."""
    clock_dir = os.path.join(profile_dir, "clocks")
    try:
        with open(os.path.join(clock_dir, "gpus.json")) as f:
            gpu_of_uuid = {normalize_uuid(u): int(i) for i, u in json.load(f).items()}
    except (OSError, ValueError):
        return {}
    rank_of_gpu = {}
    for path in glob.glob(os.path.join(profile_dir, "env", "*.json")):
        with open(path) as f:
            info = json.load(f)
        gpu = gpu_of_uuid.get(normalize_uuid(info.get("device_uuid", "")))
        if gpu is not None:
            rank_of_gpu[gpu] = info["rank"]
    samples = collections.defaultdict(list)
    for path in glob.glob(os.path.join(clock_dir, "window*.csv")):
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) != 7:
                    continue  # a row cut short when the client stopped
                rank = rank_of_gpu.get(int(parts[1]))
                if rank is not None:
                    samples[rank].append((int(parts[0]), tuple(int(v) for v in parts[2:])))
    tracks = {}
    for rank, rows in samples.items():
        rows.sort()
        tracks[rank] = ([t for t, _ in rows], [v for _, v in rows])
    return tracks


def kernel_clocks(track, offset_us, ts_us, dur_us):
    """Clock ranges over a kernel's lifetime from its GPU's sample track, or None."""
    if track is None or offset_us is None:
        return None
    times, values = track
    start = (ts_us - offset_us) * 1e3
    first = bisect.bisect_right(times, start) - 1  # state in effect at the kernel's start
    last = bisect.bisect_right(times, start + dur_us * 1e3)
    if first < 0:
        return None
    span = values[first:last]
    out = {name: [min(v[k] for v in span), max(v[k] for v in span)]
           for k, name in enumerate(CLOCK_NAMES)}
    reasons = 0
    for v in span:
        if v[4] >= 0:
            reasons |= v[4]
    out.update(event_reasons=reasons, samples=last - first - 1,
               prior_us=round((start - times[first]) / 1e3, 1))
    return out


COPY_LAUNCHER = "vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend.launch_copy"
COPY_MAX_LAG_US = 60e6


def match_copies(trace, orphans, copies, offset_us):
    """device index -> logged copy for orphan memcpys, per direction.

    Each logged copy is one memcpy on that direction's stream, executed in
    issue order after its issue. Of the order-preserving pairings with every
    copy issued before its memcpy starts (and equal bytes), take the one with
    least total issue-to-start lag: copies logged before the window whose
    memcpys ran before it, and copies still queued at its end, stay unpaired.
    """
    matched = {}
    if offset_us is None:
        return matched
    for store, marker in ((True, "DtoH"), (False, "HtoD")):
        memcpys = sorted((i for i in orphans if marker in trace.events[i]["name"]),
                         key=lambda i: trace.events[i]["ts"])
        if not memcpys:
            continue
        # The log spans the run; a copy's memcpy starts within a minute of its issue.
        lo = trace.events[memcpys[0]]["ts"] - offset_us - COPY_MAX_LAG_US
        hi = trace.events[memcpys[-1]]["ts"] - offset_us
        logged = [c for c in copies if c["store"] == store and lo <= c["t_ns"] / 1e3 <= hi]
        n, m = len(memcpys), len(logged)
        if not n or not m:
            continue
        inf = float("inf")
        cost = [[0.0] * (m + 1)] + [[inf] * (m + 1) for _ in range(n)]
        take = [[False] * (m + 1) for _ in range(n + 1)]
        for a in range(1, n + 1):
            e = trace.events[memcpys[a - 1]]
            nbytes = e.get("args", {}).get("bytes")
            for j in range(1, m + 1):
                cost[a][j] = cost[a][j - 1]
                c = logged[j - 1]
                lag = e["ts"] - (c["t_ns"] / 1e3 + offset_us)
                if lag < 0 or (nbytes is not None and c.get("bytes") is not None
                               and nbytes != c["bytes"]):
                    continue
                if cost[a - 1][j - 1] + lag < cost[a][j]:
                    cost[a][j] = cost[a - 1][j - 1] + lag
                    take[a][j] = True
        if cost[n][m] == inf:
            continue  # more memcpys than pairable copies: leave them unattributed
        a, j = n, m
        while a:
            if take[a][j]:
                matched[memcpys[a - 1]] = logged[j - 1]
                a -= 1
            j -= 1
    return matched


def extract_replay(trace, captured, rank, window, step_log, copies, clocks, out_dir, pairs):
    """Attribute every device activity of one replay trace; write one file per step.

    Returns the trace's report entry and its index entries. `pairs` accumulates
    (op, kernel) pairs by source for the cross-trace kernel-name check.
    """
    rows = []
    graph_replays = collections.defaultdict(list)  # (graph, launch) -> device indices
    orphans = []  # device activities with no CPU launch
    for i in trace.device:
        e = trace.events[i]
        args = e.get("args", {})
        launch = trace.launches.get(args.get("correlation"))
        if launch is None:
            orphans.append(i)
            continue
        ctx = trace.context(launch)
        if "graph node id" in args and "Graph" in trace.events[launch]["name"]:
            graph_replays[(ctx["marks"].get("graph_replay"), launch)].append(i)
            continue
        rows.append({"source": "eager", "device_index": i, **ctx})
        pairs["eager"][(ctx["op"], e["name"])] += 1

    offset_us = unix_to_trace_us(trace, step_log)
    offload = match_copies(trace, orphans, copies, offset_us)
    unattributed = 0
    for i in orphans:
        copy = offload.get(i)
        if copy is None:
            unattributed += 1
            rows.append({"source": "unattributed", "device_index": i})
            continue
        step = copy.get("step")
        rows.append({
            "source": "offload_copy", "device_index": i,
            "marks": {"step": step} if step is not None else {},
            "launch_api": "cuMemcpyBatchAsync", "launcher": COPY_LAUNCHER,
            "launcher_callers": copy.get("callers"), "op": None, "module_path": [],
            "copy": {"store": copy["store"], "blocks": copy["blocks"], "bytes": copy.get("bytes"),
                     "issue_lag_us": trace.events[i]["ts"] - (copy["t_ns"] / 1e3 + offset_us)},
        })

    graph_checks = collections.Counter()
    for (gid, launch), items in graph_replays.items():
        replay_ctx = trace.context(launch)
        items.sort(key=lambda i: trace.events[i]["args"]["graph node id"] & NODE_ID_MASK)
        launches = captured.get(gid) if gid is not None else None
        if launches is None or len(launches) != len(items):
            graph_checks["count_mismatch"] += 1
            for pos, i in enumerate(items):
                rows.append({"source": "graph", "graph": gid, "node_pos": pos, "device_index": i,
                             "marks": replay_ctx["marks"], "capture_join": "count_mismatch"})
            continue
        graph_checks["joined"] += 1
        for pos, (i, cap) in enumerate(zip(items, launches)):
            rows.append({"source": "graph", "graph": gid, "node_pos": pos, "device_index": i,
                         **cap, "marks": replay_ctx["marks"]})
            pairs["graph"][(cap["op"], trace.events[i]["name"])] += 1

    by_step = collections.defaultdict(list)
    for row in rows:
        e = trace.events[row.pop("device_index")]
        args = e.get("args", {})
        row.update(
            rank=rank, window=window, kernel=e["name"], cat=e["cat"],
            stream=args.get("stream", e.get("tid")), device=args.get("device", e.get("pid")),
            ts_us=e["ts"], dur_us=e.get("dur", 0), graph_node_id=args.get("graph node id"),
            grid=args.get("grid"), block=args.get("block"),
            clocks=kernel_clocks(clocks, offset_us, e["ts"], e.get("dur", 0)),
        )
        key = step_key(row.get("marks") or {})
        row["step"] = list(key) if key else None
        by_step[key].append(row)

    rank_dir = os.path.join(out_dir, f"window{window}", str(rank))
    os.makedirs(rank_dir, exist_ok=True)
    index = []
    for key in sorted(by_step, key=lambda k: (k is None, k or ("", 0))):
        kernels = sorted(by_step[key], key=lambda r: r["ts_us"])
        t0 = kernels[0]["ts_us"]
        t1 = max(r["ts_us"] + r["dur_us"] for r in kernels)
        summary = {
            "kernels": len(kernels), "busy_us": sum(r["dur_us"] for r in kernels),
            "t0_us": t0, "t1_us": t1, "span_us": t1 - t0,
            "sources": dict(collections.Counter(r["source"] for r in kernels)),
        }
        name = f"{key[0]}{key[1]:06d}.json.gz" if key else "unstepped.json.gz"
        batch = step_log.get(key) if key else None
        with gzip.open(os.path.join(rank_dir, name), "wt") as f:
            json.dump({"rank": rank, "window": window, "kind": key[0] if key else None,
                       "step": key[1] if key else None, "summary": summary, "batch": batch,
                       "kernels": kernels}, f, separators=(",", ":"))
        reqs = (batch or {}).get("reqs") or []
        index.append({
            "window": window, "rank": rank, "kind": key[0] if key else None,
            "step": key[1] if key else None,
            "file": os.path.relpath(os.path.join(rank_dir, name), out_dir),
            **summary, "total_tokens": (batch or {}).get("total_tokens"), "num_reqs": len(reqs),
            "cudagraph": [d.get("cg_mode") for d in (batch or {}).get("dispatch") or []],
        })
    entry = {
        "trace": os.path.basename(trace.path), "rank": rank, "window": window,
        "device_activities": len(rows), "unattributed": unattributed,
        "graph_replays": dict(graph_checks),
        "sources": dict(collections.Counter(r["source"] for r in rows)),
        "steps": sum(1 for k in by_step if k is not None),
        "clocks": clock_coverage(rows),
    }
    return entry, index


def clock_coverage(rows):
    """How many activities carry clocks, and how stale their start state is."""
    ages = sorted(r["clocks"]["prior_us"] for r in rows if r.get("clocks"))
    if not ages:
        return {"activities_with_clocks": 0}
    return {"activities_with_clocks": len(ages),
            "prior_us_p50": ages[len(ages) // 2], "prior_us_p99": ages[int(len(ages) * 0.99)],
            "prior_us_max": ages[-1]}


def trace_time(path):
    """The export timestamp vLLM puts in a trace's file name (<rank>.<ns>.pt.trace...)."""
    parts = os.path.basename(path).split(".")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def main():
    profile_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    report = {"traces": []}
    pairs = {"eager": collections.Counter(), "graph": collections.Counter()}
    captured = {}
    for path in sorted(glob.glob(os.path.join(profile_dir, "capture", "*.json*"))):
        trace = Trace(path)
        captured = capture_launches(trace)  # identical on every rank; the first suffices
        for op, names in eager_kernel_names(trace).items():
            for name in names:
                pairs["eager"][(op, name)] += 1
        report["capture"] = {"path": os.path.relpath(path, profile_dir), "graphs": len(captured),
                             "launches": sum(map(len, captured.values()))}
        del trace
        break
    ranks = rank_of_pid(profile_dir)
    steps = load_steps(profile_dir)
    copies = load_copies(profile_dir, ranks)
    clocks = load_clocks(profile_dir)
    traces = sorted(glob.glob(os.path.join(profile_dir, "torch", "**", "*.pt.trace.json*"),
                              recursive=True), key=trace_time)
    windows_seen = collections.Counter()
    index = []
    for path in traces:
        trace = Trace(path)
        rank = next((ranks[p] for p in trace.process_ids() if p in ranks), None)
        window = windows_seen[rank]
        windows_seen[rank] += 1
        entry, entries = extract_replay(trace, captured, rank, window, steps.get(rank, {}),
                                        copies.get(rank, []), clocks.get(rank), out_dir,
                                        pairs)
        report["traces"].append(entry)
        index += entries
        del trace
    # A graph-replayed kernel whose op never launched that kernel eagerly anywhere.
    eager_ops = {op for op, _ in pairs["eager"]}
    report["graph_kernels_unseen_eagerly"] = sorted(
        ([op, kernel, n] for (op, kernel), n in pairs["graph"].items()
         if op is not None and op in eager_ops and (op, kernel) not in pairs["eager"]),
        key=lambda item: -item[2])
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"capture": report.get("capture"), "steps": index}, f)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if traces and not sum(t["device_activities"] for t in report["traces"]):
        sys.exit(f"no device activity attributed across {len(traces)} traces")


if __name__ == "__main__":
    main()
