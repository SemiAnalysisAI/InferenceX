"""Attribute every device kernel of a profiled vLLM step to the op that launched it.

Usage: extract.py PROFILE_DIR OUT_DIR

PROFILE_DIR is an unpacked profile artifact (infx_profile/): torch/ holds the
engine's per-rank torch traces, capture/ the CUDA graph capture trace,
steps/ the per-step batch log and env/ the per-rank environment.

A kernel launched eagerly joins its CPU launch through the CUDA correlation
id. A kernel replayed from CUDA graph n carries a ``graph node id``; ordered
by node id, graph n's nodes pair with the launches recorded inside
``infx_graph_capture#n``, which carry the op, shapes and stack of the
capture. Capture ordinals follow capture order, which every rank shares.

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

MARK = re.compile(r"^infx_(step|graph_replay|graph_capture|piece)#(\d+)$")
MODULE_MARK = "infx_mod#"
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
CONTEXT_CATS = {"cpu_op", "user_annotation", "python_function"}
NODE_ID_MASK = (1 << 32) - 1


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


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
        data = load(path)
        self.events = data["traceEvents"]
        self.meta = {k: v for k, v in data.items() if k != "traceEvents"}
        self._index()

    def _index(self):
        by_thread = collections.defaultdict(list)
        self.launches = {}  # correlation -> launch event index
        self.device = []
        for i, e in enumerate(self.events):
            if e.get("ph") != "X":
                continue
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
        modules = []  # qualified names from infx_mod markers
        stack_modules = []  # class-instance names from Python stacks (capture only)
        frame = None
        for i in self.stack_of.get(launch, ()):
            e = self.events[i]
            name = e["name"]
            m = MARK.match(name)
            if m:
                marks[m.group(1)] = int(m.group(2))
            elif name.startswith(MODULE_MARK):
                modules.append(name[len(MODULE_MARK):])
            elif e["cat"] == "cpu_op" or (e["cat"] == "user_annotation" and not name.startswith("infx_")):
                ops.append(i)
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
            "input_dims": args.get("Input Dims"),
            "input_types": args.get("Input type"),
            "concrete_inputs": args.get("Concrete Inputs"),
            "kernel_file": args.get("kernel_file"),
            "module_path": modules or stack_modules,
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


def load_steps(profile_dir):
    steps = {}
    for path in glob.glob(os.path.join(profile_dir, "steps", "*.jsonl")):
        rank = os.path.basename(path)[: -len(".jsonl")]
        with open(path) as f:
            steps[rank] = {r["step"]: r for r in map(json.loads, f)}
    return steps


def extract_replay(trace, captured, known_names, rank, step_log, out_dir, report):
    """Write one row per device activity of a replay trace and per-step totals."""
    rows = []
    graph_replays = collections.defaultdict(list)  # (graph, launch) -> device indices
    unattributed = 0
    for i in trace.device:
        e = trace.events[i]
        args = e.get("args", {})
        launch = trace.launches.get(args.get("correlation"))
        if launch is None:
            unattributed += 1
            rows.append({"source": "unattributed", "device_index": i})
            continue
        if "graph node id" in args and "Graph" in trace.events[launch]["name"]:
            ctx = trace.context(launch)
            graph_replays[(ctx["marks"].get("graph_replay"), launch)].append(i)
            continue
        ctx = trace.context(launch)
        rows.append({"source": "eager", "device_index": i, **ctx})

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
            e = trace.events[i]
            name_ok = None
            if cap["op"] is not None and cap["op"] in known_names:
                name_ok = e["name"] in known_names[cap["op"]]
            elif cap.get("kernel_file") or (cap["op"] or "").startswith("triton_"):
                name_ok = e["name"] == cap["op"]
            graph_checks[f"name_{name_ok}"] += 1
            rows.append({"source": "graph", "graph": gid, "node_pos": pos, "device_index": i,
                         **cap, "marks": replay_ctx["marks"], "name_check": name_ok})

    # Device timing and step totals.
    per_step = collections.defaultdict(lambda: {"kernels": 0, "busy_us": 0.0, "t0": None, "t1": None})
    out_rows = []
    for row in rows:
        e = trace.events[row.pop("device_index")]
        args = e.get("args", {})
        row.update(
            rank=rank, kernel=e["name"], cat=e["cat"], stream=args.get("stream", e.get("tid")),
            device=args.get("device", e.get("pid")), ts_us=e["ts"], dur_us=e.get("dur", 0),
            graph_node_id=args.get("graph node id"), grid=args.get("grid"), block=args.get("block"),
        )
        step = (row.get("marks") or {}).get("step")
        row["step"] = step
        if step is not None:
            s = per_step[step]
            s["kernels"] += 1
            s["busy_us"] += row["dur_us"]
            s["t0"] = row["ts_us"] if s["t0"] is None else min(s["t0"], row["ts_us"])
            end = row["ts_us"] + row["dur_us"]
            s["t1"] = end if s["t1"] is None else max(s["t1"], end)
        out_rows.append(row)

    name = os.path.basename(trace.path).split(".pt.trace")[0]
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, f"{name}.kernels.jsonl.gz"), "wt") as f:
        for row in out_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with open(os.path.join(out_dir, f"{name}.steps.jsonl"), "w") as f:
        for step in sorted(per_step):
            s = per_step[step]
            log = step_log.get(step, {})
            f.write(json.dumps({
                "rank": rank, "step": step, "kernels": s["kernels"], "busy_us": s["busy_us"],
                "span_us": s["t1"] - s["t0"], "reqs": log.get("reqs"),
                "total_tokens": log.get("total_tokens"), "dispatch": log.get("dispatch"),
            }) + "\n")
    report[name] = {
        "rank": rank,
        "device_activities": len(out_rows),
        "unattributed": unattributed,
        "graph_replays": dict(graph_checks),
        "sources": dict(collections.Counter(r["source"] for r in out_rows)),
        "steps": len(per_step),
    }


def main():
    profile_dir, out_dir = sys.argv[1], sys.argv[2]
    report = {}
    capture_paths = sorted(glob.glob(os.path.join(profile_dir, "capture", "*.json*")))
    captured, known_names = {}, collections.defaultdict(set)
    for path in capture_paths:
        trace = Trace(path)
        captured = capture_launches(trace)  # identical on every rank; the first suffices
        for op, names in eager_kernel_names(trace).items():
            known_names[op] |= names
        report["capture"] = {
            "path": os.path.relpath(path, profile_dir),
            "graphs": len(captured),
            "launches": sum(map(len, captured.values())),
        }
        break
    ranks = rank_of_pid(profile_dir)
    steps = load_steps(profile_dir)
    for path in sorted(glob.glob(os.path.join(profile_dir, "torch", "**", "*.pt.trace.json*"), recursive=True)):
        trace = Trace(path)
        pids = trace.process_ids()
        rank = next((ranks[p] for p in pids if p in ranks), None)
        for op, names in eager_kernel_names(trace).items():
            known_names[op] |= names
        extract_replay(trace, captured, known_names, rank, steps.get(rank, {}), out_dir, report)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
