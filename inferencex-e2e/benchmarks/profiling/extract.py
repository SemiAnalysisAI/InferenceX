"""Attribute every device kernel of a profiled engine step to the op that launched it."""

import ast
import bisect
import collections
import glob
import gzip
import json
import os
import re
import shutil
import struct
import sys
import zipfile

MARK = re.compile(r"^infx_(step|dummy|graph_replay|graph_capture|piece|stream)#(\d+)$")
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
        while True:
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


def load_events(path, keep=None):
    names = {}
    return [Event(raw, names) for raw in iter_trace_events(path)
            if raw.get("ph") == "X" and (keep is None or keep(raw))]


def replayed_graphs(paths):
    """Graph ordinals the replay traces replay (their infx_graph_replay#n markers)."""
    graphs = set()
    for path in paths:
        for raw in iter_trace_events(path):
            m = MARK.match(raw.get("name", "")) if raw.get("ph") == "X" else None
            if m and m.group(1) == "graph_replay":
                graphs.add(int(m.group(2)))
    return graphs


def capture_ranges(path, graphs):
    """Sorted, merged (start, end) of the host-side infx_graph_capture#n ranges for the given ordinals."""
    ranges = []
    for raw in iter_trace_events(path):
        m = MARK.match(raw.get("name", "")) if raw.get("ph") == "X" else None
        if (m and m.group(1) == "graph_capture" and int(m.group(2)) in graphs
                and raw.get("cat") != "gpu_user_annotation"):
            ranges.append((raw["ts"], raw["ts"] + raw.get("dur", 0)))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlapping(ranges):
    """Predicate: a raw event overlaps one of the sorted, disjoint ranges."""
    starts = [a for a, _ in ranges]

    def keep(raw):
        ts = raw.get("ts", 0)
        end = ts + raw.get("dur", 0)
        k = bisect.bisect_right(starts, end) - 1
        return k >= 0 and ranges[k][1] >= ts
    return keep


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

    def __init__(self, path, keep=None):
        self.path = path
        self.events = load_events(path, keep)
        self._index()

    def _index(self):
        by_thread = collections.defaultdict(list)
        self.launches = {}
        self.device = []
        self.rocm = False
        for i, e in enumerate(self.events):
            cat = e.get("cat")
            if cat in CONTEXT_CATS:
                by_thread[(e["pid"], e["tid"])].append(i)
            elif cat in LAUNCH_CATS:
                self.rocm = self.rocm or e.get("name", "").startswith("hip")
                by_thread[(e["pid"], e["tid"])].append(i)
                corr = e.get("args", {}).get("correlation")
                if corr is not None:
                    self.launches[corr] = i
            elif cat in DEVICE_CATS:
                self.device.append(i)
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
        modules = []
        stack_modules = []
        frame = None
        module_stack = []
        launcher = None
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
                annotations.append(name)
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
    executed = {trace.events[i].get("args", {}).get("correlation") for i in trace.device}
    graphs = collections.defaultdict(list)
    order = sorted(
        (i for i in trace.stack_of if launch_kind(trace.events[i]["name"])),
        key=lambda i: trace.events[i]["ts"],
    )
    for i in order:
        if trace.events[i].get("args", {}).get("correlation") in executed:
            continue
        ctx = trace.context(i)
        gid = ctx["marks"].get("graph_capture")
        if gid is not None:
            ctx["kind"] = launch_kind(trace.events[i]["name"])
            graphs[gid].append(ctx)
    return graphs


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


def read_npz(path):
    """name -> (shape, flat values) for an .npz of little-endian int arrays (stdlib only)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            data = z.read(member)
            assert data[:6] == b"\x93NUMPY", member
            major = data[6]
            header_len = struct.unpack("<H" if major == 1 else "<I",
                                       data[8:10] if major == 1 else data[8:12])[0]
            start = (10 if major == 1 else 12) + header_len
            header = ast.literal_eval(data[(10 if major == 1 else 12):start].decode("latin1"))
            code = {"<i2": "h", "<i4": "i", "<i8": "q", "|i1": "b"}[header["descr"]]
            shape = tuple(header["shape"])
            count = 1
            for dim in shape:
                count *= dim
            values = struct.unpack(f"<{count}{code}", data[start:start + count * struct.calcsize(code)])
            out[member[: -len(".npy")]] = (shape, values)
    return out


def load_routing(profile_dir):
    """rank -> (routing dir, meta) for ranks whose MoE routing was captured."""
    found = {}
    for meta_path in glob.glob(os.path.join(profile_dir, "routing", "*", "meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        found[meta["rank"]] = (os.path.dirname(meta_path), meta)
    for rank, (path, meta) in found.items():
        if meta.get("token_shards"):
            group = re.sub(r"_tp\d+$", "", rank)
            meta["_shard_paths"] = sorted(p for r, (p, m) in found.items()
                                          if m.get("token_shards") and re.sub(r"_tp\d+$", "", r) == group)
    routing = {}
    for rank, (path, meta) in found.items():
        source = rank if meta.get("write") else re.sub(r"_tp\d+$", "_tp0", rank)
        if source in found and found[source][1].get("write"):
            routing[rank] = (source, *found[source])
    return routing


def step_routing(routing, step, out_dir):
    """A step's per-layer expert token counts; copies its routing file into out_dir."""
    if routing is None or step is None:
        return None
    source, path, meta = routing
    npz = os.path.join(path, f"step{step:06d}.npz")
    if not os.path.exists(npz):
        return None
    arrays = read_npz(npz)
    (layers, experts), counts = arrays["counts"]
    tokens = arrays["tokens"][1][0]
    valid = arrays["valid_rows"][1][0] if "valid_rows" in arrays else tokens
    shards = 1
    if meta.get("_shard_paths") and valid < tokens:
        counts = list(counts)
        for other in meta["_shard_paths"]:
            other_npz = os.path.join(other, f"step{step:06d}.npz")
            if other == path or not os.path.exists(other_npz):
                continue
            (_, other_experts), other_counts = read_npz(other_npz)["counts"]
            if other_experts == experts:
                counts = [a + b for a, b in zip(counts, other_counts)]
                shards += 1
    names = meta.get("bound", {})
    per_layer = {}
    for layer in range(layers):
        row = counts[layer * experts:(layer + 1) * experts]
        if any(row):
            per_layer[names.get(str(layer), str(layer))] = [[e, n] for e, n in enumerate(row) if n]
    dest = os.path.join(out_dir, "routing", source, os.path.basename(npz))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        shutil.copyfile(npz, dest)
    return {"source_rank": source, "tokens": tokens, "token_shards": shards, "topk": meta.get("topk"),
            "num_experts": meta.get("num_experts"), "has_ids": "ids" in arrays,
            "file": os.path.relpath(dest, out_dir), "expert_tokens": per_layer}


def load_copies(profile_dir, ranks):
    """rank -> logged CPU KV-offload copies (one memcpy each), in issue order."""
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
POWER, ENERGY = 5, 6
POWER_SCHEMA = 1
SAMPLE_GAP_PERIODS = 5


class ClockTrack(collections.namedtuple("ClockTrack", "starts ends values power energy_t energy_mj "
                                                      "sample_t sample_mw counter", defaults=((), (), ()))):
    """One GPU's polls in time order. power: positions of polls with a power reading; energy_t/energy_mj: when the energy counter changed value (between the poll that saw it and the one before) and to what. sample_t/sample_mw: the driver's power samples (NVIDIA); counter: [(t_ns, mJ)] energy counter reads at window edges."""

    @property
    def energy_period_ns(self):
        gaps = sorted(b - a for a, b in zip(self.energy_t, self.energy_t[1:]))
        return gaps[len(gaps) // 2] if gaps else None

    def energy_at(self, t_ns):
        """The energy counter at t_ns, linear between its updates, or None outside them."""
        k = bisect.bisect_right(self.energy_t, t_ns)
        if k == 0 or k == len(self.energy_t):
            return None
        t0, t1 = self.energy_t[k - 1], self.energy_t[k]
        v0, v1 = self.energy_mj[k - 1], self.energy_mj[k]
        return v0 + (v1 - v0) * (t_ns - t0) / (t1 - t0)

    @property
    def sample_period_ns(self):
        gaps = sorted(b - a for a, b in zip(self.sample_t, self.sample_t[1:]))
        return gaps[len(gaps) // 2] if gaps else None

    def sample_energy_mj(self, t0, t1, hold_ns=0):
        """Integral of the power samples (linear between them) over [t0, t1] ns, or None outside them or across a gap in them (between windows, or samples lost). hold_ns: up to this much before the first or after the last sample counts at that sample's power."""
        ts, mw = self.sample_t, self.sample_mw
        if not ts or t1 <= t0:
            return None
        period = self.sample_period_ns

        def gap(a, b):
            return a < 0 or b >= len(ts) or not period or ts[b] - ts[a] > SAMPLE_GAP_PERIODS * period
        edges = 0.0
        k = bisect.bisect_left(ts, t0)
        if k < len(ts) and t0 < ts[k] <= t0 + hold_ns and gap(k - 1, k):
            edges, t0 = edges + (ts[k] - t0) * mw[k] / 1e9, ts[k]
        j = bisect.bisect_right(ts, t1) - 1
        if j >= 0 and t1 - hold_ns <= ts[j] < t1 and gap(j, j + 1):
            edges, t1 = edges + (t1 - ts[j]) * mw[j] / 1e9, ts[j]
        if t0 < ts[0] or t1 > ts[-1] or t1 <= t0:
            return None
        first, end = max(bisect.bisect_right(ts, t0) - 1, 0), bisect.bisect_left(ts, t1)
        if period and any(ts[k + 1] - ts[k] > SAMPLE_GAP_PERIODS * period for k in range(first, min(end, len(ts) - 1))):
            return None

        def at(t):
            k = min(max(bisect.bisect_right(ts, t), 1), len(ts) - 1)
            a, b = ts[k - 1], ts[k]
            return mw[k - 1] + (mw[k] - mw[k - 1]) * (t - a) / (b - a) if b > a else mw[k]
        lo, hi = bisect.bisect_right(ts, t0), bisect.bisect_left(ts, t1)
        points = [(t0, at(t0))] + [(ts[k], mw[k]) for k in range(lo, hi)] + [(t1, at(t1))]
        return edges + sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:])) / 1e9


def clock_track(rows, power_rows=()):
    """power_rows: (t_ns, kind, value) from power<w>.csv: driver power samples and energy counter reads."""
    rows.sort()
    starts, ends, values = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    power = [k for k, v in enumerate(values) if len(v) > POWER and v[POWER] >= 0]
    energy_t, energy_mj, last = [], [], None
    for k, v in enumerate(values):
        if len(v) <= ENERGY or v[ENERGY] < 0:
            continue
        mid = (starts[k] + ends[k]) // 2
        if last is None or v[ENERGY] != last[1]:
            energy_t.append((last[0] + mid) // 2 if last else mid)
            energy_mj.append(v[ENERGY])
        last = (mid, v[ENERGY])
    samples = sorted({t: v for t, kind, v in power_rows if kind == "sample_mw"}.items())
    counter = sorted((t, v) for t, kind, v in power_rows if kind == "energy_mj")
    return ClockTrack(starts, ends, values, power, energy_t, energy_mj,
                      [t for t, _ in samples], [v for _, v in samples], counter)


def normalize_uuid(uuid):
    text = str(uuid).lower()
    for prefix in ("gpu-", "mig-"):
        text = text.removeprefix(prefix)
    return text


def load_clocks(profile_dir):
    """rank -> ClockTrack of polls (graphics, sm, mem, video MHz, event reasons[, power mW, energy mJ]; -1 where a poll has no reading)."""
    clock_dir = os.path.join(profile_dir, "clocks")
    try:
        with open(os.path.join(clock_dir, "gpus.json")) as f:
            gpus = json.load(f)
    except (OSError, ValueError):
        return {}
    gpu_of = {}
    for i, info in gpus.items():
        info = info if isinstance(info, dict) else {"uuid": info}
        for key in (info.get("uuid"), info.get("bdf")):
            if key:
                gpu_of[normalize_uuid(key)] = int(i)
    rank_of_gpu = {}
    for path in glob.glob(os.path.join(profile_dir, "env", "*.json")):
        with open(path) as f:
            info = json.load(f)
        gpu = next((gpu_of[k] for k in (normalize_uuid(info.get("device_uuid", "")),
                                        normalize_uuid(info.get("device_bdf", ""))) if k in gpu_of), None)
        if gpu is not None:
            rank_of_gpu[gpu] = info["rank"]
    samples = collections.defaultdict(list)
    for path in glob.glob(os.path.join(clock_dir, "window*.csv")):
        with open(path) as f:
            header = next(f, "").rstrip("\n").split(",")
            stamps = 2 if header[:2] == ["t0_ns", "t1_ns"] else 1
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) != len(header):
                    continue
                rank = rank_of_gpu.get(int(parts[stamps]))
                if rank is not None:
                    samples[rank].append((int(parts[0]), int(parts[stamps - 1]),
                                          tuple(int(v) for v in parts[stamps + 1:])))
    power = collections.defaultdict(list)
    for path in glob.glob(os.path.join(clock_dir, "power*.csv")):
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) == 4 and rank_of_gpu.get(int(parts[1])) is not None:
                    power[rank_of_gpu[int(parts[1])]].append((int(parts[0]), parts[2], float(parts[3])))
    return {rank: clock_track(rows, power.get(rank, ())) for rank, rows in samples.items()}


def kernel_clocks(track, offset_us, ts_us, dur_us):
    """Clock ranges over a kernel's lifetime from its GPU's sample track, or None."""
    if track is None or offset_us is None:
        return None
    starts, ends, values = track.starts, track.ends, track.values
    start = (ts_us - offset_us) * 1e3
    first = bisect.bisect_right(ends, start) - 1
    last = bisect.bisect_left(starts, start + dur_us * 1e3)
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
               prior_us=round((start - ends[first]) / 1e3, 1))
    if track.sample_t:
        out.update(nearest_readings(track.sample_t, [v / 1e3 for v in track.sample_mw], start, start + dur_us * 1e3))
    elif track.power:
        out.update(kernel_power(track, start, start + dur_us * 1e3))
    return out


def nearest_readings(times, watts, start_ns, end_ns):
    """Readings stamped during [start, end], else the nearest one: power_w [min, max], power_samples, power_nearest_us (0 when there were any)."""
    lo, hi = bisect.bisect_left(times, start_ns), bisect.bisect_right(times, end_ns)
    if hi > lo:
        span = watts[lo:hi]
        return {"power_w": [min(span), max(span)], "power_samples": hi - lo, "power_nearest_us": 0.0}
    near = [(start_ns - times[lo - 1], lo - 1)] if lo else []
    near += [(times[lo] - end_ns, lo)] if lo < len(times) else []
    if not near:
        return {}
    gap, k = min(near)
    return {"power_w": [watts[k], watts[k]], "power_samples": 0, "power_nearest_us": round(gap / 1e3, 1)}


def kernel_power(track, start_ns, end_ns):
    """Power readings taken during a kernel, else the nearest one: power_w [min, max], power_samples (readings during it), power_nearest_us (0 when there were any, else how far the reading used is from the kernel)."""
    mids = [(track.starts[k] + track.ends[k]) / 2 for k in track.power]
    lo, hi = bisect.bisect_left(mids, start_ns), bisect.bisect_right(mids, end_ns)
    if hi > lo:
        watts = [track.values[k][POWER] / 1e3 for k in track.power[lo:hi]]
        return {"power_w": [min(watts), max(watts)], "power_samples": hi - lo, "power_nearest_us": 0.0}
    near = [(start_ns - mids[lo - 1], lo - 1)] if lo else []
    near += [(mids[lo] - end_ns, lo)] if lo < len(mids) else []
    gap, k = min(near)
    watts = track.values[track.power[k]][POWER] / 1e3
    return {"power_w": [watts, watts], "power_samples": 0, "power_nearest_us": round(gap / 1e3, 1)}


def step_power(track, offset_us, t0_us, t1_us):
    """A step's power and clocks from its GPU's polls over [t0, t1]: energy (counter interpolated at both ends), power and SM clock min/avg/max, and the fraction of polls with each event-reason bit set."""
    if track is None or offset_us is None:
        return None
    t0, t1 = (t0_us - offset_us) * 1e3, (t1_us - offset_us) * 1e3
    lo, hi = bisect.bisect_left(track.starts, t0), bisect.bisect_right(track.ends, t1)
    polls = track.values[lo:hi]
    out = {"polls": len(polls)}

    def stats(xs):
        return {"min": min(xs), "avg": round(sum(xs) / len(xs), 1), "max": max(xs), "samples": len(xs)}
    sm = [v[1] for v in polls if v[1] >= 0]
    if sm:
        out["sm_mhz"] = stats(sm)
    reasons = [v[4] for v in polls if v[4] >= 0]
    if reasons:
        bits = sorted({1 << b for r in reasons for b in range(r.bit_length()) if r >> b & 1})
        out["event_reasons"] = {hex(b): round(sum(1 for r in reasons if r & b) / len(reasons), 3) for b in bits}
    if track.sample_t:
        lo, hi = bisect.bisect_left(track.sample_t, t0), bisect.bisect_right(track.sample_t, t1)
        watts = [v / 1e3 for v in track.sample_mw[lo:hi]]
        mj, period, source = track.sample_energy_mj(t0, t1), track.sample_period_ns, "power_samples"
    else:
        watts = [v[POWER] / 1e3 for v in polls if len(v) > POWER and v[POWER] >= 0]
        e0, e1, period, source = track.energy_at(t0), track.energy_at(t1), track.energy_period_ns, "energy_counter"
        mj = e1 - e0 if e0 is not None and e1 is not None else None
    if watts:
        out["power_w"] = stats(watts)
    if mj is not None:
        j = mj / 1e3
        out["energy"] = {"j": round(j, 4), "avg_power_w": round(j / ((t1 - t0) / 1e9), 1) if t1 > t0 else None,
                         "source": source, "update_period_ms": round(period / 1e6, 2) if period else None,
                         "below_resolution": period is None or t1 - t0 < period}
    return out


def energy_check(track):
    """Each window's energy counter delta against the power samples' integral over the same reads."""
    out = []
    period = track.sample_period_ns or 0
    for (a, ea), (b, eb) in zip(track.counter, track.counter[1:]):
        integral = track.sample_energy_mj(a, b, hold_ns=2 * period)
        if integral is not None and b > a:
            out.append({"from_ns": a, "to_ns": b, "counter_j": round((eb - ea) / 1e3, 3),
                        "samples_j": round(integral / 1e3, 3)})
    return out


PROFILER_MARKERS = ("infx_routing",)
COPY_LAUNCHER = "vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend.launch_copy"
COPY_MAX_LAG_US = 60e6
COPY_MAX_SPLIT = 8


def match_copies(trace, orphans, copies, offset_us):
    """device index -> logged copy for orphan memcpys, per direction."""
    matched = {}
    if offset_us is None:
        return matched
    for store, marker in ((True, "DtoH"), (False, "HtoD")):
        memcpys = sorted((i for i in orphans if marker in trace.events[i]["name"]),
                         key=lambda i: trace.events[i]["ts"])
        if not memcpys:
            continue
        lo = trace.events[memcpys[0]]["ts"] - offset_us - COPY_MAX_LAG_US
        hi = trace.events[memcpys[-1]]["ts"] - offset_us
        logged = [c for c in copies if c["store"] == store and lo <= c["t_ns"] / 1e3 <= hi]
        n, m = len(memcpys), len(logged)
        if not n or not m:
            continue
        inf = float("inf")
        sizes = [trace.events[i].get("args", {}).get("bytes") for i in memcpys]
        cost = [[0.0] * (m + 1)] + [[inf] * (m + 1) for _ in range(n)]
        take = [[0] * (m + 1) for _ in range(n + 1)]
        for a in range(1, n + 1):
            for j in range(1, m + 1):
                cost[a][j] = cost[a][j - 1]
                c = logged[j - 1]
                for k in range(1, min(a, COPY_MAX_SPLIT) + 1):
                    first = trace.events[memcpys[a - k]]
                    lag = first["ts"] - (c["t_ns"] / 1e3 + offset_us)
                    if lag < 0:
                        break
                    part = sizes[a - k:a]
                    if k > 1 and (c.get("bytes") is None or None in part):
                        break
                    if c.get("bytes") is not None and None not in part and sum(part) != c["bytes"]:
                        if sum(part) > c["bytes"]:
                            break
                        continue
                    if cost[a - k][j - 1] + lag < cost[a][j]:
                        cost[a][j] = cost[a - k][j - 1] + lag
                        take[a][j] = k
        if cost[n][m] == inf:
            continue
        a, j = n, m
        while a:
            if take[a][j]:
                for i in memcpys[a - take[a][j]:a]:
                    matched[i] = logged[j - 1]
                a -= take[a][j]
            j -= 1
    return matched


def kernel_family(name):
    """A kernel name without template arguments, tile suffixes or C++ mangling."""
    if name.startswith("_Z"):
        i, parts = (3 if name.startswith("_ZN") else 2), []
        while i < len(name) and (name[i].isdigit() or name[i] == "L" and name[i + 1:i + 2].isdigit()):
            i += name[i] == "L"
            j = i
            while j < len(name) and name[j].isdigit():
                j += 1
            parts.append(name[j:j + int(name[i:j])])
            i = j + int(name[i:j])
        if parts:
            return "::".join(parts)
    return re.split(r"[<(]|_MT\d|_GROUP_|_BLOCK_|\.kd$", name.removeprefix("void "))[0]


def activity_kind(event):
    """kernel / memcpy / memset, counting HIP's blit kernels as the copies they are."""
    if event["cat"] != "kernel":
        return event["cat"].removeprefix("gpu_")
    if event["name"].startswith("__amd_rocclr_fill"):
        return "memset"
    return "memcpy" if event["name"].startswith("__amd_rocclr_copy") else "kernel"


def join_by_stream_marks(trace, items, launches):
    """Exact order for a node-id-less replay when the capture marked each launch's stream."""
    groups = collections.defaultdict(list)
    for p, launch in enumerate(launches):
        groups[launch["marks"].get("stream")].append(p)
    streams = collections.defaultdict(list)
    for i in sorted(items, key=lambda i: trace.events[i]["ts"]):
        streams[trace.events[i]["args"].get("stream")].append(i)
    if len(groups) != len(streams):
        return None
    order, used = [None] * len(launches), set()
    for positions in groups.values():
        kinds = [launches[p]["kind"] for p in positions]
        matches = [s for s, seq in streams.items() if s not in used and len(seq) == len(positions)
                   and [activity_kind(trace.events[i]) for i in seq] == kinds]
        if len(matches) != 1:
            return None
        used.add(matches[0])
        for p, i in zip(positions, streams[matches[0]]):
            order[p] = i
    return order


def align_streams(trace, items, launches, families, beam=64):
    """Order a node-id-less graph replay's activities as its captured launches."""
    streams = collections.defaultdict(list)
    for i in sorted(items, key=lambda i: trace.events[i]["ts"]):
        streams[trace.events[i]["args"].get("stream")].append(i)
    seqs = list(streams.values())

    def score(p, i):
        cap, e = launches[p], trace.events[i]
        if (cap["kind"] == "kernel") != (activity_kind(e) == "kernel"):
            return -2
        key, family = cap["op"] or cap["launcher"], kernel_family(e["name"])
        launcher = cap["launcher"] or ""
        if (family in families.get(key, ()) or family == key
                or launcher.startswith("triton:") and launcher[len("triton:"):] in e["name"]):
            return 2
        return 0

    states = [(0, 0, (0,) * len(seqs), None)]
    for p in range(len(launches)):
        best = {}
        for total, penalty, heads, chain in states:
            ready = sorted((trace.events[seqs[s][h]]["ts"], s) for s, h in enumerate(heads)
                           if h < len(seqs[s]))
            for rank, (_, s) in enumerate(ready):
                nxt = heads[:s] + (heads[s] + 1,) + heads[s + 1:]
                state = (total + score(p, seqs[s][heads[s]]), penalty - rank, nxt, (s, chain))
                if nxt not in best or state[:2] > best[nxt][:2]:
                    best[nxt] = state
        states = sorted(best.values(), key=lambda st: st[:2], reverse=True)[:beam]
    picks, chain = [], states[0][3]
    while chain:
        picks.append(chain[0])
        chain = chain[1]
    heads, order = [0] * len(seqs), []
    for s in reversed(picks):
        order.append(seqs[s][heads[s]])
        heads[s] += 1
    return order, sum(score(p, i) > 0 for p, i in enumerate(order))


CAPTURE_OFFSET_SEARCH = 32


def ordinal_offset(graph_replays, captured):
    """Graph ordinals count each process's captures, and a rank that captured more or fewer graphs earlier (warmup, a draft model) numbers the same graphs differently: B300 TRT-LLM TP2 rank 1 was 2 ahead of rank 0. For a rank using another rank's captures, the shift under which the most replays have their capture's launch count (0 on ties), and how many replays match under it."""
    counts = collections.Counter((gid, len(items)) for (gid, _), items in graph_replays.items() if gid is not None)

    def score(d):
        return sum(n for (gid, size), n in counts.items() if len(captured.get(gid - d) or ()) == size)
    return max(((score(d), -abs(d), d) for d in range(-CAPTURE_OFFSET_SEARCH, CAPTURE_OFFSET_SEARCH + 1)))[::2]


def extract_replay(trace, captured, rank, window, step_log, copies, clocks, routing, out_dir,
                   pairs, borrowed_capture=False):
    """Attribute every device activity of one replay trace; write one file per step."""
    rows = []
    graph_replays = collections.defaultdict(list)
    orphans = []
    for i in trace.device:
        e = trace.events[i]
        args = e.get("args", {})
        launch = trace.launches.get(args.get("correlation"))
        if launch is None:
            orphans.append(i)
            continue
        ctx = trace.context(launch)
        if "Graph" in trace.events[launch]["name"]:
            graph_replays[(ctx["marks"].get("graph_replay"), launch)].append(i)
            continue
        rows.append({"source": "eager", "device_index": i, **ctx})
        pairs["eager"][(ctx["op"], e["name"])] += 1
        pairs["families"][ctx["op"] or ctx["launcher"]].add(kernel_family(e["name"]))

    offset_us = unix_to_trace_us(trace, step_log)
    offload = match_copies(trace, orphans, copies, offset_us)
    orphan_set = set(orphans)
    first_launched = min((trace.events[i]["ts"] for i in trace.device if i not in orphan_set), default=None)
    unattributed = 0
    for i in orphans:
        copy = offload.get(i)
        if copy is None:
            e = trace.events[i]
            if first_launched is not None and e["ts"] + e.get("dur", 0) <= first_launched:
                rows.append({"source": "pre_window", "device_index": i})
                continue
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
    if borrowed_capture and graph_replays:
        matched, offset = ordinal_offset(graph_replays, captured)
        if offset:
            graph_replays = {((gid - offset if gid is not None else None), launch): items
                             for (gid, launch), items in graph_replays.items()}
            graph_checks["capture_ordinal_offset"] = offset
    missing = {gid for gid, _ in graph_replays if gid is not None and gid not in captured}
    if missing:
        in_window = {gid: ls for gid, ls in capture_launches(trace).items() if gid in missing}
        if in_window:
            captured = {**captured, **in_window}
            graph_checks["captured_in_window"] += len(in_window)
    replayed = {launch for _, launch in graph_replays}
    for launch in set(trace.launches.values()) - replayed:
        if "Graph" in trace.events[launch]["name"]:
            gid = trace.context(launch)["marks"].get("graph_replay")
            if gid is None or captured.get(gid):
                graph_checks["launches_without_device_work"] += 1
    interleavings = {}
    for (gid, launch), items in graph_replays.items():
        replay_ctx = trace.context(launch)
        launches = captured.get(gid) if gid is not None else None
        join = "count_mismatch" if launches is None or len(launches) != len(items) else None
        if join is None and all("graph node id" in trace.events[i]["args"] for i in items):
            items.sort(key=lambda i: trace.events[i]["args"]["graph node id"] & NODE_ID_MASK)
        elif join is None:
            streams = collections.defaultdict(list)
            for i in sorted(items, key=lambda i: trace.events[i]["ts"]):
                streams[trace.events[i]["args"].get("stream")].append(i)
            shape = (gid, tuple(sorted((str(s), tuple(trace.events[i]["name"] for i in seq))
                                       for s, seq in streams.items())))
            if shape not in interleavings:
                order, exact = join_by_stream_marks(trace, items, launches), True
                if order is None:
                    order, evidence = align_streams(trace, items, launches, pairs["families"])
                    exact = False
                    graph_checks["stream_order_aligned"] += 1
                    graph_checks["stream_order_evidence"] += evidence
                    graph_checks["stream_order_positions"] += len(items)
                interleavings[shape] = ([trace.events[i]["args"].get("stream") for i in order], exact)
            queues = {s: iter(seq) for s, seq in streams.items()}
            items = [next(queues[s]) for s in interleavings[shape][0]]
            graph_checks["joined_by_stream_marks" if interleavings[shape][1] else "joined_by_alignment"] += 1
        if join:
            graph_checks[join] += 1
            for pos, i in enumerate(items):
                rows.append({"source": "graph", "graph": gid, "node_pos": pos, "device_index": i,
                             "marks": replay_ctx["marks"], "capture_join": join})
            continue
        graph_checks["joined"] += 1
        for pos, (i, cap) in enumerate(zip(items, launches)):
            rows.append({"source": "graph", "graph": gid, "node_pos": pos, "device_index": i,
                         **cap, "marks": replay_ctx["marks"]})
            pairs["graph"][(cap["op"], trace.events[i]["name"])] += 1

    by_step = collections.defaultdict(list)
    next_on_stream = stream_successors(trace)
    for row in rows:
        index = row.pop("device_index")
        e = trace.events[index]
        args = e.get("args", {})
        dur = e.get("dur", 0)
        nxt = next_on_stream.get(index)
        bounded = e.get("cat") in ("gpu_memcpy", "gpu_memset") or (trace.rocm and e.get("cat") == "kernel")
        if nxt is not None and e["ts"] + dur > nxt and bounded and row.get("source") != "graph":
            row["dur_us_recorded"], dur = dur, max(0.0, nxt - e["ts"])
        row.update(
            rank=rank, window=window, kernel=e["name"], cat=e["cat"],
            stream=args.get("stream", e.get("tid")), device=args.get("device", e.get("pid")),
            ts_us=e["ts"], dur_us=dur, graph_node_id=args.get("graph node id"),
            grid=args.get("grid"), block=args.get("block"),
            clocks=kernel_clocks(clocks, offset_us, e["ts"], dur),
        )
        if (any(a.startswith(PROFILER_MARKERS) for a in row.get("annotations") or ())
                or (row.get("launcher") or "").startswith("infx_prof.")):
            row["profiler"] = True
        key = step_key(row.get("marks") or {})
        row["step"] = list(key) if key else None
        by_step[key].append(row)
    placed = place_by_device_time(by_step)

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
        power = step_power(clocks, offset_us, t0, t1)
        name = f"{key[0]}{key[1]:06d}.json.gz" if key else "unstepped.json.gz"
        batch = step_log.get(key) if key else None
        moe = step_routing(routing, key[1], out_dir) if key and key[0] == "step" else None
        with gzip.open(os.path.join(rank_dir, name), "wt") as f:
            json.dump({"rank": rank, "window": window, "kind": key[0] if key else None,
                       "step": key[1] if key else None, "summary": summary, "power": power, "batch": batch,
                       "routing": moe, "kernels": kernels}, f, separators=(",", ":"))
        reqs = (batch or {}).get("reqs") or []
        index.append({
            "window": window, "rank": rank, "kind": key[0] if key else None,
            "step": key[1] if key else None,
            "file": os.path.relpath(os.path.join(rank_dir, name), out_dir),
            **summary, "total_tokens": (batch or {}).get("total_tokens"), "num_reqs": len(reqs),
            "graph": (batch or {}).get("graph"), "phase": (batch or {}).get("phase"),
            "cudagraph": [d.get("cg_mode") for d in (batch or {}).get("dispatch") or []],
            "routing": bool(moe),
            "energy_j": ((power or {}).get("energy") or {}).get("j"),
            "avg_power_w": ((power or {}).get("power_w") or {}).get("avg"),
        })
    entry = {
        "trace": os.path.basename(trace.path), "rank": rank, "window": window,
        "device_activities": len(rows), "unattributed": unattributed, "unattributed_in_steps": placed,
        "graph_replays": dict(graph_checks),
        "sources": dict(collections.Counter(r["source"] for r in rows)),
        "steps": sum(1 for k in by_step if k is not None),
        "clocks": clock_coverage(rows),
        "steps_with_routing": sum(1 for e in index if e["routing"]),
    }
    return entry, index


def place_by_device_time(by_step):
    """Device work with no CPU launch (copies a library issues through driver calls Kineto does not record, e.g. Muse Spark's copy-engine all-gathers under layernorm sequence parallelism: 64 MiB peer copies) goes in the step whose device span on this rank contains it, marked step_inferred; outside every step it stays unstepped."""
    loose = by_step.get(None) or []
    spans = sorted((min(r["ts_us"] for r in rows), max(r["ts_us"] + r["dur_us"] for r in rows), key)
                   for key, rows in by_step.items() if key is not None and rows)
    if not loose or not spans:
        return 0
    starts = [s for s, _, _ in spans]
    kept = []
    for row in loose:
        k = bisect.bisect_right(starts, row["ts_us"]) - 1
        if row["source"] == "unattributed" and k >= 0 and row["ts_us"] + row["dur_us"] <= spans[k][1]:
            key = spans[k][2]
            row["step"], row["step_inferred"] = list(key), "device_time"
            by_step[key].append(row)
        else:
            kept.append(row)
    by_step[None] = kept
    if not kept:
        del by_step[None]
    return len(loose) - len(kept)


def stream_successors(trace):
    """device activity index -> start of the next activity on its device and stream."""
    streams = collections.defaultdict(list)
    for i in trace.device:
        e = trace.events[i]
        args = e.get("args", {})
        streams[(args.get("device", e.get("pid")), args.get("stream", e.get("tid")))].append(i)
    out = {}
    for seq in streams.values():
        seq.sort(key=lambda i: trace.events[i]["ts"])
        for a, b in zip(seq, seq[1:]):
            out[a] = trace.events[b]["ts"]
    return out


def telemetry_rates(profile_dir):
    """How often each telemetry reading changed, pooled over GPUs (and the host), from telemetry<w>.csv: rows are kept only on a change, so under load a reading's row spacing is its update period (a slow-moving reading, such as a temperature, changes less often than it updates). Driver sample buffers (samples.*) keep every sample: their spacing is the driver's sampling period. stamp_ms: the same spacing by the source's own stamps."""
    gaps, stamps = collections.defaultdict(list), collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(profile_dir, "clocks", "telemetry*.csv"))):
        last = {}
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) != 7:
                    continue
                t0, t1, device, source, scope, stamp = parts[:6]
                key, pooled = (device, source, scope), ("host" if device == "host" else "gpu", source)
                mid, stamp = (int(t0) + int(t1)) / 2, int(stamp) if stamp else None
                if key in last:
                    prev_mid, prev_stamp = last[key]
                    if not source.startswith("samples."):
                        gaps[pooled].append((mid - prev_mid) / 1e6)
                    if stamp is not None and prev_stamp is not None and stamp != prev_stamp:
                        stamps[pooled].append((stamp - prev_stamp) / 1e3)
                last[key] = (mid, stamp)

    def pct(xs, q):
        return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3)
    out = {}
    for pooled in sorted(set(gaps) | set(stamps)):
        entry = {}
        for name, xs in (("change_ms", sorted(gaps.get(pooled, []))), ("stamp_ms", sorted(stamps.get(pooled, [])))):
            if xs:
                entry[name] = {"n": len(xs), "p10": pct(xs, 0.1), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9)}
        out[f"{pooled[0]}:{pooled[1]}"] = entry
    return out


def clock_coverage(rows):
    """How many activities carry clocks, and how stale their start state is."""
    ages = sorted(r["clocks"]["prior_us"] for r in rows if r.get("clocks"))
    if not ages:
        return {"activities_with_clocks": 0}
    return {"activities_with_clocks": len(ages),
            "prior_us_p50": ages[len(ages) // 2], "prior_us_p99": ages[int(len(ages) * 0.99)],
            "prior_us_max": ages[-1]}


def trace_time(path):
    """The export timestamp in a window trace's file name (<rank>.<ns>.pt.trace...)."""
    parts = os.path.basename(path).split(".")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def main():
    profile_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    report = {"traces": []}
    pairs = {"eager": collections.Counter(), "graph": collections.Counter(),
             "families": collections.defaultdict(set)}
    traces = sorted(glob.glob(os.path.join(profile_dir, "torch", "**", "*.pt.trace.json*"),
                              recursive=True), key=trace_time)
    needed = replayed_graphs(traces)
    sessions = collections.defaultdict(list)
    for path in glob.glob(os.path.join(profile_dir, "capture", "*.json*")):
        name = os.path.basename(path).split(".")
        session = int(name[1]) if len(name) > 1 and name[1].isdigit() else 0
        sessions[name[0]].append((session, path))
    captures = {}
    for capture_rank in sorted(sessions):
        captured = {}
        paths = [path for _, path in sorted(sessions[capture_rank])]
        for path in paths:
            ranges = capture_ranges(path, needed)
            trace = Trace(path, overlapping(ranges)) if ranges else None
            if trace:
                captured.update(capture_launches(trace))
            del trace
        captures[capture_rank] = captured
        report.setdefault("capture", {})[capture_rank] = {
            "paths": [os.path.relpath(p, profile_dir) for p in paths], "graphs": len(captured),
            "graphs_replayed": len(needed), "launches": sum(map(len, captured.values()))}
    default_capture = captures[min(captures)] if captures else {}
    ranks = rank_of_pid(profile_dir)
    steps = load_steps(profile_dir)
    copies = load_copies(profile_dir, ranks)
    clocks = load_clocks(profile_dir)
    routing = load_routing(profile_dir)
    report["telemetry_rates"] = telemetry_rates(profile_dir)
    report["energy_check"] = {rank: check for rank, track in sorted(clocks.items()) if (check := energy_check(track))}
    try:
        with open(os.path.join(profile_dir, "clocks", "rates.json")) as f:
            report["rate_probe"] = json.load(f)
    except (OSError, ValueError):
        pass
    report["routing"] = {rank: {"bound_layers": len(meta.get("bound", {})), "failed": meta.get("failed")}
                         for rank, (_, _, meta) in sorted(routing.items())}
    windows_seen = collections.Counter()
    index = []
    for path in traces:
        trace = Trace(path)
        rank = next((ranks[p] for p in trace.process_ids() if p in ranks), None)
        window = windows_seen[rank]
        windows_seen[rank] += 1
        entry, entries = extract_replay(trace, captures.get(rank, default_capture), rank, window,
                                        steps.get(rank, {}),
                                        copies.get(rank, []), clocks.get(rank),
                                        routing.get(rank), out_dir, pairs,
                                        borrowed_capture=rank not in captures)
        report["traces"].append(entry)
        index += entries
        del trace
    eager_ops = {op for op, _ in pairs["eager"]}
    report["graph_kernels_unseen_eagerly"] = sorted(
        ([op, kernel, n] for (op, kernel), n in pairs["graph"].items()
         if op is not None and op in eager_ops and (op, kernel) not in pairs["eager"]),
        key=lambda item: -item[2])
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        power_schema = POWER_SCHEMA if any(t.power or t.sample_t for t in clocks.values()) else None
        json.dump({"capture": report.get("capture"), "power_schema": power_schema, "steps": index}, f)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if traces and not sum(t["device_activities"] for t in report["traces"]):
        sys.exit(f"no device activity attributed across {len(traces)} traces")


if __name__ == "__main__":
    main()
