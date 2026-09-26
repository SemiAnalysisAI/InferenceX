import argparse
import collections
import concurrent.futures
import json
import re
from pathlib import Path

import orjson


def analyze(item: tuple[str, Path]) -> dict:
    label, path = item
    counts = collections.Counter()
    status = collections.Counter()
    transports = collections.Counter()
    sizes = {}
    peers = set()
    masks = bytearray()
    maxid = 0
    duplicates = 0
    first = None
    last = None
    bad_duration = 0
    footer = None
    header = None
    for line in path.open("rb", buffering=8 * 1024 * 1024):
        r = orjson.loads(line)
        event = r["event"]
        if event == "trace_start":
            header = r
            continue
        if event == "trace_end":
            footer = r
            continue
        counts[event] += 1
        rid = r["request_id"]
        maxid = max(maxid, rid)
        if rid >= len(masks):
            masks.extend(b"\0" * (rid + 65536 - len(masks)))
        bit = 1 if event == "submit" else 2
        if masks[rid] & bit:
            duplicates += 1
        masks[rid] |= bit
        peers.add((r["initiator"], r["peer"], r["operation"], r["scope"]))
        t = r["monotonic_ns"]
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        b = r["bytes"]
        if b not in sizes:
            sizes[b] = {
                "submit": 0,
                "complete": 0,
                "duration_ns_sum": 0,
                "duration_ns_max": 0,
                "first_ns": t,
                "last_ns": t,
            }
        g = sizes[b]
        g["first_ns"] = min(g["first_ns"], t)
        g["last_ns"] = max(g["last_ns"], t)
        if event in ("submit", "complete"):
            g[event] += 1
        if event == "complete":
            status[r["status"]] += 1
            transports[r["transport"]] += 1
            d = r["duration_ns"]
            bad_duration += int(d < 0)
            g["duration_ns_sum"] += d
            g["duration_ns_max"] = max(g["duration_ns_max"], d)
    mask_counts = collections.Counter(masks[1 : maxid + 1])
    result = {
        "label": label,
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "header": header,
        "footer": footer,
        "counts": dict(counts),
        "status": dict(status),
        "transports": dict(transports),
        "peers": sorted(peers),
        "max_request_id": maxid,
        "matched_pairs": mask_counts[3],
        "submit_only": mask_counts[1],
        "complete_only": mask_counts[2],
        "unobserved_ids_before_last": mask_counts[0],
        "duplicate_events": duplicates,
        "invalid_duration": bad_duration,
        "span_seconds": (last - first) / 1e9,
        "sizes": sizes,
    }
    assert footer and footer["written_records"] == sum(counts.values())
    print(label, "done", counts, "drops", footer["dropped_records"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream complete event files and audit retained request IDs."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    ROOT = args.run_dir.resolve()
    items = []
    for op in ["read", "write"]:
        for n in range(1, 4):
            label = f"{op}-{n}"
            match = re.search(
                r"TENT transfer events: (.+)", (ROOT / f"{label}.log").read_text()
            )
            items.append((label, args.events_dir / Path(match.group(1)).name))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, items))
    (ROOT / "trace-analysis.json").write_text(json.dumps(results, indent=2) + "\n")
