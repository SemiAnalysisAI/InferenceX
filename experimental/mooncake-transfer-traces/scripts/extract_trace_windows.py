import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Extract matched request windows from sequential single-thread traces."
)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--events-dir", type=Path, required=True)
args = parser.parse_args()
R = args.run_dir.resolve()
traces = json.loads((R / "trace-analysis.json").read_text())


def window(path: Path, target: int, size: int) -> tuple[list, float]:
    with path.open("rb") as f:
        lo = 0
        hi = path.stat().st_size
        while hi - lo > 16384:
            mid = (lo + hi) // 2
            f.seek(mid)
            f.readline()
            line = f.readline()
            if not line:
                hi = mid
                continue
            r = json.loads(line)
            if "monotonic_ns" not in r:
                hi = mid
                continue
            if r["monotonic_ns"] < target:
                lo = f.tell()
            else:
                hi = mid
        f.seek(lo)
        if lo:
            f.readline()
        pending = {}
        result = []
        for line in f:
            r = json.loads(line)
            if "monotonic_ns" not in r:
                continue
            if r["monotonic_ns"] < target:
                continue
            if r["bytes"] != size:
                if result:
                    break
                continue
            key = r["request_id"]
            if r["event"] == "submit":
                pending[key] = r
            elif r["event"] == "complete" and key in pending:
                s = pending.pop(key)
                assert r["duration_ns"] == r["monotonic_ns"] - s["monotonic_ns"]
                assert s["bytes"] == r["bytes"] and r["status"] == "completed"
                result.append([key, s["monotonic_ns"], r["monotonic_ns"]])
                if len(result) == 16:
                    break
        assert len(result) == 16, (path, size, len(result))
        origin = result[0][1]
        return [
            [rid, round((start - origin) / 1000, 3), round((end - origin) / 1000, 3)]
            for rid, start, end in result
        ], round((origin - target) / 1e6, 3)


output = []
for t in traces:
    origin = min(g["first_ns"] for g in t["sizes"].values())
    phases = []
    for size, g in sorted(t["sizes"].items(), key=lambda x: int(x[0])):
        size = int(size)
        target = (g["first_ns"] + g["last_ns"]) // 2
        samples, offset = window(args.events_dir / t["file"], target, size)
        phases.append(
            {
                "bytes": size,
                "start": round((g["first_ns"] - origin) / 1e9, 6),
                "end": round((g["last_ns"] - origin) / 1e9, 6),
                "completed": g["complete"],
                "mean_us": round(g["duration_ns_sum"] / g["complete"] / 1000, 3),
                "observed_GBps": round(
                    g["complete"] * size / (g["last_ns"] - g["first_ns"]), 3
                ),
                "samples": samples,
                "window_offset_ms": offset,
            }
        )
    a = t["footer"]
    output.append(
        {
            "run": t["label"],
            "file": t["file"],
            "op": t["peers"][0][2],
            "loss": round(
                100
                * a["dropped_records"]
                / (a["written_records"] + a["dropped_records"]),
                3,
            ),
            "phases": phases,
        }
    )
(R / "trace-windows.json").write_text(json.dumps(output, separators=(",", ":")) + "\n")
print(
    "Extracted",
    sum(len(p["samples"]) for t in output for p in t["phases"]),
    "validated spans across",
    sum(len(t["phases"]) for t in output),
    "phase windows",
)
