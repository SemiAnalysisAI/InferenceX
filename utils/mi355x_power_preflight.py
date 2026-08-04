#!/usr/bin/env python3
"""MI355X amd-smi power preflight — decides amd-smi vs exporter for PR5.

Run ON an MI355X node (idle is fine), takes ~90s, stdlib only:
    python3 mi355x_power_preflight.py [--samples 60] [--hz 1.0] [--out preflight_report.json]

Checks (PR4-PR5 plan section 5): amd-smi version; power/energy field inventory;
actual 1 Hz invocation latency; GPU index<->UUID/serial stability; energy counter
units + monotonicity + wrap/dropout; final boundary sample feasibility.
PASS gates: p95 invocation latency < 300 ms; every GPU present in >= 98% of
samples; energy monotonic non-decreasing on all GPUs; nonzero power fields.
"""
import argparse
import json
import shutil
import subprocess
import time


def run(cmd, timeout=30):
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return time.monotonic() - t0, p.returncode, p.stdout, p.stderr


def run_json(cmd, timeout=30):
    dt, rc, out, err = run(cmd + ["--json"], timeout)
    try:
        return dt, rc, json.loads(out), err
    except json.JSONDecodeError:
        return dt, rc, None, (err or out)[:500]


def gpu_entries(doc):
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        if isinstance(doc.get("gpu"), list):
            return doc["gpu"]
        if all(isinstance(v, dict) for v in doc.values()) and doc:
            return [dict(v, gpu=k) for k, v in doc.items()]
    return []


def num(x):
    if isinstance(x, dict):
        x = x.get("value")
    return x if isinstance(x, (int, float)) else None


def pick(sec, keys):
    if not isinstance(sec, dict):
        return None
    for k in keys:
        v = num(sec.get(k))
        if v is not None:
            return v
    return None


POWER_KEYS = ["socket_power", "average_socket_power", "current_socket_power", "power"]
ENERGY_KEYS = ["total_energy_consumption", "energy_accumulator", "total_energy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--out", default="preflight_report.json")
    args = ap.parse_args()

    report = {"checks": {}, "raw": {}}
    ck = report["checks"]

    if not shutil.which("amd-smi"):
        ck["amd_smi_present"] = False
        report["verdict"] = "FAIL: amd-smi not on PATH"
        print(json.dumps(report, indent=2))
        return
    ck["amd_smi_present"] = True

    dt, rc, ver, err = run_json(["amd-smi", "version"])
    ck["version"] = {"ok": rc == 0, "value": ver, "err": err if rc else None}

    dt, rc, static, err = run_json(["amd-smi", "static"])
    _, _, static_raw, _ = run(["amd-smi", "static", "--json"])
    report["raw"]["static_head"] = (static_raw or "")[:3000]
    identity = {}
    if rc == 0 and static:
        for gpu in gpu_entries(static):
            idx = gpu.get("gpu")
            asic = gpu.get("asic", {}) or {}
            board = gpu.get("board", {}) or {}
            identity[str(idx)] = {
                "asic_serial": asic.get("asic_serial"),
                "market_name": asic.get("market_name"),
                "product_serial": board.get("product_serial"),
            }
    ck["identity_map"] = {"ok": bool(identity), "gpus": identity}

    dt, rc, uuid_out, err = run(["amd-smi", "list", "-e"])
    report["raw"]["list_e_head"] = (uuid_out or err or "")[:800]

    dt, rc2, metric_raw, err = run(["amd-smi", "metric", "-p", "-E", "--json"])
    report["raw"]["metric_p_E_head"] = (metric_raw or err or "")[:4000]
    dt, rc, one, err = run_json(["amd-smi", "metric", "-p", "-E"])
    fields = set()
    if rc == 0 and one:
        for gpu in gpu_entries(one):
            for section, sec in gpu.items():
                if isinstance(sec, dict):
                    fields.update(f"{section}.{k}" for k in sec)
    ck["power_energy_fields"] = {"ok": bool(fields), "fields": sorted(fields)}

    interval = 1.0 / args.hz
    lat, series = [], []
    t_start = time.monotonic()
    for i in range(args.samples):
        dt, rc, snap, err = run_json(["amd-smi", "metric", "-p", "-E"], timeout=10)
        lat.append(dt)
        row = {"t": time.monotonic() - t_start, "rc": rc, "gpus": {}}
        if rc == 0 and snap:
            for gpu in gpu_entries(snap):
                idx = str(gpu.get("gpu"))
                row["gpus"][idx] = {
                    "socket_power": pick(gpu.get("power"), POWER_KEYS),
                    "energy_accumulator": pick(gpu.get("energy"), ENERGY_KEYS),
                }
        series.append(row)
        sleep_for = (i + 1) * interval - (time.monotonic() - t_start)
        if sleep_for > 0:
            time.sleep(sleep_for)

    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    ck["invocation_latency_s"] = {
        "mean": sum(lat) / len(lat), "p95": p95, "max": lat_sorted[-1],
        "ok": p95 < 0.3,
    }

    gpu_ids = sorted({g for row in series for g in row["gpus"]})
    presence = {
        g: sum(1 for row in series if row["gpus"].get(g, {}).get("socket_power") is not None)
        for g in gpu_ids
    }
    ck["presence"] = {
        "samples": len(series), "per_gpu": presence,
        "ok": bool(gpu_ids) and all(v >= 0.98 * len(series) for v in presence.values()),
    }

    mono, deltas = {}, {}
    for g in gpu_ids:
        vals = [row["gpus"][g].get("energy_accumulator") for row in series if g in row["gpus"]]
        vals = [v for v in vals if isinstance(v, (int, float))]
        ds = [b - a for a, b in zip(vals, vals[1:])]
        mono[g] = bool(ds) and all(d >= 0 for d in ds)
        deltas[g] = {"min": min(ds), "max": max(ds)} if ds else None
    ck["energy_monotonic"] = {"per_gpu": mono, "deltas": deltas, "ok": bool(mono) and all(mono.values())}

    dt, rc, final, err = run_json(["amd-smi", "metric", "-p", "-E"], timeout=10)
    ck["final_boundary_sample"] = {"ok": rc == 0, "latency_s": dt}

    powers = [
        row["gpus"][g]["socket_power"] for row in series for g in row["gpus"]
        if isinstance(row["gpus"][g].get("socket_power"), (int, float))
    ]
    ck["nonzero_power"] = {"ok": bool(powers) and max(powers) > 0,
                          "min": min(powers) if powers else None,
                          "max": max(powers) if powers else None}

    gates = ["invocation_latency_s", "presence", "energy_monotonic", "final_boundary_sample", "nonzero_power"]
    passed = all(ck[g]["ok"] for g in gates)
    report["verdict"] = "PASS: amd-smi CLI provider viable at 1 Hz" if passed else \
        "FAIL: " + ", ".join(g for g in gates if not ck[g]["ok"]) + " — evaluate exporter path (PR5A)"
    report["raw"]["series_head"] = series[:3]
    report["raw"]["series_tail"] = series[-2:]

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"verdict": report["verdict"], "checks": {k: v.get("ok") if isinstance(v, dict) else v for k, v in ck.items()}}, indent=2))
    print(f"full report: {args.out}")


if __name__ == "__main__":
    main()
