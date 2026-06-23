#!/usr/bin/env python3
"""CollectiveX spike — NCCL primitive benchmark wrapper.

Runs stock `nccl-tests` binaries (built in-container at job time — the login
nodes have no nvcc), parses the text table (NOT JSON — we do not assume the
build emits JSON), and writes a flat, provenance-tagged JSON result the plot
script and the eventual schema-freeze can consume.

Standard library only, so it runs in any minimal container.

Run (inside the container, after building nccl-tests):
    python run_nccl.py --op all_reduce \\
        --nccl-tests-dir /tmp/nccl-tests/build \\
        --world-size 8 --min-bytes 8 --max-bytes 8G \\
        --runner b200-dgxc --topology-class b200-nvlink-island --transport nvlink \\
        --env-json results/env.json --out results/b200_all_reduce.json

Verify the parser offline (no GPU needed):
    python run_nccl.py --op all_reduce --parse-only tests/fixtures/all_reduce_perf_b200_8gpu.txt \\
        --world-size 8 --runner b200-dgxc --topology-class b200-nvlink-island \\
        --out /tmp/parsed.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "nccl-tests-v1"

# op -> nccl-tests binary name
OP_BINARY = {
    "all_reduce": "all_reduce_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "alltoall": "alltoall_perf",
    "all_to_all": "alltoall_perf",
    "broadcast": "broadcast_perf",
    "sendrecv": "sendrecv_perf",
}


def _f(tok: str):
    """Parse a numeric cell; nccl-tests prints 'N/A' for #wrong when -c 0."""
    if tok in ("N/A", "n/a", "-"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse_nccl_table(text: str) -> tuple[list[dict], dict]:
    """Parse nccl-tests stdout into per-size rows + a run summary.

    Robust across ops: the column count varies (all_reduce/reduce_scatter carry
    redop+root; all_gather/alltoall do not), but every op prints the same 8
    trailing numeric columns — out-of-place (time, algbw, busbw, #wrong) then
    in-place (time, algbw, busbw, #wrong). `size` is always the first token and
    `type` the third. So we key off the first token and the last 8 tokens.
    """
    rows: list[dict] = []
    summary: dict = {"avg_busbw_gbps": None, "out_of_bounds": None, "check_passed": None}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "Avg bus bandwidth" in s:
                summary["avg_busbw_gbps"] = _f(s.split(":")[-1].strip())
            elif "Out of bounds values" in s:
                tail = s.split(":")[-1].strip()
                summary["out_of_bounds"] = tail
                summary["check_passed"] = tail.endswith("OK")
            continue
        toks = s.split()
        # Data line: first token is the byte size (all digits), and we need the
        # 8 trailing metric columns plus size+count+type up front (>=11 tokens).
        if len(toks) < 11 or not toks[0].isdigit():
            continue
        tail = toks[-8:]
        size = int(toks[0])
        dtype = toks[2] if len(toks) >= 3 else None
        oop_wrong = _f(tail[3])
        ip_wrong = _f(tail[7])
        rows.append(
            {
                "size_bytes": size,
                "dtype": dtype,
                "out_of_place": {
                    "time_us": _f(tail[0]),
                    "algbw_gbps": _f(tail[1]),
                    "busbw_gbps": _f(tail[2]),
                    "wrong": oop_wrong,
                },
                "in_place": {
                    "time_us": _f(tail[4]),
                    "algbw_gbps": _f(tail[5]),
                    "busbw_gbps": _f(tail[6]),
                    "wrong": ip_wrong,
                },
                # convenience: best (max) busbw across the two placements
                "busbw_gbps": max(
                    [b for b in (_f(tail[2]), _f(tail[6])) if b is not None],
                    default=None,
                ),
                "correct": (
                    None
                    if oop_wrong is None and ip_wrong is None
                    else ((oop_wrong or 0) == 0 and (ip_wrong or 0) == 0)
                ),
            }
        )
    return rows, summary


def comparison_key(meta: dict) -> str:
    """Machine key gating which rows may share a curve (see plan §Comparability).
    Topology-class is intentionally part of the key, so B200(IB) and
    GB200(MNNVL) are labelled distinct rather than silently overlaid."""
    parts = [
        meta["op"],
        meta["dtype"],
        str(meta["world_size"]),
        str(meta["nodes"]),
        meta["topology_class"],
        meta["comparison_class"],
        meta["measurement_contract"],
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return digest


def build_command(args, binary_path: str) -> list[str]:
    cmd: list[str] = []
    if args.launch_prefix:
        cmd += args.launch_prefix.split()
    cmd += [
        binary_path,
        "-b", str(args.min_bytes),
        "-e", str(args.max_bytes),
        "-f", str(args.factor),
        "-g", str(args.gpus_per_proc),
        "-c", str(args.check),
        "-w", str(args.warmup),
        "-n", str(args.iters),
    ]
    if args.extra_args:
        cmd += args.extra_args.split()
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX NCCL primitive runner")
    ap.add_argument("--op", required=True, choices=sorted(OP_BINARY))
    ap.add_argument("--nccl-tests-dir", help="dir containing <op>_perf binaries (build/)")
    ap.add_argument("--parse-only", help="parse this captured stdout file instead of running")
    # nccl-tests knobs
    ap.add_argument("--min-bytes", default="8")
    ap.add_argument("--max-bytes", default="8G")
    ap.add_argument("--factor", type=int, default=2, help="size step factor")
    ap.add_argument("--gpus-per-proc", type=int, default=8,
                    help="-g: GPUs per process (single-node multi-GPU). Use 1 under MPI.")
    ap.add_argument("--check", type=int, default=1, help="-c: 1 enables correctness check")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--extra-args", default="", help="extra args appended to the binary")
    ap.add_argument("--launch-prefix", default="",
                    help="e.g. 'mpirun -np 16 --hostfile hf' for multi-node; empty for single-node -g mode")
    # provenance
    ap.add_argument("--runner", required=True, help="runner label, e.g. b200-dgxc")
    ap.add_argument("--world-size", type=int, required=True, help="total ranks/GPUs in the run")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--topology-class", required=True,
                    help="e.g. b200-nvlink-island, b200-nvlink-island+cx7-ib, gb200-nvl72-mnnvl")
    ap.add_argument("--transport", default="", help="observed transport label: nvlink | ib | mnnvl")
    ap.add_argument("--comparison-class", default="standardized",
                    choices=["standardized", "backend-optimized", "framework-integrated"])
    ap.add_argument("--env-json", help="path to env_capture.py output to embed")
    ap.add_argument("--timestamp", help="ISO timestamp (default now)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    binary = OP_BINARY[args.op]
    command = None
    if args.parse_only:
        with open(args.parse_only) as fh:
            stdout = fh.read()
        ran_ok = True
    else:
        if not args.nccl_tests_dir:
            ap.error("--nccl-tests-dir is required unless --parse-only is given")
        binary_path = os.path.join(args.nccl_tests_dir, binary)
        if not os.path.exists(binary_path):
            print(f"ERROR: binary not found: {binary_path}", file=sys.stderr)
            return 2
        command = build_command(args, binary_path)
        print("running:", " ".join(command), file=sys.stderr)
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout = proc.stdout
        ran_ok = proc.returncode == 0
        if not ran_ok:
            print(stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            print(f"ERROR: {binary} exited {proc.returncode}", file=sys.stderr)

    rows, summary = parse_nccl_table(stdout)
    dtype = rows[0]["dtype"] if rows else None

    meta = {
        "op": args.op,
        "dtype": dtype,
        "world_size": args.world_size,
        "nodes": args.nodes,
        "topology_class": args.topology_class,
        "comparison_class": args.comparison_class,
        "measurement_contract": MEASUREMENT_CONTRACT,
    }

    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "family": "nccl",
        "generated_by": "run_nccl.py",
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner,
        "binary": binary,
        "command": " ".join(command) if command else f"<parse-only {args.parse_only}>",
        "transport": args.transport,
        "status": "valid" if (summary.get("check_passed") in (True, None) and ran_ok and rows) else "invalid",
        "comparison_key": comparison_key(meta),
        **meta,
        "summary": summary,
        "num_rows": len(rows),
        "rows": rows,
        "environment": env,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    print(
        f"{args.op}: parsed {len(rows)} sizes -> {args.out} "
        f"(status={doc['status']}, avg_busbw={summary.get('avg_busbw_gbps')} GB/s, "
        f"key={doc['comparison_key']})"
    )
    return 0 if doc["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
