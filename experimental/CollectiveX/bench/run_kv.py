#!/usr/bin/env python3
"""CollectiveX KV-cache transfer benchmark entrypoint (2 ranks, 1 per node).

Rank 0 is the target (owns the pool the initiator pulls from / pushes into),
rank 1 the initiator (posts every one-sided transfer and is the timed side).
The control plane is a gloo process group: payload exchange by object gather,
lockstep by barrier — no shared-FS or side-channel protocols. Data never rides
gloo.

Per (isl, page_tokens) point the initiator times `pull` then `push` over the
same prepped descriptor list; verification covers both directions (pull on the
initiator's pool, push on the target's, exchanged as verdict objects) and both
pools are repainted between points so every verify reads a clean pattern.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ep_harness  # noqa: E402  (case_id/is_case_id + atomic write; stdlib-only)
import kv_workload  # noqa: E402
from kv_backend import time_transfers  # noqa: E402

BULK_CAP = 8 << 30


def add_kv_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--workload-name", required=True, help="kv-mla | kv-gqa")
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"])
    ap.add_argument("--fabric", default="rdma", choices=["rdma", "mnnvl"],
                    help="which lane the SKU row claims; mnnvl additionally sets "
                         "UCX_CUDA_IPC_ENABLE_MNNVL=y for the UCX-backed libraries")
    ap.add_argument("--isl-ladder", default="512 4096 32768")
    ap.add_argument("--page-tokens", default="16 64")
    ap.add_argument("--ops", default="pull push")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pool-slack", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--case-id", default="", help="scheduled case ID; computed when omitted")
    ap.add_argument("--suite", default="kv-transfer")
    ap.add_argument("--version", default="1")
    ap.add_argument("--out", default="")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--scale-up-domain", type=int, default=8)
    ap.add_argument("--scale-up-transport", default="")
    ap.add_argument("--topology-class", default="")
    ap.add_argument("--socket-ifname", default=os.environ.get("COLLX_SOCKET_IFNAME", ""))
    ap.add_argument("--kv-mori-qp", type=int, default=1)
    ap.add_argument("--kv-mori-chunking", action="store_true")
    ap.add_argument("--kv-mori-port", type=int, default=48810)
    ap.add_argument("--link-gbps", type=float, default=0.0,
                    help="per-worker wire rate in Gbit/s for the %%-of-wire column; 0 = unknown")


def kv_case(args) -> dict:
    return {
        "backend": args.backend,
        "workload": args.workload_name,
        "mode": args.fabric,
        "phase": "xfer",
        "ep": 2,
        "routing": "paged",
        "precision": args.precision,
    }


def _grid(args) -> tuple[list[dict], list[int]]:
    preset = args.workload_name.removeprefix("kv-")
    isls = [int(v) for v in args.isl_ladder.split()]
    pages = [int(v) for v in args.page_tokens.split()]
    configs = [
        kv_workload.plan_config(preset, args.precision, isl, page, args.pool_slack)
        for isl in isls for page in pages
    ]
    return configs, isls


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX KV-cache transfer sweep")
    ap.add_argument("--backend", required=True, choices=["nixl", "mori-io"])
    add_kv_args(ap)
    args = ap.parse_args()

    case = kv_case(args)
    computed_case_id = ep_harness.case_id(args.runner, case)
    if args.case_id and args.case_id != computed_case_id:
        print(f"ERROR: scheduled case ID does not match factors: "
              f"{args.case_id} != {computed_case_id}", file=sys.stderr)
        return 2
    args.case_id = args.case_id or computed_case_id

    if args.fabric == "mnnvl":
        os.environ.setdefault("UCX_CUDA_IPC_ENABLE_MNNVL", "y")
    if args.socket_ifname:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", args.socket_ifname)

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    if world_size != 2:
        print(f"ERROR: kv-transfer runs exactly 2 ranks, got {world_size}", file=sys.stderr)
        return 2
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    role = "target" if rank == 0 else "initiator"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if args.backend == "mori-io":
        from kv_mori_io import MoRIIOBackend as Backend
    else:
        from kv_nixl import NIXLBackend as Backend

    configs, isls = _grid(args)
    ops = args.ops.split()
    pool_bytes = max(cfg["pool_bytes"] for cfg in configs)
    bulk_bytes = min(max(cfg["req_bytes"] for cfg in configs), BULK_CAP)
    pool = torch.empty(pool_bytes, dtype=torch.uint8, device=device)
    bulk = torch.empty(bulk_bytes, dtype=torch.uint8, device=device)

    def repaint():
        kv_workload.fill_pattern(pool)
        bulk.fill_(0xAB if role == "target" else 0xCD)
        torch.cuda.synchronize()

    repaint()
    backend = Backend(args, role, device)
    backend.register(pool, bulk)
    payloads = [None, None]
    dist.all_gather_object(payloads, backend.publish())
    backend.connect(payloads[1 - rank])
    dist.barrier()
    if rank == 1:
        print(f"[run_kv] backend={args.backend} workload={args.workload_name} "
              f"precision={args.precision} fabric={args.fabric} isls={isls} "
              f"pool={pool_bytes >> 20}MiB case={args.case_id}", flush=True)

    rows: list[dict] = []

    def measure(make, cfg_row: dict, op: str, verify_side: str, tables=None):
        """One grid point: initiator times, then the verifying side checks."""
        if role == "initiator":
            transfer_once, prep_s = make()
            samples: list[float] = []
            for _ in range(args.trials):
                samples.extend(time_transfers(transfer_once, args.warmup, args.reps))
        dist.barrier()  # transfers complete before anyone inspects pools
        verdict = {"passed": True, "detail": ""}
        if tables is not None and role == verify_side:
            dst_table, src_table = tables
            passed, detail = kv_workload.verify_transfer(pool, cfg_row["_cfg"], dst_table, src_table)
            verdict = {"passed": passed, "detail": detail}
        verdicts = [None, None]
        dist.all_gather_object(verdicts, verdict if role == verify_side else None)
        verdict = next(v for v in verdicts if v is not None)
        repaint()
        dist.barrier()
        if role != "initiator":
            return None
        stats = kv_workload.pcts(samples)
        gbps = cfg_row["req_bytes"] / stats["p50"] / 1e6
        return {
            **{k: v for k, v in cfg_row.items() if not k.startswith("_")},
            "op": op,
            "prep_ms": round(prep_s * 1e3, 3),
            "latency_ms": {k: round(v, 3) for k, v in stats.items()},
            "gbps_p50": round(gbps, 2),
            "wire_pct": round(100 * gbps / (args.link_gbps / 8), 1) if args.link_gbps else None,
            "verify": verdict,
        }

    for cfg in configs:
        target_table = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "remote"))
        initiator_table = kv_workload.block_table(cfg, kv_workload.table_seed(cfg, "local"))
        base = {
            "kind": "paged", "preset": cfg["preset"], "isl": cfg["isl"],
            "page_tokens": cfg["page_tokens"], "layers": cfg["layers"],
            "page_bytes": cfg["page_bytes"], "descs": cfg["descs"],
            "req_bytes": cfg["req_bytes"], "_cfg": cfg,
        }
        for op in ops:
            if role == "initiator":
                make = lambda op=op: backend.make_paged(cfg, op, initiator_table, target_table)
            else:
                make = None
            # pull lands on the initiator's pool; push on the target's.
            verify_side = "initiator" if op == "pull" else "target"
            tables = ((initiator_table, target_table) if op == "pull"
                      else (target_table, initiator_table))
            row = measure(make, base, op, verify_side, tables)
            if row is not None:
                rows.append(row)
                print(f"[run_kv] {json.dumps(row)}", flush=True)

    for isl in isls:
        preset = args.workload_name.removeprefix("kv-")
        cfg = kv_workload.plan_config(preset, args.precision, isl, 64, args.pool_slack)
        nbytes = min(cfg["req_bytes"], bulk_bytes)
        base = {"kind": "bulk", "preset": preset, "isl": isl, "page_tokens": None,
                "layers": cfg["layers"], "page_bytes": None, "descs": 1,
                "req_bytes": nbytes, "_cfg": cfg}
        for op in ops:
            make = (lambda op=op, n=nbytes: backend.make_bulk(n, op)) if role == "initiator" else None
            row = measure(make, base, op, verify_side="none", tables=None)
            if row is not None:
                rows.append(row)
                print(f"[run_kv] {json.dumps(row)}", flush=True)

    backend.teardown()

    gathered: list = [None, None]
    dist.all_gather_object(gathered, rows if rank == 1 else None)
    rows = gathered[1] or []
    all_ok = bool(rows) and all(r["verify"]["passed"] for r in rows)

    if rank == 0:
        doc = {
            "version": args.version,
            "record_type": "case-attempt",
            "generated_at": _dt.datetime.now().astimezone().isoformat(),
            "identity": {
                "allocation_factors": {
                    "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                    "run_id": os.environ.get("GITHUB_RUN_ID"),
                    "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
                },
                "attempt_ordinal": int(os.environ.get("COLLX_ATTEMPT_ID", "1")),
                "case_factors": {"case": {**case, "suite": args.suite}, "sku": args.runner},
                "case_id": args.case_id,
            },
            "workload": {
                "isl_ladder": isls,
                "page_tokens": [int(v) for v in args.page_tokens.split()],
                "ops": ops,
                "preset": kv_workload.PRESETS[args.workload_name.removeprefix("kv-")],
            },
            "measurement": {
                "payload_unit": "request-kv-bytes",
                "rows": rows,
                "sampling": {
                    "reps_per_trial": args.reps,
                    "trials": args.trials,
                    "warmup_per_trial": args.warmup,
                },
            },
            "implementation": {
                "name": args.backend,
                "fabric": args.fabric,
                "library_version": getattr(backend, "library_version", None),
                "maturity": getattr(backend, "maturity", "candidate"),
            },
            "topology": {
                "device_product": torch.cuda.get_device_name(device),
                "gpus_per_node": args.gpus_per_node,
                "nodes": 2,
                "ranks_per_node": 1,
                "scale_up_domain": args.scale_up_domain,
                "scale_up_transport": args.scale_up_transport or None,
                "topology_class": args.topology_class or None,
                "world_size": world_size,
            },
            "runtime": {
                "framework": str(torch.__version__),
                "vendor": "amd" if torch.version.hip else "nvidia",
            },
            "provenance": {
                "image": os.environ.get("COLLECTIVEX_IMAGE") or None,
                "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
            },
            "outcome": {
                "reasons": [] if all_ok else ["transfer verification failed"],
                "status": "success" if all_ok else "invalid",
            },
        }
        if args.out:
            ep_harness._write_json_atomic(args.out, doc)
        print(f"[run_kv] status={doc['outcome']['status']} rows={len(rows)}"
              + (f" -> {args.out}" if args.out else ""), flush=True)

    flag = torch.tensor([int(all_ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    dist.barrier()
    return 0 if int(flag.item()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
