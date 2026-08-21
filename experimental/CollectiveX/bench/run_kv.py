#!/usr/bin/env python3
"""CollectiveX KV-cache transfer benchmark entrypoint (2 ranks, 1 per node).

Rank 0 is the target (owns the pool the initiator pulls from / pushes into),
rank 1 the initiator (posts every one-sided transfer and is the timed side).
The control plane is a gloo process group: payload exchange by object gather,
lockstep by barrier — no shared-FS or side-channel protocols. Data never rides
gloo.

Per (isl, page_tokens, batch) point the initiator preps one transfer per
request in the burst (disjoint block-table slices), posts them all, then awaits
them all — a decode step admitting B requests at once. Verification covers both
directions (pull on the initiator's pool, push on the target's, exchanged as
verdict objects) and both pools are repainted between points so every verify
reads a clean pattern. Points whose pool would not fit POOL_BUDGET shed their
largest batches, so one grid covers dense GQA-bf16 and DSv4's ~2% cache alike.
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
from kv_backend import time_bursts  # noqa: E402

BULK_CAP = 8 << 30
# Pool ceiling per rank: fits the fleet's smallest HBM (h200, 141 GB) next to
# the bulk buffer; grid points shed their largest batches to stay under it.
POOL_BUDGET = 64 << 30
# Burst posting ceiling: a burst posts batch x descs descriptors, and the
# per-descriptor floor makes time linear in that product (a 512k-ISL page-16
# request alone is ~2.1M descriptors). Sized to keep every <=32k cell of the
# original grid while holding the slowest lane inside the per-case guard. The
# two smallest requested batches ride over this budget so every point keeps a
# one-to-two scaling step on the batch axis (512k page-16 runs its batch-2
# burst ~2x over budget by design — bounded, and priced into the gb-nv guard).
DESC_BUDGET = 2_250_000


def add_kv_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--workload-name", required=True, help="kv-<preset>, e.g. kv-dsv4")
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"])
    ap.add_argument("--fabric", default="rdma", choices=["rdma", "mnnvl"],
                    help="which lane the SKU row claims; mnnvl additionally sets "
                         "UCX_CUDA_IPC_ENABLE_MNNVL=y for the UCX-backed libraries")
    ap.add_argument("--isl-ladder", default="512 4096 32768")
    ap.add_argument("--page-tokens", default="16 64")
    ap.add_argument("--ops", default="pull push")
    ap.add_argument("--batch-sizes", default="1",
                    help="requests per burst; each is a separate prepped transfer, "
                         "posted together then awaited together")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pool-slack", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--case-id", default="", help="scheduled case ID; computed when omitted")
    ap.add_argument("--suite", default="kv-transfer")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--scale-up-domain", type=int, default=8)
    ap.add_argument("--scale-up-transport", default="")
    ap.add_argument("--topology-class", default="")
    ap.add_argument("--socket-ifname", default=os.environ.get("COLLX_SOCKET_IFNAME", ""))
    ap.add_argument("--kv-mori-qp", type=int, default=1)
    ap.add_argument("--kv-mori-chunking", action="store_true")
    ap.add_argument("--kv-device", default="",
                    help="engine NIC filter template; {gpu} expands to the "
                         "physical GPU index (GPU-paired NICs, e.g. Pollara)")
    ap.add_argument("--kv-mori-port", type=int, default=48810)
    ap.add_argument("--kv-mc-port", type=int, default=48830)


def export_ucx_selectors(environ=os.environ) -> None:
    """Pin the UCX fabric to the operator's validated RDMA selectors.

    UCX auto-selection is a wrong-fabric trap on several SKUs (b200-nscale's
    quad-port aux card, b300's storage IB), and the launcher's network profile
    only exports the COLLX_* names. Explicit UCX_* values always win.
    """
    devices = environ.get("COLLX_RDMA_DEVICES", "")
    if devices and "UCX_NET_DEVICES" not in environ:
        environ["UCX_NET_DEVICES"] = ",".join(
            device if ":" in device else f"{device}:1"
            for device in devices.split(",") if device)
    gid = environ.get("COLLX_IB_GID_INDEX", "")
    if gid and "UCX_IB_GID_INDEX" not in environ:
        environ["UCX_IB_GID_INDEX"] = str(gid)


def exchange_verdict(dist, role, verify_side, verify):
    """One rank verifies its destination pool; every rank returns that verdict.

    Bulk rows have no verifying side (verify_side "none"): every rank gathers
    None and the row passes by construction, without a gather-of-nothing crash.
    """
    verdict = None
    if role == verify_side:
        passed, detail = verify()
        verdict = {"passed": passed, "detail": detail}
    gathered = [None, None]
    dist.all_gather_object(gathered, verdict)
    return next((v for v in gathered if v is not None), {"passed": True, "detail": ""})


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


def _grid(args) -> tuple[list[tuple[dict, list[int]]], list[int], list[int]]:
    """(cfg, allowed_batches) per (isl, page) point. Batches whose burst would
    exceed DESC_BUDGET are shed first (the two smallest requested batches are
    always kept, so a single request stays measurable at every point and the
    batch axis keeps a one-to-two scaling step), then the point is planned for
    the largest surviving batch whose pool fits POOL_BUDGET. Smaller batches
    share that cfg (and pool), so batch is the only variable across a point's
    rows."""
    preset = args.workload_name.removeprefix("kv-")
    isls = [int(v) for v in args.isl_ladder.split()]
    pages = [int(v) for v in args.page_tokens.split()]
    batches = sorted({int(v) for v in args.batch_sizes.split()})
    points = []
    for isl in isls:
        for page in pages:
            # Per-request descriptor count is independent of batch_max.
            probe = kv_workload.plan_config(preset, args.precision, isl, page,
                                            args.pool_slack)
            allowed = [batch for batch in batches
                       if batch in batches[:2] or batch * probe["descs"] <= DESC_BUDGET]
            while allowed:
                cfg = kv_workload.plan_config(preset, args.precision, isl, page,
                                              args.pool_slack, batch_max=allowed[-1])
                if cfg["pool_bytes"] <= POOL_BUDGET:
                    break
                allowed.pop()
            if allowed:
                points.append((cfg, allowed))
    return points, isls, batches


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX KV-cache transfer sweep")
    ap.add_argument("--backend", required=True, choices=["nixl", "mori-io", "mooncake"])
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
    export_ucx_selectors()

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
    elif args.backend == "mooncake":
        from kv_mooncake import MooncakeBackend as Backend
    else:
        from kv_nixl import NIXLBackend as Backend

    points, isls, batches = _grid(args)
    ops = args.ops.split()
    pool_bytes = max(cfg["pool_bytes"] for cfg, _ in points)
    bulk_bytes = min(max(cfg["req_bytes"] for cfg, _ in points), BULK_CAP)

    # RDMA registration pins the whole pool; a small inherited soft memlock
    # limit fails it with an unhelpful ENOMEM/EIO deep inside the library
    # (Slurm propagates the SUBMITTER's limits into steps). Raise soft to hard
    # when possible; otherwise fail here with the actual numbers.
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    need = pool_bytes + bulk_bytes
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        soft = hard
    if soft != resource.RLIM_INFINITY and soft < need:
        print(f"ERROR: RLIMIT_MEMLOCK {soft} < {need} needed to register the KV pools; "
              "submit with --propagate=NONE or raise the limit", file=sys.stderr)
        return 2

    import kv_pool

    pool = kv_pool.create(args.fabric, pool_bytes, local_rank)
    bulk = kv_pool.create(args.fabric, bulk_bytes, local_rank)

    def repaint():
        pool.fill_pattern()
        bulk.fill_byte(0xAB if role == "target" else 0xCD)

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
              f"batches={batches} pool={pool_bytes >> 20}MiB case={args.case_id}",
              flush=True)
        for cfg, allowed in points:
            if allowed != batches:
                print(f"[run_kv] budgets cap isl={cfg['isl']} "
                      f"page={cfg['page_tokens']} at batch<={allowed[-1]}", flush=True)

    rows: list[dict] = []

    def verify_burst(cfg, table_pairs):
        """Every request in the burst must land: a passing request 0 says
        nothing about the others, and concurrent same-session requests are
        exactly where corruption would hide."""
        for r, (dst, src) in enumerate(table_pairs):
            passed, detail = kv_workload.verify_transfer(pool.read8, cfg, dst, src)
            if not passed:
                return False, f"request={r} {detail}"
        return True, ""

    def measure(make, cfg_row: dict, op: str, verify_side: str, table_pairs=None):
        """One grid point: initiator times bursts, then the verifying side checks."""
        if role == "initiator":
            made = make()  # one (post, wait, prep_seconds) per request in the burst
            prep_s = sum(m[2] for m in made)
            pairs = [m[:2] for m in made]
            samples: list[float] = []
            for _ in range(args.trials):
                samples.extend(time_bursts(pairs, args.warmup, args.reps))
        dist.barrier()  # transfers complete before anyone inspects pools
        verdict = exchange_verdict(
            dist, role, verify_side,
            lambda: verify_burst(cfg_row["_cfg"], table_pairs))
        repaint()
        dist.barrier()
        if role != "initiator":
            return None
        stats = kv_workload.pcts(samples)
        gbps = cfg_row["req_bytes"] * cfg_row["batch"] / stats["p50"] / 1e6
        return {
            **{k: v for k, v in cfg_row.items() if not k.startswith("_")},
            "op": op,
            "prep_ms": round(prep_s * 1e3, 3),
            "latency_ms": {k: round(v, 3) for k, v in stats.items()},
            "gbps_p50": round(gbps, 2),
            "verify": verdict,
        }

    for cfg, allowed in points:
        seed_t = kv_workload.table_seed(cfg, "remote")
        seed_i = kv_workload.table_seed(cfg, "local")
        target_tables = [kv_workload.block_table(cfg, seed_t, r) for r in range(allowed[-1])]
        initiator_tables = [kv_workload.block_table(cfg, seed_i, r) for r in range(allowed[-1])]
        base = {
            "kind": "paged", "preset": cfg["preset"], "isl": cfg["isl"],
            "page_tokens": cfg["page_tokens"], "layers": cfg["layers"],
            "page_bytes": cfg["page_bytes"], "descs": cfg["descs"],
            "req_bytes": cfg["req_bytes"], "_cfg": cfg,
        }
        for batch in allowed:
            for op in ops:
                make = None
                if role == "initiator":
                    make = lambda op=op, batch=batch: [
                        backend.make_paged(cfg, op, initiator_tables[r], target_tables[r])
                        for r in range(batch)]
                # pull lands on the initiator's pool; push on the target's.
                # Every request in the burst is checked against its own tables.
                verify_side = "initiator" if op == "pull" else "target"
                table_pairs = [
                    (initiator_tables[r], target_tables[r]) if op == "pull"
                    else (target_tables[r], initiator_tables[r])
                    for r in range(batch)]
                row = measure(make, {**base, "batch": batch}, op, verify_side,
                              table_pairs)
                if row is not None:
                    rows.append(row)
                    print(f"[run_kv] {json.dumps(row)}", flush=True)

    for isl in isls:
        preset = args.workload_name.removeprefix("kv-")
        cfg = kv_workload.plan_config(preset, args.precision, isl, 64, args.pool_slack)
        nbytes = min(cfg["req_bytes"], bulk_bytes)
        base = {"kind": "bulk", "preset": preset, "isl": isl, "page_tokens": None,
                "layers": cfg["layers"], "page_bytes": None, "descs": 1, "batch": 1,
                "req_bytes": nbytes, "_cfg": cfg}
        for op in ops:
            make = (lambda op=op, n=nbytes: [backend.make_bulk(n, op)]) if role == "initiator" else None
            row = measure(make, base, op, verify_side="none", table_pairs=None)
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
                "batch_sizes": batches,
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
