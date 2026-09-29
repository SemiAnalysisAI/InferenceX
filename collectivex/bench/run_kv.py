#!/usr/bin/env python3
"""CollectiveX KV-cache transfer benchmark entrypoint (2 ranks, 1 per node).

Rank 0 is the target (owns the pool the initiator pulls from / pushes into),
rank 1 the initiator (posts every one-sided transfer and is the timed side).
The control plane is a gloo process group: payload exchange by object gather,
lockstep by barrier — no shared-FS or side-channel protocols. Data never rides
gloo.

Per (isl, batch) point the initiator times bursts: each burst posts one
transfer per request (disjoint block-table slices, a fresh table set every
rep, as a decode worker admits requests with fresh block ids), then awaits
them all. Handle creation is part of each request's post, as in vLLM. Every
trial's last rep lands on a destination wiped to a sentinel first, so the
verify after it proves THAT timed rep moved every descriptor (first, last and
an interior word of each). Bursts are capped at --max-burst-tokens of prompt,
and points whose pool would not fit the pool budget shed their largest
batches, then their table sets; the document records what each point ran.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ep_harness  # noqa: E402  (stdlib-only; safe before torch)
import kv_workload  # noqa: E402
from kv_backend import time_bursts  # noqa: E402

BULK_CAP = 8 << 30
# Default pool ceiling per rank: fits the fleet's smallest HBM (h200, 141 GB)
# next to the bulk buffer; grid points shed their largest batches to stay under
# it. A registry kv_backends entry lowers it (`pool_budget`, reaching here as
# --pool-budget) where an engine/NIC pairing cannot register a pool this
# large: mooncake on the mi355x ionic NICs fails ibv_reg_mr with ENOMEM between
# the 20 GiB pool the mixed batch ladder planned (green) and the 53 GiB the
# power-of-two ladder plans (red on two independent allocations), while
# mori-io registers the same pool fine, so the cap is per backend, not
# fleet-wide.
POOL_BUDGET = 64 << 30
# Burst posting ceiling: a burst posts batch x descs descriptors, and the
# per-descriptor floor makes time linear in that product. On the packed
# block-major geometry a request is only ceil(isl/block) descriptors per
# group (a 512k-ISL block-256 request is ~6.1k), so the production grid sits
# far under this; the budget stays as the fail-closed guard for future
# presets or small block sizes.
DESC_BUDGET = 2_250_000
SENTINEL = 0x5A


def add_kv_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--workload-name", required=True, help="kv-<preset>, e.g. kv-dsv4")
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"])
    ap.add_argument("--fabric", default="rdma", choices=["rdma", "mnnvl"],
                    help="which lane the SKU row claims; mnnvl additionally sets "
                         "UCX_CUDA_IPC_ENABLE_MNNVL=y for the UCX-backed libraries")
    ap.add_argument("--isl-ladder", default="512 4096 32768")
    ap.add_argument("--page-tokens", default="256",
                    help="vLLM block size in tokens; dsv4 needs a multiple of "
                         "128 (HCA states) and vLLM serves it at 256")
    ap.add_argument("--ops", default="pull push")
    ap.add_argument("--batch-sizes", default="1",
                    help="requests per burst; each is a separate prepped transfer, "
                         "posted together then awaited together")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pool-slack", type=float, default=2.0)
    ap.add_argument("--max-burst-tokens", type=int, default=0,
                    help="cap on batch x isl per burst (0 = none); batch 1 always runs")
    ap.add_argument("--table-sets", type=int, default=kv_workload.DEFAULT_TABLE_SETS,
                    help="disjoint block-table sets the reps rotate through")
    ap.add_argument("--pool-budget", type=int, default=POOL_BUDGET,
                    help="per-rank pool ceiling in bytes; points shed batches to fit")
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
    ap.add_argument("--kv-device", default="",
                    help="engine NIC filter template; {gpu} expands to the "
                         "physical GPU index (GPU-paired NICs, e.g. Pollara). "
                         "For nixl it is a literal netdev comma-list pinning "
                         "UCX_NET_DEVICES below the operator inventory")
    ap.add_argument("--kv-mori-port", type=int, default=48810)
    ap.add_argument("--kv-mc-port", type=int, default=48830)


def export_ucx_selectors(environ=os.environ, device: str = "") -> None:
    """Pin the UCX fabric to the operator's validated RDMA selectors.

    UCX auto-selection is a wrong-fabric trap on several SKUs (b200-nscale's
    quad-port aux card, b300's storage IB), and the launcher's network profile
    only exports the COLLX_* names. Explicit UCX_* values always win.

    ``device`` is the case's registry NIC pin (kv_device) for a UCX-backed
    engine: a literal netdev comma-list that narrows UCX below the operator
    inventory, for rail-isolated pods where multi-rail selection is the
    variance source under measurement. Unlike the inventory it overrides a
    host-inherited UCX_NET_DEVICES: b300 ships a blanket 16-device value in
    /etc/environment (forwarded by srun --export=ALL) that would otherwise
    silently swallow the pin, the same way its UCX_TLS=rc is dropped below.
    """
    devices = device or environ.get("COLLX_RDMA_DEVICES", "")
    if devices and (device or "UCX_NET_DEVICES" not in environ):
        environ["UCX_NET_DEVICES"] = ",".join(
            dev if ":" in dev else f"{dev}:1"
            for dev in devices.split(",") if dev)
    gid = environ.get("COLLX_IB_GID_INDEX", "")
    if gid and "UCX_IB_GID_INDEX" not in environ:
        environ["UCX_IB_GID_INDEX"] = str(gid)
    # A host-inherited positive UCX_TLS list without the cuda transports (b300
    # ships UCX_TLS=rc cluster-wide in /etc/environment, forwarded by srun
    # --export=ALL) makes ucp close the cuda mds; UCX then classifies VRAM as
    # host memory and NIXL registration fails with NIXL_ERR_BACKEND. Extending
    # the list with cuda_copy,cuda_ipc is not enough: the initiator then
    # segfaults in ucp_worker_add_rkey_config resolving the cuda rkey on the
    # first ucp_get_nbx. Drop the list and let UCX auto-select; the wire stays
    # pinned through UCX_NET_DEVICES above.
    tls = environ.get("UCX_TLS", "")
    if tls and tls != "all" and not tls.startswith("^") and "cuda" not in tls:
        del environ["UCX_TLS"]


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
    """(cfg, allowed_batches) per (isl, page) point. A batch runs only if its
    burst stays under --max-burst-tokens of prompt (batch 1 always runs) and
    DESC_BUDGET descriptors; the point is then planned for the largest
    surviving batch and --table-sets whose pool fits the pool budget,
    shedding batches first and table sets after. Smaller batches share that
    cfg (and pool), so batch is the only variable across a point's rows."""
    preset = args.workload_name.removeprefix("kv-")
    isls = [int(v) for v in args.isl_ladder.split()]
    pages = [int(v) for v in args.page_tokens.split()]
    batches = sorted({int(v) for v in args.batch_sizes.split()})
    points = []
    for isl in isls:
        for page in pages:
            probe = kv_workload.plan_config(preset, args.precision, isl, page,
                                            args.pool_slack)
            allowed = [batch for batch in batches
                       if (batch == 1 or not args.max_burst_tokens
                           or batch * isl <= args.max_burst_tokens)
                       and batch * probe["descs"] <= DESC_BUDGET]
            cfg, sets = None, args.table_sets
            while allowed:
                cfg = kv_workload.plan_config(preset, args.precision, isl, page,
                                              args.pool_slack, batch_max=allowed[-1],
                                              table_sets=sets)
                if cfg["pool_bytes"] <= args.pool_budget:
                    break
                if len(allowed) > 1:
                    allowed.pop()
                elif sets > 1:
                    sets -= 1
                else:
                    allowed.pop()
            if allowed:
                points.append((cfg, allowed))
    return points, isls, batches


def _harmonize(points) -> list[tuple[int, int, int]]:
    """Give every cfg the one shared pool (the most rows any point plans) and
    return its registration layout as one (base, row_bytes, nbytes) triple.
    Every descriptor is a whole row, so a registration cut on the row grid
    never splits one (b300's former IB NICs refused cuda registrations past
    ~8 GiB; reg_spans cuts there)."""
    row_bytes = points[0][0]["row_bytes"]
    rows = max(cfg["pool_rows"] for cfg, _ in points)
    for cfg, _ in points:
        cfg["pool_rows"], cfg["pool_bytes"] = rows, rows * row_bytes
    return [(0, row_bytes, rows * row_bytes)]


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
    export_ucx_selectors(
        device=args.kv_device if args.backend == "nixl" else "")

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
    # A single grid point's timed stretch can run past gloo's 30-minute
    # default recv timeout (a slow lane's large-ISL bursts, while the target
    # rank waits silently at the next gather). Size the control-plane timeout
    # to the per-case hang guard so the guard, not gloo, decides when a run
    # died.
    grace_s = int(os.environ.get("COLLX_RUN_TIMEOUT") or "21600")
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=_dt.timedelta(seconds=grace_s))

    if args.backend == "mori-io":
        from kv_mori_io import MoRIIOBackend as Backend
    elif args.backend == "mooncake":
        from kv_mooncake import MooncakeBackend as Backend
    else:
        from kv_nixl import NIXLBackend as Backend

    points, isls, batches = _grid(args)
    if not points:
        print("ERROR: no grid point fits DESC_BUDGET and the pool budget", file=sys.stderr)
        return 2
    reg_layout = _harmonize(points)
    ops = args.ops.split()
    pool_bytes = points[0][0]["pool_bytes"]  # _harmonize gives every cfg the union pool
    row_bytes = points[0][0]["row_bytes"]
    preset = args.workload_name.removeprefix("kv-")
    pages = [int(v) for v in args.page_tokens.split()]
    bulk_bytes = min(max(cfg["req_bytes"] for cfg, _ in points), BULK_CAP)

    # RDMA registration pins the whole pool; a small inherited soft memlock
    # limit fails it with an unhelpful ENOMEM/EIO deep inside the library
    # (Slurm propagates the SUBMITTER's limits into steps). Raise soft to hard
    # when possible; otherwise fail here with the actual numbers.
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    need = pool_bytes + bulk_bytes
    if soft != hard:  # soft <= hard always, so this raises soft to hard
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
        pool.fill_pattern(salt=rank)
        bulk.fill_pattern(salt=rank)

    repaint()
    backend = Backend(args, role, device)
    backend.register(pool, bulk, row_bytes, reg_layout=reg_layout)
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
            if allowed != batches or cfg["table_sets"] != args.table_sets:
                print(f"[run_kv] budgets cap isl={cfg['isl']} at batch<={allowed[-1]} "
                      f"table_sets={cfg['table_sets']}", flush=True)

    rows: list[dict] = []

    def record(row):
        if row is not None:
            rows.append(row)
            print(f"[run_kv] {json.dumps(row)}", flush=True)

    def measure(build, cfg_row: dict, op: str, verify_side: str, verify, wipe,
                table_sets: int):
        """One grid point. Each trial times warmup + reps-1 bursts, then the
        destination owner wipes its buffer to SENTINEL, the initiator times
        the trial's last burst, and the owner verifies that burst's table set,
        so a timed rep that reported completion without landing its bytes
        fails the row. ``build(rep)`` returns the burst's (post, wait) pairs
        for table set rep % table_sets."""
        samples: list[float] = []
        request_samples: list[float] = []
        verdict = {"passed": True, "detail": ""}
        rep = 0
        for _ in range(args.trials):
            if role == "initiator":
                burst_ms, request_ms = time_bursts(build, args.warmup, args.reps - 1,
                                                   settle=backend.release, rep0=rep)
                samples += burst_ms
                request_samples += request_ms
            rep += args.warmup + args.reps - 1
            dist.barrier()
            if role == verify_side:
                wipe()
            dist.barrier()
            if role == "initiator":
                burst_ms, request_ms = time_bursts(build, 0, 1, settle=backend.release,
                                                   rep0=rep)
                samples += burst_ms
                request_samples += request_ms
            final_set = rep % table_sets
            rep += 1
            dist.barrier()  # the last burst completes before anyone inspects it
            verdict = exchange_verdict(dist, role, verify_side, lambda: verify(final_set))
            if not verdict["passed"]:
                break
        repaint()
        dist.barrier()
        if role != "initiator":
            return None
        stats = kv_workload.pcts(samples)
        request_stats = kv_workload.pcts(request_samples)
        gbps = cfg_row["req_bytes"] * cfg_row["batch"] / stats["p50"] / 1e6
        return {
            **cfg_row,
            "op": op,
            "latency_ms": {k: round(v, 3) for k, v in stats.items()},
            # Host-observed completion of each individual request within its
            # burst (waits drain in posting order, so each is an upper bound).
            "request_ms": {k: round(v, 3) for k, v in request_stats.items()},
            "gbps_p50": round(gbps, 2),
            "verify": verdict,
        }

    peer = 1 - rank
    for cfg, allowed in points:
        sets = cfg["table_sets"]
        seed_t = kv_workload.table_seed(cfg, "remote", args.seed)
        seed_i = kv_workload.table_seed(cfg, "local", args.seed)
        target_tables = [[kv_workload.block_table(cfg, seed_t, r, k) for r in range(allowed[-1])]
                         for k in range(sets)]
        initiator_tables = [[kv_workload.block_table(cfg, seed_i, r, k)
                             for r in range(allowed[-1])] for k in range(sets)]
        # What one request moves in this backend's vLLM shape (NIXL: whole
        # rows; Mooncake: per-layer pages), so rows state their own bytes.
        _, _, sizes = backend.request_entries(cfg, initiator_tables[0][0],
                                              target_tables[0][0])
        base = {
            "kind": "paged", "preset": cfg["preset"], "isl": cfg["isl"],
            "page_tokens": cfg["page_tokens"], "layers": cfg["layers"],
            "row_bytes": cfg["row_bytes"], "descs": int(len(sizes)),
            "req_bytes": int(sizes.sum()),
        }
        for batch in allowed:
            for op in ops:
                # Called by measure on the initiator only, within this iteration.
                def build(rep, batch=batch, op=op):
                    k = rep % sets
                    return backend.make_burst(cfg, op, [
                        (initiator_tables[k][r], target_tables[k][r], rep * batch + r)
                        for r in range(batch)])

                # pull lands on the initiator's pool; push on the target's.
                # Every request in the burst is checked against its own tables.
                verify_side = "initiator" if op == "pull" else "target"

                def verify(k, batch=batch, op=op):
                    for r in range(batch):
                        local, remote, sizes = backend.request_entries(
                            cfg, initiator_tables[k][r], target_tables[k][r])
                        dst, src = (local, remote) if op == "pull" else (remote, local)
                        passed, detail = kv_workload.verify_entries(
                            pool.words, dst, src, sizes, src_salt=peer)
                        if not passed:
                            return False, f"request={r} {detail}"
                    return True, ""

                record(measure(build, {**base, "batch": batch}, op, verify_side, verify,
                               lambda: pool.fill_byte(SENTINEL), sets))

    for isl in isls:
        cfg = kv_workload.plan_config(preset, args.precision, isl, pages[0], args.pool_slack)
        nbytes = min(cfg["req_bytes"], bulk_bytes)
        base = {"kind": "bulk", "preset": preset, "isl": isl, "page_tokens": None,
                "layers": cfg["layers"], "page_bytes": None, "descs": 1, "batch": 1,
                "req_bytes": nbytes}
        for op in ops:
            record(measure(
                lambda rep, op=op: [backend.make_bulk(nbytes, op)], base, op,
                "initiator" if op == "pull" else "target",
                lambda k, nbytes=nbytes: kv_workload.verify_bulk(bulk.words, nbytes,
                                                                src_salt=peer),
                lambda: bulk.fill_byte(SENTINEL), 1))

    backend.teardown()

    gathered: list = [None, None]
    dist.all_gather_object(gathered, (socket.gethostname(), rows if rank == 1 else None))
    hosts = [host for host, _ in gathered]
    rows = gathered[1][1] or []
    all_ok = bool(rows) and all(r["verify"]["passed"] for r in rows)

    if rank == 0:
        doc = ep_harness.case_attempt(
            args, {**case, "suite": args.suite}, ep_harness.git_run(),
            os.environ.get("COLLECTIVEX_IMAGE", ""), all_ok, "transfer verification failed",
            workload={
                "isl_ladder": isls,
                "page_tokens": pages,
                "batch_sizes": batches,
                "ops": ops,
                "seed": args.seed,
                "max_burst_tokens": args.max_burst_tokens or None,
                "preset": kv_workload.PRESETS[preset],
                "row_bytes": row_bytes,
            },
            measurement={
                "payload_unit": "request-kv-bytes",
                "rows": rows,
                # What each point actually ran after the burst cap and the pool
                # budget (the requested ladder is workload.batch_sizes).
                "points": [{"isl": cfg["isl"], "batches": allowed,
                            "table_sets": cfg["table_sets"]} for cfg, allowed in points],
                "pool_budget": args.pool_budget,
                "sampling": {
                    "reps_per_trial": args.reps,
                    "trials": args.trials,
                    "warmup_per_trial": args.warmup,
                },
            },
            implementation={
                "name": args.backend,
                "fabric": args.fabric,
                "library_version": backend.library_version,
                "maturity": backend.maturity,
                "nic_filter": backend.nic_filter,
                "transport": backend.transport,
                "engine_config": backend.engine_config,
            },
            topology={
                "device_product": torch.cuda.get_device_name(device),
                "gpus_per_node": args.gpus_per_node,
                "hosts": hosts,
                # The probed link layer (infiniband / roce / efa); mnnvl rows
                # ride the NVLink domain regardless.
                "network": ("mnnvl" if args.fabric == "mnnvl"
                            else os.environ.get("COLLX_RDMA_LINK_LAYER") or None),
                "nodes": 2,
                "ranks_per_node": 1,
                "scale_up_domain": args.scale_up_domain,
                "scale_up_transport": args.scale_up_transport or None,
                "topology_class": args.topology_class or None,
                "world_size": world_size,
            },
            runtime={
                "framework": str(torch.__version__),
                "vendor": "amd" if torch.version.hip else "nvidia",
            },
        )
        if args.out:
            ep_harness._write_json_atomic(args.out, doc)
        print(f"[run_kv] status={doc['outcome']['status']} rows={len(rows)}"
              + (f" -> {args.out}" if args.out else ""), flush=True)

    # all_ok is rank-invariant (both ranks hold rank 1's gathered rows); the
    # barrier keeps a failing rank 1 from exiting before rank 0 writes --out.
    dist.barrier()
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
