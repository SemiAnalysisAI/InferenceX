#!/usr/bin/env python3
"""CollectiveX KV-cache transfer entrypoint: rank 0 is the target, rank 1 the timed initiator.

Every trial's last burst lands on a destination wiped to a sentinel, so the verify
after it proves that timed burst moved every entry (docs/kv-transfer.md).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ep_harness  # noqa: E402  (stdlib-only; safe before torch)
import kv_workload  # noqa: E402

BULK_CAP = 8 << 30
SENTINEL = 0x5A
# UCX settings a row records: selectors and transports, never addresses.
UCX_RECORDED = ("UCX_NET_DEVICES", "UCX_TLS", "UCX_IB_GID_INDEX", "UCX_MAX_RNDV_RAILS",
                "UCX_RCACHE_MAX_UNRELEASED", "UCX_CUDA_IPC_ENABLE_MNNVL")


def add_kv_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--workload-name", required=True, help="kv-<preset>, e.g. kv-dsv4")
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"])
    ap.add_argument("--fabric", default="rdma", choices=["rdma", "mnnvl"],
                    help="which lane the SKU row claims; mnnvl additionally sets "
                         "UCX_CUDA_IPC_ENABLE_MNNVL=y for the UCX-backed libraries")
    ap.add_argument("--isl-ladder", default="512 4096 32768")
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
    ap.add_argument("--kv-mc-port", type=int, default=48830)


def export_ucx_selectors(environ=os.environ) -> None:
    """Pin UCX to the operator's validated RDMA selectors (auto-selection lands on
    the wrong fabric on some pools, e.g. b200-nscale's aux quad-port card); UCX_*
    values the environment already sets win."""
    devices = environ.get("COLLX_RDMA_DEVICES", "")
    if devices and "UCX_NET_DEVICES" not in environ:
        environ["UCX_NET_DEVICES"] = ",".join(
            dev if ":" in dev else f"{dev}:1" for dev in devices.split(",") if dev)
    gid = environ.get("COLLX_IB_GID_INDEX", "")
    if gid and "UCX_IB_GID_INDEX" not in environ:
        environ["UCX_IB_GID_INDEX"] = str(gid)


def exchange_verdict(dist, role, verify_side, verify, error: str = ""):
    """Every rank returns one verdict: a rank's own error first, else the verifying
    rank's check of its destination pool (a raising check fails the row)."""
    verdict = {"passed": False, "detail": error} if error else None
    if verdict is None and role == verify_side:
        try:
            passed, detail = verify()
        except Exception as exc:  # noqa: BLE001 - the row fails; both ranks must still agree
            passed, detail = False, f"verify raised {exc!r}"
        verdict = {"passed": passed, "detail": detail}
    gathered = [None, None]
    dist.all_gather_object(gathered, verdict)
    failed = [v for v in gathered if v is not None and not v["passed"]]
    return failed[0] if failed else gathered[0 if verify_side == "target" else 1]


def time_bursts(build, warmup: int, reps: int, settle,
                rep0: int = 0) -> tuple[list[float], list[float]]:
    """(burst_ms, request_ms), warmups dropped. ``build(rep)`` runs untimed; a burst
    posts every request, then drains the waits in posting order, so each request's
    completion mark is an upper bound and the burst is its last mark."""
    burst_ms: list[float] = []
    request_ms: list[float] = []
    for rep in range(warmup + reps):
        transfers = build(rep0 + rep)
        start = time.perf_counter()
        for post, _ in transfers:
            post()
        marks = []
        for _, wait in transfers:
            wait()
            marks.append((time.perf_counter() - start) * 1e3)
        settle()
        if rep >= warmup:
            burst_ms.append(marks[-1])
            request_ms.extend(marks)
    return burst_ms, request_ms


def case_attempt(args, case: dict, ok: bool, failure: str, **sections) -> dict:
    """The case-attempt document, keyed and validated the way run_ep's is: the upload checks
    record_type, identity, the cases.json version and a committed (non-dirty) source, and
    rejects any document naming a machine, so no hostname is recorded."""
    try:
        attempt_ordinal = int(os.environ.get("COLLX_ATTEMPT_ID", "1"))
    except ValueError:
        attempt_ordinal = 0
    if attempt_ordinal <= 0:
        raise ValueError("COLLX_ATTEMPT_ID must be a positive integer")
    run = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
    }
    return {
        "version": args.version,
        "record_type": "case-attempt",
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "identity": {
            "allocation_factors": {key: run[key] for key in ("run_attempt", "run_id", "source_sha")},
            "attempt_ordinal": attempt_ordinal,
            "case_factors": {"case": case, "sku": args.runner},
            "case_id": args.case_id,
        },
        **sections,
        "provenance": {
            "image": os.environ.get("COLLECTIVEX_IMAGE") or None,
            "source_sha": run["source_sha"],
            # run.sh records whether collectivex/ had uncommitted edits; such results never publish.
            "source_dirty": {"0": False, "1": True}.get(os.environ.get("COLLECTIVEX_SOURCE_DIRTY")),
        },
        "outcome": {"reasons": [] if ok else [failure], "status": "success" if ok else "invalid"},
    }


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
    """(cfg, batches) per ISL: the batches whose burst stays under
    --max-burst-tokens of prompt (batch 1 always runs), planned for the largest.
    Every cfg then shares one pool, the most rows any point plans, so batch is
    the only variable across a point's rows."""
    preset = args.workload_name.removeprefix("kv-")
    isls = [int(v) for v in args.isl_ladder.split()]
    batches = sorted({1, *(int(v) for v in args.batch_sizes.split())})
    points = []
    for isl in isls:
        allowed = [batch for batch in batches if batch == 1 or not args.max_burst_tokens
                   or batch * isl <= args.max_burst_tokens]
        points.append((kv_workload.plan_config(preset, args.precision, isl, args.pool_slack,
                                               batch_max=allowed[-1]), allowed))
    rows = max(cfg["pool_rows"] for cfg, _ in points)
    for cfg, _ in points:
        cfg["pool_rows"], cfg["pool_bytes"] = rows, rows * cfg["row_bytes"]
    return points, isls, batches


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX KV-cache transfer sweep")
    ap.add_argument("--backend", required=True, choices=["nixl", "mooncake"])
    add_kv_args(ap)
    args = ap.parse_args()

    case = kv_case(args)
    computed_case_id = ep_harness.case_id(args.runner, case)
    if args.case_id and args.case_id != computed_case_id:
        print(f"ERROR: scheduled case ID does not match factors: "
              f"{args.case_id} != {computed_case_id}", file=sys.stderr)
        return 2
    args.case_id = args.case_id or computed_case_id
    if args.backend == "mooncake" and args.ops.split() != ["push"]:
        print("ERROR: vLLM's MooncakeConnector only pushes; run mooncake with --ops push",
              file=sys.stderr)
        return 2

    if args.fabric == "mnnvl":
        os.environ.setdefault("UCX_CUDA_IPC_ENABLE_MNNVL", "y")
        # Without InfiniBand transports a failed NVLink attach fails the row instead of
        # quietly measuring the IB rails under an mnnvl label.
        os.environ["UCX_TLS"] = "cuda_copy,cuda_ipc,tcp"
    else:
        # vLLM never pins NICs; the RDMA lanes pin UCX to the operator's validated
        # selectors only because auto-selection lands on the wrong fabric on some pools.
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
    # A slow grid point can outlast gloo's 30-minute default while the target waits at
    # the next gather; the per-case hang guard (COLLX_RUN_TIMEOUT) ends a dead run, not gloo.
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=_dt.timedelta(days=1))

    if args.backend == "mooncake":
        from kv_mooncake import MooncakeBackend as Backend
    else:
        from kv_nixl import NIXLBackend as Backend

    points, isls, batches = _grid(args)
    ops = args.ops.split()
    pool_bytes = points[0][0]["pool_bytes"]  # _grid gives every cfg the one shared pool
    row_bytes = points[0][0]["row_bytes"]
    preset = args.workload_name.removeprefix("kv-")
    bulk_bytes = min(max(cfg["req_bytes"] for cfg, _ in points), BULK_CAP)

    # RDMA registration pins the whole pool, and Slurm propagates the submitter's
    # soft memlock limit into steps, which fails it deep inside the library: raise
    # soft to hard, else fail here with the numbers.
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    need = pool_bytes + bulk_bytes
    if hard != resource.RLIM_INFINITY and hard < need:
        print(f"ERROR: RLIMIT_MEMLOCK {hard} < {need} needed to register the KV pools; "
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
    backend.register(pool, bulk, row_bytes)
    payloads = [None, None]
    dist.all_gather_object(payloads, backend.publish())
    backend.connect(payloads[1 - rank])
    dist.barrier()
    if rank == 1:
        print(f"[run_kv] backend={args.backend} workload={args.workload_name} "
              f"precision={args.precision} fabric={args.fabric} isls={isls} "
              f"batches={batches} pool={pool_bytes >> 20}MiB case={args.case_id}",
              flush=True)

    rows: list[dict] = []

    def record(row):
        if row is not None:
            rows.append(row)
            print(f"[run_kv] {json.dumps(row)}", flush=True)

    def measure(build, cfg_row: dict, op: str, verify, wipe, table_sets: int):
        """One grid point: per trial, warmup + reps-1 bursts, then the destination
        owner (the initiator on pull, the target on push) wipes to SENTINEL, the
        initiator times the last burst, and the owner verifies that burst's table set."""
        verify_side = "initiator" if op == "pull" else "target"
        samples: list[float] = []
        request_samples: list[float] = []
        verdict = {"passed": True, "detail": ""}
        rep, error = 0, ""
        # With other table sets to write, an untimed burst follows the wipe so the
        # verified burst is not the first after the barriers; it lands on other rows.
        settle = 1 if table_sets > 1 else 0

        def timed(warmup, reps):
            nonlocal error
            if role != "initiator" or error:
                return
            try:
                burst_ms, request_ms = time_bursts(build, warmup, reps, backend.release, rep)
            except Exception as exc:  # noqa: BLE001 - reported through the verdict
                error = f"transfer raised {exc!r}"
                return
            samples.extend(burst_ms)
            request_samples.extend(request_ms)

        for _ in range(args.trials):
            timed(args.warmup, args.reps - 1)
            rep += args.warmup + args.reps - 1
            dist.barrier()
            if role == verify_side:
                wipe()
            dist.barrier()
            timed(settle, 1)
            rep += settle
            final_set, seed = rep % table_sets, rep
            rep += 1
            dist.barrier()  # the last burst completes before anyone inspects it
            verdict = exchange_verdict(dist, role, verify_side,
                                       lambda: verify(final_set, seed), error)
            if not verdict["passed"]:
                break
        repaint()
        dist.barrier()
        if role != "initiator":
            return None
        row = {**cfg_row, "op": op, "latency_ms": None, "request_ms": None,
               "gbps_p50": None, "verify": verdict}
        if not samples:  # the first burst failed; the verdict says why
            return row
        stats = kv_workload.pcts(samples)
        row["latency_ms"] = {k: round(v, 3) for k, v in stats.items()}
        row["gbps_p50"] = round(cfg_row["req_bytes"] * cfg_row["batch"] / stats["p50"] / 1e6, 2)
        # Each request's host-observed completion within its burst (an upper bound, as
        # waits drain in posting order), where the library completes requests one by one.
        if len(request_samples) == len(samples) * cfg_row["batch"]:
            row["request_ms"] = {k: round(v, 3) for k, v in kv_workload.pcts(request_samples).items()}
        return row

    peer = 1 - rank
    for cfg, allowed in points:
        sets = cfg["table_sets"]
        seed_t = kv_workload.table_seed(cfg, "remote", args.seed)
        seed_i = kv_workload.table_seed(cfg, "local", args.seed)
        target_tables = [[kv_workload.block_table(cfg, seed_t, r, k) for r in range(allowed[-1])]
                         for k in range(sets)]
        initiator_tables = [[kv_workload.block_table(cfg, seed_i, r, k)
                             for r in range(allowed[-1])] for k in range(sets)]
        # req_bytes is the request's KV (its block rows), the same for every
        # backend, so GB/s compares libraries on one payload; descs and
        # posted_bytes are what this backend's vLLM shape posts for it (NIXL: a
        # whole row per id; Mooncake: its transfer regions).
        _, _, sizes = backend.request_entries(cfg, initiator_tables[0][0],
                                              target_tables[0][0])
        base = {
            "kind": "paged", "preset": cfg["preset"], "isl": cfg["isl"],
            "page_tokens": cfg["page_tokens"], "layers": cfg["layers"],
            "row_bytes": cfg["row_bytes"], "descs": int(len(sizes)),
            "req_bytes": cfg["req_bytes"], "posted_bytes": int(sizes.sum()),
        }
        for batch in allowed:
            for op in ops:
                # Called by measure on the initiator only, within this iteration.
                def build(rep, batch=batch, op=op):
                    k = rep % sets
                    return backend.make_burst(cfg, op, [
                        (initiator_tables[k][r], target_tables[k][r], rep * batch + r)
                        for r in range(batch)])

                # Every request in the burst is checked against its own tables.
                def verify(k, seed, batch=batch, op=op):
                    for r in range(batch):
                        local, remote, sizes = backend.request_entries(
                            cfg, initiator_tables[k][r], target_tables[k][r])
                        dst, src = (local, remote) if op == "pull" else (remote, local)
                        passed, detail = kv_workload.verify_entries(
                            pool.words, dst, src, sizes, src_salt=peer, seed=seed * 64 + r)
                        if not passed:
                            return False, f"request={r} {detail}"
                    return True, ""

                record(measure(build, {**base, "batch": batch}, op, verify,
                               lambda: pool.fill_byte(SENTINEL), sets))

    for cfg, _ in points:
        nbytes = min(cfg["req_bytes"], bulk_bytes)
        base = {"kind": "bulk", "preset": preset, "isl": cfg["isl"], "page_tokens": None,
                "layers": cfg["layers"], "descs": 1, "batch": 1, "req_bytes": nbytes}
        for op in ops:
            record(measure(
                lambda rep, op=op: [backend.make_bulk(nbytes, op)], base, op,
                lambda k, seed, nbytes=nbytes: kv_workload.verify_bulk(
                    bulk.words, nbytes, src_salt=peer, seed=seed),
                lambda: bulk.fill_byte(SENTINEL), 1))

    backend.teardown()

    gathered: list = [None, None]
    dist.all_gather_object(gathered, rows if rank == 1 else None)
    rows = gathered[1] or []
    all_ok = bool(rows) and all(r["verify"]["passed"] for r in rows)

    if rank == 0:
        doc = case_attempt(
            args, {**case, "suite": args.suite}, all_ok, "transfer verification failed",
            workload={
                "isl_ladder": isls,
                "page_tokens": [points[0][0]["page_tokens"]],
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
                # What each point actually ran after the burst cap (the requested
                # ladder is workload.batch_sizes).
                "points": [{"isl": cfg["isl"], "batches": allowed,
                            "table_sets": cfg["table_sets"]} for cfg, allowed in points],
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
                "transport": backend.transport,
                "library_config": {
                    **backend.library_config,
                    # The UCX settings the row ran under, wherever they came from.
                    **({"ucx_env": {key: os.environ[key] for key in UCX_RECORDED
                                    if key in os.environ}}
                       if backend.transport == "ucx" else {}),
                },
            },
            topology={
                "device_product": torch.cuda.get_device_name(device),
                "gpus_per_node": args.gpus_per_node,
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
