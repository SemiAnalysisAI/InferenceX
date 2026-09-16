"""CPU-only planning and Slurm execution for the OperatorX Actions workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOLS = {"h100-dgxc": "h100_dgxc_8x", "h200-dgxc": "h200_dgxc_8x"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan(
    pool: str,
    backends: list[str],
    testlists: dict[str, list[dict]],
    images: dict[str, dict],
    world_sizes: list[int],
    chunk_size: int,
) -> dict:
    if pool not in POOLS:
        raise ValueError(f"unsupported pool: {pool}")
    if not backends or set(backends) - images.keys():
        raise ValueError("select at least one registered NVIDIA backend")
    if not world_sizes or set(world_sizes) - {1, 2, 4, 8}:
        raise ValueError("world sizes must be selected from 1,2,4,8 (single node)")
    if not 1 <= chunk_size <= 500:
        raise ValueError("chunk size must be between 1 and 500")
    groups = defaultdict(list)
    excluded = 0
    for name, shapes in sorted(testlists.items()):
        for shape in shapes:
            args = shape["args"]
            ws = args.get("world_size", 1)
            if type(ws) is not int or ws < 1:
                raise ValueError("world_size must be a positive integer")
            if ws not in world_sizes:
                excluded += 1
                continue
            moe = (
                tuple(
                    args.get(key, 1)
                    for key in (
                        "expert_parallel_size",
                        "routed_tensor_parallel_size",
                        "shared_tensor_parallel_size",
                    )
                )
                if shape["type"] == "moe_forward"
                else ()
            )
            groups[(ws, moe)].append({"testlist": name, "shape": shape})
    image_groups = defaultdict(list)
    for backend in sorted(set(backends)):
        image_groups[images[backend]["image"]].append(backend)
    cells = []
    for image, selected in sorted(image_groups.items()):
        for (ws, moe), cases in sorted(groups.items()):
            for offset in range(0, len(cases), chunk_size):
                cell = {
                    "pool": pool,
                    "cluster": POOLS[pool],
                    "nodes": 1,
                    "world_size": ws,
                    "moe": moe,
                    "image": image,
                    "backends": selected,
                    "offset": offset,
                    "cases": cases[offset : offset + chunk_size],
                }
                identity = hashlib.sha256(
                    json.dumps(cell, sort_keys=True).encode()
                ).hexdigest()[:16]
                cells.append({"id": f"{pool}-{identity}", **cell})
    if not cells:
        raise ValueError("selection contains no runnable shapes")
    if len(cells) > 256:
        raise ValueError(
            "selection exceeds 256 shards; narrow the testlists/backends/world sizes"
        )
    return {"version": 1, "excluded_shapes": excluded, "include": cells}


def digest_probe():
    path = ROOT.parent / "CollectiveX/runtime/probe.py"
    spec = importlib.util.spec_from_file_location("collectivex_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_image_digest


def command(argv: list[str], log: Path, *, env=None) -> None:
    print(f"[operatorx] phase={log.stem}", flush=True)
    with log.open("a") as stream:
        process = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=env
        )
        try:
            rc = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if rc:
            print(log.read_text()[-16000:], file=sys.stderr)
            raise subprocess.CalledProcessError(rc, argv)


def allocation_ids(root: Path) -> list[str]:
    log = root / "allocation.log"
    if not log.exists():
        return []
    return sorted(
        set(re.findall(r"(?:Granted|Pending) job allocation ([0-9]+)", log.read_text()))
    )


def cleanup(root: Path) -> None:
    for job in allocation_ids(root):
        subprocess.run(["scancel", job], check=False, timeout=15)
        for _ in range(15):
            state = subprocess.run(
                ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            with (root / "cleanup.log").open("a") as log:
                log.write(
                    f"job={job} rc={state.returncode} active={state.stdout!r} error={state.stderr!r}\n"
                )
            if state.returncode == 0 and job not in state.stdout.split():
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"allocation {job} did not terminate; retaining staged evidence"
            )


def import_image(args) -> None:
    # Runs on the allocated compute node; the cache and lock are shared with the login node.
    import fcntl

    image, digest = args.image, args.digest
    args.cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((image + digest).encode()).hexdigest()
    squash = args.cache / f"{key}.sqsh"
    with (args.cache / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            squash.exists()
            and subprocess.run(
                ["unsquashfs", "-s", str(squash)],
                stdout=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        ):
            return
        temporary = squash.with_suffix(".partial")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                ["enroot", "import", "-o", str(temporary), f"docker://{image}"],
                stdin=subprocess.DEVNULL,
                check=True,
            )
            subprocess.run(["unsquashfs", "-s", str(temporary)], check=True)
            # The importer uses the tag, just like CollectiveX. Refuse a tag that moved
            # between hosted planning and import instead of mislabelling the measurement.
            if digest_probe()(image) != digest:
                raise RuntimeError(
                    "image tag moved or digest verification failed; dispatch again"
                )
            temporary.replace(squash)
        finally:
            temporary.unlink(missing_ok=True)


def finalize(root: Path) -> None:
    if not root.exists():
        return
    cleanup(root)
    execution = root / "execution.json"
    if execution.exists():
        data = json.loads(execution.read_text())
        stage = Path(data["stage"])
        expected_parent = f".operatorx-{os.getuid()}"
        if (
            stage.parent.name != expected_parent
            or not stage.name.startswith(
                f"{data['run_id']}-{data['attempt']}-{data['cell']['id']}-"
            )
            or stage.is_symlink()
        ):
            raise ValueError("refusing unsafe stage cleanup")
        if (stage / "results").exists():
            shutil.copytree(stage / "results", root / "results", dirs_exist_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


def recover(artifacts: Path, run_id: str, pool: str, platform_config: Path) -> None:
    profile = json.loads(platform_config.read_text())["platforms"][pool]["operator"]
    base = (Path(profile["squash_dir"]).parent / f".operatorx-{os.getuid()}").resolve()
    recovered = 0
    for execution in artifacts.rglob("execution.json"):
        data = json.loads(execution.read_text())
        if data["run_id"] != run_id or data["cell"]["pool"] != pool:
            raise ValueError("recovery artifact does not match requested run/pool")
        stage = Path(data["stage"])
        if stage.parent.resolve() != base:
            raise ValueError("recovery stage does not belong to this pool/user")
        finalize(execution.parent)
        recovered += 1
    if not recovered:
        raise ValueError("no execution artifacts found to recover")
    print(f"Recovered {recovered} execution(s) from run {run_id}", flush=True)


def execute(args) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = json.loads(args.manifest.read_text())
    if manifest["source_sha"] != args.source_sha or manifest["run_id"] != args.run_id:
        raise ValueError("manifest source/run mismatch")
    cells = manifest["include"]
    cell = next(c for c in cells if c["id"] == args.shard)
    config = json.loads(args.platform_config.read_text())
    profile = config["platforms"][cell["pool"]]["operator"]
    # Both initial pools have a shared squash parent; /tmp on the submit host is not shared.
    base = Path(profile["squash_dir"]).parent / f".operatorx-{os.getuid()}"
    base.mkdir(mode=0o700, exist_ok=True)
    if (
        base.is_symlink()
        or base.stat().st_uid != os.getuid()
        or base.stat().st_mode & 0o077
    ):
        raise RuntimeError("unsafe shared OperatorX stage directory")
    stage = Path(
        tempfile.mkdtemp(prefix=f"{args.run_id}-{args.attempt}-{cell['id']}-", dir=base)
    )
    write_json(
        root / "execution.json",
        {
            "cell": cell,
            "source_sha": args.source_sha,
            "run_id": args.run_id,
            "attempt": args.attempt,
            "stage": str(stage),
        },
    )

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    rc = 1
    try:
        shutil.copytree(
            ROOT,
            stage / "source/experimental/operatorx",
            ignore=shutil.ignore_patterns("__pycache__", "results", ".venv", "tests"),
        )
        shutil.copytree(
            ROOT.parent / "CollectiveX/runtime",
            stage / "source/experimental/CollectiveX/runtime",
        )
        for name in sorted({c["testlist"] for c in cell["cases"]}):
            write_json(
                stage / "testlists" / f"{name}.json",
                [c["shape"] for c in cell["cases"] if c["testlist"] == name],
            )
        (stage / "results").mkdir()
        allocation = [
            "salloc",
            "--no-shell",
            f"--partition={profile['partition']}",
            "--nodes=1",
            "--gres=gpu:8",
            "--ntasks-per-node=8",
            "--exclusive",
            f"--time={args.time_minutes}",
            f"--job-name={args.runner_name}",
        ]
        for field, flag in (
            ("account", "account"),
            ("qos", "qos"),
            ("exclude_nodes", "exclude"),
        ):
            if profile.get(field):
                allocation.append(f"--{flag}={profile[field]}")
        command(allocation, root / "allocation.log")
        jobs = allocation_ids(root)
        if len(jobs) != 1:
            raise RuntimeError("could not identify the unique Slurm allocation")
        job = jobs[0]
        cache = base / "containers"
        launcher = stage / "source/experimental/operatorx/ci.py"
        command(
            [
                "srun",
                f"--jobid={job}",
                "--nodes=1",
                "--ntasks=1",
                "--chdir=/tmp",
                "python3",
                str(launcher),
                "import",
                "--cache",
                str(cache),
                "--image",
                cell["image"],
                "--digest",
                cell["digest"],
            ],
            root / "import.log",
        )
        key = hashlib.sha256((cell["image"] + cell["digest"]).encode()).hexdigest()
        env = dict(os.environ)
        env.pop("OPERATORX_MOE_PARALLELISM", None)
        env.update(
            OPERATORX_CLUSTER=cell["cluster"],
            OPERATORX_CONTAINER_IMAGE=cell["image"],
            OPERATORX_IMAGE_DIGEST=cell["digest"],
            OPERATORX_SOURCE_SHA=args.source_sha,
            OPERATORX_GITHUB_RUN_ID=args.run_id,
            OPERATORX_GITHUB_RUN_ATTEMPT=args.attempt,
            OPERATORX_SHARD_ID=cell["id"],
            OPERATORX_BACKENDS=",".join(cell["backends"]),
            OPERATORX_TESTLISTS=",".join(
                sorted({c["testlist"] for c in cell["cases"]})
            ),
            PYTHONPATH="/opx/source/experimental",
            PYTHONDONTWRITEBYTECODE="1",
            WORLD_SIZE=str(cell["world_size"]),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
        if cell["moe"]:
            env["OPERATORX_MOE_PARALLELISM"] = ":".join(map(str, cell["moe"]))
        run = [
            "srun",
            f"--jobid={job}",
            "--nodes=1",
            f"--ntasks={cell['world_size']}",
            f"--ntasks-per-node={cell['world_size']}",
            "--kill-on-bad-exit=1",
            "--chdir=/tmp",
            f"--container-image={cache / (key + '.sqsh')}",
            f"--container-mounts={stage}:/opx",
            "--container-workdir=/opx",
            "--no-container-mount-home",
            "--no-container-entrypoint",
            "--container-writable",
            "--export=ALL",
        ]
        if cell["pool"] == "h200-dgxc":
            run.append("--container-remap-root")
        # Python rank entrypoint avoids shell interpolation and preserves the allocated GPU mask.
        run += ["python3", "-m", "operatorx.ci", "rank"]
        command(run, root / "benchmark.log", env=env)
        rc = 0
    finally:
        # Stop writers before collecting or deleting anything. A failed cleanup retains evidence.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        finalize(root)
        write_json(
            root / "status.json", {"exit_code": rc, "slurm_jobs": allocation_ids(root)}
        )


def rank() -> None:
    for key in ("SLURM_PROCID", "SLURM_LOCALID", "WORLD_SIZE"):
        if not os.environ.get(key):
            raise ValueError(f"required rank input missing: {key}")
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    cache = (
        Path("/tmp") / f"operatorx-{os.environ['SLURM_JOB_ID']}-{os.environ['RANK']}"
    )
    cache.mkdir(mode=0o700, exist_ok=True)
    os.environ.update(
        HOME=str(cache),
        TRITON_CACHE_DIR=str(cache / "triton"),
        MPLCONFIGDIR=str(cache / "matplotlib"),
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "operatorx",
            "--strict",
            "--testlist-dir",
            "/opx/testlists",
            "--results-dir",
            "/opx/results",
        ],
    )


def summarize(manifest: dict, artifacts: Path) -> dict:
    latest = {}
    for execution in artifacts.rglob("execution.json"):
        data = json.loads(execution.read_text())
        if (
            data["run_id"] != manifest["run_id"]
            or data["source_sha"] != manifest["source_sha"]
        ):
            raise ValueError("artifact provenance does not match the requested run")
        identity = data["cell"]["id"]
        attempt = int(data["attempt"])
        if identity not in latest or attempt > latest[identity][0]:
            latest[identity] = (attempt, execution.parent)
    rows = []
    for cell in manifest["include"]:
        record = {
            "shard": cell["id"],
            "requested_shapes": len(cell["cases"]),
            "status": "missing",
            "ok": 0,
            "unsupported": 0,
            "error": 0,
        }
        if cell["id"] in latest:
            attempt, directory = latest[cell["id"]]
            record["attempt"] = attempt
            for result in (directory / "results").rglob("*.json"):
                for row in json.loads(result.read_text())["rows"]:
                    status = row["status"]
                    if status not in ("ok", "unsupported", "error"):
                        raise ValueError(f"invalid result status: {status}")
                    record[status] += 1
            status_file = directory / "status.json"
            success = (
                status_file.exists()
                and json.loads(status_file.read_text())["exit_code"] == 0
                and record["ok"] > 0
                and record["error"] == 0
            )
            record["status"] = "success" if success else "failed"
        rows.append(record)
    return {
        "shards": rows,
        "success": bool(rows) and all(r["status"] == "success" for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in (
        "pool",
        "backends",
        "testlists",
        "world-sizes",
        "run-id",
        "attempt",
        "source-sha",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--chunk-size", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("execute")
    for name in ("shard", "run-id", "attempt", "source-sha", "runner-name"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--time-minutes", required=True, type=int)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--platform-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("import")
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--image", required=True)
    p.add_argument("--digest", required=True)
    sub.add_parser("rank")
    p = sub.add_parser("finalize")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("recover")
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--pool", required=True, choices=tuple(POOLS))
    p.add_argument("--platform-config", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        import tomllib

        names = args.testlists.split(",")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
            raise ValueError("invalid testlist name")
        lists = {
            n: json.loads((ROOT / "testlists" / f"{n}.json").read_text()) for n in names
        }
        images = tomllib.loads((ROOT / "containers.toml").read_text())["nvidia"]
        result = plan(
            args.pool,
            args.backends.split(","),
            lists,
            images,
            [int(w) for w in args.world_sizes.split(",")],
            args.chunk_size,
        )
        digests = {c["image"]: "" for c in result["include"]}
        for image in digests:
            digest = digest_probe()(image)
            digests[image] = digest
            if not digest:
                raise RuntimeError(f"cannot resolve image digest: {image}")
        for cell in result["include"]:
            cell["digest"] = digests[cell["image"]]
            cell["queue-token"] = hashlib.sha256(
                f"{args.run_id}:{args.attempt}:{cell['id']}".encode()
            ).hexdigest()[:32]
        result.update(
            source_sha=args.source_sha, run_id=args.run_id, attempt=args.attempt
        )
        write_json(args.out, result)
        slim = {
            "include": [
                {k: c[k] for k in ("id", "pool", "nodes", "queue-token")}
                for c in result["include"]
            ]
        }
        print(json.dumps(slim, separators=(",", ":")))
    elif args.command == "summarize":
        report = summarize(json.loads(args.manifest.read_text()), args.artifacts)
        write_json(args.out, report)
        print(
            "| Shard | Status | Shapes requested | OK rows | Unsupported rows | Error rows |"
        )
        print("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in report["shards"]:
            print(
                f"| {row['shard']} | {row['status']} | {row['requested_shapes']} | "
                f"{row['ok']} | {row['unsupported']} | {row['error']} |"
            )
        raise SystemExit(0 if report["success"] else 1)
    elif args.command == "recover":
        recover(args.artifacts, args.run_id, args.pool, args.platform_config)
    elif args.command == "execute":
        execute(args)
    elif args.command == "finalize":
        finalize(args.output)
    elif args.command == "import":
        import_image(args)
    else:
        rank()


if __name__ == "__main__":
    main()
