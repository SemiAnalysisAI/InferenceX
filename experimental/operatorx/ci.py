"""CPU-only planning and Slurm execution for the OperatorX Actions workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import pwd
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
POOLS = {
    "h100-dgxc": "h100_dgxc_8x",
    "h200-dgxc": "h200_dgxc_8x",
    "b200-nscale": "b200_nscale_8x",
    "b300": "b300_dsxe_8x",
    "gb200": "gb200_nvl72_4x",
    "gb300": "gb300_nvl72_4x",
    "mi300x": "mi300x_amds_8x",
    "mi325x": "mi325x_amds_8x",
    "mi355x": "mi355x_8x",
}
AMD_POOLS = {"mi300x", "mi325x", "mi355x"}


def load_platforms(path: Path) -> dict:
    document = json.loads(path.read_text())
    platforms = {}
    if "base" in document:
        base = path.parent / document["base"]
        platforms.update(json.loads(base.read_text())["platforms"])
    for pool, hardware in document["platforms"].items():
        platforms[pool] = {**platforms.get(pool, {}), **hardware}
    return platforms


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
    platforms: dict[str, dict],
) -> dict:
    if pool not in POOLS:
        raise ValueError(f"unsupported pool: {pool}")
    hardware = platforms[pool]
    gpus = hardware["gpus_per_node"]
    image_platform = hardware["image_platform"]
    if type(gpus) is not int or gpus not in (4, 8):
        raise ValueError("pool must supply four or eight GPUs per physical node")
    if image_platform not in ("linux/amd64", "linux/arm64"):
        raise ValueError("unsupported image platform")
    if not backends or set(backends) - images.keys():
        raise ValueError("select at least one registered backend for this GPU platform")
    if pool in AMD_POOLS and (
        set(backends) - {"torch", "aiter", "vllm"}
        or world_sizes != [1]
        or any(
            shape["type"] not in {"gemm", "attention_mha", "attention_mla", "moe_gemm"}
            for shapes in testlists.values()
            for shape in shapes
        )
    ):
        raise ValueError(
            "AMD CI supports single-GPU torch GEMM, torch/aiter attention and vllm MoE"
        )
    if not world_sizes or set(world_sizes) - {1, 2, 4, 8}:
        raise ValueError("world sizes must be selected from 1,2,4,8 (single node)")
    if any(ws > gpus for ws in world_sizes):
        raise ValueError(f"world size exceeds the pool's {gpus}-GPU physical node")
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
                    "gpus_per_node": gpus,
                    "image_platform": image_platform,
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


def probe_module():
    path = ROOT.parent / "CollectiveX/runtime/probe.py"
    spec = importlib.util.spec_from_file_location("collectivex_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def cleanup(root: Path, timeout_seconds: int) -> None:
    if timeout_seconds <= 0:
        raise ValueError("cleanup timeout must be positive")
    for job in allocation_ids(root):
        subprocess.run(["scancel", job], check=False, timeout=15)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            with (root / "cleanup.log").open("a") as log:
                log.write(
                    f"job={job} rc={state.returncode} active={state.stdout!r} "
                    f"error={state.stderr!r}\n"
                )
            if state.returncode == 0 and job not in state.stdout.split():
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"allocation {job} did not terminate; retaining staged evidence"
            )


def image_key(image: str, digest: str, image_platform: str) -> str:
    # Preserve the already-qualified amd64 cache while isolating Arm imports.
    suffix = "" if image_platform == "linux/amd64" else f":{image_platform}"
    return hashlib.sha256((image + digest + suffix).encode()).hexdigest()


def shared_base(profile: dict, pool: str) -> Path:
    """Resolve the pool's configured/shared account storage, never temporary HOME."""
    if profile.get("stage_dir"):
        roots = [Path(profile["stage_dir"])]
    elif pool in AMD_POOLS:
        runner_temp = Path(os.environ["RUNNER_TEMP"])
        if (
            runner_temp.parts[-2:] != ("_work", "_temp")
            or not runner_temp.is_absolute()
        ):
            raise ValueError("AMD staging requires the shared runner _work/_temp path")
        roots = [runner_temp.parent.parent]
    elif pool == "b300":
        # CollectiveX uses the compute-visible account home on this pool.
        # The shared squash parent is not writable by the GHA service account.
        roots = [Path(pwd.getpwuid(os.getuid()).pw_dir)]
    elif profile.get("squash_dir"):
        roots = [Path(profile["squash_dir"]).parent]
    else:
        roots = [Path(root) for root in profile["storage_roots"]]
    for root in roots:
        if root.is_dir() and os.access(root, os.W_OK | os.X_OK):
            return root / f".operatorx-{os.getuid()}"
    raise ValueError("no writable shared storage root configured for this pool")


def import_image(args) -> None:
    # Runs on the configured import host with a compute-visible cache and lock.
    import fcntl

    image, digest = args.image, args.digest
    machines = {"linux/amd64": {"x86_64", "amd64"}, "linux/arm64": {"aarch64", "arm64"}}
    if platform.machine() not in machines[args.image_platform]:
        raise ValueError("image platform does not match the import host")
    args.cache.mkdir(parents=True, exist_ok=True)
    key = image_key(image, digest, args.image_platform)
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
            host, repository, tag = probe_module().registry_reference(image)
            uri = f"docker://{host}#{repository}:{tag}"
            # B300 login/compute homes are node-local; every importer gets private
            # temporary paths. Preserve an explicitly configured shared cache.
            with tempfile.TemporaryDirectory(prefix="operatorx-enroot-") as scratch:
                env = dict(os.environ)
                env["TMPDIR"] = scratch
                for name in ("TEMP", "DATA", "RUNTIME"):
                    directory = Path(scratch) / name.lower()
                    directory.mkdir()
                    env[f"ENROOT_{name}_PATH"] = str(directory)
                if "ENROOT_CACHE_PATH" not in env:
                    cache = Path(scratch) / "cache"
                    cache.mkdir()
                    env["ENROOT_CACHE_PATH"] = str(cache)
                subprocess.run(
                    ["enroot", "import", "-o", str(temporary), uri],
                    stdin=subprocess.DEVNULL,
                    check=True,
                    env=env,
                )
            subprocess.run(["unsquashfs", "-s", str(temporary)], check=True)
            # The importer uses the tag, just like CollectiveX. Refuse a tag that moved
            # between planning and import rather than mislabelling the measurement.
            if probe_module().resolve_image_digest(image) != digest:
                raise RuntimeError(
                    "image tag moved or digest verification failed; dispatch again"
                )
            temporary.replace(squash)
        finally:
            temporary.unlink(missing_ok=True)


def finalize(root: Path, cleanup_seconds: int) -> None:
    if not root.exists():
        return
    cleanup(root, cleanup_seconds)
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


def recover(
    artifacts: Path, run_id: str, pool: str, platform_config: Path, cleanup_seconds: int
) -> None:
    profile = load_platforms(platform_config)[pool]["operator"]
    base = shared_base(profile, pool).resolve()
    recovered = 0
    for execution in artifacts.rglob("execution.json"):
        data = json.loads(execution.read_text())
        if data["run_id"] != run_id or data["cell"]["pool"] != pool:
            raise ValueError("recovery artifact does not match requested run/pool")
        stage = Path(data["stage"])
        if stage.parent.resolve() != base:
            raise ValueError("recovery stage does not belong to this pool/user")
        finalize(execution.parent, cleanup_seconds)
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
    hardware = load_platforms(args.platform_config)[cell["pool"]]
    profile = hardware["operator"]
    gpus = hardware["gpus_per_node"]
    image_platform = hardware["image_platform"]
    if (
        cell["gpus_per_node"] != gpus
        or cell["image_platform"] != image_platform
        or cell["world_size"] > gpus
    ):
        raise ValueError("manifest hardware differs from the selected pool")
    base = shared_base(profile, cell["pool"])
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
            f"--gres=gpu:{gpus}",
            f"--ntasks-per-node={gpus}",
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
        if cell["pool"] in ("b200-nscale", "b300", "gb200", "gb300"):
            allocation.append("--mem=0")
        if cell["pool"] in ("gb200", "gb300"):
            allocation.append("--cpus-per-task=35")
        if cell["pool"] in AMD_POOLS:
            allocation.append(f"--cpus-per-task={profile['cpus_per_node'] // gpus}")
        command(allocation, root / "allocation.log")
        jobs = allocation_ids(root)
        if len(jobs) != 1:
            raise RuntimeError("could not identify the unique Slurm allocation")
        job = jobs[0]
        cache = base / "containers"
        launcher = stage / "source/experimental/operatorx/ci.py"
        import_command = [
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
            "--image-platform",
            image_platform,
        ]
        # Import on the allocated architecture, including B300: its submit host
        # lacks PyTorch extraction space, as the inference launcher notes.
        import_env = dict(os.environ)
        if profile.get("enroot_cache_path"):
            import_env["ENROOT_CACHE_PATH"] = profile["enroot_cache_path"]
        command(import_command, root / "import.log", env=import_env)
        key = image_key(cell["image"], cell["digest"], image_platform)
        env = dict(os.environ)
        env.pop("OPERATORX_MOE_PARALLELISM", None)
        env.update(
            OPERATORX_CLUSTER=cell["cluster"],
            OPERATORX_CONTAINER_IMAGE=cell["image"],
            OPERATORX_IMAGE_DIGEST=cell["digest"],
            OPERATORX_IMAGE_PLATFORM=image_platform,
            OPERATORX_GPUS_PER_NODE=str(gpus),
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
        mounts = f"{stage}:/opx"
        if cell["pool"] in ("mi300x", "mi325x"):
            mounts += ",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri"
        run = [
            "srun",
            f"--jobid={job}",
            "--nodes=1",
            f"--ntasks={cell['world_size']}",
            f"--ntasks-per-node={cell['world_size']}",
            "--kill-on-bad-exit=1",
            "--chdir=/tmp",
            f"--container-image={cache / (key + '.sqsh')}",
            f"--container-mounts={mounts}",
            "--container-workdir=/opx",
            "--no-container-mount-home",
            "--no-container-entrypoint",
            "--container-writable",
            "--export=ALL",
        ]
        if cell["pool"] in {"h200-dgxc", "b300", "gb200", "gb300"} | AMD_POOLS:
            run.append("--container-remap-root")
        if cell["pool"] == "b300":
            run.append("--mpi=none")
        # The Python entrypoint preserves the allocated GPU mask without a shell.
        run += ["python3", "-m", "operatorx.ci", "rank"]
        command(run, root / "benchmark.log", env=env)
        rc = 0
    finally:
        # Stop writers before collecting; failed cleanup retains the evidence.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        finalize(root, args.cleanup_seconds)
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
    p.add_argument("--platform-config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("execute")
    for name in ("shard", "run-id", "attempt", "source-sha", "runner-name"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--time-minutes", required=True, type=int)
    p.add_argument("--cleanup-seconds", required=True, type=int)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--platform-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("import")
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--image", required=True)
    p.add_argument("--digest", required=True)
    p.add_argument(
        "--image-platform", required=True, choices=("linux/amd64", "linux/arm64")
    )
    sub.add_parser("rank")
    p = sub.add_parser("finalize")
    p.add_argument("--cleanup-seconds", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("recover")
    p.add_argument("--cleanup-seconds", required=True, type=int)
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
        vendor = "amd" if args.pool in AMD_POOLS else "nvidia"
        images = tomllib.loads((ROOT / "containers.toml").read_text())[vendor]
        result = plan(
            args.pool,
            args.backends.split(","),
            lists,
            images,
            [int(w) for w in args.world_sizes.split(",")],
            args.chunk_size,
            load_platforms(args.platform_config),
        )
        digests = {c["image"]: "" for c in result["include"]}
        for image in digests:
            digest = probe_module().resolve_image_digest(image)
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
            "| Shard | Status | Shapes requested | OK rows | "
            "Unsupported rows | Error rows |"
        )
        print("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in report["shards"]:
            print(
                f"| {row['shard']} | {row['status']} | {row['requested_shapes']} | "
                f"{row['ok']} | {row['unsupported']} | {row['error']} |"
            )
        raise SystemExit(0 if report["success"] else 1)
    elif args.command == "recover":
        recover(
            args.artifacts,
            args.run_id,
            args.pool,
            args.platform_config,
            args.cleanup_seconds,
        )
    elif args.command == "execute":
        execute(args)
    elif args.command == "finalize":
        finalize(args.output, args.cleanup_seconds)
    elif args.command == "import":
        import_image(args)
    else:
        rank()


if __name__ == "__main__":
    main()
