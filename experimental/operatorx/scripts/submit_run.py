#!/usr/bin/env python3
"""Submit SLURM jobs to run operatorx (python -m operatorx) for every backend on a platform.

For each (image, world_size) combination, submits one srun job that launches
world_size tasks (potentially across multiple nodes). Backends sharing an image
run in the same job; world sizes are launched separately so torch.distributed
can be sized per group.

Squash files expected at /home/sa-shared/containers/<safe_image>.sqsh.
Run scripts/pull_containers.py <platform> first if any are missing.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10 fallback (e.g. b300 login)
    import tomli as tomllib  # type: ignore

import os

SQUASH_DIR = Path(os.environ.get("OPERATORX_SQUASH_DIR", "/home/sa-shared/containers"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "containers.toml"
DEFAULT_PLATFORM = "nvidia"
PARTITION = os.environ.get("OPERATORX_PARTITION", "gpu-2")
JOB_NAME = os.environ.get("OPERATORX_JOB_NAME", "benchmark")
ACCOUNT = os.environ.get("OPERATORX_ACCOUNT")  # None -> omit --account
QOS = os.environ.get("OPERATORX_QOS")          # None -> omit --qos
GPUS_PER_NODE = 8
WORLD_SIZES = [1, 2, 4, 8]  # ws>8 disabled: multi-node NCCL IB bring-up hangs on b200/b300

# Default cluster ids per platform (overridable via $OPERATORX_CLUSTER).
DEFAULT_CLUSTER = {
    "nvidia": "b200_dgx_8x",
    "amd":    "mi355x_8x",
    "tpu":    "v6e_4x",
    "trainium": "trn3_16x",
}


def safe_name(image: str) -> str:
    out = image
    for ch in "/:@#":
        out = out.replace(ch, "_")
    return out


def load_groups(platform: str) -> dict[str, list[str]]:
    """Group backends in containers.toml by their container image.

    If $OPERATORX_BACKENDS is set (comma-separated allowlist), only backends
    in that list are included; groups that end up empty are dropped. main.py
    already honors this var at run time — propagating it here also stops us
    from submitting jobs that would have produced zero new rows.
    """
    allow = {b.strip() for b in os.environ.get("OPERATORX_BACKENDS", "").split(",") if b.strip()}
    data = tomllib.loads(MANIFEST.read_text())
    groups: dict[str, list[str]] = defaultdict(list)
    for backend, entry in data.get(platform, {}).items():
        if allow and backend not in allow:
            continue
        groups[entry["image"]].append(backend)
    return dict(groups)


def slurm_layout(world_size: int) -> tuple[int, int, int]:
    """(nodes, ntasks_per_node, gpus_per_node)"""
    if world_size <= GPUS_PER_NODE:
        return 1, world_size, world_size
    nodes = (world_size + GPUS_PER_NODE - 1) // GPUS_PER_NODE
    return nodes, GPUS_PER_NODE, GPUS_PER_NODE


def _csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _scan_testlists(testlist_names: list[str]) -> tuple[set[tuple[int, int, int, int]], set[int]]:
    """Walks selected testlists. Returns
       moe_combos = {(world_size, ep, routed_tp, shared_tp)} for moe_forward shapes,
       non_moe_ws = {world_size} that has at least one non-moe_forward shape."""
    tl_dir = PROJECT_ROOT / "testlists"
    available = {p.stem: p for p in tl_dir.glob("*.json")}
    wanted = testlist_names or sorted(available)
    moe_combos: set[tuple[int, int, int, int]] = set()
    non_moe_ws: set[int] = set()
    for name in wanted:
        path = available.get(name)
        if path is None:
            continue
        for shape in json.loads(path.read_text()):
            a = shape.get("args", {})
            ws = int(a.get("world_size", 1))
            if shape.get("type") == "moe_forward":
                moe_combos.add((
                    ws,
                    int(a.get("expert_parallel_size", 1)),
                    int(a.get("routed_tensor_parallel_size", 1)),
                    int(a.get("shared_tensor_parallel_size", 1)),
                ))
            else:
                non_moe_ws.add(ws)
    return moe_combos, non_moe_ws


def submit(image: str, backends: list[str], world_size: int, platform: str, cluster: str,
           moe_parallelism: tuple[int, int, int] | None = None) -> int:
    sqsh = SQUASH_DIR / f"{safe_name(image)}.sqsh"
    if not sqsh.exists():
        print(f"[error] missing squash file: {sqsh}")
        print(f"        run scripts/pull_containers.py {platform} first")
        return 1

    project = PROJECT_ROOT
    nodes, tpn, gpn = slurm_layout(world_size)
    job_name = JOB_NAME
    backend_list = ",".join(backends)

    inner = " && ".join([
        "set -e",
        "export TRITON_CACHE_DIR=/tmp/triton_cache",
        "export MPLCONFIGDIR=/tmp/matplotlib_config",
        "export HOME=/tmp/home_tmp",
        "export NCCL_DEBUG=WARN",
        "unset NCCL_ASYNC_ERROR_HANDLING",
        "export PYTHONWARNINGS=ignore::SyntaxWarning",
        f"cd {project}",
        f"export PYTHONPATH={project}",
        "export RANK=$SLURM_PROCID",
        "export LOCAL_RANK=$SLURM_LOCALID",
        "export WORLD_SIZE=$SLURM_NTASKS",
        "export MASTER_ADDR=$(python -c 'import os,re; nl=os.environ.get(\"SLURM_NODELIST\",\"\"); m=re.match(r\"^(.+?)\\[(\\d+)\", nl); print((m.group(1)+m.group(2)) if m else nl)')",
        "export MASTER_PORT=29500",
        f"OPERATORX_BACKENDS={backend_list} OPERATORX_CLUSTER={cluster}"
        + (f" OPERATORX_TESTLISTS={os.environ['OPERATORX_TESTLISTS']}" if os.environ.get("OPERATORX_TESTLISTS") else "")
        + (f" OPERATORX_MOE_PARALLELISM={moe_parallelism[0]}:{moe_parallelism[1]}:{moe_parallelism[2]}" if moe_parallelism else "")
        + " python -m operatorx",
    ])

    # Use sbatch (async) so SLURM can schedule jobs in parallel as resources free.
    # `srun --container-image=` inside the wrap lands the actual work in the container.
    inner_srun = (
        f"srun --container-image={sqsh} --container-mounts={project}:{project} "
        f"bash -c " + shlex.quote(inner)
    )
    log_dir = project / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ws{world_size}"
    if moe_parallelism:
        tag += f"-moe{moe_parallelism[0]}_{moe_parallelism[1]}_{moe_parallelism[2]}"
    out = log_dir / f"opx-{job_name}-{platform}-{tag}-{'_'.join(backends)}-%j.out"
    err = log_dir / f"opx-{job_name}-{platform}-{tag}-{'_'.join(backends)}-%j.err"
    cmd = ["sbatch",
           f"--partition={PARTITION}",
           f"--nodes={nodes}",
           f"--ntasks={world_size}",
           f"--ntasks-per-node={tpn}",
           f"--gres=gpu:{gpn}",
           f"--time={os.environ.get('OPERATORX_TIME_MIN', '30')}",
           f"--job-name={job_name}",
           f"--output={out}",
           f"--error={err}"]
    if ACCOUNT:
        cmd.append(f"--account={ACCOUNT}")
    if QOS:
        cmd.append(f"--qos={QOS}")
    cmd += [f"--wrap={inner_srun}"]

    print(f"[submit] {job_name}")
    print(f"  image:      {image}")
    print(f"  backends:   {backends}")
    print(f"  world_size: {world_size} ({nodes} node(s) x {tpn} task(s))")
    if moe_parallelism:
        print(f"  moe parallelism: ep={moe_parallelism[0]} routed_tp={moe_parallelism[1]} shared_tp={moe_parallelism[2]}")
    return subprocess.run(cmd).returncode


def main(argv: list[str]) -> int:
    platform = argv[1] if len(argv) > 1 else DEFAULT_PLATFORM
    cluster = os.environ.get("OPERATORX_CLUSTER", DEFAULT_CLUSTER.get(platform, ""))
    if not cluster:
        print(f"no default cluster for platform={platform!r}; set $OPERATORX_CLUSTER")
        return 1
    groups = load_groups(platform)
    if not groups:
        print(f"no entries for platform={platform!r} in {MANIFEST}")
        return 1
    testlist_names = _csv(os.environ.get("OPERATORX_TESTLISTS"))
    moe_combos, non_moe_ws = _scan_testlists(testlist_names)
    moe_combos = {c for c in moe_combos if c[0] in WORLD_SIZES}
    non_moe_ws = {ws for ws in non_moe_ws if ws in WORLD_SIZES}
    print(f"{platform}: {len(groups)} container(s) on cluster={cluster}; "
          f"{len(moe_combos)} MoE combos, {len(non_moe_ws)} non-MoE ws -> "
          f"{len(groups) * (len(moe_combos) + len(non_moe_ws))} job(s)")
    print()
    rc = 0
    for image, backends in sorted(groups.items()):
        for ws, ep, r_tp, s_tp in sorted(moe_combos):
            if submit(image, sorted(backends), ws, platform, cluster,
                      moe_parallelism=(ep, r_tp, s_tp)) != 0:
                rc = 1
            print()
        for ws in sorted(non_moe_ws):
            if submit(image, sorted(backends), ws, platform, cluster) != 0:
                rc = 1
            print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
