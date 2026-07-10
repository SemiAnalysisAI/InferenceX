#!/usr/bin/env python3
"""Load private runner settings and shard controls."""

from __future__ import annotations

import argparse
import json
import os
import sys


FIELDS = {
    "partition": "COLLX_PARTITION", "account": "COLLX_ACCOUNT", "qos": "COLLX_QOS",
    "squash_dir": "COLLX_SQUASH_DIR", "stage_dir": "COLLX_STAGE_DIR",
    "enroot_cache_path": "COLLX_ENROOT_CACHE_PATH", "exclude_nodes": "COLLX_EXCLUDE_NODES",
    "nodelist": "COLLX_NODELIST", "lock_dir": "COLLX_LOCK_DIR",
    "socket_ifname": "COLLX_SOCKET_IFNAME", "rdma_devices": "COLLX_RDMA_DEVICES",
    "ib_gid_index": "COLLX_IB_GID_INDEX", "rdma_service_level": "COLLX_RDMA_SERVICE_LEVEL",
    "rdma_traffic_class": "COLLX_RDMA_TRAFFIC_CLASS",
}
REQUIRED = {
    "h100-dgxc": {"partition", "account", "squash_dir"},
    "h200-dgxc": {"partition", "squash_dir"},
    "b200-dgxc": {"partition", "account", "squash_dir"},
    "b300": {"partition", "account", "squash_dir"},
    "gb200": {"partition", "account", "storage_roots"},
    "gb300": {"partition", "account", "squash_dir", "enroot_cache_path"},
    "mi300x": {"partition", "squash_dir"},
    "mi355x": {"partition", "squash_dir"},
}


def emit(values: dict[str, object]) -> None:
    for field, value in values.items():
        name = FIELDS.get(field, field)
        sys.stdout.buffer.write(name.encode() + b"\0" + str(value).encode() + b"\0")


def operator_config(path: str, runner: str) -> None:
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        runners = document["runners"]
        selected = dict(runners[runner])
        missing = REQUIRED[runner] - set(selected)
        if missing:
            print("validation-missing-required-" + "-".join(sorted(missing)), file=sys.stderr)
            raise SystemExit(1)
        allowed = set(FIELDS) | {"storage_roots"}
        if set(selected) - allowed:
            raise ValueError
        roots = selected.pop("storage_roots", None)
        if roots:
            for root in roots:
                squash = os.path.join(root, "collectivex", "containers")
                stage = os.path.join(root, "collectivex", "stage")
                try:
                    os.makedirs(squash, mode=0o700, exist_ok=True)
                    os.makedirs(stage, mode=0o700, exist_ok=True)
                    selected.update(squash_dir=squash, stage_dir=stage)
                    break
                except OSError:
                    continue
            else:
                raise ValueError
        if any(not isinstance(value, (str, int)) or "\0" in str(value) for value in selected.values()):
            raise ValueError
        emit(selected)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        print("validation-invalid-config", file=sys.stderr)
        raise SystemExit(1)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def network_mode(path: str) -> None:
    modes = {case.get("mode", "normal") for case in load(path)["cases"]}
    if modes != {"normal"}:
        raise SystemExit(1)
    print("normal", end="")


def case_count(path: str) -> None:
    print(len(load(path)["cases"]), end="")


def _emit_argv(case: dict, version: object, runner: str, ts: str, seed: str, index: int) -> None:
    """Emit one null-delimited run_ep.py argv — the only case-to-invocation codec."""
    get = lambda key, default="": str(case.get(key) or default)
    argv = [
        "--backend", get("backend"),
        "--mode", get("mode", "normal"),
        "--phase", get("phase", "decode"),
        "--routing", get("routing", "uniform"),
        "--gpus-per-node", get("gpus_per_node", "0"),
        "--scale-up-domain", get("scale_up_domain", "0"),
        "--scope", get("scope", "scale-up"),
        "--scale-up-transport", get("scale_up_transport", "unknown"),
        "--scale-out-transport", get("scale_out_transport"),
        "--tokens-ladder", get("ladder"),
        "--hidden", get("hidden", "7168"),
        "--topk", get("topk", "8"),
        "--experts", get("experts", "256"),
        "--seed", seed,
        "--runner", runner,
        "--topology-class", get("topology_class", "manual"),
        "--transport", get("transport", "unknown"),
        "--case-id", get("case_id"),
        "--suite", get("suite"),
        "--workload-name", get("workload"),
        "--version", str(version),
    ]
    iters, trials, warmup = (get("timing", "8:64:32").split(":") + ["", "", ""])[:3]
    for flag, value in (("--iters", iters), ("--trials", trials), ("--warmup", warmup)):
        if value:
            argv += [flag, value]
    out = f"results/{runner}_{get('backend')}_{get('phase', 'decode')}_{ts}-c{index:03d}.json"
    argv += ["--out", out]
    sys.stdout.buffer.write(b"\0".join(part.encode() for part in argv) + b"\0")


def case_args(
    path: str, index: int, runner: str, ts: str, seed: str,
    ngpus: str, nodes: str, gpus_per_node: str, scale_up_domain: str,
) -> None:
    document = load(path)
    cases = document["cases"]
    if not 0 <= index < len(cases):
        raise SystemExit(1)
    case = cases[index]
    placement = tuple(
        str(case.get(field, ""))
        for field in ("ep", "nodes", "gpus_per_node", "scale_up_domain")
    )
    if placement != (ngpus, nodes, gpus_per_node, scale_up_domain):
        print(f"case placement {placement} differs from the allocation", file=sys.stderr)
        raise SystemExit(1)
    _emit_argv(case, document["version"], runner, ts, seed, index)


def manual_args(phase: str, index: int, runner: str, ts: str, seed: str) -> None:
    """Ad-hoc (shard-less) runs take one case per phase from the operator's COLLX_* env."""
    env = os.environ.get
    case = {
        "backend": env("COLLX_BENCH", ""), "mode": env("COLLX_MODE", "normal"),
        "phase": phase, "routing": env("COLLX_ROUTING", "uniform"),
        "gpus_per_node": env("COLLX_GPUS_PER_NODE", "0"),
        "scale_up_domain": env("COLLX_SCALE_UP_DOMAIN", "0"),
        "scope": env("COLLX_SCOPE", "scale-up"),
        "scale_up_transport": env("COLLX_SCALE_UP_TRANSPORT", "unknown"),
        "scale_out_transport": env("COLLX_SCALE_OUT_TRANSPORT", ""),
        "ladder": env("COLLX_TOKENS_LADDER", ""),
        "hidden": env("COLLX_HIDDEN", "7168"), "topk": env("COLLX_TOPK", "8"),
        "experts": env("COLLX_EXPERTS", "256"),
        "topology_class": env("COLLX_TOPO", "manual"),
        "transport": env("COLLX_TRANSPORT", "unknown"),
        "case_id": env("COLLX_CASE_ID", ""), "suite": env("COLLX_SUITE", ""),
        "workload": env("COLLX_WORKLOAD_NAME", ""),
        "timing": f"{env('COLLX_ITERS', '8')}:{env('COLLX_TRIALS', '64')}:{env('COLLX_WARMUP', '32')}",
    }
    _emit_argv(case, env("COLLX_VERSION", "1"), runner, ts, seed, index)


def canonical_policy(runner: str, nodes: int, gpus_per_node: int, multiarch: str, amd: str, mori: str) -> None:
    if runner in {"h100-dgxc", "h200-dgxc", "b200-dgxc", "b300"}:
        expected, allowed, family = 8, {1, 2}, "nvidia"
    elif runner in {"gb200", "gb300"}:
        expected, allowed, family = 4, {2, 4}, "gb"
    elif runner in {"mi300x", "mi355x"}:
        expected, allowed, family = 8, {1, 2}, "amd"
    else:
        raise SystemExit(1)
    if nodes not in allowed or gpus_per_node != expected:
        raise SystemExit(1)
    values = {"COLLX_NGPUS": nodes * expected, "COLLX_SEED": 67,
              "COLLX_RUN_TIMEOUT": 1800 if family == "amd" else 900,
              "COLLX_IMAGE": amd if family == "amd" else multiarch}
    if family == "gb": values["COLLX_MASTER_PORT"] = 29551
    if family == "amd":
        values.update(COLLX_MORI_KERNEL_TYPE="internode-v1" if nodes == 2 else "asyncll",
                      MORI_COMMIT=mori, MORI_DISABLE_AUTO_XGMI=0, MORI_ENABLE_SDMA=1,
                      MORI_APP_LOG_LEVEL="info", MORI_SHMEM_LOG_LEVEL="info", MORI_IO_LOG_LEVEL="info")
    emit(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, names in {
        "operator-config": ("path", "runner"), "network-mode": ("path",),
        "case-count": ("path",),
        "case-args": ("path", "index", "runner", "ts", "seed",
                      "ngpus", "nodes", "gpus_per_node", "scale_up_domain"),
        "manual-args": ("phase", "index", "runner", "ts", "seed"),
        "canonical-policy": ("runner", "nodes", "gpus_per_node", "multiarch", "amd", "mori"),
    }.items():
        command = commands.add_parser(name)
        for arg in names: command.add_argument(arg)
    args = parser.parse_args()
    if args.command == "operator-config": operator_config(args.path, args.runner)
    elif args.command == "network-mode": network_mode(args.path)
    elif args.command == "case-count": case_count(args.path)
    elif args.command == "case-args":
        case_args(args.path, int(args.index), args.runner, args.ts, args.seed,
                  args.ngpus, args.nodes, args.gpus_per_node, args.scale_up_domain)
    elif args.command == "manual-args":
        manual_args(args.phase, int(args.index), args.runner, args.ts, args.seed)
    else: canonical_policy(args.runner, int(args.nodes), int(args.gpus_per_node), args.multiarch, args.amd, args.mori)


if __name__ == "__main__":
    main()
