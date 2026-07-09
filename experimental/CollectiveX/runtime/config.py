#!/usr/bin/env python3
"""Load private runner settings and shard controls."""

from __future__ import annotations

import argparse
import json
import os
import sys


FIELDS = {
    "partition": "CX_PARTITION", "account": "CX_ACCOUNT", "qos": "CX_QOS",
    "squash_dir": "CX_SQUASH_DIR", "stage_dir": "CX_STAGE_DIR",
    "enroot_cache_path": "CX_ENROOT_CACHE_PATH", "exclude_nodes": "CX_EXCLUDE_NODES",
    "nodelist": "CX_NODELIST", "lock_dir": "CX_LOCK_DIR",
    "socket_ifname": "CX_SOCKET_IFNAME", "rdma_devices": "CX_RDMA_DEVICES",
    "ib_gid_index": "CX_IB_GID_INDEX", "rdma_service_level": "CX_RDMA_SERVICE_LEVEL",
    "rdma_traffic_class": "CX_RDMA_TRAFFIC_CLASS",
}
REQUIRED = {
    "h100-dgxc": {"partition", "account", "squash_dir"},
    "h200-dgxc": {"partition", "squash_dir"},
    "b200-dgxc": {"partition", "account", "squash_dir"},
    "b300": {"partition", "account", "squash_dir"},
    "gb200": {"partition", "account", "storage_roots"},
    "gb300": {"partition", "account", "squash_dir", "enroot_cache_path"},
    "mi300x": {"partition", "squash_dir"},
    "mi325x": {"partition", "squash_dir"},
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


def shard_version(path: str) -> None:
    print(load(path)["version"], end="")


def shard_cases(path: str) -> None:
    for case in load(path)["cases"]:
        get = lambda key, default="": str(case.get(key) or default)
        fields = (
            get("phase", "decode"), get("mode", "normal"), get("routing", "uniform"),
            get("hidden", "7168"), get("topk", "8"), get("experts", "256"), get("ladder"),
            get("suite"), get("workload"), "1" if case.get("canonical") else "",
            get("case_id"), get("ep"), get("timing", "8:64:32"), get("nodes"),
            get("gpus_per_node"), get("scale_up_domain"), get("scope"),
            get("scale_up_transport"), get("scale_out_transport"), get("transport"),
            get("topology_class"),
        )
        print("|".join(fields))


def canonical_policy(runner: str, nodes: int, gpus_per_node: int, multiarch: str, amd: str, mori: str) -> None:
    if runner in {"h100-dgxc", "h200-dgxc", "b200-dgxc", "b300"}:
        expected, allowed, family = 8, {1, 2}, "nvidia"
    elif runner in {"gb200", "gb300"}:
        expected, allowed, family = 4, {2, 4}, "gb"
    elif runner in {"mi300x", "mi325x", "mi355x"}:
        expected, allowed, family = 8, {1, 2}, "amd"
    else:
        raise SystemExit(1)
    if nodes not in allowed or gpus_per_node != expected:
        raise SystemExit(1)
    values = {"CX_NGPUS": nodes * expected, "CX_SEED": 67,
              "CX_RUN_TIMEOUT": 1800 if family == "amd" else 900,
              "CX_IMAGE": amd if family == "amd" else multiarch}
    if family == "gb": values["CX_MASTER_PORT"] = 29551
    if family == "amd":
        values.update(CX_MORI_KERNEL_TYPE="internode-v1" if nodes == 2 else "asyncll",
                      MORI_COMMIT=mori, MORI_DISABLE_AUTO_XGMI=0, MORI_ENABLE_SDMA=1,
                      MORI_APP_LOG_LEVEL="info", MORI_SHMEM_LOG_LEVEL="info", MORI_IO_LOG_LEVEL="info")
    emit(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, names in {
        "operator-config": ("path", "runner"), "network-mode": ("path",),
        "shard-version": ("path",), "shard-cases": ("path",),
        "canonical-policy": ("runner", "nodes", "gpus_per_node", "multiarch", "amd", "mori"),
    }.items():
        command = commands.add_parser(name)
        for arg in names: command.add_argument(arg)
    args = parser.parse_args()
    if args.command == "operator-config": operator_config(args.path, args.runner)
    elif args.command == "network-mode": network_mode(args.path)
    elif args.command == "shard-version": shard_version(args.path)
    elif args.command == "shard-cases": shard_cases(args.path)
    else: canonical_policy(args.runner, int(args.nodes), int(args.gpus_per_node), args.multiarch, args.amd, args.mori)


if __name__ == "__main__":
    main()
