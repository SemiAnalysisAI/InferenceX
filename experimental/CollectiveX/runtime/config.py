#!/usr/bin/env python3
"""Load private runner settings, the public backend registry, and shard controls."""

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


def _platforms() -> dict:
    """The per-SKU platform registry (configs/platform_config.json). Callers
    fail closed on a missing file, unknown SKU, or missing field."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "platform_config.json",
    )
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)["platforms"]


OPERATOR_ENV = (
    "COLLECTIVEX_OPERATOR_CONFIG_CONTENT", "COLLECTIVEX_NETWORK_CONFIG_CONTENT",
    "COLLECTIVEX_H100_CONFIG_CONTENT", "COLLECTIVEX_B300_CONFIG_CONTENT",
    "COLLECTIVEX_B200_CONFIG_CONTENT", "COLLECTIVEX_MI300_CONFIG_CONTENT",
    "COLLECTIVEX_MI355_CONFIG_CONTENT",
)


def emit(values: dict[str, object]) -> None:
    for field, value in values.items():
        name = FIELDS.get(field, field)
        sys.stdout.buffer.write(name.encode() + b"\0" + str(value).encode() + b"\0")


def _network_overlay(runner: str) -> dict[str, object]:
    """Repo-tracked per-SKU scale-out RDMA selectors — the `network` block of the
    SKU's configs/platform_config.json entry — overlaid onto the base operator
    config. Only network FIELDS are taken, so identity keys and notes are ignored;
    a missing/invalid file is a no-op fallback to the base/secret network fields."""
    try:
        block = _platforms().get(runner, {}).get("network", {})
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return {}
    return {key: value for key, value in block.items() if key in FIELDS}


def operator_config(path: str, runner: str) -> None:
    try:
        # The registry's tracked per-SKU `operator` block is the baseline
        # (de-secreted by operator decision); an operator config document, when
        # provided, overrides it per field. Path "-" means registry-only — and
        # for SKUs with no tracked block it preserves the no-config behavior
        # (emit nothing).
        selected = dict(_platforms()[runner].get("operator", {}))
        if path == "-":
            if not selected:
                return
        else:
            with open(path, encoding="utf-8") as stream:
                document = json.load(stream)
            selected.update(document["runners"].get(runner, {}))
        # Overlay repo-tracked scale-out RDMA selectors onto the base runner config;
        # SKUs without a platform_config.json network block keep their base/secret
        # network fields.
        selected.update(_network_overlay(runner))
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


def merge_operator_config(path: str) -> None:
    """Merge the base and per-cluster encrypted operator documents."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate configuration key")
            result[key] = value
        return result

    def load_env(name: str) -> dict:
        return json.loads(
            os.environ[name], object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )

    # An absent/blank base secret (all operator config de-secreted into the
    # tracked platform_config.json) yields an empty base; the per-SKU registry
    # `operator` block then supplies every field downstream. A present base is
    # still parsed strictly. Overlay secrets are already skipped when empty.
    base = (load_env(OPERATOR_ENV[0])
            if os.environ.get(OPERATOR_ENV[0], "").strip() else {"runners": {}})
    if not isinstance(base.get("runners"), dict):
        raise ValueError("invalid operator runners")
    base = {"runners": base["runners"]}
    for name in OPERATOR_ENV[1:]:
        if not os.environ.get(name):
            continue
        overlay = load_env(name)
        if not isinstance(overlay.get("runners"), dict):
            raise ValueError("invalid overlay runners")
        for runner, fields in overlay["runners"].items():
            if not isinstance(fields, dict) or not fields:
                raise ValueError("invalid overlay runner")
            base["runners"].setdefault(runner, {}).update(fields)
    payload = json.dumps(base, sort_keys=True, separators=(",", ":")) + "\n"
    if len(payload.encode()) > 65536:
        raise ValueError("merged operator configuration is too large")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def case_count(path: str) -> None:
    print(len(load(path)["cases"]), end="")


def _emit_argv(case: dict, version: object, runner: str, ts: str, index: int) -> None:
    """Emit one null-delimited run_ep.py argv — the only case-to-invocation codec."""
    get = lambda key, default="": str(case.get(key) or default)
    argv = [
        "--backend", str(case["backend"]),
        "--mode", str(case["mode"]),
        "--phase", str(case["phase"]),
        "--routing", str(case["routing"]),
        "--gpus-per-node", str(case["gpus_per_node"]),
        "--scale-up-domain", str(case["scale_up_domain"]),
        "--scope", str(case["scope"]),
        "--scale-up-transport", str(case["scale_up_transport"]),
        "--scale-out-transport", get("scale_out_transport"),
        "--tokens-ladder", str(case["ladder"]),
        "--hidden", str(case["hidden"]),
        "--topk", str(case["topk"]),
        "--experts", str(case["experts"]),
        "--seed", str(case["seed"]),
        "--runner", runner,
        "--topology-class", str(case["topology_class"]),
        "--transport", str(case["transport"]),
        "--case-id", str(case["case_id"]),
        "--suite", str(case["suite"]),
        "--workload-name", str(case["workload"]),
        "--version", str(version),
    ]
    iters, trials, warmup = str(case["timing"]).split(":")
    for flag, value in (("--iters", iters), ("--trials", trials), ("--warmup", warmup)):
        argv += [flag, value]
    out = f"results/{runner}_{case['backend']}_{case['phase']}_{ts}-c{index:03d}.json"
    argv += ["--out", out]
    sys.stdout.buffer.write(b"\0".join(part.encode() for part in argv) + b"\0")


def case_args(
    path: str, index: int, runner: str, ts: str,
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
    _emit_argv(case, document["version"], runner, ts, index)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, names in {
        "operator-config": ("path", "runner"), "merge-operator-config": ("path",),
        "case-count": ("path",),
        "case-args": ("path", "index", "runner", "ts",
                      "ngpus", "nodes", "gpus_per_node", "scale_up_domain"),
    }.items():
        command = commands.add_parser(name)
        for arg in names: command.add_argument(arg)
    args = parser.parse_args()
    if args.command == "operator-config": operator_config(args.path, args.runner)
    elif args.command == "merge-operator-config": merge_operator_config(args.path)
    elif args.command == "case-count": case_count(args.path)
    elif args.command == "case-args":
        case_args(args.path, int(args.index), args.runner, args.ts,
                  args.ngpus, args.nodes, args.gpus_per_node, args.scale_up_domain)


if __name__ == "__main__":
    main()
