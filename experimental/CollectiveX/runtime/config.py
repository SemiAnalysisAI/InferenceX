#!/usr/bin/env python3
"""Load private runner settings, the public backend registry, and shard controls."""

from __future__ import annotations

import argparse
import json
import os
import sys


OPERATOR_FIELDS = {
    "partition", "account", "qos", "squash_dir", "stage_dir",
    "enroot_cache_path", "exclude_nodes", "nodelist", "lock_dir",
}
NETWORK_FIELDS = {
    "socket_ifname", "rdma_devices", "ib_gid_index", "rdma_service_level",
    "rdma_traffic_class", "rail_isolated",
}
# Timing knobs, in the order the legacy colon-string encoded them, paired with the run_ep flag
# each one drives. The names match configs/sweep.json so a case's timing block is readable
# rather than positional.
_TIMING_FLAGS = (
    ("iters_per_trial", "--iters"),
    ("trials_per_point", "--trials"),
    ("warmup_iters_per_trial", "--warmup"),
    ("chain_iters_per_trial", "--chain-iters"),
    ("chain_trials_per_point", "--chain-trials"),
    ("chain_drop", "--chain-drop"),
)


def _migrate_timing(timing: object) -> dict:
    """The one place a legacy colon-string timing profile is decoded.

    Current shards carry an object keyed by the _TIMING_FLAGS names. A replayed pre-chain
    shard carries "iters:trials:warmup" and a replayed post-chain one all six, positionally;
    three fields emit no --chain-* flags, so run_ep's argparse defaults supply them rather
    than this file duplicating the values. Anything else fails closed.

    Worth knowing when replaying an old shard: timing was never part of case_id, so a
    pre-chain case re-run today lands under its original identity while measured with
    today's chain budget.
    """
    names = tuple(key for key, _ in _TIMING_FLAGS)
    if isinstance(timing, dict):
        if set(timing) not in (set(names), set(names[:3])):
            print(f"unrecognised timing object {timing!r}", file=sys.stderr)
            raise SystemExit(1)
        return timing
    fields = str(timing).split(":")
    if len(fields) not in (3, 6):
        print(f"unrecognised timing profile {timing!r}", file=sys.stderr)
        raise SystemExit(1)
    return dict(zip(names, fields))


def _platforms() -> dict:
    """The per-SKU platform registry (configs/platform_config.json). Callers
    fail closed on a missing file, unknown SKU, or missing field."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "platform_config.json",
    )
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)["platforms"]


def emit(values: dict[str, object]) -> None:
    for field, value in values.items():
        name = f"COLLX_{field.upper()}"
        sys.stdout.buffer.write(name.encode() + b"\0" + str(value).encode() + b"\0")


def _network_overlay(runner: str) -> dict[str, object]:
    """Repo-tracked per-SKU scale-out RDMA selectors — the `network` block of the
    SKU's configs/platform_config.json entry — overlaid onto the base operator
    config. Only NETWORK_FIELDS are taken, so identity keys and notes are ignored;
    a missing/invalid file is a no-op fallback to the base/secret network fields."""
    try:
        block = _platforms().get(runner, {}).get("network", {})
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return {}
    return {key: value for key, value in block.items() if key in NETWORK_FIELDS}


def operator_config(path: str, runner: str) -> None:
    try:
        platform = _platforms()[runner]
        # The registry's tracked per-SKU `operator` block is the baseline
        # (de-secreted by operator decision); an operator config document, when
        # provided, overrides it per field. Path "-" means registry-only.
        selected = dict(platform.get("operator", {}))
        if path != "-":
            with open(path, encoding="utf-8") as stream:
                document = json.load(stream)
            selected.update(document["runners"].get(runner, {}))
        # Overlay repo-tracked scale-out RDMA selectors onto the base runner config;
        # SKUs without a platform_config.json network block keep their base/secret
        # network fields.
        selected.update(_network_overlay(runner))
        allowed = OPERATOR_FIELDS | NETWORK_FIELDS | {"storage_roots"}
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
        selected.update(image=platform["image"], image_platform=platform["image_platform"])
        emit(selected)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        print("validation-invalid-config", file=sys.stderr)
        raise SystemExit(1)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def case_count(path: str) -> None:
    print(len(load(path)["cases"]), end="")


def _emit_kv_argv(case: dict, version: object, runner: str, ts: str, index: int) -> None:
    """Null-delimited run_kv.py argv for a kv-transfer case. The leading
    --entrypoint pair is consumed by the rank wrapper (runtime/common.sh),
    which otherwise execs run_ep.py."""
    timing = str(case["timing"]).split(":")
    if len(timing) != 3:
        print(f"unrecognised kv timing profile {case['timing']!r}", file=sys.stderr)
        raise SystemExit(1)
    out = (
        f"results/{runner}_kv-{case['backend']}_{case['precision']}_{case['mode']}"
        f"_{case['workload']}_{ts}-c{index:03d}.json"
    )
    argv = [
        "--entrypoint", "run_kv",
        "--backend", str(case["backend"]),
        "--workload-name", str(case["workload"]),
        "--precision", str(case["precision"]),
        "--fabric", str(case["mode"]),
        "--isl-ladder", str(case["isl_ladder"]),
        "--page-tokens", str(case["page_tokens"]),
        "--batch-sizes", str(case["batch_sizes"]),
        "--kv-device", str(case.get("kv_device", "")),
        "--ops", str(case["ops"]),
        "--warmup", timing[0], "--reps", timing[1], "--trials", timing[2],
        "--pool-slack", str(case["pool_slack"]),
        "--seed", str(case["seed"]),
        "--runner", runner,
        "--case-id", str(case["case_id"]),
        "--suite", str(case["suite"]),
        "--gpus-per-node", str(case["gpus_per_node"]),
        "--scale-up-domain", str(case["scale_up_domain"]),
        "--scale-up-transport", str(case["scale_up_transport"]),
        "--topology-class", str(case["topology_class"]),
        "--version", str(version),
        "--out", out,
    ]
    sys.stdout.buffer.write(b"\0".join(part.encode() for part in argv) + b"\0")


def _emit_argv(case: dict, version: object, runner: str, ts: str, index: int) -> None:
    """Emit one null-delimited benchmark argv — the only case-to-invocation codec."""
    if case.get("suite") == "kv-transfer":
        _emit_kv_argv(case, version, runner, ts, index)
        return
    get = lambda key, default="": str(case.get(key) or default)
    argv = [
        "--backend", str(case["backend"]),
        "--mode", str(case["mode"]),
        "--precision", str(case["precision"]),
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
    timing = _migrate_timing(case["timing"])
    for key, flag in _TIMING_FLAGS:
        if key in timing:
            argv += [flag, str(timing[key])]
    # case_id is the canonical identity (sku==runner, backend, workload, mode, phase, ep, routing,
    # precision), so a new identity axis cannot be omitted from the filename the way mode once was.
    # ts + the per-shard case index disambiguate legs that share one results/ directory.
    out = f"results/{case['case_id']}_{ts}-c{index:03d}.json"
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
        "operator-config": ("path", "runner"),
        "case-count": ("path",),
        "case-args": ("path", "index", "runner", "ts",
                      "ngpus", "nodes", "gpus_per_node", "scale_up_domain"),
    }.items():
        command = commands.add_parser(name)
        for arg in names: command.add_argument(arg)
    args = parser.parse_args()
    if args.command == "operator-config": operator_config(args.path, args.runner)
    elif args.command == "case-count": case_count(args.path)
    elif args.command == "case-args":
        case_args(args.path, int(args.index), args.runner, args.ts,
                  args.ngpus, args.nodes, args.gpus_per_node, args.scale_up_domain)


if __name__ == "__main__":
    main()
