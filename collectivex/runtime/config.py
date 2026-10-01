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
    "rdma_traffic_class", "rail_isolated", "single_node_rdma_devices", "rdma_fabric",
    "rdma_relaxed_ordering",
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


def _flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def _flag_pairs(case: dict, fields: str, **renamed: str) -> list[str]:
    """`--flag value` per space-separated field; the flag is spelled from the field name
    (`gpus_per_node` -> `--gpus-per-node`) and reads the case key `renamed` maps it to."""
    argv = []
    for field in fields.split():
        value = case[renamed.get(field, field)]
        argv += [_flag(field), "" if value is None else str(value)]
    return argv


def _ep_argv(case: dict, version: object, runner: str) -> list[str]:
    argv = _flag_pairs(
        case,
        "backend mode precision phase routing gpus_per_node scale_up_domain scope "
        "scale_up_transport scale_out_transport tokens_ladder hidden topk experts seed "
        "topology_class transport case_id suite workload_name",
        tokens_ladder="ladder", workload_name="workload",
    ) + ["--runner", runner, "--version", str(version)]
    timing = _migrate_timing(case["timing"])
    for key, flag in _TIMING_FLAGS:
        if key in timing:
            argv += [flag, str(timing[key])]
    return argv


def _swap_argv(case: dict, version: object, runner: str) -> list[str]:
    argv = []
    for field in ("directions", "block_bytes", "num_blocks"):  # nargs="+" lists
        argv += [_flag(field), *str(case[field]).split()]
    return argv + _flag_pairs(case, "layout seed device max_payload_bytes warmup iterations")


# suite -> (bench/<entrypoint>.py, argv codec, output flag). The rank wrapper in
# runtime/common.sh execs the entrypoint the leading --entrypoint pair names.
_SUITES = {
    "ep-core": ("run_ep", _ep_argv, "--out"),
    "swap-blocks": ("run_swap_blocks", _swap_argv, "--output"),
}


def _emit_argv(case: dict, version: object, runner: str, ts: str, index: int) -> None:
    """Emit one null-delimited benchmark argv — the only case-to-invocation codec."""
    if case.get("suite") not in _SUITES:
        print(f"unknown suite {case.get('suite')!r}", file=sys.stderr)
        raise SystemExit(1)
    entrypoint, codec, out_flag = _SUITES[case["suite"]]
    # case_id is the canonical identity, so a new identity axis cannot be omitted from the
    # filename the way mode once was. ts + the per-shard case index disambiguate legs that share
    # one results/ directory.
    out = f"results/{case['case_id']}_{ts}-c{index:03d}.json"
    argv = ["--entrypoint", entrypoint, *codec(case, version, runner), out_flag, out]
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
    # EP cases name their rank count; other suites run one rank per allocated GPU.
    ranks = case.get("ep", int(case.get("nodes", 0)) * int(case.get("gpus_per_node", 0)))
    placement = tuple(
        str(value) for value in
        (ranks, *(case.get(field, "") for field in ("nodes", "gpus_per_node", "scale_up_domain")))
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
