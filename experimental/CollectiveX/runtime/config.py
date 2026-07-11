#!/usr/bin/env python3
"""Load private runner settings, the public backend registry, and shard controls."""

from __future__ import annotations

import argparse
import json
import os
import re
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
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        runners = document["runners"]
        selected = dict(runners[runner])
        # Overlay repo-tracked scale-out RDMA selectors onto the base runner config;
        # SKUs without a platform_config.json network block keep their base/secret
        # network fields.
        selected.update(_network_overlay(runner))
        missing = set(_platforms()[runner]["operator_fields"]) - set(selected)
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

    base = load_env(OPERATOR_ENV[0])
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


# Same acceptance as common.sh collx_select_image; keep the two in sync.
_IMAGE_REF = re.compile(r"^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def backend_registry(path: str | None = None) -> None:
    """Validate the public backend registry (configs/backends.json) and emit the
    COLLX_* source-pin and image names consumed by common.sh at source time.
    Unlike the network overlay this file is required: a missing or malformed
    registry fails closed rather than falling back."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "backends.json",
        )
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        def sha(value: object) -> str:
            if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
                raise ValueError
            return value

        def image(name: str) -> str:
            ref = document["images"][name]["ref"]
            if not isinstance(ref, str) or not _IMAGE_REF.fullmatch(ref):
                raise ValueError
            return ref

        def repo(backend: dict) -> str:
            url = backend["repo"]
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError
            return url

        v2 = document["backends"]["deepep-v2"]
        emit({
            "COLLX_IMAGE_MULTIARCH": image("multiarch"),
            "COLLX_IMAGE_AMD_MORI": image("amd-mori"),
            "COLLX_MORI_COMMIT_AMD": sha(document["backends"]["mori"]["commit"]),
            "COLLX_DEEPEP_V2_REPO": repo(v2),
            "COLLX_DEEPEP_V2_COMMIT": sha(v2["commit"]),
            "COLLX_DEEPEP_V2_FMT_COMMIT": sha(v2["submodules"]["third-party/fmt"]),
        })
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        print("validation-invalid-backend-registry", file=sys.stderr)
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
        "--hidden", get("hidden"),
        "--topk", get("topk"),
        "--experts", get("experts"),
        # Scheduled cases carry the workload seed from configs/suites.yaml; the
        # positional seed only reaches ad-hoc manual runs.
        "--seed", get("seed", seed),
        "--runner", runner,
        "--topology-class", get("topology_class", "manual"),
        "--transport", get("transport", "unknown"),
        "--case-id", get("case_id"),
        "--suite", get("suite"),
        "--workload-name", get("workload"),
        "--version", str(version),
    ]
    iters, trials, warmup = (get("timing").split(":") + ["", "", ""])[:3]
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
        # No workload or timing fallbacks: a manual run states its full shape and
        # profile or run_ep.py rejects the argv. The scheduled values live in
        # configs/suites.yaml.
        "hidden": env("COLLX_HIDDEN", ""), "topk": env("COLLX_TOPK", ""),
        "experts": env("COLLX_EXPERTS", ""),
        "topology_class": env("COLLX_TOPO", "manual"),
        "transport": env("COLLX_TRANSPORT", "unknown"),
        "case_id": env("COLLX_CASE_ID", ""), "suite": env("COLLX_SUITE", ""),
        "workload": env("COLLX_WORKLOAD_NAME", ""),
        "timing": f"{env('COLLX_ITERS', '')}:{env('COLLX_TRIALS', '')}:{env('COLLX_WARMUP', '')}",
    }
    _emit_argv(case, env("COLLX_VERSION", "1"), runner, ts, seed, index)


def canonical_policy(runner: str, nodes: int, gpus_per_node: int, multiarch: str, amd: str, mori: str) -> None:
    try:
        entry = _platforms()[runner]
        expected = int(entry["gpus_per_node"])
        vendor = entry["vendor"]
        run_timeout = int(entry["run_timeout"])
        master_port = entry.get("master_port")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise SystemExit(1)
    # Node counts realizing the registered EP degrees (8 and 16, matching
    # capability._topologies) on this SKU's fixed gpus_per_node.
    allowed = {max(1, 8 // expected), 16 // expected}
    if nodes not in allowed or gpus_per_node != expected:
        raise SystemExit(1)
    values = {"COLLX_NGPUS": nodes * expected,
              "COLLX_RUN_TIMEOUT": run_timeout,
              "COLLX_IMAGE": amd if vendor == "amd" else multiarch}
    if master_port is not None:
        values["COLLX_MASTER_PORT"] = int(master_port)
    if vendor == "amd":
        # The MoRI kernel is derived by the adapter from (arch, scope); no
        # kernel-type env is emitted (COLLX_MORI_KERNEL_TYPE survives only as an
        # optional cross-check the adapter honors if a launcher still sets it).
        values.update(MORI_COMMIT=mori, MORI_DISABLE_AUTO_XGMI=0, MORI_ENABLE_SDMA=1,
                      MORI_APP_LOG_LEVEL="info", MORI_SHMEM_LOG_LEVEL="info", MORI_IO_LOG_LEVEL="info")
    emit(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, names in {
        "operator-config": ("path", "runner"), "merge-operator-config": ("path",),
        "network-mode": ("path",),
        "backend-registry": (), "case-count": ("path",),
        "case-args": ("path", "index", "runner", "ts", "seed",
                      "ngpus", "nodes", "gpus_per_node", "scale_up_domain"),
        "manual-args": ("phase", "index", "runner", "ts", "seed"),
        "canonical-policy": ("runner", "nodes", "gpus_per_node", "multiarch", "amd", "mori"),
    }.items():
        command = commands.add_parser(name)
        for arg in names: command.add_argument(arg)
    args = parser.parse_args()
    if args.command == "operator-config": operator_config(args.path, args.runner)
    elif args.command == "merge-operator-config": merge_operator_config(args.path)
    elif args.command == "network-mode": network_mode(args.path)
    elif args.command == "backend-registry": backend_registry()
    elif args.command == "case-count": case_count(args.path)
    elif args.command == "case-args":
        case_args(args.path, int(args.index), args.runner, args.ts, args.seed,
                  args.ngpus, args.nodes, args.gpus_per_node, args.scale_up_domain)
    elif args.command == "manual-args":
        manual_args(args.phase, int(args.index), args.runner, args.ts, args.seed)
    else: canonical_policy(args.runner, int(args.nodes), int(args.gpus_per_node), args.multiarch, args.amd, args.mori)


if __name__ == "__main__":
    main()
