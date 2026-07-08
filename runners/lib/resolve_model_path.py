#!/usr/bin/env python3
"""Resolve a model's on-cluster weight path from configs/runners.yaml.

The `model-paths` section of runners.yaml is the single registry of where
model checkpoints live on each cluster (NVMe / Lustre / NFS staging), so
runner shell scripts no longer hardcode if/elif ladders of paths. Entries
are matched first-to-last on (model-prefix, precision, optional framework).

Prints shell `export` lines for the caller to eval:

    export MODEL_PATH=...
    export SRT_SLURM_MODEL_PREFIX=...
    export SERVED_MODEL_NAME=...      # only when the entry declares one

Resolution order for the path itself:
  1. --env-model-path, when it points at an existing directory (lets a
     workflow dispatch override staging locations without a code change).
  2. The entry's `path`.
  3. The entry's `path-candidates`, first existing directory, else the
     first candidate (paths on compute-node-only storage may not be visible
     from the login node running this script).

Exits 3 when no entry matches and no --fallback-to-model was given.
"""

import argparse
import os
import shlex
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on unprovisioned hosts
    sys.stderr.write(
        "resolve_model_path.py: PyYAML is required (python3 -m pip install --user pyyaml)\n"
    )
    sys.exit(4)


def emit(name, value):
    print(f"export {name}={shlex.quote(str(value))}")


def entry_matches(entry, args):
    if entry.get("model-prefix") != args.model_prefix:
        return False
    if entry.get("precision") != args.precision:
        return False
    framework = entry.get("framework")
    if framework and framework != args.framework:
        return False
    return True


def resolve_path(entry, env_model_path):
    if env_model_path and os.path.isdir(env_model_path):
        return env_model_path
    if entry.get("path"):
        return entry["path"]
    candidates = entry.get("path-candidates") or []
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runners-yaml", default="configs/runners.yaml")
    parser.add_argument("--cluster", required=True,
                        help="model-paths key, e.g. cluster:gb300-nv")
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--env-model-path", default="",
                        help="pre-set MODEL_PATH override; wins when it exists")
    parser.add_argument("--fallback-to-model", default="",
                        help="HF model id to fall back to when no entry matches "
                             "(clusters that let the server download from the Hub)")
    args = parser.parse_args()

    with open(args.runners_yaml) as fh:
        config = yaml.safe_load(fh) or {}

    entries = (config.get("model-paths") or {}).get(args.cluster) or []
    for entry in entries:
        if not entry_matches(entry, args):
            continue
        path = resolve_path(entry, args.env_model_path)
        if not path:
            sys.stderr.write(
                f"model-paths entry for {args.cluster}/{args.model_prefix}/"
                f"{args.precision} declares neither path nor path-candidates\n"
            )
            sys.exit(2)
        emit("MODEL_PATH", path)
        emit("SRT_SLURM_MODEL_PREFIX", entry.get("alias") or args.model_prefix)
        if entry.get("served-model-name"):
            emit("SERVED_MODEL_NAME", entry["served-model-name"])
        return

    if args.fallback_to_model:
        emit("MODEL_PATH", args.fallback_to_model)
        emit("SRT_SLURM_MODEL_PREFIX", args.model_prefix)
        return

    supported = ", ".join(
        sorted({
            f"{e.get('model-prefix')}-{e.get('precision')}"
            + (f" ({e.get('framework')})" if e.get("framework") else "")
            for e in entries
        })
    ) or "<none registered>"
    sys.stderr.write(
        f"No model-paths entry in {args.runners_yaml} for cluster={args.cluster} "
        f"model-prefix={args.model_prefix} precision={args.precision} "
        f"framework={args.framework}. Registered: {supported}\n"
    )
    sys.exit(3)


if __name__ == "__main__":
    main()
