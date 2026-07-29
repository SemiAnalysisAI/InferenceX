"""CLI: enumerate canonical OperatorX ops from InferenceX matrix configs.

Usage:
    python -m scripts.inferencex_testlist enumerate \
        --inferencex /path/to/InferenceX \
        --config .github/configs/nvidia-master.yaml \
        --config .github/configs/amd-master.yaml \
        --models-dir /models \
        --out testlists/
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Iterable

from .emit import write_testlists
from .enumerate import ops_for_row
from .matrix import WorkloadRow, load_rows
from .models import Arch, load_arch


def _dedupe_rows(rows: Iterable[WorkloadRow]) -> list[WorkloadRow]:
    """Drop rows whose op set would be identical to one we've already enumerated."""
    seen = set()
    out: list[WorkloadRow] = []
    for r in rows:
        key = (
            r.model, r.precision, r.framework, r.tp, r.ep, r.dp_attn,
            r.isl, r.osl, r.conc, r.spec_decoding,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def cmd_enumerate(args: argparse.Namespace) -> int:
    extra_dirs = tuple(args.models_dir) if args.models_dir else ()

    rows = []
    for cfg in args.config:
        rows.extend(load_rows(args.inferencex, [cfg]))
    print(f"[load] {len(rows)} matrix rows from {len(args.config)} configs", file=sys.stderr)

    rows = _dedupe_rows(rows)
    print(f"[dedupe] {len(rows)} unique rows after collapsing concurrency repeats", file=sys.stderr)

    # Cache per-model arch lookups; we'll hit the same HF config many times.
    arch_cache: dict[str, Arch] = {}
    missing_models: dict[str, list[WorkloadRow]] = defaultdict(list)

    all_ops: list = []
    skipped = 0
    for row in rows:
        if row.model not in arch_cache:
            try:
                arch_cache[row.model] = load_arch(row.model, extra_dirs=extra_dirs)
            except FileNotFoundError as e:
                missing_models[row.model].append(row)
                skipped += 1
                continue
            except Exception as e:
                print(f"[warn] could not load arch for {row.model!r}: {e}", file=sys.stderr)
                missing_models[row.model].append(row)
                skipped += 1
                continue
        try:
            all_ops.extend(ops_for_row(row, arch_cache[row.model]))
        except Exception as e:
            print(f"[warn] enumeration failed for {row.exp_name} {row.model} tp{row.tp} conc{row.conc}: {e}",
                  file=sys.stderr)
            skipped += 1

    if missing_models:
        print(f"[warn] {len(missing_models)} models had no local config; skipped {skipped} rows", file=sys.stderr)
        for m, rs in list(missing_models.items())[:10]:
            print(f"          {m!r}: {len(rs)} rows", file=sys.stderr)

    counts = write_testlists(all_ops, args.out, merge_with_existing=args.merge_existing)
    for fname, n in sorted(counts.items()):
        print(f"[emit] {fname}.json: {n} entries", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.inferencex_testlist")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enum = sub.add_parser("enumerate", help="Build testlists from InferenceX configs")
    p_enum.add_argument("--inferencex", required=True, help="Path to InferenceX repo root")
    p_enum.add_argument("--config", required=True, action="append",
                        help="InferenceX master config path (relative to --inferencex). Repeatable.")
    p_enum.add_argument("--models-dir", action="append", default=[],
                        help="Local directory with model config.json subdirs. Repeatable.")
    p_enum.add_argument("--out", required=True, help="Output dir (operatorx/testlists/)")
    p_enum.add_argument("--no-merge-existing", dest="merge_existing", action="store_false",
                        help="Overwrite testlist files rather than unioning with existing entries.")
    p_enum.set_defaults(merge_existing=True, func=cmd_enumerate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
