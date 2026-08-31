#!/usr/bin/env python3
"""Enable InferenceX-selected eval dispatch in an srt-slurm checkout."""

from __future__ import annotations

import sys
from pathlib import Path

DO_SWEEP_ENV_BLOCK = """            "EVAL_ONLY",
            "IS_MULTINODE","""
DO_SWEEP_ENV_REPLACEMENT = """            "EVAL_ONLY",
            "EVAL_FRAMEWORK",
            "EVAL_CONC",
            "EVAL_LIMIT",
            "EVAL_SUITE",
            "SWEBENCH_GEN_MODE",
            "SWEBENCH_USE_MODAL",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "IS_AGENTIC",
            "SCENARIO_TYPE",
            "IS_MULTINODE","""
LM_EVAL_COMMAND = 'run_eval --framework lm-eval --port "$PORT" || eval_rc=$?'
GENERIC_EVAL_COMMAND = 'run_eval --port "$PORT" || eval_rc=$?'
EVAL_ARTIFACT_COPY = """cp -v results*.json /logs/eval_results/ 2>/dev/null || true
cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true"""
VERIFIER_ARTIFACT_COPY = 'stage_eval_artifacts /logs/eval_results "$PWD" || true'


def prepare_replacements(
    path: Path,
    replacements: tuple[tuple[str, str], ...],
) -> tuple[str, str, bool]:
    """Validate source replacements without mutating the checkout."""
    original = path.read_text()
    content = original
    changed = False
    for old, new in replacements:
        old_count = content.count(old)
        new_count = content.count(new)
        if old_count == 1 and new_count == 0:
            anchor = old.splitlines()[0]
            anchor_count = content.count(anchor)
            if anchor_count != 1:
                raise RuntimeError(
                    f"invalid patch state in {path}: "
                    f"anchor {anchor!r} count={anchor_count}"
                )
            content = content.replace(old, new, 1)
            changed = True
        elif old_count != 0 or new_count != 1:
            raise RuntimeError(
                f"invalid patch state in {path}: old anchor count={old_count}, "
                f"replacement count={new_count}"
            )
    return original, content, changed


def patch_checkout(root: Path) -> list[Path]:
    """Patch both post-eval sources after validating the complete checkout."""
    patches = (
        (
            root / "src/srtctl/cli/do_sweep.py",
            ((DO_SWEEP_ENV_BLOCK, DO_SWEEP_ENV_REPLACEMENT),),
        ),
        (
            root / "src/srtctl/benchmarks/scripts/lm-eval/bench.sh",
            (
                (LM_EVAL_COMMAND, GENERIC_EVAL_COMMAND),
                (EVAL_ARTIFACT_COPY, VERIFIER_ARTIFACT_COPY),
            ),
        ),
    )
    staged = [
        (path, *prepare_replacements(path, replacements))
        for path, replacements in patches
    ]
    changed = []
    written = []
    try:
        for path, original, replacement, needs_write in staged:
            if needs_write:
                path.write_text(replacement)
                written.append((path, original))
                changed.append(path)
    except OSError:
        for path, original in reversed(written):
            path.write_text(original)
        raise
    return changed


def main(argv: list[str]) -> int:
    """Patch the checkout named on the command line."""
    if len(argv) != 2:
        print(f"Usage: {argv[0]} SRT_SLURM_CHECKOUT", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    try:
        changed = patch_checkout(root)
    except (OSError, RuntimeError) as error:
        print(
            f"ERROR: failed to patch srt-slurm eval dispatch: {error}", file=sys.stderr
        )
        return 1
    if changed:
        for path in changed:
            print(f"Patched srt-slurm eval dispatch: {path}")
    else:
        print("srt-slurm eval dispatch is already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
