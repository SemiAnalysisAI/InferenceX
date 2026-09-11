from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("labels,artifacts,allowed", [
    ([], "results_bmk", True),
    (["documentation"], "bmk_agentic_example", True),
    (["sweep-enabled"], "results_bmk", True),
    (["full-sweep-enabled"], "results_bmk", True),
    ([], "run-stats", False),
    (["evals-only"], "eval_results_all", False),
    (["agentx-fast"], "bmk_agentic_example", False),
    (["full-sweep-enabled", "sweep-enabled"], "results_bmk", False),
])
def test_merge_preflight_uses_artifacts_without_requiring_a_sweep_label(
    tmp_path, labels, artifacts, allowed,
):
    """Run the real helper; fake git/gh stop before any external write or merge."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text('#!/bin/sh\nif [ "$1" = symbolic-ref ]; then echo test-branch; fi\nexit 0\n')
    git.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["TEST_REUSE_CALLS"], "a") as out:
    out.write(json.dumps(args) + "\\n")
if args[:2] == ["pr", "view"]:
    print(json.dumps({"state": "OPEN", "isCrossRepository": False,
                      "headRefName": "feature", "labels": json.loads(os.environ["TEST_REUSE_LABELS"])}))
elif args[:2] == ["pr", "comment"]:
    sys.exit(73)  # Preflight reached the first write; do not perform it.
elif args[0] == "api":
    path = args[1]
    if path.endswith("/commits"):
        print("tested-sha")
    elif "/workflows/run-sweep.yml/runs?" in path:
        print("123\\ttested-sha")
    elif "/runs/123/artifacts?" in path:
        print(os.environ["TEST_REUSE_ARTIFACTS"])
    else:
        raise AssertionError(args)
else:
    raise AssertionError(args)
''')
    gh.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "merge_with_reuse.sh"
    run = subprocess.run(
        ["bash", str(script), "7"], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "TEST_REUSE_CALLS": str(calls), "REPO": "example/project",
             "TEST_REUSE_LABELS": json.dumps([{"name": label} for label in labels]),
             "TEST_REUSE_ARTIFACTS": artifacts},
        timeout=10,
    )
    assert run.returncode == (73 if allowed else 1), run.stdout + run.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    writes = [args for args in commands if args[:2] == ["pr", "comment"]]
    assert writes == ([["pr", "comment", "7", "--repo", "example/project",
                        "--body", "/reuse-sweep-run 123"]] if allowed else [])
