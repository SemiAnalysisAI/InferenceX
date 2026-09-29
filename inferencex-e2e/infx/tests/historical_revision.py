"""A committed revision whose own tooling reads the retired ``hardware:`` runner layout.

Current code rejects that layout, so only the revision's own generator or planner can use this
history. The stubs stand in for that tooling without freezing a past matrix algorithm.
"""

import subprocess
from pathlib import Path

import yaml

from infx.matrix import validation

KEY = "fixture"
FAMILY = f"configs/nvidia-master.yaml:{KEY}"
BASE_LINK = "https://github.com/SemiAnalysisAI/InferenceX/pull/1"
HEAD_LINK = "https://github.com/SemiAnalysisAI/InferenceX/pull/42"
RUNNERS = {
    "labels": {"fixture": ["node-a"]},
    "hardware": {"fixture": {"gpus-per-node": 8, "available-cpu-dram-mib": 1024000}},
}
MASTER = {KEY: {
    "image": "example/image:stable", "model": "example/model", "model-prefix": "dsr1",
    "precision": "fp8", "framework": "sglang", "runner": "fixture", "multinode": False,
    "scenarios": {"fixed-seq-len": [{
        "isl": 8192, "osl": 1024, "search-space": [{"tp": 1, "conc-list": [2, 6]}],
    }]},
}}
# (model, conc, image) of every point the revision's own tooling generates for KEY.
POINTS = [("example/model", 2, "example/image:stable"), ("example/model", 6, "example/image:stable")]

GENERATOR = '''\
"""Generate selected configs; runner facts come from the ``hardware:`` inventory."""

import argparse
import json
import sys

import yaml


def rows(config_keys, config_files, runner_config):
    master = {}
    for path in config_files:
        with open(path) as handle:
            master.update(yaml.safe_load(handle) or {})
    with open(runner_config) as handle:
        hardware = yaml.safe_load(handle)["hardware"]
    result = []
    for key in config_keys:
        entry = master[key]
        gpus = hardware[entry["runner"]]["gpus-per-node"]
        for scenario in entry["scenarios"]["fixed-seq-len"]:
            for space in scenario["search-space"]:
                if space["tp"] > gpus:
                    sys.exit(f"{key}: tp {space['tp']} exceeds {gpus} GPUs per node")
                for conc in space["conc-list"]:
                    result.append({
                        "image": entry["image"], "model": entry["model"],
                        "model-prefix": entry["model-prefix"], "precision": entry["precision"],
                        "framework": entry["framework"], "runner": entry["runner"],
                        "isl": scenario["isl"], "osl": scenario["osl"], "tp": space["tp"],
                        "pp": 1, "dcp-size": 1, "pcp-size": 1, "conc": conc,
                        "max-model-len": scenario["isl"] + scenario["osl"] + 256, "ep": 1,
                        "dp-attn": False, "spec-decoding": "none",
                        "exp-name": entry["model-prefix"] + "_8k1k", "disagg": False,
                        "run-eval": False,
                    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test-config"])
    parser.add_argument("--config-keys", nargs="+", required=True)
    parser.add_argument("--config-files", nargs="+", required=True)
    parser.add_argument("--runner-config", default="configs/runners.yaml")
    parser.add_argument("--no-evals", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rows(args.config_keys, args.config_files, args.runner_config)))
'''

PLANNER = '''\
"""Plan changelog additions with this revision's own generator and configs."""

import argparse
import json
import subprocess

import yaml

from infx.matrix.generate import rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--changelog-file", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    args = parser.parse_args()
    diff = subprocess.run(
        ["git", "diff", args.base_ref, args.head_ref, "--", args.changelog_file],
        capture_output=True, text=True, check=True,
    ).stdout
    added = [line[1:] for line in diff.splitlines() if line[:1] == "+" and line[:3] != "+++"]
    entries = yaml.safe_load("\\n".join(added))
    keys = [key for entry in entries for key in entry["config-keys"]]
    configs = ["configs/amd-master.yaml", "configs/nvidia-master.yaml"]
    print(json.dumps({
        "single_node": {"8k1k": rows(keys, configs, "configs/runners.yaml")},
        "multi_node": {}, "evals": [], "agentic_evals": [], "multinode_evals": [],
        "multinode_agentic_evals": [],
        "changelog_metadata": {
            "base_ref": args.base_ref, "head_ref": args.head_ref, "entries": entries,
        },
    }))
'''


def forbid_current_config_parsing(monkeypatch) -> None:
    """Fail if this process parses configs with current code instead of the revision's own."""

    def parse(*_args, **_kwargs):
        raise AssertionError("current code parsed a historical revision's configs")

    for name in (
        "load_config_files", "load_runner_file", "validate_master_config", "validate_runner_config",
    ):
        monkeypatch.setattr(validation, name, parse)


def changelog_entry(link: str) -> bytes:
    return f'- config-keys:\n    - {KEY}\n  description:\n    - "Update {KEY}"\n  pr-link: {link}\n'.encode()


def commit_history(root: Path, project: str = "", *, launcher: bool = True) -> tuple[str, str]:
    """Commit a base revision and a head appending one changelog entry; return both SHAs.

    The working tree is left at the head. With ``launcher`` the revision ships its own
    ``infx.launch`` entrypoint, which e2e runs require of every measured revision.
    """
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()

    tree = root / project
    files = {
        "infx/__init__.py": "",
        "infx/matrix/__init__.py": "",
        "infx/matrix/generate.py": GENERATOR,
        "infx/matrix/plan.py": PLANNER,
        "configs/amd-master.yaml": "{}\n",
        "configs/nvidia-master.yaml": yaml.safe_dump(MASTER),
        "configs/runners.yaml": yaml.safe_dump(RUNNERS),
        **({"infx/launch/__init__.py": "", "infx/launch/__main__.py": ""} if launcher else {}),
    }
    for path, content in files.items():
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_text(content)
    (tree / "perf-changelog.yaml").write_bytes(changelog_entry(BASE_LINK))
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("add", ".")
    git("commit", "-qm", "hardware-layout base")
    base = git("rev-parse", "HEAD")
    (tree / "perf-changelog.yaml").write_bytes(
        changelog_entry(BASE_LINK) + b"\n" + changelog_entry(HEAD_LINK)
    )
    git("commit", "-qam", "hardware-layout head")
    return base, git("rev-parse", "HEAD")
