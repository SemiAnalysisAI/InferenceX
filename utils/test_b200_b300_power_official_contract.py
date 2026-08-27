"""Contract for required B200/B300 multinode dcgm-power lanes."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_ROOT = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes"
MASTER = REPO_ROOT / "configs/nvidia-master.yaml"
WORKFLOW = REPO_ROOT / ".github/workflows/test-process-result.yml"
LAUNCHERS = {
    "cluster:b200-nscale": REPO_ROOT / "runners/launch_b200-nscale-slurm.sh",
    "b300": REPO_ROOT / "runners/launch_b300-nv.sh",
}

PRODUCER_URL = "https://github.com/edwingao28/srt-slurm.git"
PRODUCER_SHA = "e5c837f06a362dc888dfea2ee588e9f19c298270"
EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
PROCESS_RESULT_RECIPE_GLOBS = (
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/**/*.yaml",
    "benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/**/*.yaml",
    "benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.6/b200-fp4/**/*.yaml",
)


def _config_files(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _config_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from _config_files(child)
    elif isinstance(value, str) and value.startswith("CONFIG_FILE="):
        yield value.removeprefix("CONFIG_FILE=").split(":", 1)[0]


def _power_recipes():
    master = yaml.safe_load(MASTER.read_text(encoding="utf-8"))
    recipes = {}
    for config in master.values():
        if not isinstance(config, dict):
            continue
        runner = config.get("runner")
        if (
            runner not in LAUNCHERS
            or config.get("model-prefix") not in {"dsv4", "kimik2.6"}
            or not config.get("disagg")
        ):
            continue
        for config_file in _config_files(config.get("scenarios", {})):
            relative = config_file.removeprefix("recipes/")
            path = RECIPE_ROOT / relative
            if path.exists() and "/agentic/" not in path.as_posix():
                recipes[path.relative_to(REPO_ROOT).as_posix()] = {
                    "runner": runner,
                    "model": config["model-prefix"],
                    "framework": config["framework"],
                }
    return recipes


POWER_RECIPES = _power_recipes()


def test_exactly_37_supported_recipes_are_selected_from_the_master_config():
    assert len(POWER_RECIPES) == 37
    assert Counter(item["runner"] for item in POWER_RECIPES.values()) == {
        "cluster:b200-nscale": 21,
        "b300": 16,
    }
    assert Counter(item["framework"] for item in POWER_RECIPES.values()) == {
        "dynamo-sglang": 18,
        "dynamo-vllm": 19,
    }
    assert all(
        "kimi-k3" not in path and "/agentic/" not in path for path in POWER_RECIPES
    )
    dedicated = sum(
        yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        .get("infra", {})
        .get("etcd_nats_dedicated_node", False)
        for path in POWER_RECIPES
    )
    assert dedicated == 15


def test_every_selected_recipe_uses_the_required_contract():
    for path, item in POWER_RECIPES.items():
        recipe = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        assert recipe["benchmark"]["type"] == "sa-bench", path
        assert recipe["telemetry"] == {
            "enabled": True,
            "provider": "dcgm-power",
            "default_frequency": 1.0,
            "storage_subdir": "power",
            "required": True,
            "startup_timeout_seconds": 120,
            "request_timeout_seconds": 2,
            "collector_join_timeout_seconds": 10,
            "dcgm_exporter": {
                "container_image": "dcgm-exporter",
                "port": 19401 if item["runner"] == "b300" else 9401,
            },
        }, path


def test_launchers_use_recipe_gating_and_the_exact_fork_commit():
    for launcher in set(LAUNCHERS.values()):
        text = launcher.read_text(encoding="utf-8")
        assert re.findall(r"^\s*USES_DCGM_POWER=(\S+)$", text, re.MULTILINE) == [
            "0",
            "1",
        ], launcher
        assert re.search(r'_RECIPE_REL="\$\{(?:_POWER_)?CONFIG_FILE%%:\*\}"', text), (
            launcher
        )
        assert "/^telemetry:/ { t = 1; next }" in text
        assert "t && /^  provider: dcgm-power$/ { p = 1 }" in text
        assert "t && /^  enabled: true$/        { e = 1 }" in text
        assert f'POWER_SRT_SLURM_URL="{PRODUCER_URL}"' in text
        assert f'POWER_SRT_SLURM_PIN="{PRODUCER_SHA}"' in text
        assert 'git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"' in text
        assert 'git checkout "$POWER_SRT_SLURM_PIN" || exit 1' in text
        assert 'test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN"' in text
        assert 'git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"' in text


def test_launcher_awk_gate_accepts_all_selected_recipes_and_rejects_non_power():
    program = """
        /^telemetry:/ { t = 1; next }
        t && /^[^ ]/  { t = 0 }
        t && /^  provider: dcgm-power$/ { p = 1 }
        t && /^  enabled: true$/        { e = 1 }
        END { exit !(p && e) }
    """
    for path in POWER_RECIPES:
        result = subprocess.run(
            ["awk", program, str(REPO_ROOT / path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (path, result.stderr)

    non_power = RECIPE_ROOT / "vllm/kimi-k3/agentic/agg-b200-dep8-mtp-agentic.yaml"
    result = subprocess.run(
        ["awk", program, str(non_power)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_launchers_provision_and_preserve_power_provenance():
    for launcher in set(LAUNCHERS.values()):
        text = launcher.read_text(encoding="utf-8")
        assert f'DCGM_EXPORTER_IMAGE="{EXPORTER_IMAGE}"' in text
        assert (
            'DCGM_EXPORTER_ENROOT_REF="${DCGM_EXPORTER_IMAGE/nvcr.io\\//nvcr.io#}"'
            in text
        ), launcher
        assert (
            'import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_ENROOT_REF"' in text
            or 'enroot import -o \\"$DCGM_EXPORTER_SQSH\\" \\"docker://$DCGM_EXPORTER_ENROOT_REF\\"'
            in text
        ), launcher
        assert 'test -r "$DCGM_EXPORTER_SQSH"' in text
        assert (
            'sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"'
            in text
        )
        assert 'dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml' in text
        assert 'grep -q "^  dcgm-exporter: " srtslurm.yaml ||' in text
        assert (
            'cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"'
            in text
        )
        assert (
            'cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt"'
            in text
        )


def test_non_power_specialized_clone_revisions_remain_available():
    expected = {
        "runners/launch_b200-nscale-slurm.sh": {
            "04e87fcc505d6d851451781a5499ca19a02ec2b4",
            "aflowers/vllm-gb200-v0.20.0",
            "c180328b98c3793ca84a1e24a030f90545eb7d5d",
            "217f9438",
        },
        "runners/launch_b300-nv.sh": {
            "aflowers/vllm-gb200-v0.20.0",
            "c180328b98c3793ca84a1e24a030f90545eb7d5d",
            "cam/sa-submission-q2-2026",
        },
    }
    for relative, revisions in expected.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert revisions <= {revision for revision in revisions if revision in text}, (
            relative
        )


def test_process_result_ci_covers_the_new_contract():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for path in (
        "configs/nvidia-master.yaml",
        "runners/launch_b200-nscale-slurm.sh",
        "runners/launch_b200-nscale-compat.sh",
        "runners/launch_b300-nv.sh",
        "utils/test_b200_b300_power_official_contract.py",
    ):
        assert f"- '{path}'" in workflow
    assert (
        "test_b200_b300_power_official_contract.py"
        in workflow.split("- name: Run pytest", 1)[1]
    )


def test_process_result_ci_globs_cover_every_selected_recipe():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    covered = set()

    for recipe_glob in PROCESS_RESULT_RECIPE_GLOBS:
        assert f"- '{recipe_glob}'" in workflow
        result = subprocess.run(
            ["git", "ls-files", f":(glob){recipe_glob}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        covered.update(result.stdout.splitlines())

    assert set(POWER_RECIPES) <= covered
