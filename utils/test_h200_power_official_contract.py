"""Static contract for the H200 multinode AgentX dcgm-power infrastructure."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "runners/launch_h200-dgxc-slurm.sh"
HELPER = REPO_ROOT / "runners/inject_srt_power_concurrencies.py"
RECIPE = REPO_ROOT / (
    "benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/agentic/"
    "disagg-h200-2p2d-pcp8-tp8-dp8-mtp.yaml"
)
DSV4_RECIPE = REPO_ROOT / (
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic/"
    "agg-h200-tp8-mtp-kvoffload.yaml"
)
WORKFLOW = REPO_ROOT / ".github/workflows/test-process-result.yml"
E2E_WORKFLOW = REPO_ROOT / ".github/workflows/e2e-tests.yml"

PRODUCER_SHA = "e5c837f06a362dc888dfea2ee588e9f19c298270"
PRODUCER_URL = "https://github.com/edwingao28/srt-slurm.git"


def test_h200_launcher_detects_recipe_opt_in_and_rejects_unvalidated_lanes():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "USES_DCGM_POWER=0" in launcher
    assert '_RECIPE_REL="${CONFIG_FILE%%:*}"' in launcher
    assert (
        '_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"'
        in launcher
    )
    assert "/^telemetry:/ { t = 1; next }" in launcher
    assert "t && /^  provider: dcgm-power$/ { p = 1 }" in launcher
    assert "t && /^  enabled: true$/        { e = 1 }" in launcher
    assert "USES_DCGM_POWER=1" in launcher
    assert '"$IS_AGENTIC" != "1"' in launcher
    assert '"$FRAMEWORK" != "dynamo-sglang"' in launcher
    assert '"$MODEL_PREFIX" != "glm5.2"' in launcher
    assert '"$MODEL_PREFIX" != "dsv4"' in launcher
    assert '"$PRECISION" != "fp8"' in launcher


def test_power_agentx_lanes_use_exact_fork_sha_and_kimi_clone_is_unchanged():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert f'POWER_SRT_SLURM_URL="{PRODUCER_URL}"' in launcher
    assert f'POWER_SRT_SLURM_PIN="{PRODUCER_SHA}"' in launcher
    assert 'git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"' in launcher
    assert 'git checkout "$POWER_SRT_SLURM_PIN" || exit 1' in launcher
    assert 'test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN"' in launcher
    assert 'git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"' in launcher

    assert '"$MODEL_PREFIX" == "dsv4"' in launcher

    # Only the power lane may swap the runtime; non-power AgentX runs keep the
    # NVIDIA release their perf-changelog provenance records.
    assert (
        "git clone --branch v1.0.44 --single-branch https://github.com/NVIDIA/srt-slurm.git"
        in launcher
    )
    assert (
        "git clone --branch v1.0.38 --single-branch https://github.com/NVIDIA/srt-slurm.git"
        in launcher
    )

    # Kimi remains deliberately outside this PR.
    assert "https://github.com/functionstackx/srt-slurm-nv.git" in launcher
    assert "df5baa93f4caf5169dea2a4236ad2cc742fe40e7" in launcher


def test_power_lane_provisions_exporter_and_injects_container_mapping():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"' in launcher
    assert 'DCGM_EXPORTER_SQSH="/data/gharunners/containers/' in launcher
    assert 'unsquashfs -l "$DCGM_EXPORTER_SQSH"' in launcher
    assert 'sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"' in launcher
    assert '"/^  nginx-sqsh:/a' in launcher
    assert 'dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml' in launcher
    assert 'grep -q "^  dcgm-exporter: " srtslurm.yaml ||' in launcher


def test_launcher_injects_exact_matrix_concurrencies_and_finalizes_each_result():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert str(HELPER.relative_to(REPO_ROOT)) in launcher
    assert 'read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"' in launcher
    assert '"${POWER_CONCURRENCIES[@]}"' in launcher
    assert '--power-dir "$POWER_LOGS_ROOT/power"' in launcher
    assert '--logs-root "$POWER_LOGS_ROOT"' in launcher
    assert '--expected-producer-sha "$POWER_SRT_SLURM_PIN"' in launcher
    assert '"$GITHUB_WORKSPACE/${RESULT_FILENAME}_conc${concurrency}.json"' in launcher
    assert '"$POWER_LOGS_ROOT/agentic/conc_${concurrency}"' in launcher
    assert 'cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"' in launcher
    assert 'cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt"' in launcher


def test_pr_b_requires_glm_and_enables_optional_dsv4_aggregate():
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    dsv4_recipe = yaml.safe_load(DSV4_RECIPE.read_text(encoding="utf-8"))

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
            "port": 9401,
        },
    }
    # The launcher injects the exact matrix point before srtctl validation.
    assert "concurrencies" not in recipe["benchmark"]
    assert recipe["benchmark"]["type"] == "custom"
    assert recipe["infra"]["etcd_nats_dedicated_node"] is True

    assert dsv4_recipe["telemetry"] == {
        "enabled": True,
        "provider": "dcgm-power",
        "default_frequency": 1.0,
        "storage_subdir": "power",
        "required": False,
        "startup_timeout_seconds": 120,
        "request_timeout_seconds": 2,
        "collector_join_timeout_seconds": 10,
        "dcgm_exporter": {
            "container_image": "dcgm-exporter",
            "port": 9401,
        },
    }
    assert "concurrencies" not in dsv4_recipe["benchmark"]
    assert dsv4_recipe["benchmark"]["type"] == "custom"
    assert dsv4_recipe["benchmark"]["env"]["IS_MULTINODE"] == "true"
    assert dsv4_recipe["resources"] == {
        "gpu_type": "h200",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
        "gpus_per_agg": 8,
    }


def test_process_result_ci_covers_h200_power_files():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for path in (
        "benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/agentic/disagg-h200-2p2d-pcp8-tp8-dp8-mtp.yaml",
        "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic/agg-h200-tp8-mtp-kvoffload.yaml",
        "runners/launch_h200-dgxc-slurm.sh",
        "runners/inject_srt_power_concurrencies.py",
        "utils/test_h200_power_official_contract.py",
        "utils/test_inject_srt_power_concurrencies.py",
    ):
        assert f"- '{path}'" in workflow
    pytest_command = workflow.split("- name: Run pytest", 1)[1]
    assert "test_h200_power_official_contract.py" in pytest_command
    assert "test_inject_srt_power_concurrencies.py" in pytest_command


def test_multinode_agentic_workflow_forwards_strict_power_inputs():
    workflow = E2E_WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("    test-sweep-multi-node-agentic:\n", 1)[1].split(
        "    test-sweep-multi-node-agentic-evals:\n", 1
    )[0]

    assert "            require-power: ${{ inputs.require-power }}\n" in job
    assert (
        "            power-producer-sha: ${{ inputs.power-producer-sha }}\n" in job
    )


def test_producer_pin_is_immutable_and_only_declared_once():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert re.fullmatch(r"[0-9a-f]{40}", PRODUCER_SHA)
    assert launcher.count(PRODUCER_SHA) == 1
    assert launcher.count(PRODUCER_URL) == 1


def test_power_adapter_module_is_importable_from_launcher_working_directory():
    result = subprocess.run(
        [sys.executable, "-m", "utils.agentic.aggregation.power_adapter", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
