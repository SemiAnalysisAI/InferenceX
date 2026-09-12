import os
from pathlib import Path
import re
import subprocess

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
RECIPE_DIR = "benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic"


@pytest.mark.parametrize("hardware", ["b200", "h200"])
@pytest.mark.parametrize("power,wrong_head", [(True, False), (False, False), (True, True)])
def test_kimi_power_selects_verified_runtime(
    tmp_path: Path, hardware: str, power: bool, wrong_head: bool,
) -> None:
    launcher_name = {
        "b200": "launch_b200-nscale-slurm.sh",
        "h200": "launch_h200-dgxc-slurm.sh",
    }[hardware]
    source = (REPO / "runners" / launcher_name).read_text()
    source = source[:source.index('echo "Installing srtctl..."')]
    if hardware == "h200":
        source += "\nfi\n"
    source = source.replace(
        'source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"',
        'source "$TEST_SLURM_UTILS"',
    )
    source = re.sub(
        r'^AGENTX_POWER_SRT_SLURM_PIN="[0-9a-f]+"$',
        'AGENTX_POWER_SRT_SLURM_PIN="' + "a" * 40 + '"',
        source, flags=re.MULTILINE,
    )
    recipe = (
        REPO / RECIPE_DIR / "agg-h200-tp16dp2ep32-latency-agentic.yaml"
        if hardware == "h200"
        else next((REPO / RECIPE_DIR).glob("agg-b200*"))
    )
    data = yaml.safe_load(recipe.read_text())
    if not power:
        data.pop("telemetry")
    destination = tmp_path / RECIPE_DIR / recipe.name
    destination.parent.mkdir(parents=True)
    destination.write_text(yaml.safe_dump(data))
    recipe_ref = f"recipes/vllm/kimi-k3/agentic/{recipe.name}"
    harness = '''
set -e
function git() {
    printf '%s\\n' "$*" >> "$TEST_ROUTE_LOG"
    case "$1" in
        clone) for target in "$@"; do :; done; mkdir -p "$target" ;;
        checkout) printf '%s\\n' "$2" > .test-head ;;
        rev-parse) if [[ "$TEST_WRONG_HEAD" == 1 ]]; then printf 'bad-head\\n'; else cat .test-head; fi ;;
        *) return 99 ;;
    esac
}
function cp() {
    if [[ "$1" == -rT ]]; then
        mkdir -p "$3"
        command cp -R "$2"/. "$3"
    else
        command cp "$@"
    fi
}
'''
    env = dict(os.environ, GITHUB_WORKSPACE=str(tmp_path), TEST_ROUTE_LOG=str(tmp_path / "route.log"),
               TEST_SLURM_UTILS=str(REPO / "runners/slurm_utils.sh"), TEST_WRONG_HEAD=str(int(wrong_head)),
               IS_MULTINODE="true", IS_AGENTIC="1", MODEL_PREFIX="kimik3", MODEL="moonshotai/Kimi-K3",
               PRECISION="fp4", FRAMEWORK="dynamo-vllm" if hardware == "b200" else "vllm",
               CONFIG_FILE=recipe_ref, SPEC_DECODING="mtp", EVAL_ONLY="false", EVAL_FRAMEWORK="lm-eval")
    result = subprocess.run(["bash"], input=harness + source, text=True, capture_output=True, cwd=tmp_path, env=env)
    stamp = tmp_path / "power-producer-sha.txt"
    if wrong_head:
        assert result.returncode != 0
        assert not stamp.exists()
        return
    assert result.returncode == 0, result.stderr
    routed_recipe = tmp_path / "srt-slurm" / recipe_ref
    assert yaml.safe_load(routed_recipe.read_text()) == data
    if power:
        assert stamp.read_text().strip() == "a" * 40
        assert "edwingao28/srt-slurm.git" in (tmp_path / "route.log").read_text()
    else:
        assert not stamp.exists()
        assert "edwingao28/srt-slurm.git" not in (tmp_path / "route.log").read_text()


@pytest.mark.parametrize("hardware", ["b200", "h200"])
def test_kimi_failed_power_stages_evidence_before_exit(tmp_path: Path, hardware: str) -> None:
    filename = {"b200": "launch_b200-nscale-slurm.sh", "h200": "launch_h200-dgxc-slurm.sh"}[hardware]
    source = (REPO / "runners" / filename).read_text()
    start = source.index('AGENTX_POWER_RC="$SRT_JOB_RC"')
    end = source.index('exit "$AGENTX_POWER_RC"', start)
    end = source.index("\n", end) + 1
    source = source[start:end] + "fi\n"
    logs = tmp_path / "source-logs"
    logs.mkdir()
    for name in ("exporter-image.sha256", "power-producer-sha.txt"):
        (tmp_path / name).write_text("retained\n")
    harness = '''
set -e
collect_agentic_power_results() {
    mkdir -p "$2/power"
    printf 'invalid telemetry\\n' > "$2/power/validation.json"
    return 42
}
bundle_server_logs() { printf 'server evidence\\n' > "$2"; }
'''
    env = dict(os.environ, SRT_JOB_RC="0", USES_AGENTX_POWER="1", USES_KIMIK3_POWER="1",
               USES_DCGM_POWER="1", EVAL_ONLY="false", JOB_ID="123", CONC_LIST="1",
               GITHUB_WORKSPACE=str(tmp_path), LOGS_DIR=str(logs), RESULT_FILENAME="kimi-test",
               SELECTED_POWER_SRT_SLURM_PIN="a" * 40)
    result = subprocess.run(["bash"], input=harness + source, text=True, capture_output=True, cwd=tmp_path, env=env)
    assert result.returncode == 42, result.stderr
    assert (tmp_path / "LOGS/power/validation.json").read_text() == "invalid telemetry\n"
    assert (tmp_path / "multinode_server_logs.tar.gz").read_text() == "server evidence\n"


def test_b200_exporter_cold_import_uses_one_registry_separator(tmp_path: Path) -> None:
    source = (REPO / "runners/launch_b200-nscale-slurm.sh").read_text()
    start = source.index("enroot_uri_for_image() {")
    end = source.index('\nimport_squash "$SQUASH_FILE"', start)
    helpers = source[start:end]
    start = source.index('if [[ "$USES_DCGM_POWER" == "1" ]]; then\n    DCGM_EXPORTER_IMAGE=')
    end = source.index("\nfi", start) + len("\nfi")
    harness = '''
set -euo pipefail
flock() { :; }
unsquashfs() { test -s "$2"; }
enroot() {
    printf '%s\\n' "$4" > "$GITHUB_WORKSPACE/import-uri"
    printf 'exporter image\\n' > "$3"
}
sha256sum() { printf 'fixture-hash  %s\\n' "$1"; }
'''
    env = dict(os.environ, GITHUB_WORKSPACE=str(tmp_path), SQUASH_DIR=str(tmp_path),
               SQUASH_LOCK_TIMEOUT="1", USES_DCGM_POWER="1")
    result = subprocess.run(["bash"], input=harness + helpers + source[start:end], text=True,
                            capture_output=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    uri = (tmp_path / "import-uri").read_text().strip()
    assert uri.startswith("docker://nvcr.io#nvidia/k8s/dcgm-exporter:")
    assert uri.count("#") == 1
    assert (tmp_path / "exporter-image.sha256").is_file()
