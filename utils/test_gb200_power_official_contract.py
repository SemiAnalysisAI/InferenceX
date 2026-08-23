"""Static contract for the official GB200 dcgm-power lane.

Power behavior is recipe-gated: a run turns power on iff the recipe it
resolves carries an enabled dcgm-power telemetry block, and only then
provisions the exporter and clones the pinned producer fork. These tests read
launcher/recipe/workflow text; nothing is executed. Cross-launcher and
repo-wide invariants live here; the GB300 twin imports the shared helpers.
"""

import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb200-fp8/8k1k/1p1d-tp4-tp4.yaml"
GB200_LAUNCHER = REPO_ROOT / "runners/launch_gb200-nv.sh"
GB300_LAUNCHER = REPO_ROOT / "runners/launch_gb300-nv.sh"
TMPL_PATH = REPO_ROOT / ".github/workflows/benchmark-multinode-tmpl.yml"

STAMP_NAME = "power-producer-sha.txt"
EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"


def launcher_power_constants(path):
    """Derive URL+PIN from a launcher — their literals live nowhere else."""
    text = path.read_text()
    url = re.search(r'^POWER_SRT_SLURM_URL="([^"]+)"$', text, re.M)
    pin = re.search(r'^POWER_SRT_SLURM_PIN="([0-9a-f]{40})"$', text, re.M)
    assert url and pin, path
    return url.group(1), pin.group(1)


FORK_URL, PRODUCER_PIN = launcher_power_constants(GB200_LAUNCHER)


def git_grep_lines(*args):
    proc = subprocess.run(
        ["git", "grep", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when nothing matches.
    assert proc.returncode in (0, 1), proc.stderr
    return proc.stdout.splitlines()


def assert_recipe_driven_detection(launcher):
    assert "USES_DCGM_POWER=0" in launcher
    assert '_RECIPE_REL="${CONFIG_FILE%%:*}"' in launcher
    assert (
        '_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"'
        in launcher
    )
    # Upstream-only recipes have no workspace mirror and must stay non-power.
    assert '-f "$_RECIPE_SRC"' in launcher
    assert "/^telemetry:/ { t = 1; next }" in launcher
    assert "t && /^  provider: dcgm-power$/ { p = 1 }" in launcher
    assert "t && /^  enabled: true$/        { e = 1 }" in launcher
    assert "USES_DCGM_POWER=1" in launcher
    assert (
        'if [[ "$USES_DCGM_POWER" == "1" && "$FRAMEWORK" != "dynamo-sglang" ]]; then'
        in launcher
    )
    # Exporter provisioning, pinned clone, srtslurm.yaml injection, and audit
    # copies are each gated so non-power lanes keep the pre-existing path.
    assert launcher.count('if [[ "$USES_DCGM_POWER" == "1" ]]; then') >= 4


def assert_pinned_clone_contract(launcher):
    assert f'POWER_SRT_SLURM_URL="{FORK_URL}"' in launcher
    assert f'POWER_SRT_SLURM_PIN="{PRODUCER_PIN}"' in launcher
    assert 'git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"' in launcher
    assert 'git checkout "$POWER_SRT_SLURM_PIN" || exit 1' in launcher
    assert 'test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN"' in launcher
    assert f'git rev-parse HEAD > "$GITHUB_WORKSPACE/{STAMP_NAME}"' in launcher


def assert_exporter_provisioning(launcher):
    assert f'DCGM_EXPORTER_IMAGE="{EXPORTER_IMAGE}"' in launcher
    assert 'import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_IMAGE"' in launcher
    assert 'test -r "$DCGM_EXPORTER_SQSH"' in launcher
    assert (
        'sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"'
        in launcher
    )
    # Container line lands after the unique nginx-sqsh key so the heredoc
    # output of non-power lanes stays byte-identical.
    assert '"/^  nginx-sqsh:/a' in launcher
    assert 'dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml' in launcher
    assert 'grep -q "^  dcgm-exporter: " srtslurm.yaml || ' in launcher
    assert (
        'cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"'
        in launcher
    )
    assert f'cp "$GITHUB_WORKSPACE/{STAMP_NAME}" "$LOGS_DIR/power/{STAMP_NAME}"' in launcher


def test_recipe_declares_enabled_dcgm_power_lane():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    telemetry = recipe["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "dcgm-power"
    assert telemetry["required"] is True
    assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter"
    assert telemetry["dcgm_exporter"]["port"] == 9401

    # Official recipes keep the full sweep ladder; only the telemetry block
    # was added.
    assert recipe["benchmark"]["concurrencies"] == "1x2x4x8x16x32x64x128"
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["resources"]["gpus_per_node"] == 4


def test_launcher_detects_power_lane_from_recipe():
    assert_recipe_driven_detection(GB200_LAUNCHER.read_text())


def test_launcher_provisions_exporter_through_squash_dir_cache():
    launcher = GB200_LAUNCHER.read_text()
    assert_exporter_provisioning(launcher)
    assert 'DCGM_EXPORTER_SQSH="${SQUASH_DIR}/' in launcher
    assert 'unsquashfs -l "$DCGM_EXPORTER_SQSH"' in launcher


def test_launcher_pins_power_producer_and_keeps_nonpower_clone():
    launcher = GB200_LAUNCHER.read_text()
    assert_pinned_clone_contract(launcher)
    # Non-power qwen3.5 keeps the upstream clone with no ref pin.
    assert 'git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"' in launcher
    assert "v1.0.25" not in launcher


def test_pin_constants_identical_across_launchers():
    assert launcher_power_constants(GB300_LAUNCHER) == (FORK_URL, PRODUCER_PIN)
    assert re.fullmatch(r"[0-9a-f]{40}", PRODUCER_PIN)
    assert FORK_URL.startswith("https://") and FORK_URL.endswith("/srt-slurm.git")


def test_pin_literal_lives_only_in_the_two_launchers():
    hits = git_grep_lines("-lF", PRODUCER_PIN)
    assert sorted(hits) == [
        "runners/launch_gb200-nv.sh",
        "runners/launch_gb300-nv.sh",
    ]


# Every recipe that opts into dcgm-power, with the exporter port its cluster
# requires (im-gb300 nodes already bind 9401; gb200 keeps the default).
POWER_RECIPES = {
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb200-1p1d-dep8-dep16-6-c512.yaml": 9401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb200-1p1d-tp8-tp8-4-c1.yaml": 9401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb200-1p2d-dep8-dep16-10-c256.yaml": 9401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb300-1p1d-dep4-dep16-5-c1024.yaml": 19401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb300-1p1d-tp4-tp4-2-c1.yaml": 19401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb200-fp8/8k1k/1p1d-tp4-tp4.yaml": 9401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp4/8k1k/disagg/stp/8k1k_stp_lowlat_0.yaml": 19401,
    "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml": 19401,
}


def test_exactly_the_declared_recipes_opt_into_dcgm_power():
    hits = git_grep_lines(
        "-lF", "provider: dcgm-power", "--", "benchmarks/multi_node/srt-slurm-recipes"
    )
    assert sorted(hits) == sorted(POWER_RECIPES)


def test_every_power_recipe_declares_the_same_telemetry_contract():
    for path, port in POWER_RECIPES.items():
        telemetry = yaml.safe_load((REPO_ROOT / path).read_text())["telemetry"]
        assert telemetry["enabled"] is True, path
        assert telemetry["provider"] == "dcgm-power", path
        assert telemetry["default_frequency"] == 1.0, path
        assert telemetry["required"] is True, path
        assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter", path
        assert telemetry["dcgm_exporter"]["port"] == port, path


def test_lane_and_pin_have_no_env_override_backdoor():
    # All assignments are literals: no environment variable can flip the
    # lane or redirect the pinned clone.
    for path in (GB200_LAUNCHER, GB300_LAUNCHER):
        text = path.read_text()
        assert re.findall(r"^\s*USES_DCGM_POWER=(\S+)$", text, re.M) == ["0", "1"], path
        for name in ("POWER_SRT_SLURM_URL", "POWER_SRT_SLURM_PIN"):
            values = re.findall(rf"^\s*{name}=(.*)$", text, re.M)
            assert len(values) == 1 and "$" not in values[0], (path, name)


def test_workflows_carry_no_producer_sha_literal():
    offenders = []
    for wf in sorted((REPO_ROOT / ".github/workflows").iterdir()):
        if wf.suffix not in (".yml", ".yaml"):
            continue
        for lineno, line in enumerate(wf.read_text().splitlines(), 1):
            # Action pins ("uses: ...@<sha>") are the only allowed 40-hex.
            if "uses:" in line:
                continue
            if re.search(r"\b[0-9a-f]{40}\b", line):
                offenders.append(f"{wf.name}:{lineno}")
    assert offenders == []


def test_stamp_filename_agrees_between_launchers_and_template():
    stamp_writes = set()
    for path in (GB200_LAUNCHER, GB300_LAUNCHER):
        match = re.search(
            r'git rev-parse HEAD > "\$GITHUB_WORKSPACE/([^"]+)"', path.read_text()
        )
        assert match, path
        stamp_writes.add(match.group(1))

    tmpl = TMPL_PATH.read_text()
    read_match = re.search(r'export POWER_PRODUCER_SHA="\$\(cat ([^)]+)\)"', tmpl)
    assert read_match
    assert stamp_writes == {STAMP_NAME} == {read_match.group(1)}

    # The manual input stays the override: the stamp only fills an empty env.
    assert f'if [ -z "$POWER_PRODUCER_SHA" ] && [ -f {STAMP_NAME} ]; then' in tmpl
    assert "POWER_PRODUCER_SHA: ${{ inputs.power-producer-sha }}" in tmpl
