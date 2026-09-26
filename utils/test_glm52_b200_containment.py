"""C48-only launcher/setup and actual AgentX command construction contracts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = "recipes/glm5.2/sglang/b200-fp4/agentx/"
C48 = RECIPE_DIR + "disagg-variants.yaml:override_1p4d_tp4_c48"
C64 = RECIPE_DIR + "disagg-variants.yaml:override_1p1d_c64"
SETUP = "glm52-b200-c48-containment.sh"


def executable(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def launch(
    tmp_path,
    *,
    recipe=C48,
    eval_only="false",
    missing=None,
    use_default=False,
    hash_ok=True,
):
    workspace = tmp_path / "workspace"
    runners = workspace / "runners"
    configs = workspace / "benchmarks/multi_node/srt-slurm-recipes/configs"
    runners.mkdir(parents=True)
    configs.mkdir(parents=True)
    shutil.copy(ROOT / "runners/launch_b200-nscale-slurm.sh", runners)
    shutil.copy(ROOT / "runners/watch_glm52_b200_c48_cleanup.py", runners)
    shutil.copy(ROOT / "benchmarks/benchmark_lib.sh", workspace / "benchmarks")
    if missing != "setup":
        shutil.copy(
            ROOT / "benchmarks/multi_node/srt-slurm-recipes/configs" / SETUP, configs
        )
    home = tmp_path / "home"
    payload = (
        home
        / ".cache/inferencex/glm52-b200-c48/27aa9a6eb223616d956dd7d507c0e26839cadcfb84339179898ff3a502ccbcba"
        if use_default
        else tmp_path / "payload"
    )
    for name in (
        "install_native_containment.py",
        "inputs/integrated-candidate-file-hashes.json",
        "wheels/build-output-receipt.json",
        "wheels/ai_dynamo-1.5.0.dev20260909-py3-none-any.whl",
        "wheels/ai_dynamo_runtime-1.5.0.dev20260909-cp310-abi3-manylinux_2_39_x86_64.whl",
    ):
        if name == missing:
            continue
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    fixture = tmp_path / "fixture"
    config = fixture / recipe.split(":", 1)[0]
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({
        "base": {
            "name": "fixture-base",
            "health_check": {"max_attempts": 17, "interval_seconds": 3},
            "frontend": {"env": {}},
            "roles": {"prefill": {"env": {}}, "decode": {"env": {}}},
            "benchmark": {"env": {}},
        },
        recipe.split(":", 1)[1]: {"name": "fixture-variant"},
    }))
    (runners / "slurm_utils.sh").write_text(r"""
setup_srt_slurm() { command mkdir -p "$1"; cp -R "$FIXTURE/." "$1/"; cd "$1"; }
install_srt_slurm() { :; }
prepare_srt_power() { SRTCTL_RECIPE_ARGS=(--set 'telemetry.required=true'); }
run_srt_setup() { :; }
write_srt_cluster_config() {
    python3 -c 'import json,os,sys;open(os.environ["CLUSTER_ARGS"],"w").write(json.dumps(sys.argv[1:]))' "$@"
    printf '{}\n' > "$2"
}
apply_srt_recipe() {
    python3 -c 'import json,os,sys;open(os.environ["SUBMISSION"],"w").write(json.dumps(sys.argv[1:]))' "$@"
    printf '✅ Job 12345 submitted\n'
}
stream_slurm_job_log() { exit 0; }
python3() {
    if [[ "$1" == */watch_glm52_b200_c48_cleanup.py ]]; then
        command python3 -c 'import json,os,sys;open(os.environ["WATCHER_ARGS"],"w").write(json.dumps(sys.argv[1:]))' "$@"
        exit 0
    fi
    command python3 "$@"
}
# Refuse the launcher's external shared cache paths; use its workspace fallback.
mkdir() { case "$*" in *'/data/'*) return 1 ;; *) command mkdir "$@" ;; esac; }
chmod() { :; }
""")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("curl", "srtctl", "unsquashfs", "flock"):
        executable(bin_dir / name, "exit 0\n")
    # GNU sed differs on the local macOS test host. Ignore only its in-place
    # edit of the controlled recipe; all reads use the real sed implementation.
    executable(
        bin_dir / "sed",
        'if [[ "$1" == -i ]]; then exit 0; fi\nexec /usr/bin/sed "$@"\n',
    )
    executable(
        bin_dir / "grep",
        'if [[ "$1" == -oP ]]; then /usr/bin/sed -n \'s/.*Job \\([0-9][0-9]*\\) submitted.*/\\1/p\' | tail -1; else exec /usr/bin/grep "$@"; fi\n',
    )
    executable(bin_dir / "sha256sum", f"cat >/dev/null; exit {0 if hash_ok else 1}\n")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{Path(sys.executable).parent}:" + os.environ["PATH"],
        "HOME": str(home),
        "GITHUB_WORKSPACE": str(workspace),
        "FIXTURE": str(fixture),
        "SUBMISSION": str(tmp_path / "submission.json"),
        "CLUSTER_ARGS": str(tmp_path / "cluster.json"),
        "WATCHER_ARGS": str(tmp_path / "watcher.json"),
        "MODEL_PREFIX": "glm5.2",
        "FRAMEWORK": "dynamo-sglang",
        "PRECISION": "fp4",
        "SPEC_DECODING": "mtp",
        "MODEL": "controlled-model",
        "MODEL_PATH": "/controlled-weights",
        "IMAGE": "controlled/image:tag",
        "EVAL_ONLY": eval_only,
        "IS_AGENTIC": "1",
        "IS_MULTINODE": "true",
        "RUN_EVAL": "false",
        "SLURM_ACCOUNT": "controlled",
        "SLURM_PARTITION": "controlled",
        "CONFIG_FILE": recipe,
        "RUNNER_NAME": "controlled-runner",
        "ISL": "1",
        "OSL": "1",
        "CONC_LIST": "48",
    }
    env.pop("EVAL_CONFIG_FILE", None)
    env.pop("GLM52_CONTAINMENT_PAYLOAD", None)
    if not use_default:
        env["GLM52_CONTAINMENT_PAYLOAD"] = str(payload)
    result = subprocess.run(
        ["bash", str(runners / "launch_b200-nscale-slurm.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    submission = (
        json.loads(Path(env["SUBMISSION"]).read_text())
        if Path(env["SUBMISSION"]).exists()
        else None
    )
    cluster = (
        json.loads(Path(env["CLUSTER_ARGS"]).read_text())
        if Path(env["CLUSTER_ARGS"]).exists()
        else None
    )
    return result, submission, cluster, payload


@pytest.mark.parametrize("use_default", [False, True])
def test_c48_performance_submits_all_role_containment(tmp_path, use_default):
    result, submitted, cluster, payload = launch(tmp_path, use_default=use_default)
    assert result.returncode == 0, result.stdout + result.stderr
    assert submitted[submitted.index("--setup-script") + 1] == SETUP
    watcher = json.loads((tmp_path / "watcher.json").read_text())
    assert watcher[1:] == [
        "12345",
        "outputs/12345/logs/sweep_12345.log",
        "controlled-runner",
    ]
    overrides = [
        submitted[i + 1] for i, arg in enumerate(submitted[:-1]) if arg == "--set"
    ]
    assert set(overrides) == {
        "telemetry.required=true",
        'name="controlled-runner"',
        "dynamo.install=false",
        'frontend.env.DYN_GLM52_PREFILL_FAILURE_CONTAINMENT="1"',
        'roles.prefill.env.DYN_GLM52_PREFILL_FAILURE_CONTAINMENT="1"',
        'roles.decode.env.DYN_GLM52_PREFILL_FAILURE_CONTAINMENT="1"',
        'roles.prefill.env.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE="1"',
        'roles.decode.env.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE="1"',
        'benchmark.env.AIPERF_REQUEST_TIMEOUT_SECONDS="1800"',
    }
    mount = cluster.index(str(payload))
    assert cluster[mount - 1 : mount + 2] == [
        "--mount",
        str(payload),
        "/glm52-containment:ro",
    ]


@pytest.mark.parametrize(
    "missing",
    [
        "setup",
        "install_native_containment.py",
        "inputs/integrated-candidate-file-hashes.json",
        "wheels/build-output-receipt.json",
        "wheels/ai_dynamo-1.5.0.dev20260909-py3-none-any.whl",
        "wheels/ai_dynamo_runtime-1.5.0.dev20260909-cp310-abi3-manylinux_2_39_x86_64.whl",
    ],
)
def test_missing_containment_input_prevents_submission(tmp_path, missing):
    result, submitted, cluster, _ = launch(tmp_path, missing=missing)
    assert result.returncode != 0
    assert "containment input is missing" in result.stderr
    assert submitted is None and cluster is None


def test_foreign_containment_payload_prevents_submission(tmp_path):
    result, submitted, cluster, _ = launch(tmp_path, hash_ok=False)
    assert result.returncode != 0 and "payload hash mismatch" in result.stderr
    assert submitted is None and cluster is None


@pytest.mark.parametrize(
    ("recipe", "eval_only"),
    [
        (C48, "true"),
        (
            C64,
            "false",
        ),
    ],
)
def test_unaffected_cells_do_not_require_or_activate_payload(
    tmp_path, recipe, eval_only
):
    result, submitted, cluster, payload = launch(
        tmp_path,
        recipe=recipe,
        eval_only=eval_only,
        missing="install_native_containment.py",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--setup-script" not in submitted
    assert not any(
        "CONTAINMENT" in value
        or "REQUEST_TIMEOUT" in value
        or "DEFERRED_DECODE" in value
        for value in submitted
    )
    assert str(payload) not in cluster


@pytest.mark.parametrize(
    ("recipe", "eval_only"),
    [(RECIPE_DIR + f"agg-variants.yaml:override_c{c}", "false") for c in (1, 4, 8)]
    + [(C48, "false"), (C64, "false"), (C48, "true"), (C64, "true")],
)
def test_canonical_variants_preserve_runner_identity_and_health_budget(
    tmp_path, recipe, eval_only
):
    sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
    from srtctl.core.config import generate_override_configs
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides

    result, submitted, _, _ = launch(tmp_path, recipe=recipe, eval_only=eval_only)
    assert result.returncode == 0, result.stdout + result.stderr
    source = tmp_path / "fixture" / recipe.split(":", 1)[0]
    raw = yaml.safe_load(source.read_text())
    overrides = [
        submitted[i + 1] for i, arg in enumerate(submitted[:-1]) if arg == "--set"
    ]
    apply_overrides_to_recipe(raw, parse_overrides(overrides, []))
    resolved = generate_override_configs(raw, selector=recipe.split(":", 1)[1])[0][1]
    assert resolved["name"] == "controlled-runner"
    assert resolved["health_check"] == {"max_attempts": 17, "interval_seconds": 3}
    active = recipe == C48 and eval_only == "false"
    assert ("--setup-script" in submitted) is active
    assert (
        "DYN_GLM52_PREFILL_FAILURE_CONTAINMENT"
        in resolved["frontend"].get("env", {})
    ) is active
    assert ("AIPERF_REQUEST_TIMEOUT_SECONDS" in resolved["benchmark"]["env"]) is active


@pytest.mark.parametrize(
    "timeout,expected",
    [
        ("", None),
        ("1800", 1800.0),
        ("1.25", 1.25),
        ("0", "error"),
        ("-1", "error"),
        ("nan", "error"),
        ("inf", "error"),
        ("1e999", "error"),
        ("2;echo unsafe", "error"),
    ],
)
def test_replay_timeout_validates_and_preserves_measurement(
    tmp_path, timeout, expected
):
    output = tmp_path / "command.json"
    cli = tmp_path / "aiperf"
    cli.write_text(
        f'#!{sys.executable}\nimport json,sys\nopen({str(output)!r},"w").write(json.dumps(sys.argv[1:]))\n'
    )
    cli.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}:" + os.environ["PATH"],
        "INFMAX_CONTAINER_WORKSPACE": str(tmp_path),
        "MODEL": "controlled-model",
        "PORT": "8000",
        "CONC": "48",
        "DURATION": "3600",
        "AIPERF_FAILED_REQUEST_THRESHOLD": "0",
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": "0.1",
        "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300",
        "AGENTIC_WARMUP_GRACE_PERIOD": "1800",
        "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES": "0",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "true",
        "AIPERF_UNSAFE_OVERRIDE": "false",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
        "AIPERF_WARMUP_REQUESTS_PER_LANE": "10",
        "AIPERF_REQUEST_TIMEOUT_SECONDS": timeout,
        "AIPERF_CLI": str(cli),
    }
    result = subprocess.run(
        [
            "bash",
            "-ec",
            'source "$1"; AIPERF_CLI="$3"; build_replay_cmd "$2"; eval "$REPLAY_CMD"',
            "bash",
            str(ROOT / "benchmarks/benchmark_lib.sh"),
            str(tmp_path),
            str(cli),
        ],
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    if expected == "error":
        assert result.returncode != 0 and "positive and finite" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        args = json.loads(output.read_text())
        if expected is None:
            assert "--request-timeout-seconds" not in args
        else:
            assert float(args[args.index("--request-timeout-seconds") + 1]) == expected
        assert args[args.index("--benchmark-duration") + 1] == "3600"
        assert args[args.index("--concurrency") + 1] == "48"
        assert args[args.index("--warmup-requests-per-lane") + 1] == "10"
        assert "--unsafe-override" not in args


@pytest.mark.parametrize("hash_ok", [True, False])
def test_setup_verifies_installer_before_workload_python(tmp_path, hash_ok):
    events = tmp_path / "events"
    script = r"""
sha256sum() { cat >/dev/null; return "$HASH_RC"; }
function /opt/sglang/bin/python3() { printf 'install %s\n' "$*" >> "$EVENTS"; }
cat() { if [[ "$#" == 0 ]]; then command cat; else printf 'readback %s\n' "$*" >> "$EVENTS"; fi; }
source "$1"
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "bash",
            str(ROOT / "benchmarks/multi_node/srt-slurm-recipes/configs" / SETUP),
        ],
        env={**os.environ, "EVENTS": str(events), "HASH_RC": "0" if hash_ok else "1"},
        capture_output=True,
        check=False,
        text=True,
    )
    if not hash_ok:
        assert result.returncode != 0 and not events.exists()
    else:
        assert result.returncode == 0, result.stderr
        lines = events.read_text().splitlines()
        assert lines[0].startswith(
            "install /glm52-containment/install_native_containment.py --inputs /glm52-containment/inputs --wheels /glm52-containment/wheels "
        )
        assert lines[1] == "readback /tmp/glm52-containment-install.json"
