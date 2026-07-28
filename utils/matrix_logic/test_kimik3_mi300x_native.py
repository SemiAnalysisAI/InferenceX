"""CPU-only gates for the Kimi K3 MI300X native multi-node AgentX path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "utils" / "matrix_logic"))

from generate_sweep_configs import generate_test_config_sweep  # noqa: E402
from validation import load_config_files, load_runner_file  # noqa: E402

CONFIG_KEY = "kimik3-fp4-mi300x-vllm-agentic"
SERVER_SCRIPT = (
    REPO_ROOT / "benchmarks" / "multi_node" / "agentic" / "kimik3_fp4_mi300x_vllm.sh"
)
PREFLIGHT_SCRIPT = REPO_ROOT / "runners" / "mi300x_native_node_preflight.sh"
DEFAULT_LAUNCHER = REPO_ROOT / "runners" / "launch_mi300x-amds.sh"
NATIVE_LAUNCHER = REPO_ROOT / "runners" / "launch_mi300x-amds-native-multinode.sh"
IMAGE = "vllm/vllm-openai-rocm:kimi-k3"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
STAGING_COMMANDS = ("hf download", "huggingface-cli download", "wget", "curl")
JOB_ID = "4242"
RESULT_FILENAME = "kimik3_agentic_prefill-tp8-pp2_conc4_mi300x-amds_00"


def generate_kimik3_matrix() -> list[dict]:
    configs = load_config_files([str(REPO_ROOT / "configs" / "amd-master.yaml")])
    runners = load_runner_file(str(REPO_ROOT / "configs" / "runners.yaml"))
    args = argparse.Namespace(
        config_keys=[CONFIG_KEY],
        seq_lens=None,
        conc=None,
        scenario_type=["agentic-coding"],
        runner_node_filter=None,
    )
    return generate_test_config_sweep(args, configs, runners)


def test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs() -> None:
    rows = generate_kimik3_matrix()

    assert [row["conc"] for row in rows] == [[1], [2], [4], [8]]
    assert {row["runner"] for row in rows} == {"cluster:mi300x-amds"}
    assert {row["framework"] for row in rows} == {"vllm"}
    assert {row["disagg"] for row in rows} == {False}
    assert {
        (
            row["prefill"]["num-worker"],
            row["prefill"]["tp"],
            row["prefill"]["pp"],
            row["prefill"]["ep"],
            row["prefill"]["dp-attn"],
            row["decode"]["num-worker"],
            row["decode"]["tp"],
            row["decode"]["pp"],
            row["decode"]["ep"],
            row["decode"]["dp-attn"],
        )
        for row in rows
    } == {(1, 8, 2, 1, False, 0, 8, 2, 1, False)}
    settings = rows[0]["prefill"]["additional-settings"]
    assert settings == ["NATIVE_MULTINODE=1"]
    assert all("AITER_SITUV2_A8W4" not in setting for setting in settings)


def server_env(rank: int = 0) -> dict[str, str]:
    env = {
        **os.environ,
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PATH": "/models/Kimi-K3",
        "PORT": "8888",
        "CONC_LIST": "4",
        # benchmark_lib.sh requires this for every agentic/ caller; the workflow
        # always exports it from the matrix row.
        "KV_OFFLOADING": "none",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "PREFILL_PP_SIZE": "2",
        "PREFILL_EP": "1",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "0",
        "MULTINODE_NODE_COUNT": "2",
        "MULTINODE_GPUS_PER_NODE": "8",
        "MULTINODE_NODE_RANK": str(rank),
        "MULTINODE_MASTER_ADDR": "node-a",
        "KIMIK3_VLLM_DRY_RUN": "1",
    }
    # The unset-AITER case is a real assertion, so never inherit the ambient value.
    env.pop("AITER_SITUV2_A8W4", None)
    return env


def run_server(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SERVER_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_rank_zero_serves_tp8_pp2_without_headless() -> None:
    result = run_server(server_env(0))
    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--pipeline-parallel-size 2" in result.stdout
    assert "--nnodes 2" in result.stdout
    assert "--node-rank 0" in result.stdout
    assert "--master-addr node-a" in result.stdout
    assert "--headless" not in result.stdout
    assert "FLASHMLA" not in result.stdout
    assert "FLASHINFER" not in result.stdout


def test_rank_one_is_headless() -> None:
    result = run_server(server_env(1))
    assert result.returncode == 0, result.stderr
    assert "--node-rank 1" in result.stdout
    assert "--headless" in result.stdout


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PREFILL_TP", "16", "TP8 x PP2"),
        ("PREFILL_PP_SIZE", "1", "TP8 x PP2"),
        ("PREFILL_EP", "8", "EP1"),
        ("DECODE_NUM_WORKERS", "1", "aggregated"),
        ("CONC_LIST", "4 8", "one concurrency"),
        ("CONC_LIST", "16", "1, 2, 4, or 8"),
        ("AITER_SITUV2_A8W4", "auto", "0 or 1"),
    ],
)
def test_server_rejects_out_of_contract_values(
    name: str, value: str, message: str
) -> None:
    env = server_env()
    env[name] = value
    result = run_server(env)
    assert result.returncode != 0
    assert message in result.stderr


def test_aiter_mode_is_not_defaulted_and_accepts_both_modes() -> None:
    unset_result = run_server(server_env())
    assert "AITER_SITUV2_A8W4=unset" in unset_result.stdout
    for value in ("0", "1"):
        env = server_env()
        env["AITER_SITUV2_A8W4"] = value
        result = run_server(env)
        assert result.returncode == 0
        assert f"AITER_SITUV2_A8W4={value}" in result.stdout


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def rocminfo_output(gpu_count: int, gpu_arch: str) -> str:
    """Mimic rocminfo closely enough to catch sloppy agent counting."""
    blocks = [
        "HSA Agents\n==========\n",
        "Agent 1\n*******\n"
        "  Name:                    AMD EPYC 9654 96-Core Processor\n"
        "  Marketing Name:          AMD EPYC 9654 96-Core Processor\n"
        "  Device Type:             CPU\n",
    ]
    for index in range(gpu_count):
        blocks.append(
            f"Agent {index + 2}\n*******\n"
            f"  Name:                    {gpu_arch}\n"
            "  Marketing Name:          AMD Instinct MI300X\n"
            "  Device Type:             GPU\n"
        )
    # The ISA block repeats the arch in a different shape; counting it would
    # double every GPU.
    blocks.append(
        "*** Done ***\nISA Info:\n  ISA 1\n"
        f"    Name:                    amdgcn-amd-amdhsa--{gpu_arch}:sramecc+:xnack-\n"
    )
    return "".join(blocks)


def make_node(
    tmp_path: Path,
    *,
    gpu_count: int = 8,
    gpu_arch: str = "gfx942",
    with_main_ref: bool = True,
    with_weight_index: bool = True,
    with_indexed_shard: bool = True,
) -> dict[str, object]:
    """Build one node's model cache, squash tree, and external-command fakes."""
    cache_root = tmp_path / "raid" / "hf-hub-cache" / "models--moonshotai--Kimi-K3"
    snapshot_dir = cache_root / "snapshots" / REVISION
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text('{"model_type": "kimi_k3"}\n')
    if with_main_ref:
        (cache_root / "refs").mkdir(parents=True)
        (cache_root / "refs" / "main").write_text(f"{REVISION}\n")
    if with_weight_index:
        (snapshot_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.layers.0.weight": "model-00001-of-00001.safetensors"
                    },
                }
            )
        )
    if with_indexed_shard:
        (snapshot_dir / "model-00001-of-00001.safetensors").write_text("weights\n")

    squash_dir = tmp_path / "raid" / "hf-hub-cache" / "inferencex" / "squash"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cmd_log = tmp_path / "commands.log"
    cmd_log.write_text("")

    write_executable(
        bin_dir / "rocminfo",
        "#!/usr/bin/env bash\ncat <<'ROCMINFO_EOF'\n"
        + rocminfo_output(gpu_count, gpu_arch)
        + "ROCMINFO_EOF\n",
    )
    write_executable(
        bin_dir / "unsquashfs",
        "#!/usr/bin/env bash\n"
        'echo "unsquashfs $*" >> "$FAKE_CMD_LOG"\n'
        "for arg in \"$@\"; do target=\"$arg\"; done\n"
        '[[ -s "$target" ]] || exit 1\n'
        "exit 0\n",
    )
    write_executable(
        bin_dir / "enroot",
        "#!/usr/bin/env bash\n"
        'echo "enroot $*" >> "$FAKE_CMD_LOG"\n'
        '[[ "${1:-}" == "import" ]] || exit 1\n'
        "out=\"\"\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        '    -o) out="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        '[[ -n "$out" ]] || exit 1\n'
        "printf 'fake-squash-image\\n' > \"$out\"\n",
    )
    # Linux runners use the real flock; macOS has none, so shim only there.
    if shutil.which("flock") is None:
        write_executable(
            bin_dir / "flock",
            "#!/usr/bin/env bash\n"
            'echo "flock $*" >> "$FAKE_CMD_LOG"\n'
            "exit 0\n",
        )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "KIMIK3_MODEL_CACHE_ROOT": str(cache_root),
        "KIMIK3_SQUASH_DIR": str(squash_dir),
        "IMAGE": IMAGE,
        "FAKE_CMD_LOG": str(cmd_log),
        "KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS": "5",
    }
    return {"squash_dir": squash_dir, "cmd_log": cmd_log, "env": env}


def run_preflight(node: dict[str, object]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(PREFLIGHT_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=node["env"],
        capture_output=True,
        text=True,
    )


def preflight_records(stdout: str) -> list[str]:
    return [
        line
        for line in stdout.splitlines()
        if line.startswith("INFERENCEX_KIMIK3_PREFLIGHT ")
    ]


def test_preflight_imports_and_validates_image_in_node_local_tree(
    tmp_path: Path,
) -> None:
    node = make_node(tmp_path)
    result = run_preflight(node)

    assert result.returncode == 0, result.stderr
    records = preflight_records(result.stdout)
    assert len(records) == 1
    record = records[0]
    assert f"revision={REVISION}" in record
    assert "gpu_count=8" in record
    assert "gpu_arch=gfx942" in record

    squashes = sorted(node["squash_dir"].glob("*.sqsh"))
    assert len(squashes) == 1
    assert squashes[0].stat().st_size > 0
    assert f"squash_size_bytes={squashes[0].stat().st_size}" in record

    command_log = node["cmd_log"].read_text()
    assert f"docker://{IMAGE}" in command_log
    assert not any(command in command_log for command in STAGING_COMMANDS)
    script_source = PREFLIGHT_SCRIPT.read_text()
    assert not any(command in script_source for command in STAGING_COMMANDS)


def test_preflight_reuses_a_valid_squash_without_import(tmp_path: Path) -> None:
    node = make_node(tmp_path)
    assert run_preflight(node).returncode == 0

    node["cmd_log"].write_text("")
    result = run_preflight(node)

    assert result.returncode == 0, result.stderr
    assert len(preflight_records(result.stdout)) == 1
    assert "enroot import" not in node["cmd_log"].read_text()


def test_preflight_rejects_seven_gpus(tmp_path: Path) -> None:
    result = run_preflight(make_node(tmp_path, gpu_count=7))
    assert result.returncode != 0
    assert "exactly 8 gfx942" in result.stderr


def test_preflight_rejects_wrong_architecture(tmp_path: Path) -> None:
    result = run_preflight(make_node(tmp_path, gpu_arch="gfx950"))
    assert result.returncode != 0
    assert "exactly 8 gfx942" in result.stderr


def test_preflight_rejects_missing_main_ref(tmp_path: Path) -> None:
    result = run_preflight(make_node(tmp_path, with_main_ref=False))
    assert result.returncode != 0
    assert "refs/main" in result.stderr


def test_preflight_rejects_missing_weight_index(tmp_path: Path) -> None:
    result = run_preflight(make_node(tmp_path, with_weight_index=False))
    assert result.returncode != 0
    assert "model.safetensors.index.json" in result.stderr


def test_preflight_rejects_missing_indexed_shard(tmp_path: Path) -> None:
    result = run_preflight(make_node(tmp_path, with_indexed_shard=False))
    assert result.returncode != 0
    assert "missing weight shard" in result.stderr


# --- Slurm lifecycle -------------------------------------------------------
#
# The fakes below stand in for external commands only. Every assertion targets
# launcher behaviour -- what it allocated, what it refused, what it cleaned up --
# never the fakes themselves.

FAKE_SALLOC = """#!/usr/bin/env bash
echo "salloc $*" >> "$FAKE_CMD_LOG"
echo "salloc: Granted job allocation {job_id}" >&2
if [[ "$*" == *--parsable* ]]; then
    echo "{job_id}"
fi
"""

FAKE_SCANCEL = """#!/usr/bin/env bash
echo "scancel $*" >> "$FAKE_CMD_LOG"
"""

FAKE_SQUEUE = """#!/usr/bin/env bash
echo "squeue $*" >> "$FAKE_CMD_LOG"
"""

FAKE_CURL = """#!/usr/bin/env bash
echo "curl $*" >> "$FAKE_CMD_LOG"
[[ "${FAKE_HEALTH:-ok}" == "ok" ]]
"""

FAKE_SRUN = """#!/usr/bin/env bash
args="$*"
# One invocation is one log entry, even when a bash -c body spans lines.
printf 'srun %s\\n' "$(printf '%s' "$args" | tr '\\n' ' ')" >> "$FAKE_CMD_LOG"

if [[ "$args" == *mi300x_native_node_preflight.sh* ]]; then
    cat "$FAKE_PREFLIGHT_OUTPUT"
    exit 0
fi

if [[ "$args" == *--kill-on-bad-exit=1* ]]; then
    mode="${FAKE_SERVER_MODE:-alive}"
    if [[ "$mode" == exit:* ]]; then
        exit "${mode#exit:}"
    fi
    # Outlive the launcher's health polling until the allocation is cancelled.
    sleep 600
    exit 0
fi

if [[ "$args" == *agentic_srt.sh* ]]; then
    # Stand in for the rank-0 container: write the bounded handoff archive into
    # the workspace file the host pre-created for us.
    handoff="$GITHUB_WORKSPACE/$(basename "$KIMIK3_HANDOFF_PATH")"
    staging=$(mktemp -d)
    mkdir -p "$staging/output" "$staging/agentic/conc_${CONC_LIST}/aiperf_artifacts"
    printf '{"num_requests_successful": 12}\\n' \
        > "$staging/output/${RESULT_FILENAME}_conc${CONC_LIST}.json"
    printf '{}\\n' \
        > "$staging/agentic/conc_${CONC_LIST}/aiperf_artifacts/profile_export.json"
    ( cd "$staging" && tar czf - output agentic ) > "$handoff"
    rm -rf "$staging"
    sleep "${FAKE_CLIENT_SLEEP_SECONDS:-0}"
    exit "${FAKE_CLIENT_EXIT_CODE:-0}"
fi

if [[ "$args" == *hostname* ]]; then
    echo "node-a"
    exit 0
fi

exit 0
"""


def preflight_record(hostname: str, revision: str) -> str:
    return (
        f"INFERENCEX_KIMIK3_PREFLIGHT hostname={hostname} revision={revision} "
        "gpu_count=8 gpu_arch=gfx942 squash_size_bytes=33076838400"
    )


def make_cluster(
    tmp_path: Path,
    *,
    preflight_node_count: int = 2,
    mismatched_revisions: bool = False,
) -> dict[str, object]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cmd_log = tmp_path / "commands.log"
    cmd_log.write_text("")

    write_executable(bin_dir / "salloc", FAKE_SALLOC.format(job_id=JOB_ID))
    write_executable(bin_dir / "scancel", FAKE_SCANCEL)
    write_executable(bin_dir / "squeue", FAKE_SQUEUE)
    write_executable(bin_dir / "curl", FAKE_CURL)
    write_executable(bin_dir / "srun", FAKE_SRUN)

    hostnames = ["node-a", "node-b"][:preflight_node_count]
    revisions = [REVISION, OTHER_REVISION if mismatched_revisions else REVISION]
    preflight_output = tmp_path / "preflight.txt"
    preflight_output.write_text(
        "".join(
            f"{preflight_record(host, revisions[index])}\n"
            for index, host in enumerate(hostnames)
        )
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(workspace),
        "FAKE_CMD_LOG": str(cmd_log),
        "FAKE_PREFLIGHT_OUTPUT": str(preflight_output),
        "RUNNER_NAME": "mi300x-amds_00",
        "IMAGE": IMAGE,
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PREFIX": "kimik3",
        "PRECISION": "fp4",
        "FRAMEWORK": "vllm",
        "EXP_NAME": "kimik3_p1x8_d0x8_conc4",
        "RESULT_FILENAME": RESULT_FILENAME,
        "SCENARIO_TYPE": "agentic-coding",
        "SCENARIO_SUBDIR": "agentic/",
        "IS_AGENTIC": "1",
        "IS_MULTINODE": "true",
        "DISAGG": "false",
        "SPEC_DECODING": "none",
        "KV_OFFLOADING": "none",
        "CONC_LIST": "4",
        "PORT": "8888",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "PREFILL_PP_SIZE": "2",
        "PREFILL_EP": "1",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "0",
        "DECODE_TP": "8",
        "DECODE_PP_SIZE": "2",
        "DECODE_EP": "1",
        "DECODE_DP_ATTN": "false",
        "KIMIK3_MODEL_CACHE_ROOT": str(tmp_path / "raid" / "models--moonshotai--Kimi-K3"),
        "KIMIK3_SQUASH_DIR": str(tmp_path / "raid" / "squash"),
        "KIMIK3_STARTUP_TIMEOUT_SECONDS": "60",
        "KIMIK3_HEALTH_POLL_SECONDS": "1",
        "KIMIK3_CLEANUP_TIMEOUT_SECONDS": "5",
        "KIMIK3_CLEANUP_POLL_SECONDS": "1",
        "FAKE_HEALTH": "ok",
    }
    env.pop("NATIVE_MULTINODE", None)
    return {"workspace": workspace, "cmd_log": cmd_log, "env": env}


def run_launcher(
    cluster: dict[str, object], script: Path = NATIVE_LAUNCHER, timeout: int = 120
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(REPO_ROOT),
        env=cluster["env"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def grep_supports_perl_regex() -> bool:
    return (
        subprocess.run(
            ["grep", "-oP", "x"], input="x", capture_output=True, text=True
        ).returncode
        == 0
    )


@pytest.mark.skipif(
    not grep_supports_perl_regex(),
    reason="the pre-existing single-node launcher parses salloc with GNU grep -P",
)
def test_default_launcher_keeps_existing_single_node_path(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path)
    cluster["env"].update(
        {
            "TP": "8",
            "HF_HUB_CACHE": "/hf-hub-cache",
            "SCENARIO_SUBDIR": "fixed_seq_len/",
            "IS_MULTINODE": "false",
            "IS_AGENTIC": "0",
        }
    )
    result = run_launcher(cluster, script=DEFAULT_LAUNCHER)

    assert result.returncode == 0, result.stderr
    command_log = cluster["cmd_log"].read_text()
    assert "--nodes=2" not in command_log
    assert "mi300x_native_node_preflight.sh" not in command_log


def test_native_launcher_uses_two_full_nodes_and_all_node_preflight(
    tmp_path: Path,
) -> None:
    cluster = make_cluster(tmp_path)
    cluster["env"]["NATIVE_MULTINODE"] = "1"
    result = run_launcher(cluster, script=DEFAULT_LAUNCHER)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = cluster["cmd_log"].read_text().splitlines()

    allocation = next(line for line in lines if line.startswith("salloc "))
    assert "--nodes=2" in allocation
    assert "--gres=gpu:8" in allocation

    preflight = next(
        line for line in lines if "mi300x_native_node_preflight.sh" in line
    )
    assert "--ntasks=2" in preflight

    server = next(line for line in lines if "--kill-on-bad-exit=1" in line)
    assert "kimik3_fp4_mi300x_vllm.sh" in server

    client = next(line for line in lines if "agentic_srt.sh" in line)
    assert "--overlap" in client
    assert "--nodelist=node-a" in client

    assert f"scancel {JOB_ID}" in cluster["cmd_log"].read_text()


def test_native_launcher_rejects_topology_before_salloc(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path)
    cluster["env"]["PREFILL_PP_SIZE"] = "1"
    result = run_launcher(cluster)

    assert result.returncode != 0
    assert "TP8 x PP2" in result.stderr
    assert "salloc " not in cluster["cmd_log"].read_text()


def test_native_launcher_rejects_one_preflight_record(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path, preflight_node_count=1)
    result = run_launcher(cluster)

    assert result.returncode != 0
    assert "--kill-on-bad-exit=1" not in cluster["cmd_log"].read_text()


def test_native_launcher_rejects_mismatched_revisions(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path, mismatched_revisions=True)
    result = run_launcher(cluster)

    assert result.returncode != 0
    assert "--kill-on-bad-exit=1" not in cluster["cmd_log"].read_text()


def test_server_failure_preserves_failure_and_cancels_allocation(
    tmp_path: Path,
) -> None:
    cluster = make_cluster(tmp_path)
    cluster["env"]["FAKE_SERVER_MODE"] = "exit:23"
    cluster["env"]["FAKE_HEALTH"] = "down"
    result = run_launcher(cluster)

    assert result.returncode != 0
    assert "exited" in (result.stdout + result.stderr)
    assert "agentic_srt.sh" not in cluster["cmd_log"].read_text()
    assert f"scancel {JOB_ID}" in cluster["cmd_log"].read_text()


def test_sigterm_returns_143_and_reaps_server_and_allocation(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path)
    cluster["env"]["FAKE_HEALTH"] = "down"
    cluster["env"]["KIMIK3_STARTUP_TIMEOUT_SECONDS"] = "300"

    process = subprocess.Popen(
        ["bash", str(NATIVE_LAUNCHER)],
        cwd=str(REPO_ROOT),
        env=cluster["env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if "curl " in cluster["cmd_log"].read_text():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("launcher never reached server health polling")
        process.terminate()
        output = process.communicate(timeout=10)[0]
    except BaseException:
        process.kill()
        raise

    assert process.returncode == 143, output
    assert "stopping server step" in output
    assert f"scancel {JOB_ID}" in cluster["cmd_log"].read_text()


def test_success_extracts_only_host_owned_bounded_artifacts(tmp_path: Path) -> None:
    cluster = make_cluster(tmp_path)
    workspace = cluster["workspace"]
    result = run_launcher(cluster)

    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = workspace / f"{RESULT_FILENAME}_conc4.json"
    raw_artifact = (
        workspace / "LOGS" / "agentic" / "conc_4" / "aiperf_artifacts"
        / "profile_export.json"
    )
    server_logs = workspace / "multinode_server_logs.tar.gz"
    for path in (aggregate, raw_artifact, server_logs):
        assert path.is_file(), f"missing {path}"
        assert path.stat().st_uid == os.getuid()

    handoffs = sorted(workspace.glob("*handoff*"))
    assert handoffs == []
