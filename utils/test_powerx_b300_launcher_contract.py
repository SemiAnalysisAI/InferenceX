"""Contract tests for the PowerX B200/B300 single-node comparison.

The campaign intentionally reuses the already-measured B200 lane, so the
B300 follow-up must keep the request and server semantics aligned.  The only
allowed launcher differences are the B300 cluster's local checkpoint and
tokenizer path plumbing.
"""

import shlex
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONFIG = REPO_ROOT / "configs/powerx-dense-ladder.yaml"
B200_SCRIPT = (
    REPO_ROOT / "benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_b200.sh"
)
B300_SCRIPT = (
    REPO_ROOT / "benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_b300.sh"
)
B300_RUNNER = REPO_ROOT / "runners/launch_b300-nv.sh"


def _continued_command(script: str, marker: str) -> str:
    """Return one shell command, joining its backslash-continuation lines."""

    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    command_lines = []
    for line in lines[start:]:
        stripped = line.rstrip()
        command_lines.append(stripped[:-1] if stripped.endswith("\\") else stripped)
        if not stripped.endswith("\\"):
            break
    return " ".join(command_lines)


def _shell_options(command: str) -> dict[str, tuple[str, ...]]:
    """Parse long shell options while preserving command-substitution tokens."""

    tokens = shlex.split(command)
    options: dict[str, tuple[str, ...]] = {}
    index = next(i for i, token in enumerate(tokens) if token.startswith("--"))
    while index < len(tokens):
        token = tokens[index]
        if token in {">", "&"}:
            break
        assert token.startswith("--"), (command, token)
        if "=" in token:
            name, value = token.split("=", 1)
            values = [value]
        else:
            name = token
            values = []
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            if tokens[index] in {">", "&"}:
                index = len(tokens)
                break
            values.append(tokens[index])
            index += 1
        options[name] = tuple(values)
    return options


def _launch_options(path: Path) -> dict[str, tuple[str, ...]]:
    command = _continued_command(path.read_text(), "python3 -m sglang.launch_server")
    return _shell_options(command)


def _request_options(path: Path) -> dict[str, tuple[str, ...]]:
    command = _continued_command(path.read_text(), "run_benchmark_serving")
    return _shell_options(command)


def test_campaign_b300_lane_matches_frozen_b200_workload_contract():
    config = yaml.safe_load(CAMPAIGN_CONFIG.read_text())
    b200 = config["qwen3.5-fp8-b200-sglang"]
    b300 = config["qwen3.5-fp8-b300-sglang"]

    for field in ("image", "model", "model-prefix", "precision", "framework", "multinode"):
        assert b300[field] == b200[field], field
    assert b300["scenarios"] == b200["scenarios"]
    assert b300["model"].split("/")[-1] == "Qwen3.5-397B-A17B-FP8"


def test_b300_request_options_match_b200_exactly():
    assert _request_options(B300_SCRIPT) == _request_options(B200_SCRIPT)


def test_b300_server_options_match_b200_except_local_path_plumbing():
    b200 = _launch_options(B200_SCRIPT)
    b300 = _launch_options(B300_SCRIPT)

    assert b200.pop("--model-path") == ("$MODEL",)
    assert "--served-model-name" not in b200
    assert "--tokenizer-path" not in b200

    assert b300.pop("--model-path") == ("$MODEL_PATH",)
    assert b300.pop("--served-model-name") == ("$MODEL",)
    assert b300.pop("--tokenizer-path") == ("$MODEL_PATH",)

    assert b300 == b200


def test_b300_runner_derives_local_checkpoint_from_campaign_model_basename():
    runner = B300_RUNNER.read_text()

    assert 'MODEL_BASENAME="${MODEL##*/}"' in runner
    assert "        Qwen3.5-397B-A17B-FP8" in runner
    assert (
        'export MODEL_PATH="${HF_HUB_CACHE_MOUNT%/}/${MODEL_BASENAME}"' in runner
    )
    assert 'MODEL_PATH="/scratch/models/Qwen3.5-397B-A17B-FP8"' not in runner
