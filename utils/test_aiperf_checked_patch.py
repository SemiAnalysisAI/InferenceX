"""Exercise exact-source patch admission without a network or GPU dependency."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent / "aiperf-patches/apply_checked_patch.sh"


@pytest.fixture
def patch_case(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    original = {"one.py": "old one\n", "two.py": "old two\n"}
    candidate = {"one.py": "new one\n", "two.py": "new two\n"}
    for name, contents in original.items():
        (source / name).write_text(contents)
    for label, content_map in [("before", original), ("after", candidate)]:
        (tmp_path / label).write_text(
            "".join(
                f"{hashlib.sha256(content.encode()).hexdigest()}  {name}\n"
                for name, content in content_map.items()
            )
        )
    patch = tmp_path / "change.patch"
    patch.write_text(
        "".join(
            f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-{original[name]}+{candidate[name]}"
            for name in original
        )
    )
    command = [
        "bash",
        str(HELPER),
        str(source),
        str(patch),
        str(tmp_path / "before"),
        str(tmp_path / "after"),
    ]
    return source, patch, original, candidate, command


def invoke(command):
    return subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=10
    )


def test_apply_and_repeat_are_exact_and_idempotent(patch_case):
    source, _, _, candidate, command = patch_case
    for _ in range(2):
        result = invoke(command)
        assert result.returncode == 0, result.stderr
        assert {name: (source / name).read_text() for name in candidate} == candidate


@pytest.mark.parametrize("replacement", ["unrelated edit\n", "new one\n"])
def test_source_drift_or_partial_patch_rejects_without_more_mutation(
    patch_case, replacement
):
    source, _, original, _, command = patch_case
    (source / "one.py").write_text(replacement)
    result = invoke(command)
    assert result.returncode != 0
    assert (source / "one.py").read_text() == replacement
    assert (source / "two.py").read_text() == original["two.py"]


def test_failed_dry_run_does_not_partially_apply(patch_case):
    source, patch, original, _, command = patch_case
    patch.write_text(patch.read_text().replace("-old two", "-wrong two"))
    assert invoke(command).returncode != 0
    assert {name: (source / name).read_text() for name in original} == original


def test_wrong_candidate_hash_is_not_accepted(patch_case):
    _, _, _, _, command = patch_case
    Path(command[-1]).write_text(f"{'0' * 64}  one.py\n")
    assert invoke(command).returncode != 0


def test_concurrent_installers_serialize(patch_case):
    source, _, _, candidate, command = patch_case
    children = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for _ in range(2)
    ]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, (stdout, stderr)
        assert {name: (source / name).read_text() for name in candidate} == candidate
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()


@pytest.mark.parametrize(
    "enabled,prior_patch_rc,drift,uv_rc,expected",
    [
        ("0", 0, False, 0, "0,1"),
        ("1", 0, False, 0, "0,1"),
        ("1", 19, False, 0, "19,0"),
        ("1", 0, True, 0, "1,0"),
        ("invalid", 0, False, 0, "1,0"),
        ("1", 0, False, 23, "23,0"),
    ],
)
def test_dependency_setup_only_marks_ready_after_checked_patch(
    tmp_path, patch_case, enabled, prior_patch_rc, drift, uv_rc, expected
):
    source, patch, original, candidate, _ = patch_case
    repo = HELPER.parents[2]
    body = (repo / "benchmarks/benchmark_lib.sh").read_text()
    body = (
        "install_agentic_deps() {"
        + body.split("install_agentic_deps() {", 1)[1].split("\n}\n", 1)[0]
        + "\n}\n"
    )
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    script = benchmarks / "install.sh"
    script.write_text(body)
    delivery = tmp_path / "utils/aiperf-patches"
    delivery.mkdir(parents=True)
    for src, name in [
        (HELPER, "apply_checked_patch.sh"),
        (patch, "cancel-wire-drain.patch"),
        (tmp_path / "before", "cancel-wire-drain.before.sha256"),
        (tmp_path / "after", "cancel-wire-drain.after.sha256"),
    ]:
        shutil.copyfile(src, delivery / name)
    if drift:
        (source / "one.py").write_text("unrelated edit\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
check_env_vars() { :; }
test_uv() { return "$UV_RC"; }
ensure_agentic_uv() { AIPERF_UV_BIN=test_uv; }
_patch_aiperf_dataset_config_race() { return "$PRIOR_PATCH_RC"; }
AIPERF_DEPS_READY=0
AIPERF_VENV="$TEST_ROOT/dependency-venv"
AIPERF_RUNTIME_DIR="$TEST_ROOT/runtime"
AIPERF_UV_CACHE_DIR="$TEST_ROOT/cache"
AIPERF_CLI=$(type -P true)
AIPERF_HF_CLI="$AIPERF_CLI"
install_agentic_deps
rc=$?
printf 'RESULT=%s,%s\n' "$rc" "$AIPERF_DEPS_READY"
""",
            "test",
            str(script),
        ],
        env={
            **os.environ,
            "TEST_ROOT": str(tmp_path),
            "AIPERF_DIR": str(source),
            "PRIOR_PATCH_RC": str(prior_patch_rc),
            "UV_RC": str(uv_rc),
            "AIPERF_CANCEL_WIRE_DRAIN_FIX": enabled,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "environment is incomplete" not in result.stderr
    assert f"RESULT={expected}" in result.stdout
    if not drift:
        # The root Docker adapter must never patch its shared checkout.
        assert {name: (source / name).read_text() for name in original} == original
        if enabled == "1" and prior_patch_rc == 0 and uv_rc == 0:
            copies = list((tmp_path / "runtime").glob("aiperf-source.*"))
            assert len(copies) == 1
            assert {
                name: (copies[0] / name).read_text() for name in candidate
            } == candidate
