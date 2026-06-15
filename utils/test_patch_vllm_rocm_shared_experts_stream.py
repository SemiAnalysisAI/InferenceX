from pathlib import Path

import pytest

import utils.patch_vllm_rocm_shared_experts_stream as stream_patch


REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_SOURCE = """def select(hidden_states):
        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0] <= threshold
        )
"""


def test_patch_source_enables_rocm_shared_expert_stream():
    patched = stream_patch.patch_source(SAMPLE_SOURCE)

    assert "current_platform.is_cuda_alike()" in patched
    assert "current_platform.is_cuda()" not in patched
    assert stream_patch.patch_source(patched) == patched


def test_apply_runtime_patch_is_fingerprinted_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / stream_patch.RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE_SOURCE, encoding="utf-8")
    patched = stream_patch.patch_source(SAMPLE_SOURCE)

    monkeypatch.setattr(
        stream_patch,
        "EXPECTED_SOURCE_SHA256",
        stream_patch._sha256(SAMPLE_SOURCE),
    )
    monkeypatch.setattr(
        stream_patch,
        "EXPECTED_PATCHED_SHA256",
        stream_patch._sha256(patched),
    )

    stream_patch.apply_runtime_patch(tmp_path)
    stream_patch.apply_runtime_patch(tmp_path, check_only=True)
    assert path.read_text(encoding="utf-8") == patched


def test_apply_runtime_patch_rejects_source_drift(tmp_path, monkeypatch):
    path = tmp_path / stream_patch.RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE_SOURCE + "# drift\n", encoding="utf-8")
    monkeypatch.setattr(
        stream_patch,
        "EXPECTED_SOURCE_SHA256",
        stream_patch._sha256(SAMPLE_SOURCE),
    )

    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        stream_patch.apply_runtime_patch(tmp_path, check_only=True)


def test_mi300x_workflows_wire_shared_expert_stream_mode():
    profile = (REPO_ROOT / ".github/workflows/profile.yml").read_text(
        encoding="utf-8"
    )
    e2e = (REPO_ROOT / ".github/workflows/e2e-tests.yml").read_text(
        encoding="utf-8"
    )
    benchmark = (REPO_ROOT / ".github/workflows/benchmark-tmpl.yml").read_text(
        encoding="utf-8"
    )
    recipe = (
        REPO_ROOT
        / "benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x.sh"
    ).read_text(encoding="utf-8")

    for source in (profile, e2e, benchmark):
        assert "m3-shared-expert-stream-mode" in source
    assert "M3_SHARED_EXPERT_STREAM_MODE" in recipe
    assert "patch_vllm_rocm_shared_experts_stream.py" in recipe
