from pathlib import Path

import pytest

import utils.patch_vllm_mi300x_rank0_profiler as profiler_patch


REPO_ROOT = Path(__file__).resolve().parents[1]


SAMPLE_SOURCE = '''class GPUWorker:
    def profile(self):
        self.profiler = TorchProfilerWrapper(
            self.profiler_config,
            worker_name="worker",
            local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
        )
'''


def test_patch_source_limits_gpu_tracing_to_rank_zero():
    patched = profiler_patch.patch_source(SAMPLE_SOURCE)

    assert profiler_patch.PATCH_MARKER in patched
    assert '["CPU", "CUDA"]' in patched
    assert "if self.local_rank == 0" in patched
    assert 'else ["CPU"]' in patched
    assert profiler_patch.patch_source(patched) == patched


def test_apply_runtime_patch_is_fingerprinted_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / profiler_patch.RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE_SOURCE, encoding="utf-8")
    patched = profiler_patch.patch_source(SAMPLE_SOURCE)

    monkeypatch.setattr(
        profiler_patch,
        "EXPECTED_SOURCE_SHA256",
        profiler_patch._sha256(SAMPLE_SOURCE),
    )
    monkeypatch.setattr(
        profiler_patch,
        "EXPECTED_PATCHED_SHA256",
        profiler_patch._sha256(patched),
    )

    profiler_patch.apply_runtime_patch(tmp_path)
    profiler_patch.apply_runtime_patch(tmp_path, check_only=True)
    assert path.read_text(encoding="utf-8") == patched


def test_apply_runtime_patch_rejects_source_drift(tmp_path, monkeypatch):
    path = tmp_path / profiler_patch.RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE_SOURCE + "# drift\n", encoding="utf-8")
    monkeypatch.setattr(
        profiler_patch,
        "EXPECTED_SOURCE_SHA256",
        profiler_patch._sha256(SAMPLE_SOURCE),
    )

    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        profiler_patch.apply_runtime_patch(tmp_path, check_only=True)


def test_mi300x_profile_scratch_does_not_create_workspace_directories():
    workflow = (REPO_ROOT / ".github/workflows/profile.yml").read_text(
        encoding="utf-8"
    )
    recipe = (
        REPO_ROOT
        / "benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x.sh"
    ).read_text(encoding="utf-8")
    benchmark_lib = (REPO_ROOT / "benchmarks/benchmark_lib.sh").read_text(
        encoding="utf-8"
    )

    assert "/workspace/profile_traces" not in workflow
    assert "/workspace/profile_traces" not in recipe
    assert "/tmp/inferencex-profile/${res_name}" in workflow
    assert "/tmp/inferencex-profile/${RESULT_FILENAME}" in recipe
    assert "_windowed" in workflow
    assert 'profile_active_iterations=5' in recipe
    assert (
        "profile_output_len=$((profile_delay + profile_active_iterations + "
        "profile_tail_margin))"
        in recipe
    )
    assert 'benchmark_random_range_ratio="1.0"' in recipe
    assert '--output-len "$benchmark_output_len"' in recipe
    assert 'benchmark_num_warmups="$CONC"' in recipe
    assert '--num-warmups "$benchmark_num_warmups"' in recipe
    assert '--num-warmups "$num_warmups"' in benchmark_lib
