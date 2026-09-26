"""Unit tests for infx.runners.srt_launch helper functions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------


class TestImageToSquashKey:
    """Verify _image_to_squash_key produces the same result as the sed expression."""

    def test_docker_hub_image(self):
        from infx.runners.srt_launch import _image_to_squash_key

        result = _image_to_squash_key("vllm/vllm-openai:v0.8.5")
        assert result == "vllm_vllm-openai_v0.8.5"

    def test_nvcr_image(self):
        from infx.runners.srt_launch import _image_to_squash_key

        result = _image_to_squash_key("nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3")
        assert result == "nvcr.io_nvidia_tritonserver_24.08-trtllm-python-py3"

    def test_image_with_hash(self):
        from infx.runners.srt_launch import _image_to_squash_key

        result = _image_to_squash_key("repo/image@sha256:abc123")
        assert result == "repo_image_sha256_abc123"

    def test_colons_and_slashes(self):
        from infx.runners.srt_launch import _image_to_squash_key

        result = _image_to_squash_key("a/b:c@d#e")
        assert result == "a_b_c_d_e"


class TestResolveH100SrtContainer:
    """Verify container resolution for different frameworks."""

    def test_dynamo_sglang(self):
        from infx.runners.srt_launch import resolve_h100_srt_container

        squash, key = resolve_h100_srt_container("nvcr.io/test:v1", "dynamo-sglang")
        assert squash == "/mnt/nfs/lustre/containers/nvcr.io_test_v1.sqsh"
        assert key == "nvcr.io#test:v1"

    def test_dynamo_trt(self):
        from infx.runners.srt_launch import resolve_h100_srt_container

        squash, key = resolve_h100_srt_container("nvcr.io/nvidia/test:v1", "dynamo-trt")
        assert squash == "/mnt/nfs/sa-shared/containers/nvidia+test+v1.sqsh"
        assert key == "nvcr.io#nvidia/test:v1"

    def test_invalid_framework_raises(self):
        from infx.runners.srt_launch import resolve_h100_srt_container

        with pytest.raises(ValueError, match="Unsupported framework"):
            resolve_h100_srt_container("test:v1", "unsupported")

    def test_empty_image_raises(self):
        from infx.runners.srt_launch import resolve_h100_srt_container

        with pytest.raises(ValueError, match="Invalid image"):
            resolve_h100_srt_container("", "dynamo-sglang")

    def test_whitespace_image_raises(self):
        from infx.runners.srt_launch import resolve_h100_srt_container

        with pytest.raises(ValueError, match="Invalid image"):
            resolve_h100_srt_container("has space", "dynamo-sglang")


class TestBuildEvalPassthrough:
    """Verify eval passthrough JSON construction."""

    def test_includes_standard_names(self, monkeypatch):
        monkeypatch.setenv("INFERENCEX_RUNTIME_ENV_VARS", "EXTRA_VAR1 EXTRA_VAR2")
        import json

        from infx.runners.srt_launch import _build_eval_passthrough

        result = json.loads(_build_eval_passthrough())
        assert "EVAL_FRAMEWORK" in result
        assert "CONC" in result
        assert "EXTRA_VAR1" in result
        assert "EXTRA_VAR2" in result

    def test_empty_runtime_vars(self, monkeypatch):
        monkeypatch.setenv("INFERENCEX_RUNTIME_ENV_VARS", "")
        import json

        from infx.runners.srt_launch import _build_eval_passthrough

        result = json.loads(_build_eval_passthrough())
        # Should have the 17 standard names but no extras (empty string split -> [""]).
        assert "EVAL_FRAMEWORK" in result


class TestCopyToWorkspace:
    """Verify copy_to_workspace edge cases."""

    def test_copies_file(self, tmp_path):
        from infx.runners.srt_launch import copy_to_workspace

        src = tmp_path / "source.json"
        src.write_text('{"data": 1}')
        dst = tmp_path / "dest.json"
        copy_to_workspace(str(src), str(dst))
        assert dst.read_text() == '{"data": 1}'

    def test_same_file_skips(self, tmp_path):
        from infx.runners.srt_launch import copy_to_workspace

        src = tmp_path / "same.json"
        src.write_text('{"data": 1}')
        # Same path => same inode.
        copy_to_workspace(str(src), str(src))
        assert src.read_text() == '{"data": 1}'


class TestBundleServerLogs:
    """Verify bundle_server_logs creates an archive."""

    def test_creates_archive(self, tmp_path):
        from infx.runners.srt_launch import bundle_server_logs

        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "output.log").write_text("hello")
        archive = tmp_path / "logs.tar.gz"
        bundle_server_logs(str(logs), str(archive))
        assert archive.exists()
        assert archive.stat().st_size > 0

    def test_empty_dir_no_archive(self, tmp_path):
        from infx.runners.srt_launch import bundle_server_logs

        logs = tmp_path / "empty"
        logs.mkdir()
        archive = tmp_path / "logs.tar.gz"
        bundle_server_logs(str(logs), str(archive))
        assert not archive.exists()

    def test_missing_dir_no_error(self, tmp_path):
        from infx.runners.srt_launch import bundle_server_logs

        archive = tmp_path / "logs.tar.gz"
        bundle_server_logs("/nonexistent/path", str(archive))
        assert not archive.exists()


class TestEnvHelper:
    """Verify _env raises on missing/empty vars."""

    def test_missing_var_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        from infx.runners.srt_launch import _env

        with pytest.raises(EnvironmentError, match="NONEXISTENT_VAR_12345"):
            _env("NONEXISTENT_VAR_12345")

    def test_empty_var_raises(self, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR_TEST", "")
        from infx.runners.srt_launch import _env

        with pytest.raises(EnvironmentError, match="EMPTY_VAR_TEST"):
            _env("EMPTY_VAR_TEST")

    def test_present_var_returns(self, monkeypatch):
        monkeypatch.setenv("PRESENT_VAR_TEST", "hello")
        from infx.runners.srt_launch import _env

        assert _env("PRESENT_VAR_TEST") == "hello"


class TestCheckStagedSrtAssets:
    """Verify check_staged_srt_assets validates model and container."""

    def test_missing_config_raises(self, tmp_path):
        from infx.runners.srt_launch import check_staged_srt_assets

        model = tmp_path / "model"
        model.mkdir()
        # No config.json inside model dir.
        with pytest.raises(RuntimeError, match="readiness-blocked"):
            check_staged_srt_assets(str(model), "/nonexistent.sqsh")

    def test_valid_model_invalid_squash(self, tmp_path):
        from infx.runners.srt_launch import check_staged_srt_assets

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")

        with (
            patch("subprocess.run") as mock_run,
            pytest.raises(RuntimeError, match="readiness-blocked"),
        ):
            mock_run.return_value = MagicMock(returncode=1)
            check_staged_srt_assets(str(model), "/nonexistent.sqsh")


class _StopAfterPrepare(Exception):
    pass


def test_srt_commands_run_in_the_srtctl_venv(tmp_path, monkeypatch):
    """Like slurm_utils.sh, infx.srt_slurm commands run under the activated srtctl venv.

    The runner's system python3 has no PyYAML; the first CI run failed with
    ModuleNotFoundError when prepare ran under sys.executable.
    """
    from infx.runners import srt_launch

    env = {
        "GITHUB_WORKSPACE": str(tmp_path), "SRT_RECIPE": "recipe.yaml", "FRAMEWORK": "sglang",
        "MODEL": "m", "MODEL_PREFIX": "p", "IMAGE": "img:1", "PRECISION": "fp8", "TP": "8",
        "PP_SIZE": "1", "DCP_SIZE": "1", "PCP_SIZE": "1", "EP_SIZE": "1", "DP_ATTENTION": "false",
        "GPU_COUNT": "8", "IS_AGENTIC": "1", "SPEC_DECODING": "none", "CONC": "1", "ISL": "1",
        "OSL": "1", "RANDOM_RANGE_RATIO": "1", "RESULT_FILENAME": "r", "GPU_MONITOR_INTERVAL": "1",
        "SRT_MODEL_PATH": "hf:m", "HF_HUB_CACHE_MOUNT": "/h", "HF_HUB_CACHE": "/c",
        "SALLOC_TIME_LIMIT": "480", "PATH": "/usr/bin:/bin",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    def fake_setup(root, *_args, **_kwargs):
        Path(root).mkdir(parents=True)
        return "[]"

    calls = []

    def fake_run(argv, *_args, **_kwargs):
        calls.append((list(argv), os.environ["PATH"], os.getcwd()))
        if argv[:3] == ["python3", "-m", "infx.srt_slurm.single_node"]:
            raise _StopAfterPrepare
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(srt_launch, "setup_srt_slurm", fake_setup)
    monkeypatch.setattr(srt_launch, "_ensure_uv", lambda: None)
    monkeypatch.setattr(srt_launch.subprocess, "run", fake_run)

    with pytest.raises(_StopAfterPrepare):
        srt_launch.launch_srt_single_node("h100-dgxc-slurm")

    argv, path, cwd = calls[-1]
    assert argv[:4] == ["python3", "-m", "infx.srt_slurm.single_node", "prepare"]
    assert path.split(":")[0].endswith("/checkout/.venv/bin")
    # srtctl finds srtslurm.yaml in the working directory, as after the bash `cd`.
    assert cwd.endswith("/checkout")
