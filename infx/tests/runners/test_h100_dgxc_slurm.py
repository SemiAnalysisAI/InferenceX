"""Unit tests for infx.runners.clusters.h100_dgxc_slurm execution-path dispatch."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_pyslurm():
    """Ensure fake pyslurm is loadable."""
    if "pyslurm" not in sys.modules:
        fake = types.ModuleType("pyslurm")
        fake.RPCError = type("RPCError", (Exception,), {})
        fake.Job = MagicMock()
        fake.JobSubmitDescription = MagicMock()
        db = types.ModuleType("pyslurm.db")
        db.Job = MagicMock()
        fake.db = db
        sys.modules["pyslurm"] = fake
        sys.modules["pyslurm.db"] = db


_install_fake_pyslurm()

# Patch SlurmClient at module level so imports don't fail on macOS.
import infx.runners.slurm as slurm_mod

slurm_mod._pyslurm = sys.modules["pyslurm"]
slurm_mod._pyslurm_available = True

from infx.runners.slurm import SlurmClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _base_env(monkeypatch):
    """Set the minimum environment for the launcher's initial check_env_vars."""
    required = {
        "EVAL_ONLY": "false",
        "IS_MULTINODE": "false",
        "RUN_EVAL": "false",
        "SALLOC_TIME_LIMIT": "180",
        "IS_AGENTIC": "0",
        "SRT_RECIPE": "recipes/test.yaml",
        "GITHUB_WORKSPACE": "/workspace",
        "RUNNER_NAME": "h100-dgxc-slurm_00",
        "MODEL": "test-model",
        "MODEL_PREFIX": "dsr1",
        "IMAGE": "nvcr.io/test:latest",
        "PRECISION": "fp8",
        "TP": "8",
        "PP_SIZE": "1",
        "DCP_SIZE": "1",
        "PCP_SIZE": "1",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
        "GPU_COUNT": "8",
        "SPEC_DECODING": "none",
        "CONC": "32",
        "ISL": "8192",
        "OSL": "1024",
        "RANDOM_RANGE_RATIO": "0.5",
        "RESULT_FILENAME": "test_result",
        "GPU_MONITOR_INTERVAL": "10",
        "HF_HUB_CACHE": "/hf_cache",
        "FRAMEWORK": "sglang",
        "INFERENCEX_RUNTIME_ENV_VARS": "",
        "SRT_MODEL_PATH": "hf:test-model",
        "HF_HUB_CACHE_MOUNT": "/mnt/hf",
    }
    for key, val in required.items():
        monkeypatch.setenv(key, val)


# ---------------------------------------------------------------------------
# Execution path routing
# ---------------------------------------------------------------------------


class TestExecutionPathRouting:
    """Verify launch() selects the correct execution path."""

    def test_native_single_node_when_srt_recipe(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_AGENTIC", "0")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.setenv("SRT_RECIPE", "recipes/test.yaml")

        from infx.runners.clusters.h100_dgxc_slurm import launch

        with patch(
            "infx.runners.clusters.h100_dgxc_slurm._launch_native_single_node",
        ) as mock:
            mock.return_value = 0
            rc = launch()
            mock.assert_called_once()
            assert rc == 0

    def test_multinode_when_is_multinode(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_MULTINODE", "true")

        from infx.runners.clusters.h100_dgxc_slurm import launch

        with patch(
            "infx.runners.clusters.h100_dgxc_slurm._launch_multinode",
        ) as mock:
            mock.return_value = 0
            rc = launch()
            mock.assert_called_once()
            assert rc == 0

    def test_agentic_when_is_agentic_1(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_AGENTIC", "1")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.delenv("SRT_RECIPE", raising=False)

        from infx.runners.clusters.h100_dgxc_slurm import launch

        with patch(
            "infx.runners.clusters.h100_dgxc_slurm._launch_agentic",
        ) as mock:
            mock.return_value = 0
            rc = launch()
            mock.assert_called_once()
            assert rc == 0


# ---------------------------------------------------------------------------
# Agentic path — srun argv construction
# ---------------------------------------------------------------------------


class TestAgenticSrunArgv:
    """Verify the agentic path builds srun commands with identical Pyxis flags to the bash."""

    def test_srun_container_step_flags(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_AGENTIC", "1")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.setenv("GPU_COUNT", "8")
        monkeypatch.setenv("EXP_NAME", "test_exp")
        monkeypatch.setenv("SCENARIO_SUBDIR", "")
        monkeypatch.delenv("SRT_RECIPE", raising=False)

        recorded_calls = []

        def fake_submit_holder(**kwargs):
            return 99999

        def fake_wait_for_running(job_id, **kwargs):
            pass

        def fake_run_step(job_id, argv, **kwargs):
            recorded_calls.append((job_id, argv, kwargs))
            return MagicMock(returncode=0)

        def fake_cancel(job_id):
            pass

        from infx.runners.clusters import h100_dgxc_slurm as mod

        with (
            patch.object(SlurmClient, "submit_holder_job", side_effect=fake_submit_holder),
            patch.object(SlurmClient, "wait_for_running", side_effect=fake_wait_for_running),
            patch.object(SlurmClient, "run_step", side_effect=fake_run_step),
            patch.object(SlurmClient, "cancel", side_effect=fake_cancel),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            mod._launch_agentic()

        # Should have two srun calls: squash import + benchmark.
        assert len(recorded_calls) == 2

        # Check the benchmark step (second call).
        _, _, kwargs = recorded_calls[1]
        image = os.environ["IMAGE"]
        squash = f"/mnt/nfs/lustre/containers/{_image_to_squash_key(image)}.sqsh"
        assert kwargs["container"] == squash
        assert "--no-container-mount-home" not in str(kwargs) or kwargs.get(
            "no_container_mount_home"
        )
        assert kwargs["no_container_entrypoint"] is True
        assert "PORT=8888" in kwargs["export"]
        assert "AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache" in kwargs["export"]

    def test_dsv41flash_uses_ix_mount(self, _base_env, monkeypatch):
        """DeepSeek-V4.1-Flash must use /ix instead of /workspace."""
        monkeypatch.setenv("IS_AGENTIC", "1")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.setenv("MODEL_PREFIX", "dsv41flash")
        monkeypatch.setenv("GPU_COUNT", "8")
        monkeypatch.setenv("EXP_NAME", "test_exp")
        monkeypatch.setenv("SCENARIO_SUBDIR", "")
        monkeypatch.delenv("SRT_RECIPE", raising=False)

        recorded_calls = []

        def fake_run_step(job_id, argv, **kwargs):
            recorded_calls.append((job_id, argv, kwargs))
            return MagicMock(returncode=0)

        from infx.runners.clusters import h100_dgxc_slurm as mod

        with (
            patch.object(SlurmClient, "submit_holder_job", return_value=99999),
            patch.object(SlurmClient, "wait_for_running"),
            patch.object(SlurmClient, "run_step", side_effect=fake_run_step),
            patch.object(SlurmClient, "cancel"),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            mod._launch_agentic()

        # Benchmark step should mount at /ix, not /workspace.
        _, _, kwargs = recorded_calls[1]
        assert kwargs["container_workdir"] == "/ix/"
        assert "/ix/" in kwargs["container_mounts"]


# ---------------------------------------------------------------------------
# Squash import locking
# ---------------------------------------------------------------------------


class TestSquashImportLocking:
    """Verify the squash import step uses flock serialization."""

    def test_squash_import_uses_flock(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_AGENTIC", "1")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.setenv("GPU_COUNT", "8")
        monkeypatch.setenv("EXP_NAME", "test_exp")
        monkeypatch.setenv("SCENARIO_SUBDIR", "")
        monkeypatch.delenv("SRT_RECIPE", raising=False)

        import_scripts = []

        def capture_run_step(job_id, argv, **kwargs):
            import_scripts.append(argv)
            return MagicMock(returncode=0)

        from infx.runners.clusters import h100_dgxc_slurm as mod

        with (
            patch.object(SlurmClient, "submit_holder_job", return_value=99999),
            patch.object(SlurmClient, "wait_for_running"),
            patch.object(SlurmClient, "run_step", side_effect=capture_run_step),
            patch.object(SlurmClient, "cancel"),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            mod._launch_agentic()

        # First srun step should be the squash import with flock.
        assert len(import_scripts) >= 1
        import_argv = import_scripts[0]
        # It's ["bash", "-c", <script>].
        assert import_argv[0] == "bash"
        assert import_argv[1] == "-c"
        script = import_argv[2]
        assert "flock" in script
        assert "unsquashfs" in script
        assert "enroot import" in script


# ---------------------------------------------------------------------------
# Cancellation on error
# ---------------------------------------------------------------------------


class TestCancellationOnError:
    """Verify the holder job is cancelled when an error occurs."""

    def test_cancel_on_srun_failure(self, _base_env, monkeypatch):
        monkeypatch.setenv("IS_AGENTIC", "1")
        monkeypatch.setenv("IS_MULTINODE", "false")
        monkeypatch.setenv("GPU_COUNT", "8")
        monkeypatch.setenv("EXP_NAME", "test_exp")
        monkeypatch.setenv("SCENARIO_SUBDIR", "")
        monkeypatch.delenv("SRT_RECIPE", raising=False)

        cancelled_ids = []

        def fail_run_step(job_id, argv, **kwargs):
            raise subprocess.CalledProcessError(1, "srun")

        def track_cancel(job_id):
            cancelled_ids.append(job_id)

        import subprocess

        from infx.runners.clusters import h100_dgxc_slurm as mod

        with (
            patch.object(SlurmClient, "submit_holder_job", return_value=99999),
            patch.object(SlurmClient, "wait_for_running"),
            patch.object(SlurmClient, "run_step", side_effect=fail_run_step),
            patch.object(SlurmClient, "cancel", side_effect=track_cancel),
            pytest.raises(subprocess.CalledProcessError),
        ):
            mod._launch_agentic()

        assert 99999 in cancelled_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestMainEntry:
    """Verify ``python3 -m infx.runners`` dispatch."""

    def test_unknown_cluster_exits_1(self):
        from infx.runners.__main__ import main

        with (
            patch("sys.argv", ["infx.runners", "nonexistent-cluster"]),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_known_cluster_calls_launch(self):
        from infx.runners.__main__ import main

        with (
            patch("sys.argv", ["infx.runners", "h100-dgxc-slurm"]),
            patch(
                "infx.runners.clusters.h100_dgxc_slurm.launch",
                return_value=0,
            ) as mock_launch,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_launch.assert_called_once()


def _image_to_squash_key(image: str) -> str:
    out = image
    for ch in "/\\:@#":
        out = out.replace(ch, "_")
    return out
