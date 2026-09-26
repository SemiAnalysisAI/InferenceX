"""Unit tests for infx.runners.slurm with a fake pyslurm module."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake pyslurm — injected before importing the module under test
# ---------------------------------------------------------------------------


def _make_fake_pyslurm():
    """Build a mock pyslurm module with Job, JobSubmitDescription, db.Job."""
    pyslurm = types.ModuleType("pyslurm")
    pyslurm.RPCError = type("RPCError", (Exception,), {})

    # Core Job
    class FakeJob:
        def __init__(self, job_id=0):
            self._id = job_id
            self._state = "PENDING"
            self._exit_code = None
            self._exit_code_signal = None

        @property
        def id(self):
            return self._id

        @property
        def state(self):
            return self._state

        @property
        def exit_code(self):
            return self._exit_code

        @property
        def exit_code_signal(self):
            return self._exit_code_signal

        @staticmethod
        def load(job_id):
            job = FakeJob(job_id)
            # State is controlled per-test via the class variable.
            job._state = FakeJob._loaded_state
            job._exit_code = FakeJob._loaded_exit_code
            job._exit_code_signal = FakeJob._loaded_exit_code_signal
            return job

        def cancel(self):
            FakeJob._cancelled_ids.append(self._id)

        _loaded_state = "RUNNING"
        _loaded_exit_code = 0
        _loaded_exit_code_signal = 0
        _cancelled_ids: list[int] = []

    pyslurm.Job = FakeJob

    # JobSubmitDescription
    class FakeJobSubmitDescription:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._submitted_id = 12345

        def submit(self):
            FakeJobSubmitDescription._last_submission = self.kwargs
            return self._submitted_id

        _last_submission: dict | None = None

    pyslurm.JobSubmitDescription = FakeJobSubmitDescription

    # db module
    db_module = types.ModuleType("pyslurm.db")

    class FakeDbJob:
        def __init__(self, job_id=0, **kwargs):
            self._id = job_id
            self._state = "COMPLETED"
            self._exit_code = 0
            self._exit_code_signal = 0

        @property
        def state(self):
            return self._state

        @property
        def exit_code(self):
            return self._exit_code

        @property
        def exit_code_signal(self):
            return self._exit_code_signal

        @staticmethod
        def load(job_id, **kwargs):
            job = FakeDbJob(job_id)
            job._state = FakeDbJob._loaded_state
            job._exit_code = FakeDbJob._loaded_exit_code
            job._exit_code_signal = FakeDbJob._loaded_exit_code_signal
            return job

        _loaded_state = "COMPLETED"
        _loaded_exit_code = 0
        _loaded_exit_code_signal = 0

    db_module.Job = FakeDbJob
    pyslurm.db = db_module
    sys.modules["pyslurm.db"] = db_module

    return pyslurm, FakeJob, FakeJobSubmitDescription, FakeDbJob


@pytest.fixture(autouse=True)
def _inject_fake_pyslurm():
    """Install fake pyslurm before each test, remove it after."""
    fake_pyslurm, fake_job, fake_jsd, fake_db_job = _make_fake_pyslurm()

    # Reset class-level state.
    fake_job._loaded_state = "RUNNING"
    fake_job._loaded_exit_code = 0
    fake_job._loaded_exit_code_signal = 0
    fake_job._cancelled_ids = []
    fake_jsd._last_submission = None
    fake_db_job._loaded_state = "COMPLETED"
    fake_db_job._loaded_exit_code = 0
    fake_db_job._loaded_exit_code_signal = 0

    old = sys.modules.get("pyslurm")
    sys.modules["pyslurm"] = fake_pyslurm

    # Force re-import of the module under test to pick up the fake.
    import infx.runners.slurm as slurm_mod

    slurm_mod._pyslurm = fake_pyslurm
    slurm_mod._pyslurm_available = True

    yield fake_pyslurm, fake_job, fake_jsd, fake_db_job

    # Restore.
    if old is not None:
        sys.modules["pyslurm"] = old
    else:
        sys.modules.pop("pyslurm", None)
    sys.modules.pop("pyslurm.db", None)
    slurm_mod._pyslurm = None
    slurm_mod._pyslurm_available = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubmitHolderJob:
    def test_submits_with_correct_kwargs(self, _inject_fake_pyslurm):
        _, _, fake_jsd, _ = _inject_fake_pyslurm
        from infx.runners.slurm import SlurmClient

        job_id = SlurmClient.submit_holder_job(
            partition="hpc-gpu-1",
            account="customer",
            gres="gpu:8",
            time_limit="180",
            job_name="test-runner",
            exclusive=True,
        )
        assert job_id == 12345
        submission = fake_jsd._last_submission
        assert submission is not None
        assert submission["name"] == "test-runner"
        assert submission["partitions"] == "hpc-gpu-1"
        assert submission["account"] == "customer"
        assert submission["gres_per_node"] == "gpu:8"
        assert submission["time_limit"] == 180  # bare minutes become an int for pyslurm
        assert submission["resource_sharing"] == "no"
        assert "sleep infinity" in submission["script"]
        # Compute pods may not see the submit directory; the holder writes nowhere.
        assert submission["standard_output"] == "/dev/null"
        assert submission["standard_error"] == "/dev/null"
        assert submission["working_directory"] == "/"

    def test_clock_time_limit_passes_through(self, _inject_fake_pyslurm):
        _, _, fake_jsd, _ = _inject_fake_pyslurm
        from infx.runners.slurm import SlurmClient

        SlurmClient.submit_holder_job(
            partition="p", account="a", gres="gpu:8", time_limit="08:00:00", job_name="n",
        )
        assert fake_jsd._last_submission["time_limit"] == "08:00:00"


class TestJobState:
    def test_returns_running(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "RUNNING"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_state(100) == "RUNNING"

    def test_returns_completed(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "COMPLETED"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_state(100) == "COMPLETED"


class TestJobIsActive:
    def test_active_when_running(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "RUNNING"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_is_active(100) is True

    def test_inactive_when_completed(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "COMPLETED"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_is_active(100) is False

    def test_inactive_when_failed(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "FAILED"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_is_active(100) is False

    def test_active_when_pending(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "PENDING"
        from infx.runners.slurm import SlurmClient

        assert SlurmClient.job_is_active(100) is True


class TestCancel:
    def test_cancel_records_id(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        from infx.runners.slurm import SlurmClient

        SlurmClient.cancel(42)
        assert 42 in fake_job._cancelled_ids


class TestWaitForRunning:
    def test_immediate_running(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "RUNNING"
        from infx.runners.slurm import SlurmClient

        # Should not raise.
        SlurmClient.wait_for_running(100, timeout=1, poll=0.01)

    def test_raises_on_terminal(self, _inject_fake_pyslurm):
        _, fake_job, _, _ = _inject_fake_pyslurm
        fake_job._loaded_state = "FAILED"
        from infx.runners.slurm import SlurmClient

        with pytest.raises(RuntimeError, match="terminal state FAILED"):
            SlurmClient.wait_for_running(100, timeout=1, poll=0.01)


class TestAccountingState:
    def test_returns_completed(self, _inject_fake_pyslurm):
        _, _, _, fake_db_job = _inject_fake_pyslurm
        fake_db_job._loaded_state = "COMPLETED"
        fake_db_job._loaded_exit_code = 0
        fake_db_job._loaded_exit_code_signal = 0
        from infx.runners.slurm import SlurmClient

        result = SlurmClient.accounting_state(100)
        assert result == ("COMPLETED", "0:0")

    def test_returns_failed(self, _inject_fake_pyslurm):
        _, _, _, fake_db_job = _inject_fake_pyslurm
        fake_db_job._loaded_state = "FAILED"
        fake_db_job._loaded_exit_code = 1
        fake_db_job._loaded_exit_code_signal = 0
        from infx.runners.slurm import SlurmClient

        result = SlurmClient.accounting_state(100)
        assert result == ("FAILED", "1:0")


class TestVerifyJobStatus:
    def test_completed_ok(self, _inject_fake_pyslurm):
        _, _, _, fake_db_job = _inject_fake_pyslurm
        fake_db_job._loaded_state = "COMPLETED"
        fake_db_job._loaded_exit_code = 0
        fake_db_job._loaded_exit_code_signal = 0
        from infx.runners.slurm import SlurmClient

        # Should not raise.
        SlurmClient.verify_job_status(100)

    def test_failed_raises(self, _inject_fake_pyslurm):
        _, _, _, fake_db_job = _inject_fake_pyslurm
        fake_db_job._loaded_state = "FAILED"
        fake_db_job._loaded_exit_code = 1
        fake_db_job._loaded_exit_code_signal = 0
        from infx.runners.slurm import SlurmClient

        with pytest.raises(RuntimeError, match="state=FAILED"):
            SlurmClient.verify_job_status(100)

    def test_nonzero_exit_code_raises(self, _inject_fake_pyslurm):
        _, _, _, fake_db_job = _inject_fake_pyslurm
        fake_db_job._loaded_state = "COMPLETED"
        fake_db_job._loaded_exit_code = 1
        fake_db_job._loaded_exit_code_signal = 0
        from infx.runners.slurm import SlurmClient

        with pytest.raises(RuntimeError, match="exit_code=1:0"):
            SlurmClient.verify_job_status(100)


class TestRunStep:
    def test_builds_correct_srun_argv(self, _inject_fake_pyslurm):
        from infx.runners.slurm import SlurmClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            SlurmClient.run_step(
                42,
                ["bash", "-c", "echo hello"],
                container="/path/to.sqsh",
                container_mounts="/src:/dst",
                container_workdir="/workspace/",
                no_container_mount_home=True,
                no_container_entrypoint=True,
                export="ALL,PORT=8888",
            )
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "srun"
            assert "--jobid=42" in cmd
            assert "--overlap" in cmd  # steps must share the holder allocation
            assert "--container-image=/path/to.sqsh" in cmd
            assert "--container-mounts=/src:/dst" in cmd
            assert "--container-workdir=/workspace/" in cmd
            assert "--no-container-mount-home" in cmd
            assert "--no-container-entrypoint" in cmd
            assert "--export=ALL,PORT=8888" in cmd
            assert cmd[-3:] == ["bash", "-c", "echo hello"]

    def test_extra_srun_args_passed(self, _inject_fake_pyslurm):
        from infx.runners.slurm import SlurmClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            SlurmClient.run_step(
                42,
                ["hostname"],
                extra_srun_args=["--job-name=test"],
            )
            cmd = mock_run.call_args[0][0]
            assert "--job-name=test" in cmd
            assert "hostname" in cmd


class TestPyslurmUnavailable:
    def test_clear_error_when_missing(self):
        """Verify a clear error is raised when pyslurm is not installed."""
        import infx.runners.slurm as slurm_mod

        slurm_mod._pyslurm = None
        slurm_mod._pyslurm_available = None
        # Temporarily hide pyslurm from sys.modules.
        old = sys.modules.pop("pyslurm", None)
        old_db = sys.modules.pop("pyslurm.db", None)
        try:
            with pytest.raises(
                slurm_mod.PyslurmUnavailableError,
                match="pyslurm is not installed",
            ):
                slurm_mod.SlurmClient.job_state(1)
        finally:
            if old is not None:
                sys.modules["pyslurm"] = old
            if old_db is not None:
                sys.modules["pyslurm.db"] = old_db
            slurm_mod._pyslurm = None
            slurm_mod._pyslurm_available = None
