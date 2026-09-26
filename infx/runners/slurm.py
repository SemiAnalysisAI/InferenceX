"""Thin Slurm client backed by pyslurm for job lifecycle management.

The client wraps pyslurm 25.11's public API (``Job``, ``Jobs``,
``JobSubmitDescription``, ``db.Job``) so callers never import pyslurm directly.
When pyslurm is unavailable (e.g. on macOS during development), import succeeds
but every method raises :class:`PyslurmUnavailableError`.

Design — holder jobs and ``srun --jobid`` attach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classic ``salloc --no-shell`` acquires a Slurm allocation and prints the job ID;
subsequent ``srun --jobid=<id>`` steps attach to that allocation.  In pyslurm
there is no ``salloc`` equivalent — only ``JobSubmitDescription.submit()``, which
submits a *batch* job.

To replicate the salloc pattern we submit a **holder job**: a batch script whose
only purpose is to hold the allocation open (``sleep infinity``).  Once Slurm
schedules it and it enters RUNNING, the allocation exists and ``srun --jobid=<id>
--overlap`` can attach interactive steps exactly as it does with salloc
allocations.  (Slurm allows ``srun --jobid`` to attach additional steps to any
running job the caller owns, including batch jobs — the ``--overlap`` flag lets
steps share allocated resources with the batch script.)

When all steps are done, the holder job is cancelled (``Job(id).cancel()``),
releasing the allocation immediately instead of waiting for the sleep to expire.

``srun`` is invoked as a subprocess because Pyxis container flags
(``--container-image``, ``--container-mounts``, etc.) are srun-spank plugins with
no pyslurm binding.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy pyslurm import — succeeds even without the C extension
# ---------------------------------------------------------------------------

_pyslurm: Any = None
_pyslurm_available: bool | None = None


class PyslurmUnavailableError(RuntimeError):
    """Raised when pyslurm is required but could not be imported."""


def _ensure_pyslurm() -> Any:
    global _pyslurm, _pyslurm_available  # noqa: PLW0603
    if _pyslurm_available is None:
        try:
            import pyslurm  # type: ignore[import-untyped]

            _pyslurm = pyslurm
            _pyslurm_available = True
        except ImportError as exc:
            _pyslurm_available = False
            raise PyslurmUnavailableError(
                "pyslurm is not installed.  Set INFERENCEX_PYSLURM_PATH to the "
                "build output directory or install pyslurm into the Python "
                "environment.  See infx/runners/pyslurm_build.py for the build "
                "helper."
            ) from exc
    if not _pyslurm_available:
        raise PyslurmUnavailableError("pyslurm is not installed")
    return _pyslurm


# ---------------------------------------------------------------------------
# Terminal-state helpers
# ---------------------------------------------------------------------------

_ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "REQUEUED",
        "SUSPENDED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
    }
)


def _is_terminal(state: str) -> bool:
    return state.upper() not in _ACTIVE_STATES


# ---------------------------------------------------------------------------
# SlurmClient
# ---------------------------------------------------------------------------


class SlurmClient:
    """High-level Slurm operations backed by pyslurm.

    All methods require pyslurm to be importable at call time (not at class
    construction).  This lets callers build the client early and defer the
    pyslurm import until the first RPC.
    """

    # -- Job submission ------------------------------------------------------

    @staticmethod
    def submit_holder_job(
        *,
        partition: str,
        account: str,
        gres: str,
        time_limit: str,
        job_name: str,
        num_nodes: int = 1,
        exclusive: bool = True,
    ) -> int:
        """Submit a batch holder job that sleeps until cancelled.

        Returns the Slurm job ID once submission succeeds.  The job is not
        necessarily RUNNING yet — call :meth:`wait_for_running` to block until
        the allocation is usable.
        """
        pyslurm = _ensure_pyslurm()
        desc = pyslurm.JobSubmitDescription(
            name=job_name,
            script="#!/bin/bash\nexec sleep infinity\n",
            # The holder only sleeps. Its default slurm-%j.out lands in the submit
            # directory, which compute pods may not mount (h100-dgxc: the batch
            # step dies with signal 53), so write nowhere and start in /.
            standard_output="/dev/null",
            standard_error="/dev/null",
            working_directory="/",
            partitions=partition,
            account=account,
            gres_per_node=gres,
            # salloc --time takes bare minutes ("480"); pyslurm needs an int for that form.
            time_limit=int(time_limit) if time_limit.strip().isdigit() else time_limit,
            nodes=num_nodes,
            resource_sharing="no" if exclusive else None,
        )
        job_id: int = desc.submit()
        log.info("Submitted holder job %d (partition=%s, gres=%s)", job_id, partition, gres)
        return job_id

    # -- Job queries ---------------------------------------------------------

    @staticmethod
    def job_state(job_id: int) -> str:
        """Return the current Slurm state string (e.g. ``RUNNING``, ``COMPLETED``)."""
        pyslurm = _ensure_pyslurm()
        job = pyslurm.Job.load(job_id)
        return str(job.state).upper()

    @staticmethod
    def job_exit_code(job_id: int) -> int | None:
        """Return the exit code of a completed job, or ``None`` if still active."""
        pyslurm = _ensure_pyslurm()
        job = pyslurm.Job.load(job_id)
        state = str(job.state).upper()
        if not _is_terminal(state):
            return None
        return int(job.exit_code) if job.exit_code is not None else None

    @staticmethod
    def job_is_active(job_id: int) -> bool:
        """Return ``True`` if the job is still queued, running, or completing."""
        pyslurm = _ensure_pyslurm()
        try:
            job = pyslurm.Job.load(job_id)
            return not _is_terminal(str(job.state).upper())
        except Exception:  # noqa: BLE001 — RPCError when job is purged
            return False

    @staticmethod
    def wait_for_running(job_id: int, *, timeout: float = 3600, poll: float = 5) -> None:
        """Block until *job_id* enters ``RUNNING`` (or a terminal state)."""
        deadline = time.monotonic() + timeout
        while True:
            state = SlurmClient.job_state(job_id)
            if state == "RUNNING":
                log.info("Job %d is RUNNING", job_id)
                return
            if _is_terminal(state):
                msg = f"Job {job_id} reached terminal state {state} before RUNNING"
                raise RuntimeError(msg)
            if time.monotonic() > deadline:
                msg = f"Timed out waiting for job {job_id} to start (last state: {state})"
                raise TimeoutError(msg)
            time.sleep(poll)

    # -- Accounting ----------------------------------------------------------

    @staticmethod
    def accounting_state(job_id: int) -> tuple[str, str] | None:
        """Query slurmdbd for ``(state, exit_code)``; return ``None`` if unavailable.

        The exit code string uses the ``M:N`` format (exit_code:signal).
        Falls back to the slurmctld job record when slurmdbd is unreachable.
        """
        pyslurm = _ensure_pyslurm()
        try:
            db_job = pyslurm.db.Job.load(job_id)
            state = str(db_job.state).upper()
            ec = db_job.exit_code
            sig = db_job.exit_code_signal
            exit_str = f"{ec or 0}:{sig or 0}"
            return state, exit_str
        except Exception:  # noqa: BLE001
            # slurmdbd may be unavailable; fall back to slurmctld
            try:
                job = pyslurm.Job.load(job_id)
                state = str(job.state).upper()
                ec = job.exit_code or 0
                sig = job.exit_code_signal or 0
                return state, f"{ec}:{sig}"
            except Exception:  # noqa: BLE001
                return None

    # -- Job control ---------------------------------------------------------

    @staticmethod
    def cancel(job_id: int) -> None:
        """Cancel a job (idempotent — does not raise if already done)."""
        pyslurm = _ensure_pyslurm()
        try:
            pyslurm.Job(job_id).cancel()
            log.info("Cancelled job %d", job_id)
        except Exception:  # noqa: BLE001 — already done or purged
            log.debug("cancel(%d) ignored (job may already be done)", job_id)

    # -- Log streaming -------------------------------------------------------

    @staticmethod
    def stream_job_log(job_id: int, log_file: str, *, poll: float = 5) -> None:
        """Tail *log_file* until *job_id* leaves the queue.

        Mirrors the bash ``stream_slurm_job_log`` helper: waits for the file to
        appear, then uses ``tail -F`` with a background poller that exits when
        the job is no longer active.
        """
        while not Path(log_file).is_file():
            if not SlurmClient.job_is_active(job_id):
                # Show slurmctld state for debugging.
                state = "unknown"
                with contextlib.suppress(Exception):
                    state = SlurmClient.job_state(job_id)
                msg = f"Job {job_id} ended (state={state}) before creating {log_file}"
                raise RuntimeError(msg)
            time.sleep(poll)

        # Background: poll until job leaves queue, then exit (so tail --pid stops).
        poller = subprocess.Popen(
            ["bash", "-c", _POLL_SCRIPT.format(job_id=job_id, poll=int(poll))],
        )
        log.info("Tailing %s", log_file)
        try:
            subprocess.run(
                ["tail", "-F", "-s", "2", "-n+1", log_file, f"--pid={poller.pid}"],
                check=False,
            )
        finally:
            poller.wait()

    # -- Step execution (srun) -----------------------------------------------

    @staticmethod
    def run_step(
        job_id: int,
        argv: list[str],
        *,
        container: str | None = None,
        container_mounts: str | None = None,
        container_workdir: str | None = None,
        no_container_mount_home: bool = False,
        no_container_entrypoint: bool = False,
        export: str | None = None,
        extra_srun_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run an ``srun --jobid`` step, optionally inside a Pyxis container.

        This shells out to ``srun`` because Pyxis flags are srun-spank plugins
        with no pyslurm binding.  The step overlaps the holder batch job.
        """
        # The holder batch step owns the allocation; steps must overlap it or they queue forever.
        cmd: list[str] = ["srun", f"--jobid={job_id}", "--overlap"]
        if container:
            cmd.append(f"--container-image={container}")
        if container_mounts:
            cmd.append(f"--container-mounts={container_mounts}")
        if container_workdir:
            cmd.append(f"--container-workdir={container_workdir}")
        if no_container_mount_home:
            cmd.append("--no-container-mount-home")
        if no_container_entrypoint:
            cmd.append("--no-container-entrypoint")
        if export:
            cmd.append(f"--export={export}")
        if extra_srun_args:
            cmd.extend(extra_srun_args)
        cmd.extend(argv)

        log.info("srun step: %s", " ".join(cmd))
        return subprocess.run(cmd, check=True)

    # -- Job status verification (mirrors verify_slurm_job_status) -----------

    @staticmethod
    def verify_job_status(job_id: int, *, retries: int = 10) -> None:
        """Verify a completed job exited cleanly.

        Mirrors ``verify_slurm_job_status`` from slurm_utils.sh: tries accounting
        first, falls back to slurmctld, retries transient gaps.
        """
        for attempt in range(1, retries + 1):
            result = SlurmClient.accounting_state(job_id)
            if result is None:
                if attempt < retries:
                    time.sleep(1)
                    continue
                msg = f"Could not verify terminal Slurm status for job {job_id}"
                raise RuntimeError(msg)

            state, exit_code = result

            if state == "COMPLETED" and exit_code == "0:0":
                return

            if state in {"", "PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                if attempt < retries:
                    time.sleep(1)
                    continue
                msg = f"Could not verify terminal Slurm status for job {job_id}"
                raise RuntimeError(msg)

            msg = f"Slurm job {job_id} ended with state={state} exit_code={exit_code}"
            raise RuntimeError(msg)

        msg = f"Could not verify terminal Slurm status for job {job_id}"
        raise RuntimeError(msg)


# Inline bash snippet used by stream_job_log's background poller.
_POLL_SCRIPT = """\
while python3 -c "
import sys
sys.path.insert(0, '.')
from infx.runners.slurm import SlurmClient
sys.exit(0 if SlurmClient.job_is_active({job_id}) else 1)
" 2>/dev/null; do
    sleep {poll}
done
"""
