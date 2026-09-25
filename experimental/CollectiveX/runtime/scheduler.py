"""simple-slurm job steps with the existing explicit salloc allocation lifecycle."""
from __future__ import annotations

import io
from contextlib import suppress
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile

from .process import command_context, log, log_path, log_tail, run


HOST_EXPORTS = "HOME,PATH,USER,XDG_CACHE_HOME,ENROOT_CACHE_PATH"

# A host utility must also work before a container can read the staged source. Send a small
# stdlib-only zipapp over stdin, as the old probes sent probe.py and image-import code. This
# preserves login/compute filesystem independence without generating a shell program.
REMOTE_BOOTSTRAP = """import pathlib, runpy, sys, tempfile
with tempfile.TemporaryDirectory(prefix='collectivex-remote-') as directory:
    archive = pathlib.Path(directory) / 'remote.pyz'
    archive.write_bytes(sys.stdin.buffer.read())
    sys.argv[0] = str(archive)
    runpy.run_path(str(archive), run_name='__main__')
"""


def remote_archive() -> bytes:
    """Bundle compute-host utilities; Slurm and GPU dependencies stay outside the archive."""
    content = io.BytesIO()
    root = Path(__file__).parent
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.glob("*.py")):
            archive.write(path, f"runtime/{path.name}")
        archive.writestr("__main__.py", "from runtime.node import main\nraise SystemExit(main())\n")
    return content.getvalue()


class SlurmAllocation:
    """One allocated shard; persistent job IDs let workflow cleanup outlive its launcher."""

    def __init__(self, root: Path, env: dict[str, str]):
        self.root, self.env = root, env
        self.job_id: str | None = None

    def allocate(self, arguments: list[str], attempt: int = 1) -> str:
        """Obtain the same no-shell allocation and record it before any job step."""
        name = self.env.get("RUNNER_NAME", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            name = "collectivex"
        suffix = "" if attempt == 1 else f"-a{attempt}"
        path = log_path(self.root, f"scheduler-allocation{suffix}")
        log("scheduler-request=submit")
        try:
            run(["salloc", f"--job-name={name}", *arguments, "--no-shell"], path=path, env=self.env)
        except subprocess.CalledProcessError:
            log_tail(path)
            raise
        match = re.search(r"Granted job allocation ([1-9][0-9]*)", path.read_text())
        if match is None:
            raise RuntimeError("could not resolve allocated JOB_ID from salloc")
        self.job_id = match[1]
        (self.root / "jobid").write_text(self.job_id + "\n")
        os.chmod(self.root / "jobid", 0o600)
        return self.job_id

    def step(self, options: dict, argv: list[str], path: Path, *,
             timeout: int | None = None, stdin: bytes | None = None,
             kill_after: int | None = 30) -> int:
        """Run a step through simple-slurm, retaining literal Pyxis and boolean options.

        simple-slurm 0.3.6 drops empty-valued srun flags and models sbatch's option set.
        Pass bare flags and plugin options through its command tail. Quote all data before
        its shell-string API; a real executable fixture tests spaces and shell metacharacters.
        """
        if not self.job_id or not re.fullmatch(r"[1-9][0-9]*", self.job_id):
            raise ValueError("a job step requires an existing allocation")
        env = dict(self.env)
        # The library constructs a queue helper even for srun. Our queue queries always use
        # an explicit format, so an unrelated inherited SQUEUE_FORMAT must not affect steps.
        env.pop("SQUEUE_FORMAT", None)
        with command_context(env, path, stdin):
            from simple_slurm import Slurm

            slurm = Slurm()
            extra = []
            for key, value in {"jobid": self.job_id, **options}.items():
                if value is None or value is False:
                    continue
                if value is True:
                    extra.append(f"--{key.replace('_', '-')}")
                elif f"--{key}" in slurm.parser._option_string_actions:
                    slurm.add_arguments(**{key: shlex.quote(str(value))})
                else:
                    extra.append(f"--{key.replace('_', '-')}={value}")
            executable = ["srun"]
            if timeout is not None:
                kill = ["-k", str(kill_after)] if kill_after is not None else []
                executable = ["timeout", *kill, str(timeout), *executable]
            return slurm.srun(shlex.join([*extra, *map(str, argv)]), srun_cmd=shlex.join(executable))

    def host(self, nodes: int, command: list[str], path: Path, **options) -> int:
        """Run a stdlib node utility once per node, without accessing its filesystem first."""
        return self.step(
            {"nodes": nodes, "ntasks": nodes, "ntasks_per_node": 1,
             "chdir": "/tmp", "export": HOST_EXPORTS, "input": "all", **options},
            ["python3", "-c", REMOTE_BOOTSTRAP, *command], path, stdin=remote_archive(),
        )

    def nodes(self) -> str:
        """Return the allocated host list for quarantine/retry, validating every name."""
        expression = run(["squeue", "-h", "-j", self.job_id, "-o", "%N"], env=self.env).stdout.strip()
        if not re.fullmatch(r"[][A-Za-z0-9._,-]+", expression):
            raise RuntimeError("invalid allocation node list")
        names = run(["scontrol", "show", "hostnames", expression], env=self.env).stdout.splitlines()
        if not names or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) for name in names):
            raise RuntimeError("cannot identify nodes from a rejected allocation")
        return ",".join(names)

    def release(self) -> None:
        """Cancel only this allocation and retain its recovery record until it has stopped."""
        record = self.root / "jobid"
        job_id = self.job_id or (record.read_text().strip() if record.is_file() else "")
        if not job_id:
            return
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise RuntimeError("invalid cleanup allocation")
        with suppress(OSError, subprocess.SubprocessError):
            run(["scancel", job_id], env=self.env, check=False)
        for _ in range(30):
            try:
                state = run(["squeue", "-h", "-j", job_id, "-o", "%A"], env=self.env, check=False)
                finished = state.returncode == 0 and not state.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                finished = False
            if finished:
                record.unlink(missing_ok=True)
                self.job_id = None
                return
            time.sleep(1)
        raise RuntimeError("scheduled allocation did not terminate during cleanup")
