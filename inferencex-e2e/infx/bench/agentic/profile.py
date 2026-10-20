"""Op-attribution profiling of an AgentX point (``INFX_PROFILE``): torch windows on every worker."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from infx.bench import proc

PROFILING = proc.REPO_ROOT / "benchmarks" / "profiling"


def enabled(env: Mapping[str, str]) -> bool:
    return bool(env.get("INFX_PROFILE_WINDOWS"))


def synthetic(env: Mapping[str, str]) -> bool:
    return enabled(env) and env.get("INFX_PROFILE_MODE") == "synthetic"


def servers(env: Mapping[str, str], url: str) -> list[str]:
    """srt-slurm's logical worker endpoints, each engine server's control port; else the frontend."""
    endpoints = [e for e in env.get("SRT_AGG_ENDPOINTS", "").split(",") if e]
    return [f"http://{e}" for e in endpoints] or [url]


def start_windows(
    env: Mapping[str, str], url: str, conc: int, phase_log: Path
) -> subprocess.Popen | None:
    """The window client, following ``phase_log`` (AIPerf's log, or the synthetic load's phases)."""
    if not enabled(env):
        return None
    prof_dir = Path(env["INFX_PROF_DIR"])
    prof_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(PROFILING / "profile_windows.py"),
        env["INFX_PROFILE_WINDOWS"],
        str(prof_dir / f"windows_conc{conc}.jsonl"),
        str(phase_log),
        str(prof_dir / "steps"),
        *servers(env, url),
    ]
    proc.echo(argv)
    return subprocess.Popen(argv, env=dict(env))


def stop_windows(client: subprocess.Popen | None) -> None:
    if client is None:
        return
    client.terminate()
    try:
        client.wait(timeout=30)
    except subprocess.TimeoutExpired:
        client.kill()
        client.wait()


def run_synthetic(env: Mapping[str, str], url: str, conc: int, result_json: Path) -> int:
    """Synthetic load in place of the replay; the windows decide when it may stop."""
    prof_dir = Path(env["INFX_PROF_DIR"])
    prof_dir.mkdir(parents=True, exist_ok=True)
    phases = prof_dir / "synthetic_phases.log"
    stop_file = prof_dir / "synthetic.stop"
    phases.write_text("")
    stop_file.unlink(missing_ok=True)
    child_env = {**env, "INFX_SYNTH_STOP_FILE": str(stop_file)}
    windows = start_windows(child_env, url, conc, phases)
    load_argv = [
        sys.executable,
        str(PROFILING / "synthetic_load.py"),
        url,
        str(phases),
        str(prof_dir / f"windows_conc{conc}.jsonl"),
        str(result_json),
    ]
    proc.echo(load_argv)
    load = subprocess.Popen(load_argv, env=child_env)
    if windows is not None:
        windows.wait()
    stop_file.touch()
    return proc.status(load.wait())
