#!/usr/bin/env python3
"""Resolve a PR / workflow run / job to its live Slurm jobs and open a log viewer per job.

    live_logs.py 3523
    live_logs.py https://github.com/SemiAnalysisAI/InferenceX/pull/3523
    live_logs.py https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36345723679
    live_logs.py https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36345723679/job/108694606739
    live_logs.py --stop            # stop every viewer this script started

Cluster login hosts are NOT stored in this repo. They are read from
~/.config/infx-live-logs/clusters.json, e.g. {"b300-dsxe": {"host": "my-b300-login-alias"}},
or from INFX_LIVE_LOGS_HOST_<CLUSTER> (cluster upper-cased, '-' -> '_').
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import webbrowser

REPO = os.environ.get("INFX_LIVE_LOGS_REPO", "SemiAnalysisAI/InferenceX")
CONFIG = os.path.expanduser("~/.config/infx-live-logs/clusters.json")
STATE_DIR = os.path.expanduser("~/.cache/infx-live-logs")
HERE = os.path.dirname(os.path.abspath(__file__))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
BANNER = ("Access to this system", "All activity is logged")


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    if out.returncode:
        sys.exit(f"gh api {path} failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def resolve_jobs(target):
    """Return [(run_id, job)] for in-progress, non-setup jobs of the target."""
    m = re.search(r"/actions/runs/(\d+)/job/(\d+)", target)
    if m:
        return [(m.group(1), gh(f"repos/{REPO}/actions/jobs/{m.group(2)}"))]
    m = re.search(r"/actions/runs/(\d+)", target)
    if m:
        run_ids = [m.group(1)]
    else:
        m = re.search(r"(?:/pull/)?(\d+)\s*$", target.strip())
        if not m:
            sys.exit(f"cannot parse {target!r}: pass a PR number, PR URL, run URL or job URL")
        pr = gh(f"repos/{REPO}/pulls/{m.group(1)}")
        runs = gh(f"repos/{REPO}/actions/runs?head_sha={pr['head']['sha']}&per_page=50")["workflow_runs"]
        runs = [r for r in runs if r["name"].startswith("Run Sweep") and r["status"] != "completed"] or \
               [r for r in runs if r["name"].startswith("Run Sweep")][:1]
        if not runs:
            sys.exit(f"PR #{m.group(1)} has no Run Sweep run on head {pr['head']['sha'][:8]}")
        run_ids = [str(r["id"]) for r in runs]
    jobs = []
    for rid in run_ids:
        for j in gh(f"repos/{REPO}/actions/runs/{rid}/jobs?per_page=100")["jobs"]:
            if j["name"] != "setup" and j["status"] == "in_progress" and j.get("runner_name"):
                jobs.append((rid, j))
    return jobs


def cluster_host(cluster):
    env = os.environ.get("INFX_LIVE_LOGS_HOST_" + re.sub(r"[^A-Z0-9]", "_", cluster.upper()))
    if env:
        return env
    try:
        with open(CONFIG) as f:
            return json.load(f)[cluster]["host"]
    except (OSError, KeyError, ValueError):
        return None


def remote(host, script):
    out = subprocess.run(SSH + [host, script], capture_output=True, text=True, timeout=90, check=False).stdout
    return [l for l in out.splitlines() if l and not l.startswith(BANNER)]


def slurm_job_for_runner(host, runner):
    """srtctl names the Slurm job after the GitHub runner (e.g. b300-dsxe_01)."""
    lines = remote(host, f"squeue -h -n {runner} -o '%i %T' | sort -n | tail -1; "
                         f"sacct -n -X -P --name {runner} -S now-1days -o JobID | grep -E '^[0-9]+$' | sort -n | tail -1")
    ids = [l.split()[0] for l in lines if l.split()[0].isdigit()]
    if not ids:
        # Some launchers name the job after the worker (e.g. worker-2); fall back to the runner's
        # gharunnerNN work directory.
        suffix = runner.rsplit("_", 1)[-1]
        lines = remote(host, f"squeue -h -o '%i %Z' | grep '/gharunner{suffix}/' | sort -n | tail -1")
        ids = [l.split()[0] for l in lines if l.split()[0].isdigit()]
    if not ids:
        return None, None
    job = ids[0]
    # srtctl points the job's StdOut at <outputs>/<job>/logs/sweep_<job>.log, which is the most
    # reliable way to find the logs (WorkDir can be a checkout next to outputs/, not its parent).
    info = remote(host, f"scontrol show job {job} 2>/dev/null | grep -oE '(WorkDir|StdOut)=[^ ]+'")
    kv = dict(x.split("=", 1) for x in info if "=" in x)
    out = kv.get("StdOut", "")
    if os.path.basename(out).startswith("sweep_"):
        return job, os.path.dirname(out)
    wd = kv.get("WorkDir") or next(iter(remote(host, f"sacct -n -X -P -j {job} -o WorkDir")), None)
    return job, (f"{wd}/outputs/{job}/logs" if wd else None)


def free_port(start=8765):
    for p in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)):
                return p
    sys.exit("no free local port")


def start_viewer(host, job, logdir, title, history="all"):
    os.makedirs(STATE_DIR, exist_ok=True)
    port = free_port()
    with open(os.path.join(STATE_DIR, f"viewer-{job}.log"), "w") as log:
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--host", host, "--job", job,
                              "--logdir", logdir, "--title", title, "--port", str(port), "--history", history],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(os.path.join(STATE_DIR, "pids"), "a") as f:
        f.write(f"{p.pid} {port} {job}\n")
    url = f"http://127.0.0.1:{port}/"
    for _ in range(50):  # wait until the server accepts connections
        with socket.socket() as s:
            if not s.connect_ex(("127.0.0.1", port)):
                break
        time.sleep(0.1)
    return url


def stop_all():
    path = os.path.join(STATE_DIR, "pids")
    if not os.path.exists(path):
        print("no viewers recorded")
        return
    with open(path) as f:
        lines = f.read().split("\n")
    for line in filter(None, lines):
        pid, port, job = line.split()
        try:
            os.killpg(int(pid), signal.SIGTERM)
            print(f"stopped viewer for Slurm {job} (port {port})")
        except ProcessLookupError:
            pass
    os.remove(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="PR number, PR URL, workflow run URL or job URL")
    ap.add_argument("--stop", action="store_true", help="stop every viewer started by this script")
    ap.add_argument("--no-open", action="store_true", help="print URLs without opening a browser")
    ap.add_argument("--host", help="manual mode: ssh target of the login node (with --job and --logdir)")
    ap.add_argument("--job", help="manual mode: Slurm job id")
    ap.add_argument("--logdir", help="manual mode: the job's logs directory on the cluster")
    ap.add_argument("--history", default="all", help="'all' (default) or N: only fetch the last N lines of each log (faster on huge logs)")
    args = ap.parse_args()
    if args.stop:
        return stop_all()
    if args.host and args.job and args.logdir:
        url = start_viewer(args.host, args.job, args.logdir, f"Slurm {args.job}", args.history)
        print(url)
        if not args.no_open:
            webbrowser.open(url)
        return None
    if not args.target:
        ap.error("target is required (or --host/--job/--logdir for manual mode)")

    jobs = resolve_jobs(args.target)
    if not jobs:
        sys.exit("no in-progress jobs with a runner found (queued, finished, or setup-only)")
    missing = set()
    for rid, j in jobs:
        runner = j["runner_name"]
        cluster = runner.rsplit("_", 1)[0]
        host = cluster_host(cluster)
        if not host:
            missing.add(cluster)
            continue
        job, logdir = slurm_job_for_runner(host, runner)
        if not job or not logdir:
            print(f"- {j['name'][:90]}: no Slurm job named {runner} on {cluster} (single-node or not submitted yet)")
            continue
        title = f"{cluster} · Slurm {job} · {j['name'].split('|')[-1].strip()[:80]}"
        url = start_viewer(host, job, logdir, title, args.history)
        print(f"- {j['name'][:90]}\n  runner {runner} → Slurm {job}\n  {url}  (GitHub job {j['html_url']})")
        if not args.no_open:
            webbrowser.open(url)
    if missing:
        sys.exit(f"no login host configured for: {', '.join(sorted(missing))}. Add it to {CONFIG} "
                 '(e.g. {"b300-dsxe": {"host": "<ssh alias>"}}); see SKILL.md for where to find it.')


if __name__ == "__main__":
    main()
