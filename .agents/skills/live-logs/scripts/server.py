#!/usr/bin/env python3
"""Stream one Slurm job's srt-slurm logs to a local browser page.

One ssh `tail -F` per discovered log file feeds an in-memory buffer; the page
subscribes over Server-Sent Events. New log files (for example aiperf output once the benchmark
starts) are discovered every 30 s and get their own pane.
"""
import argparse
import collections
import json
import os
import queue
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15"]
BANNER = ("Access to this system", "All activity is logged")
# Worker, orchestrator and benchmark logs; exporters and configs are left out on purpose.
FIND_EXPR = (
    "\\( -name '*_prefill_w*.out' -o -name '*_decode_w*.out' -o -name '*_agg_w*.out' "
    "-o -name 'sweep_*.log' -o -name '*_frontend_*.out' -o -name 'service_mooncake-master.out' "
    "-o -name 'service_etcd.out' -o -name 'aiperf.log' -o -name 'benchmark.log' \\)"
)


def page_regex(name):
    """Reuse the page's NOISE / ERR regex so server- and client-side filtering agree."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")) as f:
        page = f.read()
    m = re.search(rf"^const {name} = /(.*)/;$", page, re.MULTILINE)
    return re.compile(m.group(1))


NOISE_RE = page_regex("NOISE")
ERR_RE = page_regex("ERR")


def wanted(line, noise, erronly):
    if erronly:
        return bool(ERR_RE.search(line))
    return not (noise and NOISE_RE.search(line) and not ERR_RE.search(line))


def last_matching(buf, n, noise, erronly):
    """Newest n lines of buf that pass the filters (n=None means all), oldest first."""
    if n is None:
        return [l for l in buf if wanted(l, noise, erronly)]
    out = []
    for l in reversed(buf):
        if wanted(l, noise, erronly):
            out.append(l)
            if len(out) >= n:
                break
    out.reverse()
    return out


def group_of(rel):
    if re.search(r"_(prefill|decode|agg)_w\d+\.out$", rel):
        return 0
    if rel.endswith(("aiperf.log", "benchmark.log")):
        return 2
    return 1


def label_of(rel):
    base = os.path.basename(rel)
    m = re.match(r".*-(gpu-\d+|[a-z0-9]+-\d+)_(prefill|decode|agg)_w(\d+)\.out$", base)
    if m:
        return f"{m.group(2).capitalize()} w{m.group(3)} · {m.group(1)}"
    m = re.match(r".*-(gpu-\d+|[a-z0-9]+-\d+)_frontend_(\d+)\.out$", base)
    if m:
        return f"Dynamo frontend · {m.group(1)}"
    if base.startswith("sweep_"):
        return "srtctl sweep"
    if base.startswith("service_"):
        return base[len("service_"):-len(".out")]
    m = re.search(r"(conc_\d+)/.*?(aiperf|benchmark)\.log$", rel)
    if m:
        return f"{m.group(2)} · {m.group(1)}"
    parts = rel.split("/")
    return "/".join(parts[-3:]) if len(parts) > 1 else base


class State:
    def __init__(self):
        self.files = []  # [rel, label, group]
        self.buffers = {}
        self.totals = collections.Counter()
        self.status = {"ssh": "connecting", "squeue": "", "mem": {}}
        self.subscribers = []
        self.lock = threading.Lock()

    def publish(self, ev):
        with self.lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass


def remote(host, script, timeout=60):
    out = subprocess.run(SSH + [host, script], capture_output=True, text=True, timeout=timeout, check=False).stdout
    return [l for l in out.splitlines() if l and not l.startswith(BANNER)]


def tail_batch(args, st, rels):
    """Follow a fixed set of files from line 1; reconnect and re-read on ssh drops."""
    by_base = {}
    for r in rels:
        by_base[r] = r
    while True:
        cmd = SSH + [args.host, f"cd {args.logdir} && exec tail -n {args.tail_from} -F " + " ".join(rels)]
        st.status["ssh"] = "connected"
        st.publish({"t": "status", "s": st.status})
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace", bufsize=1)
        cur = rels[0] if len(rels) == 1 else None  # tail prints "==> f <==" headers only for several files
        for line in p.stdout:
            line = ANSI.sub("", line.rstrip("\n"))
            m = re.match(r"^==> (.+) <==$", line)
            if m:
                cur = by_base.get(m.group(1), cur)
                continue
            if cur is None or not line:
                continue
            st.buffers[cur].append(line)
            st.totals[cur] += 1
            st.publish({"t": "line", "k": cur, "l": line})
        p.wait()
        with st.lock:
            for r in rels:
                st.buffers[r].clear()
                st.totals[r] = 0
        st.publish({"t": "reset", "k": rels})
        st.status["ssh"] = f"reconnecting (exit {p.returncode})"
        st.publish({"t": "status", "s": st.status})
        time.sleep(5)


def discover_loop(args, st):
    while True:
        try:
            found = remote(args.host, f"cd {args.logdir} 2>/dev/null && find . -maxdepth 5 -type f {FIND_EXPR} | sed 's#^./##' | sort")
            new = [r for r in found if r not in st.buffers]
            if new:
                with st.lock:
                    for r in new:
                        st.buffers[r] = collections.deque(maxlen=args.max_lines)
                        st.files.append([r, label_of(r), group_of(r)])
                st.publish({"t": "files", "f": st.files})
                # one tail per file: a multi-file tail prints each file in full before the next,
                # so a huge prefill log would hold every other pane empty until it finished
                for r in new:
                    threading.Thread(target=tail_batch, args=(args, st, [r]), daemon=True).start()
        except (OSError, subprocess.SubprocessError, ValueError) as e:  # flaky network: keep going
            st.status["ssh"] = f"discovery error: {e}"
        time.sleep(30)


def status_loop(args, st):
    while True:
        try:
            script = (
                f"squeue -j {args.job} -h -o '%T %M %N'; "
                f"for n in $(scontrol show hostnames $(squeue -j {args.job} -h -o %N) 2>/dev/null); do "
                "echo \"NODE $n $(scontrol show node $n | grep -oE 'FreeMem=[0-9]+|RealMemory=[0-9]+|CPULoad=[0-9.]+' | tr '\\n' ' ')\"; done"
            )
            out = remote(args.host, script)
            sq = [l for l in out if not l.startswith("NODE ")]
            st.status["squeue"] = sq[0] if sq else "job not in queue (finished?)"
            mem = {}
            for l in out:
                if l.startswith("NODE "):
                    _, name, *kvs = l.split()
                    mem[name] = dict(x.split("=", 1) for x in kvs if "=" in x)
            st.status["mem"] = mem
            st.status["totals"] = dict(st.totals)
            st.publish({"t": "status", "s": st.status})
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            st.status["squeue"] = f"status poll error: {e}"
        time.sleep(20)


def build_zip(args):
    """Pull every text log under <job>/logs (full files, no tachometer metrics) as a gzipped tar and repack it as a zip."""
    name = f"slurm-{args.job}"
    pick = (f"cd {args.logdir} && find . -type f \\( -name '*.out' -o -name '*.log' -o -name '*.txt' \\) "
            "! -path './tachometer/*' | sed 's#^./##' | tar czf - -T -")
    p = subprocess.Popen(SSH + [args.host, pick],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fd, out = tempfile.mkstemp(prefix=name + "-", suffix=".zip")
    os.close(fd)
    with tarfile.open(fileobj=p.stdout, mode="r|gz") as tar, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in tar:
            if not m.isfile():
                continue
            src = tar.extractfile(m)
            with zf.open(f"{name}-logs/{m.name}", "w") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
    p.wait()
    return out


def make_handler(args, st):
    page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_file(self, rel):
            """Stream one full log straight from the cluster; only files the viewer discovered are allowed."""
            if rel not in st.buffers:
                self.send_error(404, "unknown log file")
                return
            p = subprocess.Popen(SSH + [args.host, f"cat {args.logdir}/{rel}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="slurm-{args.job}-{os.path.basename(rel)}"')
            self.end_headers()
            try:
                while True:
                    chunk = p.stdout.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                p.kill()
            finally:
                p.wait()

        def do_GET(self):
            if self.path == "/":
                with open(page_path) as f:
                    page = f.read()
                body = page.replace("__TITLE__", args.title).replace("__JOB__", json.dumps(str(args.job))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/download.zip"):
                try:
                    path = build_zip(args)
                except (OSError, tarfile.TarError, subprocess.SubprocessError) as e:
                    self.send_error(502, f"could not fetch logs: {e}")
                    return
                try:
                    size = os.path.getsize(path)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", f'attachment; filename="slurm-{args.job}-logs.zip"')
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    with open(path, "rb") as f:
                        while True:
                            chunk = f.read(1 << 20)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    os.remove(path)
                return
            url = urlparse(self.path)
            if url.path == "/file":
                self.send_file(parse_qs(url.query).get("f", [""])[0])
                return
            if url.path != "/events":
                self.send_error(404)
                return
            qs = parse_qs(url.query)
            n_arg = qs.get("n", ["4000"])[0]
            n = None if n_arg == "all" else max(1, int(n_arg))
            noise = qs.get("noise", ["1"])[0] == "1"
            erronly = qs.get("err", ["0"])[0] == "1"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue(maxsize=50000)
            with st.lock:
                snap = {k: last_matching(v, n, noise, erronly) for k, v in st.buffers.items()}
                st.status["totals"] = dict(st.totals)
                files = [list(f) for f in st.files]
                st.subscribers.append(q)
            try:
                self.wfile.write(f"data: {json.dumps({'t': 'snapshot', 'f': files, 'b': snap, 's': st.status})}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        batch = [q.get(timeout=15)]
                        while len(batch) < 1000:
                            try:
                                batch.append(q.get_nowait())
                            except queue.Empty:
                                break
                        batch = [e for e in batch if e["t"] != "line" or wanted(e["l"], noise, erronly)]
                        if not batch:
                            continue
                        self.wfile.write(f"data: {json.dumps({'t': 'batch', 'e': batch})}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with st.lock:
                    st.subscribers.remove(q)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="ssh target of the cluster login node")
    ap.add_argument("--job", required=True, help="Slurm job id")
    ap.add_argument("--logdir", required=True, help="<WorkDir>/outputs/<job>/logs on the cluster")
    ap.add_argument("--title", default="")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-lines", type=int, default=1_000_000, help="lines kept per file")
    ap.add_argument("--history", default="all", help="'all' (read every file from line 1) or N (only the last N lines of each file)")
    args = ap.parse_args()
    args.title = args.title or f"Slurm {args.job}"
    args.tail_from = "+1" if args.history == "all" else str(int(args.history))
    st = State()
    threading.Thread(target=discover_loop, args=(args, st), daemon=True).start()
    threading.Thread(target=status_loop, args=(args, st), daemon=True).start()
    print(f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, st)).serve_forever()


if __name__ == "__main__":
    main()
