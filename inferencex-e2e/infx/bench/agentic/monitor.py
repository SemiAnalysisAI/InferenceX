"""Diagnostic sampler for the AgentX client node: memory, OOM events, sockets, CPU.

Writes one JSON object per line to ``<out>/monitor.jsonl``:
every 5 s node memory, cgroup memory and OOM counters, TCP socket states and kernel TCP
counters; every 15 s per-process RSS/CPU/fd for processes above 32 MiB; every 60 s
per-process PSS/USS from smaps_rollup. Stdlib only; any read error is recorded and skipped.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
TICK = os.sysconf("SC_CLK_TCK")
TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
    "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
    "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
}
TCP_EXT = (
    "ListenOverflows", "ListenDrops", "TCPTimeouts", "TCPAbortOnTimeout", "TCPAbortOnData",
    "TCPAbortOnClose", "TCPAbortOnMemory", "TCPAbortFailed", "TCPMemoryPressures",
    "TCPBacklogDrop", "TCPReqQFullDrop", "TCPSynRetrans", "TW", "TWRecycled", "TCPOrphanQueued",
)
TCP_SNMP = ("ActiveOpens", "PassiveOpens", "AttemptFails", "EstabResets", "CurrEstab", "RetransSegs", "InErrs", "OutRsts")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _kv(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def meminfo() -> dict[str, int]:
    keys = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem", "AnonPages", "Mapped", "SwapFree", "Dirty")
    info = _kv(_read("/proc/meminfo"))
    return {k: info.get(k, -1) // 1024 for k in keys}  # MiB


def cgroup_dirs() -> list[str]:
    """This process's cgroup v2 directory and its ancestors up to the Slurm job level."""
    rel = ""
    for line in _read("/proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip()
    dirs, path = [], Path("/sys/fs/cgroup" + rel)
    while str(path).startswith("/sys/fs/cgroup") and path != Path("/sys/fs/cgroup"):
        if (path / "memory.current").exists():
            dirs.append(str(path))
        if path.name.startswith("job_"):
            break
        path = path.parent
    return dirs


def cgroup(dirs: list[str]) -> list[dict]:
    rows = []
    for d in dirs:
        row: dict = {"dir": d[len("/sys/fs/cgroup"):]}
        for name in ("memory.current", "memory.max", "memory.peak", "memory.swap.current"):
            try:
                v = _read(f"{d}/{name}").strip()
                row[name] = v if v == "max" else int(v) // (1 << 20)
            except OSError:
                pass
        try:
            row["events"] = _kv(_read(f"{d}/memory.events"))
            stat = _kv(_read(f"{d}/memory.stat"))
            row["anon"] = stat.get("anon", 0) >> 20
            row["file"] = stat.get("file", 0) >> 20
            row["sock"] = stat.get("sock", 0) >> 20
        except OSError:
            pass
        rows.append(row)
    return rows


def tcp() -> dict:
    states: dict[str, int] = {}
    remotes: dict[str, dict[str, int]] = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as f:
                next(f)
                for line in f:
                    p = line.split(None, 4)
                    st = TCP_STATES.get(p[3], p[3])
                    states[st] = states.get(st, 0) + 1
                    if st != "LISTEN":
                        rem = p[2][-4:]  # remote port (hex)
                        per = remotes.setdefault(rem, {})
                        per[st] = per.get(st, 0) + 1
        except OSError:
            pass
    top = sorted(remotes.items(), key=lambda kv: -sum(kv[1].values()))[:4]
    snmp = _read("/proc/net/snmp").splitlines()
    netstat = _read("/proc/net/netstat").splitlines()

    def pairs(lines: list[str], prefix: str) -> dict[str, int]:
        heads = [l.split() for l in lines if l.startswith(prefix)]
        return dict(zip(heads[0][1:], map(int, heads[1][1:]))) if len(heads) >= 2 else {}

    ext, tcp_snmp = pairs(netstat, "TcpExt:"), pairs(snmp, "Tcp:")
    sockstat = {}
    for line in _read("/proc/net/sockstat").splitlines():
        if line.startswith("TCP:"):
            f = line.split()[1:]
            sockstat = dict(zip(f[::2], map(int, f[1::2])))
    return {
        "states": states,
        "by_remote_port": {str(int(k, 16)): v for k, v in top},
        "sockstat": sockstat,
        "ext": {k: ext.get(k, -1) for k in TCP_EXT},
        "snmp": {k: tcp_snmp.get(k, -1) for k in TCP_SNMP},
        "port_range": _read("/proc/sys/net/ipv4/ip_local_port_range").split(),
    }


def procs(detail: bool) -> list[dict]:
    rows = []
    uid = os.getuid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        base = f"/proc/{pid}"
        try:
            if os.stat(base).st_uid != uid:
                continue
            status = _kv(_read(f"{base}/status"))
            rss = status.get("VmRSS", 0) // 1024
            if rss < 32:
                continue
            stat = _read(f"{base}/stat")
            fields = stat[stat.rfind(")") + 2 :].split()
            cmd = _read(f"{base}/cmdline").replace("\0", " ").strip()[:120]
            row = {
                "pid": int(pid), "ppid": int(fields[1]), "cmd": cmd,
                "rss": rss, "anon": status.get("RssAnon", 0) // 1024,
                "file": status.get("RssFile", 0) // 1024, "shmem": status.get("RssShmem", 0) // 1024,
                "threads": status.get("Threads", 0),
                "cpu_s": round((int(fields[11]) + int(fields[12])) / TICK, 1),
            }
            if detail:
                roll = _kv(_read(f"{base}/smaps_rollup"))
                row["pss"] = roll.get("Pss", 0) // 1024
                row["uss"] = (roll.get("Private_Clean", 0) + roll.get("Private_Dirty", 0)) // 1024
                row["fds"] = len(os.listdir(f"{base}/fd"))
            rows.append(row)
        except (OSError, IndexError, ValueError):
            continue
    return rows


def main(out_dir: str, stop_file: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = cgroup_dirs()
    with (out / "monitor.jsonl").open("a") as sink:
        sink.write(json.dumps({"t": time.time(), "start": True, "cgroups": dirs, "ncpu": os.cpu_count(),
                               "host": os.uname().nodename}) + "\n")
        n = 0
        while not os.path.exists(stop_file):
            t0 = time.time()
            row: dict = {"t": round(t0, 3)}
            for key, fn in (("mem", meminfo), ("cg", lambda: cgroup(dirs)), ("tcp", tcp),
                            ("load", lambda: _read("/proc/loadavg").split()[:3])):
                try:
                    row[key] = fn()
                except Exception as exc:  # noqa: BLE001 - diagnostics must never stop the run
                    row[f"{key}_err"] = repr(exc)
            if n % 3 == 0:
                try:
                    row["procs"] = procs(detail=n % 12 == 0)
                except Exception as exc:  # noqa: BLE001
                    row["procs_err"] = repr(exc)
            row["dt"] = round(time.time() - t0, 3)
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            n += 1
            time.sleep(max(0.5, 5 - (time.time() - t0)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
