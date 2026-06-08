#!/usr/bin/env python3
"""Poll one or more Prometheus /metrics endpoints at a fixed interval and append
timestamped raw snapshots to a file. Stdlib only (urllib) so it runs in any
vLLM/sglang serving container. Runs until SIGTERM/SIGINT or --duration.

Output format: one block per scrape per url:
    # ==== ts=<iso8601> url=<url> ====
    <raw prometheus text>
parseable later with awk/grep on the delimiter line.

Usage:
    python scrape_metrics.py --urls http://localhost:8888/metrics \
        --out results/dsv4_b1/server_metrics.prom --interval 15
"""
from __future__ import annotations

import argparse
import signal
import time
import urllib.request
from datetime import datetime, timezone


def fetch(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return f"# FETCH_ERROR {type(e).__name__}: {e}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", required=True,
                    help="comma-separated Prometheus /metrics URLs to poll")
    ap.add_argument("--out", required=True, help="append timestamped snapshots here")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between scrapes")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = run until killed")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-request timeout")
    a = ap.parse_args()

    urls = [u.strip() for u in a.urls.split(",") if u.strip()]
    stop = {"v": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(v=True))

    print(f"[scrape_metrics] polling {urls} every {a.interval}s -> {a.out}", flush=True)
    t0 = time.time()
    n = 0
    with open(a.out, "a") as f:
        while not stop["v"]:
            ts = datetime.now(timezone.utc).isoformat()
            for u in urls:
                txt = fetch(u, a.timeout)
                f.write(f"# ==== ts={ts} url={u} ====\n{txt}\n")
            f.flush()
            n += 1
            if a.duration and (time.time() - t0) >= a.duration:
                break
            # responsive sleep so SIGTERM stops us promptly
            slept = 0.0
            while slept < a.interval and not stop["v"]:
                step = min(1.0, a.interval - slept)
                time.sleep(step)
                slept += step
    print(f"[scrape_metrics] stopped after {n} scrape(s); wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
