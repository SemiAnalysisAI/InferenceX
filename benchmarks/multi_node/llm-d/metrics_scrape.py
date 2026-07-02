#!/usr/bin/env python3
"""Scrape vLLM /metrics from EVERY prefill and decode engine (plus EPP) over the
benchmark window, into one time-correlated CSV that survives a killed run.

Why this exists (vs the old leader-only bash scraper):
  * Each vLLM node's /metrics only reports its LOCAL data-parallel ranks, so
    scraping just PREFILL_LEADER_IP is blind to the leader's own worker node AND
    to every other prefill engine. With Option B (PREFILL_WORKERS>1) that means
    we saw 1 of N prefill engines. To answer "is EPP fanning out to ALL
    prefillers?" we must scrape every prefill node and compare per-engine load.
  * Decode is scraped on the vLLM port (VLLM_PORT), NOT the pd-sidecar port -
    we want the engine's own queue/running counts, not the proxy's.

Targets are derived from ALL_IPS (rank-ordered: [0:PREFILL_NODES] prefill,
[PREFILL_NODES:PREFILL_NODES+DECODE_NODES] decode), all on VLLM_PORT. EPP raw
/metrics (per-endpoint routing counts) is dumped to a sibling .prom file.

Output CSV columns:
  ts_iso, epoch, role, name, addr, conc, <one column per metric in WANT>
Counter metrics are summed across the node's engine labels; gauges too (so a
node's row is the node-wide value). Rows are appended line-buffered so a run
that is later killed still leaves a complete-up-to-that-point CSV on the shared
filesystem.

All config via env; no args required:
  ALL_IPS, PREFILL_NODES, DECODE_NODES, VLLM_PORT (default 8200),
  EPP_METRICS_HOST (default 127.0.0.1), EPP_METRICS_PORT (default 9090),
  METRICS_SCRAPE_INTERVAL_S (default 2),
  SCRAPE_OUT_CSV (required), SCRAPE_EPP_OUT (optional .prom dump),
  SCRAPE_CONC_FILE (optional; its contents tag the `conc` column).
"""
import datetime
import os
import time
import urllib.request

# Diagnostic-focused metric set. Counters (…_total, …_sum, …_count) are
# monotonic; take deltas between rows when analyzing. Gauges are instantaneous.
WANT = [
    "vllm:num_requests_running",        # in-flight -> EPP fan-out + decode-slot saturation
    "vllm:num_requests_waiting",        # queue depth -> the TTFT-explosion signal
    "vllm:kv_cache_usage_perc",         # KV pressure per engine
    "vllm:num_preemptions_total",       # decode KV thrash (recompute)
    "vllm:prompt_tokens_total",         # prefill work per engine (fan-out cross-check)
    "vllm:generation_tokens_total",     # decode work per engine
    "vllm:request_queue_time_seconds_sum",    # with _count -> avg queue wait
    "vllm:request_queue_time_seconds_count",
    "vllm:nixl_num_failed_transfers_total",   # KV-transfer health
    "vllm:nixl_num_kv_expired_reqs_total",
]


def sum_metric(text, name):
    """Sum every sample of `name` (across engine/model labels). None if absent."""
    total = None
    prefix_brace = name + "{"
    prefix_space = name + " "
    for line in text.splitlines():
        if line.startswith(prefix_brace) or line.startswith(prefix_space):
            try:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return total


def build_targets():
    ips = [x for x in os.environ.get("ALL_IPS", "").split(",") if x]
    pn = int(os.environ.get("PREFILL_NODES", "1"))
    dn = int(os.environ.get("DECODE_NODES", "1"))
    pw = int(os.environ.get("PREFILL_WORKERS", "1"))
    dw = int(os.environ.get("DECODE_WORKERS", "1"))
    gpn = int(os.environ.get("GPUS_PER_NODE", "4"))
    port = int(os.environ.get("VLLM_PORT", "8200"))
    lb = os.environ.get("LLMD_LB_MODE", "hybrid")
    targets = []

    def add(role, node_ips, workers):
        # hybrid: one target per node at VLLM_PORT (the node's api-server exposes
        # all its local ranks there). multi-port: the vLLM ranks each serve on
        # VLLM_PORT + (rank within engine), so scrape every local rank port.
        # Always the vLLM port, never the decode sidecar.
        nodes_per_engine = max(1, len(node_ips) // max(1, workers))
        for i, ip in enumerate(node_ips):
            if lb == "multi-port":
                start = (i % nodes_per_engine) * gpn
                for k in range(gpn):
                    targets.append((role, f"{role}-{i}-r{start + k}", f"{ip}:{port + start + k}"))
            else:
                targets.append((role, f"{role}-{i}", f"{ip}:{port}"))

    add("prefill", ips[:pn], pw)
    add("decode", ips[pn:pn + dn], dw)
    return targets


def read_conc(path):
    if not path:
        return ""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def main():
    out_csv = os.environ["SCRAPE_OUT_CSV"]
    epp_out = os.environ.get("SCRAPE_EPP_OUT", "")
    epp_host = os.environ.get("EPP_METRICS_HOST", "127.0.0.1")
    epp_port = os.environ.get("EPP_METRICS_PORT", "9090")
    conc_file = os.environ.get("SCRAPE_CONC_FILE", "")
    interval = float(os.environ.get("METRICS_SCRAPE_INTERVAL_S", "2"))
    targets = build_targets()

    with open(out_csv, "a", buffering=1) as f:
        f.write("# ts_iso,epoch,role,name,addr,conc," + ",".join(WANT) + "\n")
        while True:
            now = datetime.datetime.now()
            ts = now.isoformat(timespec="milliseconds")
            epoch = f"{now.timestamp():.3f}"
            conc = read_conc(conc_file)
            for role, name, addr in targets:
                try:
                    txt = urllib.request.urlopen(
                        f"http://{addr}/metrics", timeout=2).read().decode()
                    vals = [sum_metric(txt, m) for m in WANT]
                    cells = ["" if v is None else f"{v:.3f}" for v in vals]
                    f.write(f"{ts},{epoch},{role},{name},{addr},{conc}," +
                            ",".join(cells) + "\n")
                except Exception as ex:  # noqa: BLE001 - log-and-continue scraper
                    f.write(f"{ts},{epoch},{role},{name},{addr},{conc},"
                            f"ERR:{type(ex).__name__}\n")
            if epp_out:
                try:
                    txt = urllib.request.urlopen(
                        f"http://{epp_host}:{epp_port}/metrics", timeout=2).read().decode()
                    with open(epp_out, "a", buffering=1) as ef:
                        ef.write(f"# scrape_ts={epoch} conc={conc}\n{txt}")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(interval)


if __name__ == "__main__":
    main()
