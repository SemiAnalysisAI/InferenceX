#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only custom replay sweep for srt-slurm multinode (disagg) jobs.
# srt-slurm owns server startup; this runs as benchmark.type=custom against
# the already-healthy dynamo frontend on the head node. Modeled on
# agentic_srt.sh but runs OUR replay sweep instead of the aiperf client.
#
# IMPORTANT pathing: the benchmark container only has the srt-slurm
# default_mounts (/configs/dynamo-wheels, /aiperf_mmap_cache) plus the repo
# workspace at $INFMAX_CONTAINER_WORKSPACE (=/infmax-workspace) and the managed
# output dir /logs. /mnt/home is NOT mounted here, so we read the replayer +
# dataset from the workspace mount and write results to /logs (collected by
# srtctl into outputs/<job>/logs/).

WS="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
REPLAY_DIR="$WS/utils/custom_replay"
DATASET="${DATASET:-$REPLAY_DIR/batch_3x.replay.jsonl}"
CONCURRENCIES="${CONCURRENCIES:-4,8,16}"
mkdir -p "$RESULT_DIR"

# Discover the live OpenAI endpoint + the model id the frontend advertises,
# rather than hardcoding (the dynamo/nginx frontend port differs across the
# sglang vs vllm recipes). Probe PORT first, then the usual candidates.
read -r BASE MODEL_ID < <(PORT="${PORT:-}" python3 - <<'PY'
import json, os, urllib.request
cands = [os.environ.get("PORT", "")] + ["8000", "8888", "8080", "9000", "8001"]
seen = set()
for p in cands:
    if not p or p in seen:
        continue
    seen.add(p)
    try:
        with urllib.request.urlopen(f"http://localhost:{p}/v1/models", timeout=5) as r:
            mid = json.load(r)["data"][0]["id"]
            print(f"http://localhost:{p} {mid}")
            break
    except Exception:
        pass
PY
) || true

if [ -z "${BASE:-}" ]; then
    echo "[replay] ERROR: no OpenAI /v1/models endpoint found on candidate ports (PORT=${PORT:-unset})" >&2
    exit 1
fi
echo "[replay] endpoint=$BASE  model=$MODEL_ID  dataset=$DATASET  result=$RESULT_DIR"

# Replayer deps in an isolated venv (matches the colocated launcher pattern).
if python3 -m venv --system-site-packages "$RESULT_DIR/.rvenv" 2>/dev/null; then
    "$RESULT_DIR/.rvenv/bin/pip" install -q -r "$REPLAY_DIR/requirements.txt" || true
    REPLAY_PY="$RESULT_DIR/.rvenv/bin/python"
else
    python3 -m pip install --break-system-packages -q -r "$REPLAY_DIR/requirements.txt" || true
    REPLAY_PY=python3
fi

# Background Prometheus scrape for the whole run: the dynamo frontend at $BASE
# plus any worker engine /metrics in WORKER_METRICS_URLS (comma-separated; filled
# from the recipe once prefill/decode worker host:ports are known). Killed on exit.
SCRAPE_URLS="$BASE/metrics"
[ -n "${WORKER_METRICS_URLS:-}" ] && SCRAPE_URLS="$SCRAPE_URLS,$WORKER_METRICS_URLS"
echo "[replay] scraping metrics: $SCRAPE_URLS -> $RESULT_DIR/server_metrics.prom"
"$REPLAY_PY" "$REPLAY_DIR/scrape_metrics.py" --urls "$SCRAPE_URLS" \
    --out "$RESULT_DIR/server_metrics.prom" --interval 15 &
SCRAPE_PID=$!
trap 'kill "$SCRAPE_PID" 2>/dev/null || true' EXIT

"$REPLAY_PY" "$REPLAY_DIR/sweep_pareto.py" \
    --dataset "$DATASET" \
    --base-url "$BASE" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_ID" \
    --concurrencies "$CONCURRENCIES" \
    --warmup \
    --result-dir "$RESULT_DIR"
