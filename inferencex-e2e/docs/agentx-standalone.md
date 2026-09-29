# Run AgentX-Harness Standalone

<div align="center">

**English** | [中文](agentx-standalone_zh.md)

</div>

Run the AgentX client against an already-running OpenAI-compatible server.

## Install

With Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed,
clone the harness at the pinned commit. Its CLI is named `aiperf`.

```bash
git clone --filter=blob:none --no-checkout https://github.com/SemiAnalysisAI/agentx-harness.git
cd agentx-harness
git checkout --detach 754356e9a39acc6cc6afb242d123bb57c3fb6f75
uv venv --python 3.11 .venv-agentx
uv pip install --python .venv-agentx/bin/python -e . 'datasets>=4.7.0'
source .venv-agentx/bin/activate
```

## Run

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
TOKENIZER="moonshotai/Kimi-K3"
DATASET="semianalysis_cc_traces_weka_062126"
CONC=8
OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "$OUTPUT_DIR"
aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --trace-idle-gap-cap-seconds 300 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

This profiles for one hour, plus dataset preparation, warmup, and drain time.
To sweep, change `CONC` and rerun with a fresh output directory after prior
requests drain; restart the server if its recipe changes. Match the target
recipe's duration and warmup settings when reproducing a result.

Add `--server-metrics "${SERVER_URL}/metrics"` if the server exposes metrics.

## Optional session-affinity flags

For DP-attention runs requiring router affinity, export these before
`aiperf profile`. They add headers without replacing `X-Correlation-ID`.

```bash
# Adds X-Dynamo-Session-ID; subagents also get X-Dynamo-Parent-Session-ID.
# Only this option sends the parent ID, preserving forked-agent lineage.
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
# Also sends X-Session-ID, if a front-end router needs it.
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
```
