# Run AgentX with AIPerf Standalone

Run the AgentX client against an already-running OpenAI-compatible server.

## Install

Install the pinned upstream AIPerf commit in a Python 3.11–3.13 virtual environment:

```bash
python3 -m venv .venv-agentx
source .venv-agentx/bin/activate
python -m pip install 'aiperf @ https://github.com/ai-dynamo/aiperf/archive/8aa7039f872cd029f05a4c313fb8759e9a482cd3.tar.gz'
```

The commit matches `inferencex-e2e/infx/bench/agentic/requirements.txt`.

## Run

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
TOKENIZER="moonshotai/Kimi-K3"
DATASET="semianalysis_cc_traces_weka_062126"
CONC=8
OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "$OUTPUT_DIR"
aiperf profile --scenario inferencex-agentx-mvp \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration 3600 \
  --endpoint-type chat --streaming --stats-interval 30 --random-seed 42 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --trace-idle-gap-cap-seconds 300 --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --failed-request-threshold 0.10 --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

The `inferencex-agentx-mvp` scenario locks the replay invariants: agentic replay
timing, streaming, `ignore_eos`, no input truncation, first-turn cache busting,
the accepted trace loaders and a 900-second minimum profile. The remaining flags
and `AIPERF_*` variables are the AgentX run settings InferenceX passes for every
point; keep them when reproducing a result. Model, tokenizer, dataset,
concurrency, and output remain caller inputs; `--tokenizer-trust-remote-code` is
needed only for tokenizers with custom code.

InferenceX's `python3 -m infx.bench agentic` wrapper defaults to the full,
uncapped corpus (`semianalysis_cc_traces_weka_062126`). Model prefixes beginning
with `dsr1`, `qwen3.5`, or `qwen3.8next` explicitly select the
256k-capped corpus (`semianalysis_cc_traces_weka_062126_256k`). New model families
use the full corpus unless added to `CAPPED_FAMILIES` in
`infx/bench/agentic/traces.py`. `WEKA_LOADER_OVERRIDE` can pin either loader
regardless of model family; it does not configure the standalone `aiperf` CLI.

The random seed is **42**. Keep this seed when reproducing AgentX results,
or pass `--random-seed 123` to use a different seed.

Allow additional time for dataset preparation, warmup, and draining requests.
To sweep, restart the server for every `CONC` value and use a fresh output
directory. Do not flush caches or reuse a live server across concurrency points. Match the target
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

AIPerf sends `X-Session-Affinity` with the correlation ID by default; set
`AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID=false` to omit it.
