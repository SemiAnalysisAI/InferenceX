# Run the AIPerf Agentic Client Standalone

Run the aiperf client against an already-running OpenAI-compatible server.

## Install

Install the pinned PyPI nightly in a Python 3.11–3.13 virtual environment:

```bash
python3 -m venv .venv-aiperf
source .venv-aiperf/bin/activate
python -m pip install 'aiperf==0.13.0.dev20261027'
```

The nightly is required: released `aiperf==0.13.0` rejects the serving scenarios'
`phases[].concurrency.warmupRequestsPerLane` and `traceIdleGapCapSeconds` keys.

## Run

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
TOKENIZER="moonshotai/Kimi-K3"
DATASET="semianalysis_cc_traces_weka_062126"
CONC=8
OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUTPUT_DIR"
aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --streaming --use-server-token-count \
  --stats-interval 30 --random-seed 42 \
  --trace-idle-gap-cap-seconds 300 \
  --num-dataset-entries 393 --slice-duration 1.0 --no-gpu-telemetry \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --failed-request-threshold 0.10 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

The flags after `--concurrency` replicate the defaults the retired
`agentx-harness` wrapper injected via its `agentx` scenario; `--scenario` is
mandatory (without it aiperf silently no-ops the scenario and stops replaying
traces). Explicit CLI arguments and `AIPERF_*` environment variables override
them. Model, tokenizer, dataset, concurrency, and output remain caller inputs;
`--tokenizer-trust-remote-code` is needed only for tokenizers with custom code.

InferenceX's `python3 -m infx.bench agentic` wrapper defaults to the full,
uncapped corpus (`semianalysis_cc_traces_weka_062126`). Model prefixes beginning
with `dsr1`, `qwen3.5`, or `qwen3.8next` explicitly select the
256k-capped corpus (`semianalysis_cc_traces_weka_062126_256k`). New model families
use the full corpus unless added to `CAPPED_FAMILIES` in
`infx/bench/agentic/traces.py`. `WEKA_LOADER_OVERRIDE` can pin either loader
regardless of model family; it does not configure the standalone `agentx` CLI.

The default random seed is **42**. Keep this seed when reproducing AgentX results,
or pass `--random-seed 123` to use a different seed.

Allow additional time for dataset preparation, warmup, and draining requests.
To sweep, restart the server for every `CONC` value and use a fresh output
directory. Do not flush caches or reuse a live server across concurrency points. Match the target
recipe's duration and warmup settings when reproducing a result.

Add `--server-metrics "${SERVER_URL}/metrics"` if the server exposes metrics.

## Optional session-affinity flags

For DP-attention runs requiring router affinity, export these before
`agentx profile`. They add headers without replacing `X-Correlation-ID`.

```bash
# Adds X-Dynamo-Session-ID; subagents also get X-Dynamo-Parent-Session-ID.
# Only this option sends the parent ID, preserving forked-agent lineage.
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
# Also sends X-Session-ID, if a front-end router needs it.
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
```
