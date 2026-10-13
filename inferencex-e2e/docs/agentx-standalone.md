# Run AgentX-Harness Standalone

Run the AgentX client against an already-running OpenAI-compatible server.

## Install

Install the pinned PyPI release in a Python 3.11–3.13 virtual environment:

```bash
python3 -m venv .venv-agentx
source .venv-agentx/bin/activate
python -m pip install 'agentx-harness==1.0.7'
```

The package version matches the `agentx-v1.0.7` GitHub release. The `agentx`
command selects the `agentx` scenario by default.

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
agentx profile \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

The `agentx` preset supplies a one-hour profile, ten warmup requests per lane,
and shared reporting and runtime settings. Explicit CLI arguments and `AIPERF_*`
environment variables override preset defaults. Model, tokenizer, dataset,
concurrency, and output remain caller inputs; `--tokenizer-trust-remote-code` is
needed only for tokenizers with custom code.

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
