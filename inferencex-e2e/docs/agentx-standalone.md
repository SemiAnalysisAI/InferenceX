# Run AgentX-Harness Standalone

Run the AgentX client against an already-running OpenAI-compatible server.

## Install

Install the pinned PyPI release in a Python 3.11–3.13 virtual environment:

```bash
python3 -m venv .venv-agentx
source .venv-agentx/bin/activate
python -m pip install 'agentx-harness==1.0.14'
```

The package version matches the `agentx-v1.0.14` GitHub release. The `agentx`
command selects the `agentx` scenario by default.

## Run

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="deepseek-ai/DeepSeek-V4.1-Flash"
TOKENIZER="deepseek-ai/DeepSeek-V4.1-Flash"
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

## Snapshot a published curve's recipes

To find the recipe behind each point of a published curve, run from `inferencex-e2e/`:

```bash
uv run --locked python -m infx.results.published_curve --model GLM-5.3 \
  --hf-model nvidia/GLM-5.3-NVFP4 --hardware gb300 --framework dynamo-sglang \
  --precision fp4 --agentic --output-dir published-glm5.3-gb300
```

`--model` takes a dashboard model or a raw model key; the dashboard pools `glm5.2` and
`glm5.3` under GLM-5.2, so the snapshot holds the pooled rows and each keeps its own `model`.
`--hf-model` names the model key by the checkpoint the matrix runs. `snapshot.json` lists
every row of the family from `/api/v1/benchmarks` with its curve, Pareto-frontier membership,
physical GPU count, tok/$ and engine source, plus the model's evaluations. Rows of the
requested curve also carry their producing commit from `/api/v1/workflow-info` and the
variant of the matrix point at that commit with the row's recipe fingerprint and concurrency.
`recipes/` holds each variant as srtctl expands it and `sources/<commit>/` its recipe file as
committed. A row that cannot be bound keeps its values, commit and run URL, with the reason in
`unbound` and `hints` of where to look at its commit. The API contract is
[openapi.json](https://inferencex.semianalysis.com/api/openapi.json).

`on_frontier` marks the dashboard's frontier: P90 interactivity against tok/s/GPU on each line
the chart draws. To mark a frontier on other axes too, add `--frontier X,Y` once per pair, e.g.
`--frontier p90_e2el,tok_per_usd`; each row then has `frontiers: {"p90_e2el,tok_per_usd": ...}`,
null where it lacks either metric. The axes are the median, p75 or p90 `intvty`, `ttft` or
`e2el`, `tput_per_gpu`, `input_tput_per_gpu`, `output_tput_per_gpu`, `tput_per_physical_gpu`
and `tok_per_usd`; latencies count lower as better, the rest higher.

The frontier, `physical_gpus`, `tput_per_physical_gpu` and `tok_per_usd` are computed by the
script from each row's published fields to match the dashboard's charts. They are not read from
the dashboard, so they can differ from it if its formulas change.

Binding fetches each producing commit and runs that commit's own matrix generator, so it
executes code from GitHub on your machine. Only runs of `--repository` (default
`SemiAnalysisAI/InferenceX`) are bound, but a commit fetched through its pull request head may
come from a fork.
