# Run the AgentX harness by itself

<div align="center">

**English** | [中文](agentx-standalone_zh.md)

</div>

Run AgentX against an already-running OpenAI-compatible server with the
`aiperf profile` command below. The client does not need GPUs, Slurm, GitHub
Actions, or an InferenceX server launcher. You start and configure the server
separately, as in [ATOM's AgentX recipe](https://github.com/ROCm/ATOM/blob/main/recipes/Agentic-Kimi-K3.md).

The replay settings here follow InferenceX's
[`build_replay_cmd()`](../benchmarks/benchmark_lib.sh) and
[runtime settings](../benchmarks/runtime_settings.sh). The checked-out source
wins if those settings change. For workflow-driven runs and evals, use
[Evaluation and AgentX Procedures](eval-agentx-procedures.md).

## Install the pinned client

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
network access to install packages and download the public trace dataset,
and access to the served model's tokenizer. The client can run on the server
host or another machine that can reach it.

From a directory where you want the checkout:

```bash
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git
git -C InferenceX submodule update --init --depth 1 inferencex-e2e/utils/aiperf
cd InferenceX/inferencex-e2e

uv venv --python 3.11 .venv-agentx
uv pip install --python .venv-agentx/bin/python \
  -e ./utils/aiperf 'datasets>=4.7.0'
source .venv-agentx/bin/activate

git rev-parse HEAD
git -C utils/aiperf rev-parse HEAD
aiperf --version
```

For an existing checkout, run the submodule update from its root, then the
commands beginning with `cd inferencex-e2e`. To reproduce a particular
InferenceX run, check out its revision before updating the submodule.
Use the AIPerf revision pinned by that checkout; a generic `pip install aiperf`
or a moving AIPerf branch is not the same version guarantee. This installs only
the replay client and its dependencies, without installing the InferenceX
orchestration tooling or changing the serving environment.

Keep the following commands in the same activated shell, in `inferencex-e2e/`.
If the tokenizer is gated, authenticate with Hugging Face before running.
`--tokenizer-trust-remote-code` permits tokenizer repository code to execute;
use a tokenizer source you trust.

## Point the client at your server

Edit these values to match the server you have already started:

```bash
export SERVER_URL="http://127.0.0.1:8000"
export SERVED_MODEL_NAME="moonshotai/Kimi-K3"
export TOKENIZER="moonshotai/Kimi-K3"
export DATASET="semianalysis_cc_traces_weka_062126"
export CONC=8
export DURATION=3600
export OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

curl --fail --silent --show-error "${SERVER_URL}/v1/models"
```

`SERVER_URL` is the base URL, without `/v1` or `/v1/chat/completions`.
For a remote server, replace `127.0.0.1` with its reachable address.
`SERVED_MODEL_NAME` must match a model ID returned by `/v1/models`;
`TOKENIZER` is a Hugging Face ID or a tokenizer directory on the client.
It need not be the server's filesystem path. For an authenticated endpoint,
add `--api-key "$OPENAI_API_KEY"` to AIPerf and the matching
`Authorization: Bearer` header to curl, with the key supplied securely.

Choose the corpus before profiling. The current
[`resolve_trace_source()`](../benchmarks/benchmark_lib.sh) mapping is:

| InferenceX model prefix | `DATASET` |
| --- | --- |
| `dsv4*`, `glm5.2*`, `glm5.3*`, `minimaxm3*`, `kimik3*` | `semianalysis_cc_traces_weka_062126` |
| Other prefixes, including Qwen3.5 and Qwen3.8-Flash-Next | `semianalysis_cc_traces_weka_062126_256k` |

The client downloads and caches the selected public Hugging Face dataset.
The `_256k` corpus is a different, prefiltered workload, not a server setting.
Keep the date-pinned corpus identical when comparing runs. If the server has
an explicit smaller context limit, add `--max-context-length <TOKENS>` to match
that limit and disclose the filtering; do not silently truncate prompts or
present a filtered run as the full-corpus result.

The server must support streaming chat completions, the scenario's
`ignore_eos=true` request field, and the selected context lengths. Configure
prefix caching, KV offload, parallelism, and speculative decoding in the
server's own recipe. This client command does not configure any of them.
For speculative InferenceX comparisons, follow the
[golden acceptance-length rules](../infx/golden_al_distribution/README.md);
synthetic acceptance is for throughput measurement, not correctness evals.

## Run one concurrency point

This example is a one-hour profile with the shared InferenceX replay settings.
When reproducing a particular run, match its recipe's duration and warmup grace
period. Dataset preparation, cache warmup, and draining add to its wall-clock time.

```bash
set -eo pipefail

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/inferencex-revision.txt"
git -C utils/aiperf rev-parse HEAD > "$OUTPUT_DIR/aiperf-revision.txt"
uv pip freeze --python .venv-agentx/bin/python > "$OUTPUT_DIR/client-packages.txt"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$SERVER_URL" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "$SERVED_MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --benchmark-duration "$DURATION" \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

`--concurrency` counts live session trees, including their subagents, rather
than imposing a fixed cap on simultaneous HTTP requests. Recorded assistant
responses construct later turns; live responses are measured but are not fed
back into the next prompt with the environment setting above.

The scenario enforces streaming, first-turn prefix cache busting,
`ignore_eos=true`, and a 10-second whole-system idle-gap cap. The explicit
`0.25`/`0.75` trajectory-start ratios, seed `42`, and 300-second per-tree idle
cap above match InferenceX's wrapper, not all of the generic AIPerf tutorial
defaults. Do not substitute those defaults when reproducing InferenceX.

Optional additions to the same command:

- **Prometheus metrics:** add `--server-metrics "${SERVER_URL}/metrics"` if the
  server exposes it. For a router, supply the workers' reachable metrics URLs
  after one `--server-metrics` flag. `--no-gpu-telemetry` disables AIPerf GPU
  telemetry, not these engine metrics; the standalone command does not launch
  InferenceX's separate power collector.
- **Chat-template accounting:** add `--apply-chat-template` when the matching
  server recipe uses it, as ATOM's example does. It enables chat-template-based
  client token accounting; it does not launch or configure the server.
- **Router affinity:** match the router's session-affinity configuration.
  See the pinned [AgentX routing guide](../utils/aiperf/docs/benchmark-modes/semianalysis-agentx-faq.md)
  before running through multiple replicas.

For a sweep, rerun the profile block once per `CONC`, assigning a new
`OUTPUT_DIR` each time. Wait for outstanding server requests to drain before
starting the next point. Restart the server when its per-concurrency recipe
changes. Keep the corpus, client revision, seed, and replay settings fixed;
record any server changes alongside each result.

For a short diagnostic only, change `DURATION` to `60` and add
`--unsafe-override`. Such a run is marked `submission_valid=false`.
The scenario's normal minimum is 900 seconds; passing that minimum alone
does not make a smoke test equivalent to the configured comparison run.

## Check and preserve the results

AIPerf prints the exported paths at completion. Keep the entire output
directory, including `aiperf.log`, `profile_export_aiperf.json`,
`profile_export.jsonl`, and any `server_metrics_export.*` files produced.
Exports can appear directly under `aiperf_artifacts/` or in per-run
subdirectories; use the actual paths reported in the log.

Run the same request-error check used by InferenceX:

```bash
python -m infx.results.agentic.validate_agentic_result \
  "$OUTPUT_DIR/aiperf_artifacts" \
  --failed-request-threshold 0.10
```

This checks completed requests and the failure fraction, not scenario validity.
Also inspect `metadata.submission_valid` and the accompanying validity details
in `profile_export_aiperf.json`, and review errors, token counts, throughput,
TTFT, and request latency. A `true` validity stamp does not verify that the
server recipe or every replay setting matches another run.

Record the server image digest/revision, full launch command, model/tokenizer
revision, hardware, parallelism, KV-cache/offload settings, and speculative
settings. The raw client exports are not an InferenceX dashboard submission:
this path does not run evals, produce the wrapper's normalized result JSON,
or upload results. Use the [result and ingestion guide](results-and-ingestion.md)
for that pipeline.

## If the run fails

- **Unknown scenario or flags:** check that the active `aiperf` executable
  comes from `.venv-agentx`, and that the submodule matches the checkout.
- **HTTP 400/404:** verify the served model ID, endpoint, context limit, and
  support for `ignore_eos`. Preserve the server's error response.
- **Dataset configuration timeout:** confirm Hugging Face connectivity,
  tokenizer access, writable cache space, and available client CPU/RAM.
  Both configuration timeouts above are 1,800 seconds.
- **Warmup timeout:** inspect server logs and request progress. The grace
  period is a drain deadline, not a fixed warmup sleep; some high-concurrency
  recipes explicitly use a longer one.
- **Nonzero exit or invalid submission:** retain the logs and validity
  details, fix the cause, and rerun. Do not treat a partially exported
  metrics file as a successful benchmark.
