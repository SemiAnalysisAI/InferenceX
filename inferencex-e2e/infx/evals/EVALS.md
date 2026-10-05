# Evals

Graded QA jobs (`gsm8k`, `gpqa`) catch accuracy regressions from parallelism,
concurrency, kernels, and other throughput optimizations. They run separately
from throughput. Selection lives in `mark_eval_entries()` in
`infx.matrix.generate`.

## Selection

- **Fixed-sequence, single-node:** 8k1k only, at the highest and median
  concurrency for every model, runner, framework, precision, TP, and decoding
  configuration.
- **Fixed-sequence, multi-node:** 8k1k only, with one job per parallelism
  topology at its highest eligible concurrency. Rows differing only by
  concurrency share a topology.
- **Agentic GSM8K (every model, including Kimi K3 and MiniMax M3):** selected
  by default as a separate eval-only job at the highest concurrency of each
  single-node group of model, runner, framework, precision, spec-decoding,
  dp-attn and image (the 8k1k keys plus image). TP/EP and KV offloading do not
  split groups. Multi-node rows use the highest eligible concurrency per
  topology; a deployment with no topology at concurrency 16 or above gets one
  eval at its highest concurrency. Throughput for every agentic point still
  runs. Scores use the same GSM8K floors in `thresholds.yaml` as
  fixed-sequence 8k1k evals.
- **Kimi K3 agentic, in addition:** every generated point automatically runs
  `kimi-vendor` with `kimi_tool_call_schema_full` (204 schema cases in two
  stream modes, 408 checks). The two-check smoke requires an explicit override.
- **MiniMax M3 agentic, in addition:** every generated point automatically
  runs `minimax-vendor` with `minimax_m3_full` (102 provider cases). The
  one-case smoke requires an explicit override.
- **BFCL:** explicit only. No automatic model mapping selects BFCL.

Every mode, including each sweep's combined changelog entries, schedules at
most one eval per InferenceX-app eval identity: model, runner, framework,
precision, spec-decoding, disaggregation, per-role TP/EP/DP-attention/workers,
eval suite, sequence lengths, and concurrency. The app stores one result per
identity in a run, so rows that differ only by image, KV offloading, or recipe
file would overwrite each other's result there, and InferenceX's rerun
deduplication would delete one raw result. The first row without KV offloading
keeps the eval; a batched multi-node eval keeps only its unclaimed
concurrencies. Throughput coverage is unchanged.

Generator eval modes:

- Default: throughput plus the fixed-sequence subset, the agentic GSM8K
  subset, and every automatically selected Kimi K3 or MiniMax M3 vendor eval.
- `--no-evals`: throughput only, including no automatic vendor evals.
- `--evals-only`: selected evals only.
- `--all-evals`: every eligible fixed-sequence and agentic eval. This is
  equivalent to `--evals-only --all-evals`. Multi-node fixed-sequence
  topologies run all `conc-list` values sequentially on one engine.
- `--trim-conc`: after eval selection, retain the minimum concurrency for each
  single-node or multi-node deployment shape and move that shape's selected eval
  to the retained row. Standalone eval-only rows (the Kimi K3 and MiniMax M3
  GSM8K evals) keep their own concurrency. This is the deployment smoke mode,
  not a throughput sweep.

Changelog entries use `evals-only: true` and `all-evals: true`. The `all-evals`
setting implies eval-only there. On PRs, the same names are modifier labels:
`all-evals` expands coverage without suppressing throughput, while `evals-only`
suppresses it. `all-evals` runs remain reusable, but `evals-only` and
`agentx-fast` runs are not.

Deduplication is scenario-aware: fixed-sequence coverage does not suppress
agentic coverage, and `all-evals` wins over default eval coverage.

### Tool-use support contract

The tool-use adapters are backend-independent clients of the local
OpenAI-compatible endpoint. The deployment-smoke target set contains every
generated Kimi K3 and MiniMax M3 agentic configuration in the NVIDIA and AMD
master configs, including their single-node and multi-node vLLM and
Dynamo-vLLM recipes. A configuration is verified only when the current PR head
launches its tool-aware endpoint, the matching vendor smoke and `bfcl_smoke`
complete their expected sample counts, and the native and
`inferencex-eval-v1` artifacts are collected without `integration_error`.

Infrastructure support does not mean every model must pass every quality
threshold. A completed result with a positive effective sample count can score
below its threshold and fail the quality gate without being a deployment
failure. Missing parser support, transport errors, timeouts, malformed output,
missing samples, and missing artifacts are infrastructure failures.

Generator coverage and static parser checks do not prove a live backend. Before
claiming complete deployment support, run both smoke suites on every row from
the matrices below at the current PR head. Run each full vendor or BFCL
model-quality suite on at least one matching deployment; these longer suites do
not need to repeat on every equivalent parser topology.

Generate the complete deployment-smoke matrices with:

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml configs/amd-master.yaml \
  --model-prefix kimik3 \
  --scenario-type agentic-coding \
  --evals-only --all-evals --trim-conc

uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml configs/amd-master.yaml \
  --model-prefix minimaxm3 \
  --scenario-type agentic-coding \
  --evals-only --all-evals --trim-conc
```

Capacity-limited campaigns can split a `test-config` result with `--conc` and
`--exp-names`. Each requested experiment name must match exactly one generated
row, so a shard cannot silently include another deployment that shares the same
configuration key and concurrency.

Kimi and MiniMax matrix rows select their full vendor suites automatically, including with
`--trim-conc`; that option trims deployment points, not schema cases. For an
explicit deployment smoke, override `eval-framework: kimi-vendor` and
`eval-suite: kimi_tool_call_schema`; for MiniMax, use `eval-framework: minimax-vendor`
and `eval-suite: minimax_m3_smoke`. Run `bfcl_smoke` explicitly as needed. Full suites use the same endpoint and
artifact paths; a smoke result does not establish full-suite quality.

### Artifact reuse

Full sweeps with default or `all-evals` eval selection may reuse their eval
artifacts. Source coverage is
authoritative. Raw `meta_env.json` identities must match `eval_results_all`,
and batched evals use `completed_eval_concs`. Policy drift is allowed, but
malformed metadata, duplicates, and raw/aggregate mismatches are not. See
[workflow reuse](../../../.github/workflows/README.md#reusing-an-approved-pr-full-sweep).

## How?

`python3 -m infx.bench eval` ([`infx/bench/eval/`](../bench/eval/__init__.py)) runs
the selected eval framework against a ready server. `e2e-tests.yml` defaults
`eval-framework` to `auto`, then reads the concrete framework and suite from each
eval matrix row. Fixed-sequence and opted-in generic agentic evals use
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`lm-eval`) with GSM8K. Workflow inputs can explicitly override the
matrix-selected framework or suite for manual diagnostics.

To run it by hand, use Python 3.10 or newer from `inferencex-e2e/`, normally inside
the serving container:

```text
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint URL --concurrency "N [N ...]" --stage-to DIR [--framework NAME]
```

The command requires `MODEL` and the `true`/`false` flags `EVAL_ONLY` and
`IS_MULTINODE`. `MODEL_NAME` is the served name sent in requests and defaults to
`MODEL`. The framework is `EVAL_FRAMEWORK`, else `--framework`, else `lm-eval`.
lm-eval also requires `OPENAI_API_KEY` (the workflows set `EMPTY`) and installs its
pinned harness into the running `python3` with `uv pip`. Each run writes into a fresh temporary
directory, then copies the allow-listed artifacts into `--stage-to` and writes
`meta_env.json` there, also when the eval fails. `EVAL_CONCURRENT_REQUESTS` and
`EVAL_RESULT_DIR` are no longer read.

The Kimi full suite runs automatically for every generated `kimik3` agentic point.
The matrix selects `eval-framework: kimi-vendor` and
`eval-suite: kimi_tool_call_schema_full`. To invoke the full suite manually from the
`inferencex-e2e/` directory after a server is ready:

```bash
export MODEL='<HF model ID>' MODEL_NAME='<served model identifier>'
export MODEL_PREFIX='<model prefix>' PORT='<server port>' CONC='<benchmark concurrency>'
export EVAL_ONLY=false IS_MULTINODE=false
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_SUITE=kimi_tool_call_schema_full
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency "$CONC" --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

Vendor suites do not use the concurrency for their requests. For them,
`--concurrency` only records the point's `conc` in `meta_env.json`.

For a short endpoint check, explicitly set `EVAL_SUITE=kimi_tool_call_schema`
instead (or the same `eval-suite` workflow input). Historical smoke artifacts
retain their original task name and sample counts; they are not full-suite
results.

| Kimi task | Selection | Unique schema cases | Reported checks (`n_eff`) |
| --- | --- | ---: | ---: |
| `kimi_tool_call_schema` | Explicit smoke | 1 | 2 |
| `kimi_tool_call_schema_full` | Automatic AgentX evaluation | 204 | 408 |

Each schema case runs once in streaming and once in non-streaming mode. A
smoke score of `1.0` therefore means 2/2 checks passed on one schema case.
Read the task and effective sample count with the score. Both tasks measure
tool-call argument schema conformance, not GSM8K accuracy or overall agent
quality. The full suite retains its `0.0` quality threshold: a completed score
is diagnostic, while missing outcomes and integration failures fail the job.

The framework selects a suite-specific subprocess adapter, while the suite
selects a case set understood by that adapter. Each adapter owns its endpoint
format, native report, metrics, and integration-failure policy. Kimi, MiniMax,
and BFCL are data entries in `PROVIDERS` in
[`infx/bench/eval/vendor.py`](../bench/eval/vendor.py), run by one generic runner
that provisions the verifier interpreter, installs the suite's pinned runtime,
runs the adapter under the suite deadline, and has the adapter write an
integration-error result when a step fails. There is no shared request or report
abstraction. Automatic selection chooses only the Kimi and MiniMax vendor cases;
BFCL remains an explicit workflow override.
Agentic eval jobs forward the matrix `spec-decoding` value, so MTP entries
launch their existing `*_mtp.sh` server instead of silently falling back to STP.

### Stock Kimi tool-call schema smoke

The smoke runs the unmodified
[MoonshotAI/Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier)
at commit `3dad65a760a8867cda72f6dd8848d876a4e851b4`. Each run downloads and
SHA256-verifies the fresh pinned GitHub source archive, then safely extracts
only the upstream pytest configuration, tool-call schema tests, and bundled
Walle cases. InferenceX does not install the verifier package or reimplement
its request, streaming, or validation logic.

System Python 3.12 or newer is preferred and used directly. On older images,
the runner provisions an isolated Python 3.12 virtual environment with `uv`
(from `PATH`, else Astral's standalone installer). `uv pip install --target`
installs the minimal pinned verifier runtime (`httpx[http2]`, `openai`,
`jsonschema`, and `pytest`) for the selected interpreter into a separate
temporary package directory, then runs upstream
`tests/tool_call_json_schema/test_tool_call_json_schema.py` with:

- the local OpenAI-compatible endpoint, `EMPTY` API key, and served model name;
- `--think-mode none` for other models, or `--think-mode opensource --thinking`
  for `dsv4`, plus `--selection object --max-cases 1 --max-tokens 2048`;
- the bundled Walle case directory and `--tool-json-report`.

The temporary Python runtime, package directory, and verifier checkout are
removed after both successful and failed runs.

The selection is `TestAdditionalProperties:1`, parametrized upstream in
non-streaming and streaming modes. Each mode runs once through the unchanged
upstream pytest harness. The unchanged native report remains one final outcome
per mode. It is uploaded as `kimi_vendor_report.json`, and
`infx/evals/kimi_vendor_eval.py` projects those two outcomes into the existing
eval result shape. Both outcomes are recorded with a `0.0`
`kimi_tool_call_schema` threshold, so model quality remains diagnostic. Setup,
timeout, and collection failures emit a zero-score result with error metadata.
The adapter's 900-second global timeout bounds the entire upstream pytest
process.

This smoke validates one object-schema tool call. It does not cover tool choice,
parallel calls, multi-turn execution, or general agent quality. Multi-value
batched concurrency is unsupported. Multi-node aggregate jobs run the same
two-check smoke against their OpenAI-compatible frontend when explicitly
selected. Eval-only launchers restore real block verification before submitting
recipes that otherwise use synthetic acceptance for throughput.

### Kimi full tool-call schema diagnostic

`kimi_tool_call_schema_full` runs the same pinned upstream test module with
`--selection all`. It evaluates all 204 selected Walle schema cases in
non-streaming and streaming modes, for 408 reported outcomes. Eight pytest
workers run without an adapter-level whole-suite deadline. The native report must declare
the exact selected suite and line identities, contain both modes for every
case, and reconcile all outcome counts before projection.

The Python adapter's optional `--timeout-seconds <positive-seconds>` argument
sets an explicit whole-suite deadline when needed. The smoke retains its
900-second default deadline. Stock verifier request timeouts, engine readiness
deadlines, and workflow/scheduler allocation limits still apply; removing the
full-suite adapter deadline does not create an unlimited GPU allocation.

Automatic Kimi K3 AgentX eval rows select this suite for both AMD and NVIDIA,
including single-node and multi-node deployments. Manual dispatch can also
select `eval-framework: kimi-vendor` and
`eval-suite: kimi_tool_call_schema_full`. Its threshold is `0.0`, so model
quality is diagnostic while setup, timeout, malformed-report, and integration
failures still fail through the standard zero-effective-sample error path. The
full suite reuses the smoke's pinned checkout, stock invocation, result
envelope, artifact staging, collector, and dashboard path.

### MiniMax provider compatibility smoke

The MiniMax smoke is an explicit one-case endpoint check. Automatic `minimaxm3`
agentic points select the full 102-case suite. To invoke the smoke manually from
the `inferencex-e2e/` directory against an already-ready server:

```bash
export MODEL='<HF model ID>' MODEL_NAME="<served MiniMax-M3 model identifier>"
export PORT='<server port>' CONC='<benchmark concurrency>'
export EVAL_ONLY=false IS_MULTINODE=false
export EVAL_FRAMEWORK=minimax-vendor
export EVAL_SUITE=minimax_m3_smoke
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency "$CONC" --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

`infx/evals/minimax_m3_smoke.json` is derived from
[MiniMax-AI/MiniMax-Provider-Verifier](https://github.com/MiniMax-AI/MiniMax-Provider-Verifier)
`sample.jsonl` at commit
`c899f95e17bfc4a338ddd4cb1638279125885e55`. The vendored fixture retains
the full upstream MIT copyright, permission, and warranty notice. It contains
only upstream zero-based row 71, an `expected_tool_call: true` request
exercising tool-call trigger and argument-schema validation.

Each run downloads and hash-verifies the pinned upstream `verify.py`, complete
`sample.jsonl`, and validator modules. InferenceX writes row 71 unchanged to a
temporary JSONL input and invokes the stock verifier with its documented CLI.
The invocation uses concurrency one, the stock 600-second request timeout and
three-retry setting, and the documented `--extra-body` override
`{\"temperature\":0,\"top_p\":1,\"max_tokens\":40960}`. A one-hour outer
process deadline bounds the stock harness without changing its request,
response, retry, or scoring code.

`minimax_vendor_report.json` and `minimax_vendor_results.jsonl` are the
unchanged stock summary and detailed result artifacts. The adapter additionally
writes exactly one timestamped `results_minimax_vendor_*.json` compatibility
artifact. Its `result_format` is `inferencex-eval-v1`, `eval_adapter` is
`minimax-provider-verifier`, task is `minimax_m3_smoke`, and primary metric is
`exact_match,strict-match`. A completed run records original and effective
sample counts of one. Its score is the minimum of the stock verifier's
tool-call match rate, tool-call schema accuracy, and one minus its
error-only-reasoning rate. A successful request that emits no tool calls has
zero schema accuracy, so it remains an effective model-quality result rather
than an integration failure. The `minimax_m3_smoke` threshold is `0.0`, so its
model quality remains diagnostic.

Setup, transport, timeout, malformed native output, and collection failures
emit a zero-effective-sample compatibility artifact with integration-error
metadata. A complete stock result remains a model-quality outcome rather than
an integration failure.

This is a fixed single-case provider compatibility smoke, not the full
102-case MiniMax Provider Verifier, BFCL, or a cross-model quality comparison.
It does not estimate the upstream dataset's aggregate rates, stochastic
pass-at-k behavior, streaming behavior, parallel-call behavior, multi-turn tool
execution, language following, scenario key-order recall, or general agent
quality.

### MiniMax M3 full provider diagnostic

`minimax_m3_full` runs automatically for every generated `minimaxm3` AgentX
eval point on AMD and NVIDIA, including single-node and multi-node deployments.
It covers all 102 rows in the pinned MiniMax Provider Verifier dataset; completed
quality scores remain diagnostic. It can also be selected explicitly:

```bash
export MODEL='<HF model ID>' MODEL_NAME='<served model identifier>'
export PORT='<server port>' CONC='<benchmark concurrency>'
export EVAL_ONLY=false IS_MULTINODE=false
export EVAL_FRAMEWORK=minimax-vendor
export EVAL_SUITE=minimax_m3_full
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency "$CONC" --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

The runner downloads only the eight source and validator files allowlisted in
`infx/evals/minimax_m3_full_eval.py` at commit
`c899f95e17bfc4a338ddd4cb1638279125885e55`, verifies each SHA256, and executes
the pinned `verify.py` once. It uses five workers, a 600-second request timeout,
three upstream retries, and a seven-hour whole-suite timeout. The workflow
retains at least one hour for artifact staging, score validation, and cleanup.

The native files are `minimax_vendor_report.json` and
`minimax_vendor_results.jsonl`. The compatibility result publishes task
`minimax_m3_full` with the native `tool_calls_match_rate`, requires exactly 102
successful result rows, and rejects transport failures or inconsistent
summaries. Its threshold is `0.0`, so this suite is diagnostic during the first
rollout. Setup, transport, timeout, malformed-output, and integration failures
still fail through the standard zero-effective-sample error path.

### BFCL V4 deterministic tool-use smoke

The BFCL smoke is opt-in for models served through an OpenAI-compatible
chat-completions endpoint. Select `eval-framework: bfcl` and
`eval-suite: bfcl_smoke` in `e2e-tests.yml`, or run it from `inferencex-e2e/`
against an already-ready server:

```bash
export MODEL='<HF model ID>' MODEL_NAME="<served model identifier>"
export PORT='<server port>' CONC='<benchmark concurrency>'
export EVAL_ONLY=false IS_MULTINODE=false
export EVAL_FRAMEWORK=bfcl
export EVAL_SUITE=bfcl_smoke
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency "$CONC" --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

The validator reads BFCL's declared `acc` metric from the compatibility result,
so workflows do not need a framework-specific metric override.

The runtime pins
[`bfcl-eval==2026.3.23`](https://pypi.org/project/bfcl-eval/2026.3.23/), built
from Gorilla commit
[`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`](https://github.com/ShishirPatil/gorilla/commit/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8).
It downloads the exact
[`bfcl_eval-2026.3.23-py3-none-any.whl`](https://files.pythonhosted.org/packages/ba/41/ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/bfcl_eval-2026.3.23-py3-none-any.whl)
and verifies SHA256
`3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a`
before installation. The integration follows the pinned
[vLLM perf-eval BFCL runner](https://github.com/vllm-project/perf-eval/blob/7ecb11405df86b202f4c5cca322bd133052fee82/lib/run_bfcl.py),
but uses a fixed four-case V4 partial evaluation:

| BFCL category | Exact upstream case ID | Projected task |
|---------------|------------------------|----------------|
| `simple_python` | `simple_python_141` | `bfcl_simple_python` |
| `multiple` | `multiple_38` | `bfcl_multiple` |
| `parallel` | `parallel_1` | `bfcl_parallel` |
| `irrelevance` | `irrelevance_0` | `bfcl_irrelevance` |

`uv` installs the verified wheel and its undeclared `soundfile==0.13.1` import
dependency into a temporary Python 3.10-or-newer virtual environment with system
site packages enabled, excluding every package the image already provides so the
image's existing Torch/Transformers stack is reused; it never mutates the global
Python environment. Those image versions win over BFCL's own pins, such as `numpy==1.26.4`.
The temporary environment and BFCL project root are removed after
the run. Once package installation finishes, evaluation is local-only: BFCL
skips its server setup and uses only the already-running local API root,
typically `http://127.0.0.1:$PORT/v1`. The OpenAI SDK appends
`/chat/completions`; the adapter base URL is not the full endpoint. BFCL does
not download a model or call a remote inference API.

The smoke fixes temperature to `0` and uses four BFCL worker threads. Request
construction, response interpretation, and retry behavior remain those of the
pinned stock BFCL OpenAI-completions handler and OpenAI SDK. The adapter only
registers the served model against that stock handler. A 900-second external
process deadline bounds the smoke; each full suite uses its declared deadline.
Dependency installation is separately bounded at 600 seconds. Dependency,
setup, transport, timeout, and collection failures write
zero-score artifacts with integration-error metadata and fail the runner
nonzero. A completed evaluation exits independently of model quality; the
workflow score-validation step applies the threshold afterward.

The endpoint must implement OpenAI chat completions at `/v1/chat/completions`,
accept `tools` and the tool-selection fields emitted by BFCL, and return the
served model's OpenAI tool-call shape. In particular, assistant tool calls need
function names and JSON-encoded `function.arguments`; the response must also
support a normal no-tool answer for the irrelevance case. Starting a nominally
OpenAI-compatible server is not sufficient if it cannot parse that model's
native tool-call syntax.

Configure the server's model-specific function-calling parser and, when the
model's default template does not render tools correctly, its tool-aware chat
template. For vLLM, automatic calls require `--enable-auto-tool-choice` plus
`--tool-call-parser`, with `--chat-template` when needed. SGLang uses its
corresponding `--tool-call-parser`; TensorRT-LLM uses `--tool_parser`, plus the
matching reasoning-parser option when the model requires one. Parser names are
engine-specific. Current common mappings are Kimi K3 (`kimi_k3`), MiniMax M3
(`minimax_m3` in vLLM/TRT-LLM and `minimax-m3` in SGLang), and DeepSeek V4
(`deepseek_v4` in vLLM/TRT-LLM and `deepseekv4` in SGLang). GLM-4.5 uses
`glm45` in vLLM/SGLang, while GLM-4.7 uses `glm47`; Qwen3-Coder uses
`qwen3_coder` in vLLM/SGLang, with `qwen3_xml` for vLLM's XML variant and
`qwen3` for the corresponding TensorRT-LLM parser. The model recipe and
installed engine version are authoritative; BFCL does not replace a missing or
mismatched parser/chat template.

`bfcl_report.json` is the native report. `results_bfcl.json` is the
`inferencex-eval-v1` compatibility result consumed by the eval command's artifact
staging, the workflow upload, the collector, and the score validator. It projects
the four-case aggregate as task `bfcl_smoke` and the four one-case diagnostic
tasks shown above. Every row uses lm-eval-compatible `acc,none` (plus
`acc_stderr,none`); BFCL workflows therefore validate with metric prefix
`acc,` rather than the default exact-match prefix.

The `bfcl_smoke` and four `bfcl_<category>` thresholds are `0.0`, so all five
scores remain diagnostic. Dependency, endpoint, timeout, malformed-output,
missing-sample, and integration failures still fail through the standard
zero-effective-sample path. BFCL reuses the existing eval job, upload paths,
aggregation, and validation instead of adding a parallel workflow or artifact
route.

#### BFCL V4 model-quality suites

Two explicit BFCL suites extend the four-case endpoint smoke into broader
model-quality diagnostics:

| Suite | Selected BFCL V4 categories | Requests |
|-------|-----------------------------|----------|
| `bfcl_vllm_minimax_m3` | `simple_python` (400), `multiple` (200), `parallel` (200), `parallel_multiple` (200) | 1000 |
| `bfcl_vllm_kimi` | The same 1000 single-turn cases plus 60 each from `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, and `multi_turn_long_context` | 1240 |

These are the model-specific non-live and multi-turn slices used by the pinned
BFCL vLLM integration, not every BFCL V4 leaderboard category. They exclude
the V4 agentic web-search and memory evaluations.

Select these suites explicitly with `eval-framework: bfcl`; `bfcl_smoke`
remains the framework default. Both suites use BFCL's OpenAI completions
handler against the local endpoint rather than a hosted-provider handler. They
fix temperature to `0.001` and retain the stock handler's request construction,
response interpretation, and retry behavior. A transport-only subclass pins
the OpenAI SDK to two retries and a 180-second per-attempt timeout. MiniMax uses
eight worker threads and a two-hour whole-suite timeout. Kimi uses 16 threads,
caps multi-turn cases at ten steps, and uses a four-hour whole-suite timeout.

The adapter builds a deterministic run-ID map from the pinned BFCL dataset.
Single-turn suites select every case in their named categories. The Kimi
multi-turn selection sorts each leaf category and takes its first 60 cases.
Although upstream BFCL evaluates these subsets with `partial_eval`, the
adapter rejects missing, unexpected, or duplicate result IDs and score headers
whose counts or accuracy do not reconcile with the selected corpus.

The compatibility result publishes `bfcl_vllm_minimax_m3` or
`bfcl_vllm_kimi` as the aggregate task and preserves per-category
`bfcl_<category>` tasks. Kimi also publishes a combined `bfcl_multi_turn`
task. Full-suite thresholds are `0.0`; they are diagnostic until repeated
hardware runs establish model, precision, and backend baselines. A completed
zero-score run therefore passes threshold validation, while dependency,
transport, timeout, malformed-output, and integration failures still fail.

`bfcl_upstream_artifacts.tar.gz` preserves the pinned upstream result and
failure-only score JSONL files, exact selected-ID map, file locks, provenance
manifest, and Apache 2.0 license copy for debugging and attribution.
The native `bfcl_report.json` includes the package version, wheel hash, source
revision, integration revision, per-category score headers,
case IDs, failure records, and sampling settings. The compatibility
`results_bfcl.json` remains the only input to the normal InferenceX eval
collector and dashboard path.

### Benchmark and eval flow

Every lane runs its eval with `python3 -m infx.bench eval` against the job's own
server. In combined mode (`RUN_EVAL=true`, `EVAL_ONLY=false`) the server starts,
throughput runs, and then the eval runs against the same server. In eval-only mode
(`EVAL_ONLY=true`) the server starts with its eval-only settings, throughput is
skipped, and the eval runs.

| Lane | Eval entrypoint | `--concurrency` | `--stage-to` |
|------|-----------------|-----------------|--------------|
| srt-slurm single-node | `post_eval.command` runs `benchmarks/single_node/srt_eval.sh <endpoint> /logs/infx-eval-exit-code` | `CONC` | The checkout root |
| srt-slurm multi-node | `post_eval.command` runs `benchmarks/multi_node/srt_eval.sh <endpoint> /infmax-workspace` | `EVAL_CONC` | `/logs/eval_results` |

Key eval modules in `infx/bench/eval/`:

| Module | Description |
|--------|-------------|
| `__init__.py` (`evaluate`) | The `eval` command. Selects the framework, checks `EVAL_SUITE`, waits for the chat route before vendor evals in eval-only jobs, batches lm-eval concurrencies, stages artifacts, writes `meta_env.json`, and sets the exit code |
| `lm_eval.py` | lm-eval framework. `install` installs the pinned harness commit, `context_length` and `native_context_length` size each request, and `run` drives `local-chat-completions` with the sitecustomize patch |
| `vendor.py` | One generic `run` for the Kimi, MiniMax, and BFCL entries in `PROVIDERS`. `_provision` finds `uv` and chooses the verifier interpreter (the image `python3` when new enough, otherwise a `uv` venv, and always a system-site-packages `uv` venv for BFCL). The `_prepare_kimi`, `_prepare_minimax`, and `_prepare_bfcl` hooks install the pinned runtimes with `uv pip`, and failed steps get integration-error results |
| `meta.py` | `build` and `write` produce `meta_env.json`. `_disaggregated` maps the multi-node `PREFILL_*`/`DECODE_*` topology, and `refresh` serves the host-side srt collector |
| `stage.py` | `copy` applies the artifact allow-list and the `_conc<N>` suffixes |
| `context.py` | `EvalContext` and `EvalOutcome`, the contract each framework's `run(ctx)` implements |

The exit code is the framework's (a child killed by signal N reports 128+N), else 1
when metadata or staging failed. A batched run exits 0 and records failed
concurrencies in `meta_env.json`. An invalid environment input or flag value exits 1
with an `ERROR:` line before anything is staged, and a malformed command line exits 2.

`EVAL_FRAMEWORK` is the orchestration-level selection and takes precedence over a
`--framework` argument. Without that environment variable, `--framework` selects the
framework, and the default is `lm-eval`.

### Single-node
Single-node jobs run through srt-slurm recipes. For a fixed-sequence eval-only job, `runtime_arguments` in `infx/srt_slurm/single_node.py` starts the server with the matrix `MAX_MODEL_LEN` (`isl + osl + 256`) as its context (`context-length` for SGLang, `max_seq_len` and `max_num_tokens` for TRT-LLM, `max-model-len` for vLLM and ATOM), and srt-slurm skips the benchmark stage. AgentX evals keep the recipe context and receive `MAX_MODEL_LEN=0`. lm-eval sizes each request from `EVAL_MAX_MODEL_LEN` when set, otherwise from `MAX_MODEL_LEN` capped at the model's native maximum (`0` means the native maximum), and falls back to 16384 when neither is known. The shim writes the eval's exit code to `/logs/infx-eval-exit-code`, and the srt collector fails the job unless it is `0`.

### Multi-node
Multi-node evals on AMD and NVIDIA Slurm clusters run through [srt-slurm](https://github.com/NVIDIA/srt-slurm) at the shared Git submodule revision at `utils/srt-slurm`. Native `post_eval.command` and `post_eval.passthrough_env` select the InferenceX eval entrypoint without modifying the upstream checkout.
- `do_sweep.py` skips the benchmark stage when `EVAL_ONLY=true`, runs `_run_post_eval()` directly
- In eval-only mode, uses the full `wait_for_model()` health check (same as benchmark stage) since the benchmark health check was skipped
- InferenceX always sets `post_eval.command` to `benchmarks/multi_node/srt_eval.sh` (single-node jobs use `benchmarks/single_node/srt_eval.sh`), which runs `python3 -m infx.bench eval` from the mounted workspace (`/infmax-workspace`). srt-slurm falls back to its own registered `lm-eval` runner only when `post_eval.command` is unset, which InferenceX launches never do, and that runner expects a shell helper this repository no longer ships. `EVAL_FRAMEWORK` and `EVAL_SUITE` reach the eval through `post_eval.passthrough_env`, so vendor frameworks need no hook changes
- Eval artifacts written to `/logs/eval_results/` inside the container, collected by `infx/launch/drivers/srt/collect.py` when `RUN_EVAL=true` or `EVAL_ONLY=true`
- The srt driver always collects server logs for debugging but skips benchmark result collection when `EVAL_ONLY=true`
- Env vars threaded: `RUN_EVAL`, `EVAL_ONLY`, `EVAL_FRAMEWORK`, `EVAL_SUITE`, `IS_MULTINODE`, `FRAMEWORK`, `PRECISION`, `MODEL_PREFIX`, `RUNNER_TYPE`, `RESULT_FILENAME`, `SPEC_DECODING`, `ISL`, `OSL`, `PREFILL_TP/EP/NUM_WORKERS/DP_ATTN`, `DECODE_TP/EP/NUM_WORKERS/DP_ATTN`, `MODEL_NAME`, `EVAL_CONC`

For multi-node `all-evals`, `EVAL_CONC` is a space-separated list. When it contains multiple values, `python3 -m infx.bench eval` runs those concurrency points sequentially against the same live engine, stages each result with a `_concN` filename suffix, and records expected/completed/failed points in `meta_env.json`.

### Workflow structure
- `e2e-tests.yml`: `test-sweep-evals` (single-node fixed-seq-len), `test-sweep-multi-node-evals`
  (multi-node fixed-seq-len), `test-sweep-agentic-evals` (single-node agentic), and
  `test-sweep-multi-node-agentic-evals` (multi-node agentic)
- `run-sweep.yml`: `sweep-evals`, `sweep-multi-node-evals`, `sweep-agentic-evals`, and
  `sweep-multi-node-agentic-evals` (same four-way split)
- All four use their respective benchmark templates (`benchmark-tmpl.yml` for single-node,
  `benchmark-multinode-tmpl.yml` for multi-node) with `eval-only: true`, `run-eval: true`
- `collect-evals` depends on all four eval jobs; `collect-results` only runs when benchmark jobs ran
- `infx.matrix.plan` splits eval results by node count and scenario type into `evals`
  (single-node fixed-seq-len), `agentic_evals` (single-node agentic), `multinode_evals`
  (multi-node fixed-seq-len), and `multinode_agentic_evals` (multi-node agentic)

### Result collection

Eval results are collected by `.github/workflows/collect-evals.yml`:

1. Downloads all `eval_*` artifacts
2. Runs `infx/results/collect_eval_results.py` to aggregate results
3. Outputs `agg_eval_<exp_name>.json` with all eval metrics
4. Publishes a summary table to GitHub Step Summary

Fetch and inspect eval results:

```bash
# Download eval results artifact
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n eval_results_all -D ./evals

# View eval summary
cat ./evals/agg_eval_all.json | jq -r '
  .[] | [.hw, .framework, .precision, .tp, .conc, .task,
    (if .infrastructure_success then ((.score * 100 | round) / 100)
     else .integration_error.type end)]
  | @tsv' | column -t

# Filter to specific hardware
cat ./evals/agg_eval_all.json | jq '[.[] | select(.hw == "B200")]'
```

### Metrics

| Field | Description |
|-------|-------------|
| `score` | Primary metric (exact match for GSM8K) |
| `em_strict` | Strict exact match (requires `####` format) |
| `em_flexible` | Flexible extraction (looser number matching) |
| `n_eff` | Number of samples evaluated |
| `task` | Eval task name (e.g., `gsm8k`) |
| `eval_suite` | Explicit suite identity used for collection and artifact reuse |
| `infrastructure_success` | `false` when setup, transport, timeout, sample-count, or score validation failed |
| `integration_error` | Structured infrastructure failure type and message, otherwise `null` |

Collection retains the latest attempt for each artifact or batched concurrency.
Raw compatibility artifacts encode infrastructure failures with `score: 0`,
`n_eff: 0`, and `integration_error`. Aggregation preserves the failure row but
sets `score: null` and `infrastructure_success: false`, so dashboards cannot
mistake an endpoint failure for measured model quality. An older successful
attempt cannot replace a newer failed retry.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVAL` | `false` | Enable eval after throughput benchmark |
| `EVAL_ONLY` | `false` | Skip throughput, only run evals (set by workflow). The eval command requires `true` or `false` |
| `EVAL_FRAMEWORK` | Workflow: `auto`; eval command: `lm-eval` | Eval runner (`lm-eval`, `kimi-vendor`, `minimax-vendor`, or `bfcl`). `auto` resolves from matrix metadata before reusable workflow dispatch |
| `EVAL_SUITE` | Matrix-selected for automatic vendor evals; otherwise the framework's default suite | Vendor suite selector, accepted only with `kimi-vendor`, `minimax-vendor`, or `bfcl`. `meta_env.json` records the completed suite as `eval_suite`, which for lm-eval is the task file's stem (`gsm8k`). Explicit workflow overrides remain supported |
| `EVAL_TASKS_DIR` | `infx/evals/gsm8k.yaml` | lm-eval task YAML (or task name), relative to `inferencex-e2e/` |
| `EVAL_MAX_MODEL_LEN` | unset | Explicit lm-eval request budget. Unset means `MAX_MODEL_LEN` capped at the model's native maximum, or 16384 when neither is known |
| `EVAL_CONC` | Workflow-selected; srt-slurm falls back to the highest benchmark concurrency | Multi-node eval concurrency, passed as `--concurrency`. A space-separated list enables sequential batched evals against one live engine |
| `EVAL_LIMIT` | empty | Limit eval to first N instances (smoke tests). Empty means the full set |

### Score validation
`infx/evals/validate_scores.py` checks eval results against thresholds in `infx/evals/thresholds.yaml`. Runs as a separate workflow step after artifact upload so results are preserved even if validation fails.

### Adding a new eval task

1. Create a task YAML in `infx/evals/` following the lm-eval task format.
2. Set `EVAL_TASKS_DIR=infx/evals/<your_task>.yaml` when running benchmarks.
3. Update `infx/results/collect_eval_results.py` if new metrics need extraction.

### Adding a provider verifier

1. Add a provider-specific adapter under `infx/evals/`.
2. Add a `Provider` entry, with its `Suite` specs and `prepare` hook, to `PROVIDERS`
   in `infx/bench/eval/vendor.py`. `infx.bench.eval.FRAMEWORKS` registers it with
   the eval command. Keep suite-specific request and report policy in the adapter.
3. Install dependencies with `uv pip` (`infx.bench.uv`) in a provider-specific isolated
   runtime from the `prepare` hook.
4. Emit `result_format: inferencex-eval-v1`, name the native report so it matches the
   staging allow-list in `infx/bench/eval/stage.py` and the workflow upload paths,
   set `EVAL_SUITE`, and add a threshold.
5. Keep the adapter's integration-error path stdlib-only and Python 3.10
   compatible. When provisioning fails, the runner invokes it under the image's
   `python3`.

### Runtime patches (`infx/evals/patches/`)

`infx.bench.eval.lm_eval` applies this standalone patch to the pinned lm-eval.

- `lm_eval_sitecustomize.py`: reasoning-token handling
  (extracts `reasoning_content` when `message.content` is empty) and TRT
  compatibility (no `{"type": "text"}` injection for non-HF tokenizers).
  Each lm-eval run copies it into a temp dir as `sitecustomize.py` on `PYTHONPATH`.

## Task files
The following files are task definitions from lm-eval. More information on changes lives within the files:
- `infx/evals/gsm8k.yaml`
- `infx/evals/gpqa_diamond.yaml`
