# Evals

Graded QA jobs (`gsm8k`, `gpqa`) catch accuracy regressions from parallelism,
concurrency, kernels, and other throughput optimizations. They run separately
from throughput. Selection lives in `mark_eval_entries()` in
`utils/matrix_logic/generate_sweep_configs.py`.

## Selection

- **Single-node:** 8k1k only, at the highest and median concurrency for every model,
  runner, framework, precision, TP, and decoding configuration.
- **Multi-node:** 8k1k only, with one job per parallelism topology at its highest
  eligible concurrency. Rows differing only by concurrency share a topology.
- **Agentic (GSM8K), single-node:** highest-conc entry per (model, runner,
  framework, precision) group.
- **Agentic (GSM8K), multi-node:** highest eligible concurrency per
  parallelism topology.

Generator eval modes:

- Default: throughput plus the selected eval subset.
- `--no-evals`: throughput only.
- `--evals-only`: selected evals only.
- `--all-evals`: every eligible fixed-sequence and agentic eval. This is
  equivalent to `--evals-only --all-evals`. Multi-node fixed-sequence
  topologies run all `conc-list` values sequentially on one engine.
- `--trim-conc`: after eval selection, retain the minimum concurrency for each
  single-node or multi-node deployment shape and move that shape's selected eval
  to the retained row. This is the deployment smoke mode, not a throughput sweep.

Changelog entries use `evals-only: true` and `all-evals: true`. The `all-evals`
setting implies eval-only there. On PRs, the same names are modifier labels:
`all-evals` expands coverage without suppressing throughput, while `evals-only`
suppresses it. Modifier runs cannot be reused.

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
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml configs/amd-master.yaml \
  --model-prefix kimik3 \
  --scenario-type agentic-coding \
  --evals-only --all-evals --trim-conc

uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml configs/amd-master.yaml \
  --model-prefix minimaxm3 \
  --scenario-type agentic-coding \
  --evals-only --all-evals --trim-conc
```

Capacity-limited campaigns can split a `test-config` result with `--conc` and
`--exp-names`. Each requested experiment name must match exactly one generated
row, so a shard cannot silently include another deployment that shares the same
configuration key and concurrency.

Run each generated matrix with the matching vendor smoke and `bfcl_smoke`.
The full Kimi, MiniMax, and BFCL suites use the same endpoint and artifact
paths, but are diagnostic model-quality campaigns rather than a replacement
for the per-topology deployment smoke.

### Artifact reuse

Default full sweeps may reuse their eval subset. Source coverage is
authoritative. Raw `meta_env.json` identities must match `eval_results_all`,
and batched evals use `completed_eval_concs`. Policy drift is allowed, but
malformed metadata, duplicates, and raw/aggregate mismatches are not. See
[workflow reuse](../../.github/workflows/README.md#reusing-an-approved-pr-full-sweep).

## How?
`run_eval` in `benchmarks/benchmark_lib.sh` dispatches to the selected eval
runner. Existing jobs continue to use lm-eval with GSM8K by default.

The default eval framework is [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`). Agentic eval-only matrix jobs inherit this default and therefore run the same GSM8K task as 8k1k. Explicit agentic runs can still select SWE-bench.

The Kimi smoke is opt-in and applies to supported models exposing a
tool-aware OpenAI-compatible chat-completions API. Select
`eval-framework: kimi-vendor` and `eval-suite: kimi_tool_call_schema` on
`e2e-tests.yml`, or invoke it from the repository root after a server is ready:

```bash
source benchmarks/benchmark_lib.sh
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_SUITE=kimi_tool_call_schema
export EVAL_RESULT_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
run_eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py
```

The framework selects a suite-specific subprocess adapter, while the suite
selects a case set understood by that adapter. Each adapter owns its endpoint
format, dependencies, native report, metrics, and integration-failure policy.
Kimi, MiniMax, and BFCL use separate explicit `run_eval` cases rather than a
shared request or report abstraction.
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
the runner uses the existing system `pip` to install pinned `uv==0.11.33` under
a temporary prefix, then provisions an isolated Python 3.12 virtual environment.
The selected interpreter installs the minimal pinned verifier runtime
(`httpx[http2]`, `openai`, `jsonschema`, and `pytest`) into a separate
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
`utils/evals/kimi_vendor_eval.py` projects those two outcomes into the existing
eval result shape. Both must pass, so the `kimi_tool_call_schema` threshold is
`1.0`. Setup, timeout, and collection failures emit a zero-score result with
error metadata. The adapter's 900-second global timeout bounds the entire
upstream pytest process.

This smoke validates one object-schema tool call. It does not cover tool choice,
parallel calls, multi-turn execution, or general agent quality. Multi-value
batched concurrency is unsupported. Multi-node aggregate jobs run the same
two-case smoke against their OpenAI-compatible frontend. Eval-only launchers
restore real block verification before submitting recipes that otherwise use
synthetic acceptance for throughput.

### Kimi full tool-call schema diagnostic

`kimi_tool_call_schema_full` runs the same pinned upstream test module with
`--selection all`. It evaluates all 204 selected Walle schema cases in
non-streaming and streaming modes, for 408 reported outcomes. Eight pytest
workers share a two-hour whole-suite timeout. The native report must declare
the exact selected suite and line identities, contain both modes for every
case, and reconcile all outcome counts before projection.

Select it explicitly with `eval-framework: kimi-vendor` and
`eval-suite: kimi_tool_call_schema_full`. Its threshold is `0.0`, so model
quality is diagnostic while setup, timeout, malformed-report, and integration
failures still fail through the standard zero-effective-sample error path. The
full suite reuses the smoke's pinned checkout, stock invocation, result
envelope, artifact staging, collector, and dashboard path.

### MiniMax provider compatibility smoke

The Phase 1 MiniMax smoke is opt-in and applies to supported models exposing
an OpenAI-compatible chat-completions API. Select
`eval-framework: minimax-vendor` and `eval-suite: minimax_m3_smoke` in
`e2e-tests.yml`, or run it from the repository root against an already-ready
server:

```bash
source benchmarks/benchmark_lib.sh
export EVAL_FRAMEWORK=minimax-vendor
export MODEL_NAME="<served MiniMax-M3 model identifier>"
export EVAL_SUITE=minimax_m3_smoke
export EVAL_RESULT_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
run_eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py
```

`utils/evals/minimax_m3_smoke.json` is derived from
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
than an integration failure. The `minimax_m3_smoke` threshold remains `1.0`.

Setup, transport, timeout, malformed native output, and collection failures
emit a zero-effective-sample compatibility artifact with integration-error
metadata. A complete stock result below the threshold remains a model-quality
outcome rather than an integration failure.

This is a fixed single-case provider compatibility smoke, not the full
102-case MiniMax Provider Verifier, BFCL, or a cross-model quality comparison.
It does not estimate the upstream dataset's aggregate rates, stochastic
pass-at-k behavior, streaming behavior, parallel-call behavior, multi-turn tool
execution, language following, scenario key-order recall, or general agent
quality.

### MiniMax M3 full provider diagnostic

`minimax_m3_full` is an explicit, non-gating expansion of the smoke to all 102
rows in the pinned MiniMax Provider Verifier dataset:

```bash
source benchmarks/benchmark_lib.sh
export EVAL_FRAMEWORK=minimax-vendor
export EVAL_SUITE=minimax_m3_full
export MODEL_NAME='<served model identifier>'
run_eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py
```

The runner downloads only the eight source and validator files allowlisted in
`utils/evals/minimax_m3_full_eval.py` at commit
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
`eval-suite: bfcl_smoke` in `e2e-tests.yml`, or run it from the repository root
against an already-ready server:

```bash
source benchmarks/benchmark_lib.sh
export EVAL_FRAMEWORK=bfcl
export MODEL_NAME="<served model identifier>"
export EVAL_SUITE=bfcl_smoke
export EVAL_RESULT_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
run_eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py
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

The verified wheel and its undeclared `soundfile==0.13.1` import dependency are
installed into a temporary Python 3.10-or-newer virtual environment with system
site packages enabled so the image's existing Torch/Transformers stack can be
reused; it never mutates the global Python environment. The temporary
environment and BFCL project root are removed after
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
`inferencex-eval-v1` compatibility result consumed by the existing artifact
upload, `append_lm_eval_summary`, collector, and score validator. It projects
the four-case aggregate as task `bfcl_smoke` and the four one-case diagnostic
tasks shown above. Every row uses lm-eval-compatible `acc,none` (plus
`acc_stderr,none`); BFCL workflows therefore validate with metric prefix
`acc,` rather than the default exact-match prefix.

Only `bfcl_smoke` gates the run: its `0.75` threshold requires at least three
of the four fixed upstream cases to be correct. The four `bfcl_<category>`
thresholds are `0.0`, so their one-case scores remain diagnostic and a single
failed category does not become a second gate. BFCL reuses the existing eval
job, upload paths, aggregation, and validation instead of adding a parallel
workflow or artifact route.

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

### Benchmark script flow

All benchmark scripts in `benchmarks/` follow one of two flows:

```bash
# Combined mode (benchmark + eval):
# 1. Start server (with context-length expansion if EVAL_ONLY=true)
# 2. wait_for_server_ready
# 3. run_benchmark_serving (skipped automatically when EVAL_ONLY=true)
# 4. Run evals:
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary  # Writes meta_env.json and stages artifacts
fi

# Eval-only mode (EVAL_ONLY=true):
# 1. Compute eval context via compute_eval_context_length
# 2. Start server with that context (--context-length or --max-model-len)
# 3. wait_for_server_ready
# 4. run_benchmark_serving returns immediately (skipped)
# 5. run_eval + append_lm_eval_summary
```

Key eval functions in `benchmarks/benchmark_lib.sh`:

| Function | Description |
|----------|-------------|
| `run_eval` | Unified entrypoint - dispatches to framework-specific runner |
| `run_lm_eval` | Runs lm-eval harness against the OpenAI-compatible endpoint |
| `run_kimi_vendor_eval` | Selects and runs a pinned Kimi Vendor Verifier suite |
| `run_minimax_vendor_eval` | Selects the pinned MiniMax smoke or full diagnostic |
| `run_bfcl_eval` | Selects a pinned BFCL V4 smoke or model-quality suite |
| `append_lm_eval_summary` | Writes `meta_env.json` and stages eval artifacts in the workspace |
| `_install_lm_eval_deps` | Installs lm-eval dependencies |
| `_prepare_vendor_verifier_python` | Uses system Python 3.12+ or provisions an isolated pinned Python 3.12 runtime for provider verifiers |
| `_prepare_kimi_vendor_runtime` | Installs the pinned verifier dependencies in an isolated temp path |
| `_prepare_minimax_m3_full_runtime` | Downloads hash-verified stock MiniMax sources and installs their pinned dependencies for smoke and full suites |
| `_prepare_bfcl_runtime` | Installs the verified BFCL wheel in a temporary virtual environment |
| `_install_bfcl_eval_deps` | Downloads, verifies, and installs the pinned BFCL wheel |
| `_prepare_kimi_vendor_verifier` | Downloads, hash-verifies, and safely extracts a fresh subset of the pinned source archive |
| `_patch_lm_eval` | Patches lm-eval for reasoning tokens and TRT compatibility |
| `compute_eval_context_length` | Computes eval context length (requested benchmark context, capped at model native max) |
| `get_native_max_context_length` | Extracts model's native max context length from HF config |

`EVAL_FRAMEWORK` is the orchestration-level selection and takes precedence over
legacy `--framework lm-eval` arguments embedded in fixed-sequence recipes.
Without that environment variable, an explicit `--framework` argument takes
precedence over the scenario default.

### Single-node
For default lm-eval jobs in eval-only mode (`EVAL_ONLY=true`), the benchmark script computes `EVAL_MAX_MODEL_LEN` via `compute_eval_context_length`, starts the server with that context length, skips throughput, and runs lm-eval. Each framework wires that context differently (`--context-length` for SGLang, `--max_seq_len` for TRT-LLM).

### Multi-node
Multi-node evals support two hardware paths:

**MI355X (AMD)** — `benchmarks/multi_node/amd_utils/server_sglang.sh`
- Skips throughput when `EVAL_ONLY=true`
- Fixed-seq-len: runs lm-eval via `run_eval --framework lm-eval` against the router on port 30000
- Agentic-coding (disaggregated, `IS_AGENTIC=1`): follows the same GSM8K/lm-eval path via
  `run_eval --framework lm-eval`. Since there's no single "TP" for a disaggregated topology,
  and the workflow spells a couple of metadata fields differently
  (`PREFILL_DP_ATTN`/`DECODE_DP_ATTN`) than `append_lm_eval_summary` expects
  (`PREFILL_DP_ATTENTION`/`DECODE_DP_ATTENTION`), the agentic branch bridges those before
  calling `run_eval`; `append_lm_eval_summary` itself runs automatically inside `run_eval()`
  (same `EVAL_ONLY=true && IS_AGENTIC` auto-staging as single-node), not as a separate call.
- Concurrency uses workflow-provided `EVAL_CONC` when set, otherwise falls back to max of `BENCH_MAX_CONCURRENCY` (x-separated values)
- Eval artifacts copied to `/run_logs/slurm_job-*/eval_results/`
- `runners/launch_mi355x-amds.sh` skips benchmark result collection when `EVAL_ONLY=true` and uses `find` to locate eval results

**NVIDIA Slurm multi-node (GB200, GB300, B200, B300, H100, H200)** runs through [srt-slurm](https://github.com/NVIDIA/srt-slurm) on the `sa-submission-q2-2026` branch.
- `do_sweep.py` skips the benchmark stage when `EVAL_ONLY=true`, runs `_run_post_eval()` directly
- In eval-only mode, uses the full `wait_for_model()` health check (same as benchmark stage) since the benchmark health check was skipped
- The registered srt-slurm `lm-eval` post-runner sources InferenceX's `benchmark_lib.sh` from the mounted workspace (`/infmax-workspace`). Kimi-selected launches patch that hook to use generic `run_eval` dispatch while preserving lm-eval as the default.
- Eval artifacts written to `/logs/eval_results/` inside the container, collected by launch scripts
- NVIDIA Slurm launch scripts always collect server logs for debugging but skip benchmark result collection when `EVAL_ONLY=true`
- Env vars threaded: `RUN_EVAL`, `EVAL_ONLY`, `EVAL_FRAMEWORK`, `EVAL_SUITE`, `IS_MULTINODE`, `FRAMEWORK`, `PRECISION`, `MODEL_PREFIX`, `RUNNER_TYPE`, `RESULT_FILENAME`, `SPEC_DECODING`, `ISL`, `OSL`, `PREFILL_TP/EP/NUM_WORKERS/DP_ATTN`, `DECODE_TP/EP/NUM_WORKERS/DP_ATTN`, `MODEL_NAME`, `EVAL_CONC`

For multi-node `all-evals`, `EVAL_CONC` is a space-separated list. When it contains multiple values, `run_eval` runs those concurrency points sequentially against the same live engine, stages each result with a `_concN` filename suffix, and records expected/completed/failed points in `meta_env.json`.

### Workflow structure
- `e2e-tests.yml`: `test-sweep-evals` (single-node fixed-seq-len), `test-sweep-multi-node-evals`
  (multi-node fixed-seq-len), `test-sweep-agentic-evals` (single-node agentic), and
  `test-sweep-multi-node-agentic-evals` (multi-node agentic)
- `run-sweep.yml`: `sweep-evals`, `sweep-multi-node-evals`, `sweep-agentic-evals`, and
  `sweep-multi-node-agentic-evals` (same four-way split)
- All four use their respective benchmark templates (`benchmark-tmpl.yml` for single-node,
  `benchmark-multinode-tmpl.yml` for multi-node) with `eval-only: true`, `run-eval: true`
- `collect-evals` depends on all four eval jobs; `collect-results` only runs when benchmark jobs ran
- `process_changelog.py` splits eval results by node count and scenario type into `evals`
  (single-node fixed-seq-len), `agentic_evals` (single-node agentic), `multinode_evals`
  (multi-node fixed-seq-len), and `multinode_agentic_evals` (multi-node agentic)

### Result collection

Eval results are collected by `.github/workflows/collect-evals.yml`:

1. Downloads all `eval_*` artifacts
2. Runs `utils/collect_eval_results.py` to aggregate results
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
| `EVAL_ONLY` | `false` | Skip throughput, only run evals (set by workflow) |
| `EVAL_FRAMEWORK` | `lm-eval` | Eval runner (`lm-eval`, `swebench`, `kimi-vendor`, or `minimax-vendor`) |
| `EVAL_SUITE` | basename of `EVAL_TASKS_DIR`, else `gsm8k` | Provider suite selector and artifact identity. External overrides are supported by `kimi-vendor` and `minimax-vendor`; other runners derive it from their task |
| `EVAL_TASKS_DIR` | `utils/evals/gsm8k.yaml` | Path to lm-eval task YAML |
| `EVAL_RESULT_DIR` | `/tmp/eval_out-*` | Output directory for eval results |
| `EVAL_MAX_MODEL_LEN` | `16384` | Max context for eval (set by `compute_eval_context_length`) |
| `EVAL_CONCURRENT_REQUESTS` | `64` | Concurrent requests during eval. A space-separated list enables sequential batched evals against one live engine |
| `EVAL_LIMIT` | empty | Limit eval to first N instances (smoke tests). Empty means the full set |

### Score validation
`utils/evals/validate_scores.py` checks eval results against thresholds in `utils/evals/thresholds.yaml`. Runs as a separate workflow step after artifact upload so results are preserved even if validation fails.

### Adding a new eval task

1. Create a task YAML in `utils/evals/` following the lm-eval task format.
2. Set `EVAL_TASKS_DIR=utils/evals/<your_task>.yaml` when running benchmarks.
3. Update `utils/collect_eval_results.py` if new metrics need extraction.

### Adding a provider verifier

1. Add a provider-specific adapter under `utils/evals/`.
2. Add an explicit framework case in `run_eval`; keep suite-specific policy in
   that adapter's shell runner.
3. Install dependencies in a provider-specific isolated runtime.
4. Emit `result_format: inferencex-eval-v1`, preserve the native report in an
   explicitly uploaded suite-specific path, set `EVAL_SUITE`, and add a threshold.

### Runtime patches (`utils/evals/patches/`)

The benchmark helpers invoke these standalone scripts against pinned dependencies.
Source rewrites are anchor-checked, idempotent, and atomic.

- `lm_eval_sitecustomize.py` (`_patch_lm_eval`): reasoning-token handling
  (extracts `reasoning_content` when `message.content` is empty) and TRT
  compatibility (no `{"type": "text"}` injection for non-HF tokenizers).
  Copied into a temp dir as `sitecustomize.py` on `PYTHONPATH`.
- `patch_swebench_agent.py` (`_patch_swebench_agent`): mini-swe-agent/swe-rex
  sandbox lifecycle cleanup, budget-exhaustion submission fallback, and the
  [SWE-ReX #281](https://github.com/SWE-agent/SWE-ReX/pull/281) closed-stdin fix.
- `patch_swebench_scoring.py` (`_patch_swebench_scoring`): swebench Modal
  scorer reserved-CPU reduction + sandbox termination on instance completion.

### SWE-bench Lite (`--framework swebench`)

SWE-bench requires applying each generated patch and running repository tests.
The dedicated framework uses mini-swe-agent for agentic generation by default,
then scores predictions with the official SWE-bench harness. It emits
`exact_match,resolved` in the existing lm-eval result shape so collection and
validation remain shared with the other evals.

```bash
run_eval --framework swebench --port "$PORT"
append_lm_eval_summary
```

- Task metadata and single-shot prompt: `utils/evals/swebench_lite.yaml`.
- Scoring: `utils/evals/swebench_score.py` (diff extraction → `predictions.jsonl` →
  `python -m swebench.harness.run_evaluation` → resolved-rate → results JSON). Offline
  `--report` mode skips Docker for testing.
- Generation modes (`SWEBENCH_GEN_MODE`) include `agentic`, the default, which runs the
  mini-swe-agent loop against the local endpoint. Each instance's shell runs in a Modal
  sandbox via swe-rex, matching the real SWE-bench setting. The `single-shot` mode uses
  lm-eval with one prompt per instance. It provides a roughly 10% floor baseline and is
  kept only as an explicit debugging escape hatch. Agentic knobs include `SWEBENCH_AGENT_WORKERS`
  (default: the config's `CONC`, else 64), `SWEBENCH_AGENT_STEP_LIMIT` (250),
  `SWEBENCH_AGENT_CMD_TIMEOUT` (per command, 300s), `SWEBENCH_AGENT_TIMEOUT` (6h),
  `SWEBENCH_AGENT_SANDBOX_CPU` (unset = Modal default), and `SWEBENCH_MODAL_APP_NAME`
  (`infx-evals-swe`).
- Run size: an empty `EVAL_LIMIT` runs the full split of roughly 300 instances. A positive integer runs the
  first N as an explicit smoke-test slice. `EVAL_LIMIT=full` (or `0`) also selects the full split.
- Scoring knobs: `SWEBENCH_TASK_NAME` (selects the YAML), `SWEBENCH_MAX_WORKERS`,
  `SWEBENCH_EVAL_SANDBOX_CPU` (cores per scoring sandbox, default 2), `SWEBENCH_EVAL_TIMEOUT`
  (per-instance test timeout, default 900s), `SWEBENCH_NAMESPACE` (pass `""` on arm/Mac),
  `SWEBENCH_SKIP_SCORE=true` (generate-only), `SWEBENCH_USE_MODAL=true` (score on Modal remote
  sandboxes instead of local Docker, as used in CI). For Modal credentials, set
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (e.g. from a GitHub secret) or provide `~/.modal.toml`.
  If the file is absent, the env vars are bootstrapped into it automatically. The scoring dataset
  is derived from the YAML's `dataset_path`, which keeps generation and scoring aligned.
  If `SWEBENCH_DATASET` is set, it must match or the run fails fast.
- Scoring runs on Modal remote sandboxes in CI (`SWEBENCH_USE_MODAL=true`, no Docker on the GPU
  nodes). Local Docker scoring needs about 120 GB of disk. The `thresholds.yaml` gate is `0.50`,
  calibrated from full-split runs that scored 54%. Historical 50-instance slices scored 62–76%.

## Task files
The following files are task definitions from lm-eval. More information on changes lives within the files:
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`
- `utils/evals/swebench_lite.yaml` (generation only, scored by `swebench_score.py`)
