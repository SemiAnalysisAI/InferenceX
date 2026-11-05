# Golden Acceptance-Length Distributions

English | [中文](README_zh.md)

This directory contains the golden acceptance-length (AL) curves used to standardize speculative decoding moving forward in InferenceX. Each YAML maps a model, thinking mode, and draft length (`num_speculative_tokens`) to a mean AL measured on the **coding** category of the SPEED-Bench Qualitative split, and records the [collection settings](#collection-settings) that produced those values.

## Why SPEED-Bench

[SPEED-Bench](https://arxiv.org/abs/2604.09557) is a unified benchmark for speculative decoding across diverse semantic domains and realistic serving regimes. Its Qualitative split contains 880 semantically diverse prompts, with 80 prompts in each of 11 categories. It is designed to measure acceptance rate (AR) and acceptance length (AL). Its Throughput splits cover fixed 1K–32K input lengths and multiple entropy regimes for system-level evaluation. The benchmark uses real prompts because random-token inputs can distort acceptance behavior, expert routing, and measured throughput.

SPEED-Bench is a practical cross-engine standard rather than an InferenceX-only workload:

- vLLM maintainers merged [native SPEED-Bench support](https://github.com/vllm-project/vllm/pull/36029) into `vllm bench serve` for Qualitative AR/AL measurement and Throughput evaluation. It is documented in the [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.22.0/benchmarking/cli/).
- SGLang exposes a native [`speed-bench` dataset adapter](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/speed_bench.py) and documents it in its [serving benchmark guide](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/developer_guide/bench_serving.mdx).
- The SPEED-Bench paper evaluates production-grade engines including vLLM, SGLang, and TensorRT-LLM, so its methodology is suitable for cross-runtime comparisons.

This upstream maintainer adoption is why InferenceX uses SPEED-Bench as the common measurement substrate for golden AL collection.

## Why the coding category

AL is workload-dependent: a draft model's predictions are easier to accept in some domains than others. Agentic coding accounts for the largest share of its token volume. We therefore calibrate synthetic acceptance against SPEED-Bench's `coding` category instead of averaging unrelated domains such as roleplay, translation, or creative writing.

## Fairness Guidelines for AgentX

Under the AgentX Guidelines, each model, thinking mode, and draft length has one committed golden AL. Once synthetic acceptance is enabled for a benchmark scenario, a submission may choose any supported draft length, but it may not substitute a different acceptance target. Different models keep their own SPEED-Bench-derived curves. All submissions evaluating the same model and mode use the same curve.

For Qwen3.5 srt-slurm launches, the srt driver runs the
`infx/srt_slurm/synthetic_acceptance.py` connector, which selects the committed curve
from the generation role's `speculative-num-steps` and `THINKING_MODE` after
caller overrides. The verification token is not an additional MTP step. For
SGLang MTP synthetic throughput, the connector rejects a generation role whose
`speculative-num-draft-tokens` is missing or is not an integer equal to
`speculative-num-steps + 1`. Manual acceptance-length settings are not needed.
Eval-only runs remove synthetic settings and use real verification.

TokenSpeed MTP recipes use the same automatic selection from
`speculative-num-steps`, with `speculative-num-draft-tokens` equal to the step
count plus one and `speculative-eagle-topk: 1`. The connector sets
`synthetic-acceptance-length` and the required `sampling-backend: greedy` for
AgentX throughput. Eval-only and fixed-sequence runs remove that length and
select `sampling-backend: flashinfer` for real verification. TokenSpeed's
connector accepts explicit MTP flags; other speculative algorithms and the
JSON `speculative-config` shorthand are rejected.

vLLM supports this through synthetic rejection sampling. For example, an EAGLE3 run can inject the selected YAML value through `synthetic_acceptance_length`:

```bash
vllm serve MODEL \
  --speculative-config '{
    "method": "eagle3",
    "model": "DRAFT_MODEL",
    "num_speculative_tokens": 4,
    "rejection_sample_method": "synthetic",
    "synthetic_acceptance_length": 3.24
  }'
```

The option was unified across vLLM model runners in [vllm-project/vllm#40662](https://github.com/vllm-project/vllm/pull/40662).

SGLang supports the same policy through its simulated-acceptance environment variables, set in the server environment (the `aggregated_environment` / `decode_environment` sections of srt-slurm YAMLs, or exported before launch in benchmark scripts):

```yaml
SGLANG_SIMULATE_ACC_LEN: '3.24'   # AL from the committed golden YAML
SGLANG_SIMULATE_ACC_METHOD: match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE: real-draft-token
```

TensorRT-LLM supports it through [`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`](https://github.com/NVIDIA/TensorRT-LLM/blob/2cbdaa0ffa36fbef7960a0ad9f0458373025fa9f/tensorrt_llm/_torch/speculative/interface.py#L1065-L1078). **Note the off-by-one:** this variable counts accepted *draft* tokens only and excludes the bonus/verification token, so set it to the golden AL **minus 1**. Fractional values are supported. The integer part is accepted every iteration, and the fractional part is the probability of accepting one additional draft token. For example, a golden AL of `3.5` becomes:

```bash
TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=2.5
```

ATOM supports it through `--spec-decode-acceptance-length`, which takes the golden AL directly:

```bash
python -m atom.entrypoints.openai_server \
  --model MODEL \
  --draft-model DRAFT_MODEL \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 3.78
```

Acceptance length counts the target's guaranteed verification token, so the golden AL goes in unchanged — no off-by-one — matching vLLM's `synthetic_acceptance_length` and SGLang's `SGLANG_SIMULATE_ACC_LEN`. The value must fall in `[1, num_speculative_tokens + 1]`, and it resolves to the same minimum-variance per-position schedule vLLM derives, so the accepted-length *distribution* matches and not merely its mean. Accepted positions emit the real draft tokens, equivalent to SGLang's `real-draft-token`. The knob has an equivalent rate spelling, `--spec-decode-acceptance-rate`, which takes `(AL - 1) / num_speculative_tokens`; the two are mutually exclusive. Support landed in [ROCm/ATOM#1948](https://github.com/ROCm/ATOM/pull/1948).

Read the realized value back from `average_tokens_per_forward` on the server's `/debug/mtp_stats`, or the `atom:mtp_average_tokens_per_forward` Prometheus metric. ATOM computes it as `1 + accepted draft tokens / verification steps`, the same formula used to collect the golden curves, so the two are directly comparable.

**DSpark note:** ATOM rejects forced acceptance combined with the DSpark confidence scheduler (`--dspark-config '{"confidence_schedule": true}'`), which sizes each request's verify length at runtime and can cap acceptance below the requested length with no way to detect it. The golden DSpark curves were themselves collected without adaptive verification, so leaving it off is also the comparable configuration.

llama.cpp supports it through `llama-server --spec-synth-len`, which takes the golden AL directly:

```bash
llama-server \
  --model MODEL.gguf \
  --model-draft MTP_HEAD.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-synth-len 2.48
```

srtctl runs no llama.cpp, so the launcher's [serve driver](../launch/drivers/serve.py), not the connector, sets this flag, and it refuses a recipe that sets `spec-synth-len` or `spec-synth-rates`.

This policy follows the same broad principle as MLPerf Inference: prescribe the workload rules needed for comparable system measurements. InferenceX is evaluating inference-system performance, not the ability to fine-tune a benchmark-specific speculative head.

## How a golden AL curve is collected

The push-button [`speedbench-al.yml`](../../../.github/workflows/speedbench-al.yml) workflow, introduced in [InferenceX#1650](https://github.com/SemiAnalysisAI/InferenceX/pull/1650) and extended to additional MTP and EAGLE3 models in [InferenceX#1706](https://github.com/SemiAnalysisAI/InferenceX/pull/1706), performs the following process. It superseded the early manually assembled reference in [InferenceX#1592](https://github.com/SemiAnalysisAI/InferenceX/pull/1592), making the exact commands, logs, outputs, and generated YAML auditable from one run.

1. A maintainer dispatches the workflow with a model, model prefix, serving framework (`vllm` or `sglang`), image, draft lengths (normally 1–8), thinking modes, `category=coding`, and `output-len=4096`.
2. The workflow launches the model on a B300 runner and selects the matching `<model-prefix>_fp4_b300_<framework>.sh` collector under [`benchmarks/single_node/speedbench/`](../../benchmarks/single_node/speedbench). A non-empty `collector-variant` input selects `<model-prefix>_fp4_b300_<framework>_<collector-variant>.sh` instead. For example, `model-prefix=kimik3` with `collector-variant=probabilistic_sample_method_block_rejection_sample_method` runs the Kimi K3 probabilistic-draft collector. The run fails before launch if the selected script does not exist.
3. For every `(thinking mode, draft length)` cell, the collector starts a clean vLLM or SGLang server with real speculative decoding and the model's production sampling/chat-template settings.
4. A vLLM collector snapshots vLLM's cumulative accepted-token and verification-draft counters, runs every prompt in the SPEED-Bench Qualitative `coding` category through `vllm bench serve`, and snapshots the counters again. The SGLang collector runs the same prompts through `sglang.bench_serving` and records the `accept_length` it reports.
5. A vLLM collector computes the mean acceptance length as:

   ```text
   AL = 1 + (delta accepted draft tokens / delta verification drafts)
   ```

   The `1` is the target model's guaranteed verification token. Values are rounded to two decimal places.
6. The collector emits a YAML matrix. The workflow publishes it in the GitHub Actions step summary and uploads it as a `speedbench-reference-al-<model-prefix>` artifact.
7. Server logs and detailed per-request results are retained so reviewers can confirm sensible output, correct thinking mode, and the absence of silent server or chat-template failures.
8. After review, the matrix is committed here with its exact sampling metadata, its [collection settings](#collection-settings), and source Actions run URL. With `open-pr=true` and `golden-curve=<curve stem>` (for example `deepseek-v4-pro__mtp__native__default-default`), the workflow writes `<curve stem>.yaml` here with the source run URL in its `source_run` key. It loads the file through `python -m infx.golden_al_distribution show` and then opens the PR. The whole file is replaced, so cells the run did not collect are dropped.

The collection measures real head quality only to establish the golden curve. AgentX then uses that committed curve as the synthetic acceptance target for every comparable submission.

## Collection settings

Each curve YAML records the SPEED-Bench client settings of its collection in a reserved top-level `collection` key beside the model key, and `load_curve(name).collection` returns them (`None` for a curve without the block). Every agentic speculative eval runs the report-only acceptance-length gate first: it replays these settings with real verification on the eval's own server and compares the measured AL with the golden AL; see [the speculative acceptance gate](../evals/EVALS.md#speculative-acceptance-gate).

```yaml
collection:
  renderer: chat_template  # or deepseek_v4, deepseek_v41
  modes:                   # one entry per thinking mode with AL values
    thinking_on:
      chat_template_kwargs: {enable_thinking: true}  # null when the collector passed none
      sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, presence_penalty: 0.0, repetition_penalty: 1.0}
```

- `renderer` is how the client turned each prompt into text. `chat_template` applies the served model's Hugging Face chat template to one user message, with the generation prompt and `chat_template_kwargs`. `deepseek_v4` and `deepseek_v41` are vLLM's tokenizer modes of those names; the gate uses the equivalent encoders in this repository ([V4](../bench_serving/encoding_dsv4.py), [V4.1](../bench_serving/encoding_dsv41.py)), with `chat_template_kwargs` as their thinking options. The V4.1 encoder is vLLM's from the week of the collection: later vLLM renders reasoning effort `high` as 75, not 50.
- `sampling` lists every sampling parameter the requests carried, including the `repetition_penalty: 1.0` that `vllm bench serve` adds. A parameter that is not listed was not sent, so the server used its own default.
- Everything else is the same for every collection: the 80 prompts of the SPEED-Bench Qualitative `coding` category, in the order vLLM's loader sends them, as streaming `/v1/completions` requests with `max_tokens` 4096 and no `ignore_eos`. Collections ran at different concurrencies; the gate always replays at 64, treating AL as independent of concurrency.
- New curves must include the block; the gate skips a curve without one. [`muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml`](muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml) was collected with SGLang's `bench_serving` client on its native `/generate` endpoint, which sends no `repetition_penalty`; the gate replays the same rendered prompts and sampling on `/v1/completions`. [`glm-5.3-nvfp4__mtp__native__default-default.yaml`](glm-5.3-nvfp4__mtp__native__default-default.yaml) was collected the same way and has no block yet.

## Reproducing a collection

Test the model-specific collector and image first, then dispatch the workflow from the branch containing that collector:

```bash
gh workflow run speedbench-al.yml \
  --repo SemiAnalysisAI/InferenceX \
  --ref BRANCH \
  -f runner=b300 \
  -f model=HF_MODEL_ID \
  -f model-prefix=MODEL_PREFIX \
  -f image=VLLM_IMAGE \
  -f 'mtp-list=1 2 3 4 5 6 7 8' \
  -f 'thinking-modes=off on' \
  -f category=coding \
  -f output-len=4096 \
  -f open-pr=false
```

Before accepting an updated curve, reviewers should verify:

- Every requested draft length and thinking mode completed.
- Detailed outputs are coherent and use the intended thinking mode.
- Server logs contain no fallback, draft-disable, or chat-template errors.
- The YAML metadata matches the dispatched image, sampling settings, model, and speculative method.
- The `collection` block matches the collector's concurrency, prompt rendering, and per-mode settings.
- `source_run` is the Actions run that produced the artifact.
- The committed values exactly match the workflow artifact.

## Current golden curves

| Model | Method | Golden YAML |
| --- | --- | --- |
| DeepSeek V4 Pro | MTP | [`deepseek-v4-pro__mtp__native__default-default.yaml`](deepseek-v4-pro__mtp__native__default-default.yaml) |
| DeepSeek V4 Pro 0813 | DSpark (probabilistic draft) | [`deepseek-v4-pro-0813__dspark__native__probabilistic-default.yaml`](deepseek-v4-pro-0813__dspark__native__probabilistic-default.yaml) |
| DeepSeek V4.1 Flash | DSpark (probabilistic draft, block verify) | [`deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml`](deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml) |
| Qwen3.5 397B-A17B | MTP | [`qwen3.5-397b-a17b-nvfp4__mtp__native__default-default.yaml`](qwen3.5-397b-a17b-nvfp4__mtp__native__default-default.yaml) |
| Kimi K3 | DSpark | [`kimi-k3__dspark__kimi-k3-dspark__default-default.yaml`](kimi-k3__dspark__kimi-k3-dspark__default-default.yaml) |
| Kimi K3 | DSpark (probabilistic draft, block verify) | [`kimi-k3__dspark__kimi-k3-dspark__probabilistic-block.yaml`](kimi-k3__dspark__kimi-k3-dspark__probabilistic-block.yaml) |
| Kimi K3 | DSpark (probabilistic draft) | [`kimi-k3__dspark__kimi-k3-dspark__probabilistic-default.yaml`](kimi-k3__dspark__kimi-k3-dspark__probabilistic-default.yaml) |
| Kimi K3.1 Pro | DSpark (probabilistic draft, block verify) | [`kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block.yaml`](kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block.yaml) |
| Kimi K3.1 Pro | DFLASH (RadixArk draft, block verify) | [`kimi-k3.1-pro__dflash__kimi-k3.1-dflash2-preview__default-block.yaml`](kimi-k3.1-pro__dflash__kimi-k3.1-dflash2-preview__default-block.yaml) |
| MiniMax-M3 | EAGLE3 | [`minimax-m3__eagle3__minimax-m3-eagle3__default-default.yaml`](minimax-m3__eagle3__minimax-m3-eagle3__default-default.yaml) |
| MiniMax-M3 | EAGLE3 (GQA) | [`minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default.yaml`](minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default.yaml) |
| MiniMax-M3 | EAGLE3 (GQA, block verify) | [`minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-block.yaml`](minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-block.yaml) |
| GLM-5.2 | MTP | [`glm-5.2-fp8__mtp__native__default-default.yaml`](glm-5.2-fp8__mtp__native__default-default.yaml) |
| GLM-5.3 | MTP (SGLang) | [`glm-5.3-nvfp4__mtp__native__default-default.yaml`](glm-5.3-nvfp4__mtp__native__default-default.yaml) |
| GLM-5.3-Flash | MTP | [`glm-5.3-flash__mtp__native__default-default.yaml`](glm-5.3-flash__mtp__native__default-default.yaml) |
| Qwen3.8-Flash-Next | MTP (native) | [`qwen3.8-flash-next-fp8__mtp__native__default-default.yaml`](qwen3.8-flash-next-fp8__mtp__native__default-default.yaml) |
| Qwen3.8-27B | MTP (native) | [`qwen3.8-27b-nvfp4__mtp__native__default-default.yaml`](qwen3.8-27b-nvfp4__mtp__native__default-default.yaml) ([placeholder](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/0)) |
| Muse Spark Open | DFlash2 | [`muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml`](muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml) |

## Using golden curves in srt-slurm runs

- Do not hard-code synthetic acceptance lengths in SRT recipes, master configs, or launchers. Do not add manual `SYNTHETIC_ACCEPTANCE_LENGTH`, vLLM `synthetic_acceptance_length`, SGLang `SGLANG_SIMULATE_ACC_LEN`, or TRT-LLM `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` settings. The engine examples above show what the connector injects, not what a recipe should set.
- Submit recipes through the srt driver (`infx/launch/drivers/srt/`). It runs the [`infx/srt_slurm` connector](../srt_slurm/synthetic_acceptance.py), which applies native SRT `--set` / `--unset` overrides. Calling upstream `srtctl` directly skips this selection.
- Keep the speculative method, draft model, draft-token count, and relevant sampling settings explicit in the recipe. The connector combines the generation role's settings (decode, otherwise aggregated), after caller overrides, with `MODEL_PREFIX` and `THINKING_MODE` to select the curve. For Kimi DSpark, set `draft_sample_method` to `greedy` or `probabilistic`.
- Eval-only and non-AgentX runs use real verification, and the connector removes stale synthetic settings. Non-speculative roles receive no simulation settings. `RUN_EVAL` does not disable simulation for the throughput portion.
- Missing golden curves or unmeasured draft lengths fail before submission. Add measured golden data for a new combination instead of a guessed or hard-coded length.

## Curve identities and names

Every curve file is named `<model>__<method>__<draft>__<draft_sample>-<verify>.yaml`: the curve's model key, its method (`mtp` for a model-native head, `eagle3`, `dspark`, `dflash`), its draft (`native` or the draft model), and the draft sampler and verifier the collector set (`default` when it left the engine default).

[`identities.yaml`](identities.yaml) gives each curve two blocks. A new curve needs both before it merges; a test checks each file name against them.

`select` decides which runs use the curve. `curve_name` returns the most specific matching curve (the one declaring the most of `draft_family`, `draft_sample` and `verify`) and fails when none match or the most specific ones tie.

| Key | Meaning |
| --- | --- |
| `model_prefixes` | InferenceX model prefixes that use the curve |
| `method` | `mtp`, `eagle3`, `dspark` or `dflash`. SGLang `EAGLE` and `NEXTN` mean the model's native `mtp` head, except MiniMax-M3's EAGLE, which is `eagle3` |
| `draft_family` | Only where one model's curves differ by draft: `gqa` when the draft model id contains `gqa`, else `default` |
| `draft_sample` | Only where one model's curves differ by draft sampler: the run's `draft_sample_method` must equal it |
| `verify` | Only where one model's curves differ by verifier: `standard`, `block` or `relaxed`. The run's verifier is read as the audit reads it; an unknown verifier (ATOM, TileRT) never matches a curve that sets `verify` |

`collection` records how the curve was measured: `engine`, `draft`, `draft_sample` (`greedy`, `probabilistic`), `verify` (`standard`, `block`, `relaxed`), `adaptive`, `topology` (`chain`, `tree-topk<N>`), `draft_precision` (the draft weights' quantization), `draft_kv_cache_dtype` and `kv_cache_dtype` (the target's). `provisional: true` marks a curve not measured for its model. Each fact is `{value, source}`:

| Source | Meaning |
| --- | --- |
| `explicit` | Set by the collector or recipe |
| `documented` | Fixed by the connector or engine: a recipe's synthetic sampler becomes `block`, and block, relaxed and adaptive verification are opt-in, so unset means `standard` and not adaptive |
| `assumed` | The connector's own reading, such as ATOM DSpark drafting probabilistically |
| `default` | Unset, so the engine default; `value` is null |
| `unknown` | Not readable, such as ATOM's verifier; `value` is null |

[`infx/srt_slurm/spec_identity.py`](../srt_slurm/spec_identity.py) describes a run's generation role in the same vocabulary and audits it against its selected curve without changing the selection. KV cache dtypes are normalized: `fp8_e4m3` is `fp8`, `bfloat16` is `bf16` and `auto` is the engine default. Findings are `mismatch` (both values known and different), `unverified` (an engine default or unknown on either side, other than the same engine's default on both), `adaptive` (no fixed-k curve applies), `provisional` and `engine`. To audit every speculative role of the sweep matrix:

```bash
python -m infx.matrix.generate full-sweep --config-files configs/nvidia-master.yaml configs/amd-master.yaml configs/apple-master.yaml --scenario-type agentic-coding > matrix.json
python -m infx.srt_slurm.spec_identity snapshot matrix.json --root .
```

## Querying the curves

This directory is the `infx.golden_al_distribution` Python package. `golden_length(model_prefix, spec, thinking)` returns the value AgentX synthetic acceptance applies, and `curve_name` shows which YAML a speculative config resolves to. The same lookup runs from a shell:

```bash
python -m infx.golden_al_distribution list                          # every curve and its measured draft lengths
python -m infx.golden_al_distribution show qwen3.5-397b-a17b-nvfp4__mtp__native__default-default              # every AL in one curve
python -m infx.golden_al_distribution lookup qwen3.5 mtp 3          # 3.39 (thinking_on by default)
python -m infx.golden_al_distribution lookup kimik3 dspark 7 --draft-sample-method probabilistic --json
```

`lookup` takes the InferenceX model prefix, the engine's method name (`mtp`, `eagle`, `nextn`, `eagle3`, `dspark`, `dflash`) and `num_speculative_tokens`. `--draft-model` selects the MiniMax GQA curve, and `--draft-sample-method` chooses between the two Kimi DSpark curves. A missing curve or cell exits non-zero, exactly as a launch would fail.

## Primary references

- [Similar Efforts from AMD @haic0 PR Reference 1](https://github.com/SemiAnalysisAI/InferenceX/pull/1633)
- [Similar Efforts from AMD @haic0 Reference 2](https://github.com/SemiAnalysisAI/InferenceX/pull/1115#issuecomment-4295024377)
- [SPEED-Bench paper](https://arxiv.org/abs/2604.09557)
- [SPEED-Bench dataset and dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- [vLLM SPEED-Bench integration](https://github.com/vllm-project/vllm/pull/36029)
- [vLLM synthetic acceptance support](https://github.com/vllm-project/vllm/pull/40662)
- [ATOM forced acceptance-length support](https://github.com/ROCm/ATOM/pull/1948)
- [InferenceX synthetic-acceptance tracking issue](https://github.com/SemiAnalysisAI/InferenceX/issues/1651)
- [InferenceX SPEED-Bench workflow](../../../.github/workflows/speedbench-al.yml)
- [InferenceX early reference-alignment PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1592)
- [InferenceX initial AL collector PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1650)
- [InferenceX multi-model AL collectors PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1706)
- [InferenceX multi-node synthetic-acceptance bring-up](https://github.com/SemiAnalysisAI/InferenceX/pull/1789)

DeepSeek V4.1 Flash includes measured draft lengths 1–5 for thinking off/on. The same image rejected lengths 6–8 before serving in [run 34494319147](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34494319147); no AL values are assigned to those lengths.
