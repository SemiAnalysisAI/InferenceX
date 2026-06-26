# Golden Acceptance-Length Distributions

English | [中文](README_zh.md)

This directory contains the golden acceptance-length (AL) curves used to standardize speculative decoding moving forward in InferenceX. Each YAML maps a model, thinking mode, and draft length (`num_speculative_tokens`) to a mean AL measured on the **coding** category of the SPEED-Bench Qualitative split.

## Why SPEED-Bench

[SPEED-Bench](https://arxiv.org/abs/2604.09557) is a unified benchmark for speculative decoding across diverse semantic domains and realistic serving regimes. Its Qualitative split contains 880 semantically diverse prompts—80 prompts in each of 11 categories—and is designed to measure acceptance rate (AR) and acceptance length (AL). Its Throughput splits cover fixed 1K–32K input lengths and multiple entropy regimes for system-level evaluation. The benchmark uses real prompts because random-token inputs can distort acceptance behavior, expert routing, and measured throughput.

SPEED-Bench is a practical cross-engine standard rather than an InferenceX-only workload:

- vLLM maintainers merged [native SPEED-Bench support](https://github.com/vllm-project/vllm/pull/36029) into `vllm bench serve` for Qualitative AR/AL measurement and Throughput evaluation. It is documented in the [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.22.0/benchmarking/cli/).
- SGLang exposes a native [`speed-bench` dataset adapter](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/speed_bench.py) and documents it in its [serving benchmark guide](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/developer_guide/bench_serving.mdx).
- The SPEED-Bench paper evaluates production-grade engines including vLLM, SGLang, and TensorRT-LLM, so its methodology is suitable for cross-runtime comparisons.

This upstream maintainer adoption is why InferenceX uses SPEED-Bench as the common measurement substrate for golden AL collection.

## Why the coding category

AL is workload-dependent: a draft model's predictions are easier to accept in some domains than others. Agentic coding accounts for the largest share of its token volume. We therefore calibrate synthetic acceptance against SPEED-Bench's `coding` category instead of averaging unrelated domains such as roleplay, translation, or creative writing.

## Fairness Guidelines for AgentX

Under the AgentX Guidelines, each model, thinking mode, and draft length has one committed golden AL. Once synthetic acceptance is enabled for a benchmark scenario, a submission may choose any supported draft length, but it may not substitute a different acceptance target. Different models keep their own SPEED-Bench-derived curves; all submissions evaluating the same model and mode use the same curve.

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

This policy follows the same broad principle as MLPerf Inference: prescribe the workload rules needed for comparable system measurements. InferenceX is evaluating inference-system performance, not the ability to fine-tune a benchmark-specific speculative head.

## How a golden AL curve is collected

The push-button [`speedbench-al.yml`](../.github/workflows/speedbench-al.yml) workflow, introduced in [InferenceX#1650](https://github.com/SemiAnalysisAI/InferenceX/pull/1650) and extended to additional MTP and EAGLE3 models in [InferenceX#1706](https://github.com/SemiAnalysisAI/InferenceX/pull/1706), performs the following process. It superseded the early manually assembled reference in [InferenceX#1592](https://github.com/SemiAnalysisAI/InferenceX/pull/1592), making the exact commands, logs, outputs, and generated YAML auditable from one run.

1. A maintainer dispatches the workflow with a model, model prefix, vLLM image, draft lengths (normally 1–8), thinking modes, `category=coding`, and `output-len=4096`.
2. The workflow launches the model on a B300 runner and selects the matching collector under [`benchmarks/single_node/speedbench/`](../benchmarks/single_node/speedbench/).
3. For every `(thinking mode, draft length)` cell, the collector starts a clean vLLM server with real MTP or EAGLE3 decoding and the model's production sampling/chat-template settings.
4. The collector snapshots vLLM's cumulative accepted-token and verification-draft counters, runs every prompt in the SPEED-Bench Qualitative `coding` category through `vllm bench serve`, and snapshots the counters again.
5. It computes the mean acceptance length as:

   ```text
   AL = 1 + (delta accepted draft tokens / delta verification drafts)
   ```

   The `1` is the target model's guaranteed verification token. Values are rounded to two decimal places.
6. The collector emits a YAML matrix. The workflow publishes it in the GitHub Actions step summary and uploads it as a `speedbench-reference-al-<model-prefix>` artifact.
7. Server logs and detailed per-request results are retained so reviewers can confirm sensible output, correct thinking mode, and the absence of silent server or chat-template failures.
8. After review, the matrix is committed here with its exact sampling metadata and source Actions run URL.

The collection measures real head quality only to establish the golden curve. AgentX then uses that committed curve as the synthetic acceptance target for every comparable submission.

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

- every requested draft length and thinking mode completed;
- detailed outputs are coherent and use the intended thinking mode;
- server logs contain no fallback, draft-disable, or chat-template errors;
- the YAML metadata matches the dispatched image, sampling settings, model, and speculative method;
- the source Actions run is linked at the first line of the YAML; and
- the committed values exactly match the workflow artifact.

## Current golden curves

| Model | Method | Golden YAML | Source run |
| --- | --- | --- | --- |
| DeepSeek V4 Pro | MTP | [`dsv4_mtp.yaml`](dsv4_mtp.yaml) | [27180633016](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27180633016) |
| Qwen3.5 397B-A17B | MTP | [`qwen3.5_mtp.yaml`](qwen3.5_mtp.yaml) | [27317114007](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27317114007) |
| Kimi K2.5 | EAGLE3 | [`kimik2.5_eagle3.yaml`](kimik2.5_eagle3.yaml) | [28122195822](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28122195822) |
| MiniMax-M3 | EAGLE3 | [`minimaxm3_eagle3.yaml`](minimaxm3_eagle3.yaml) | [28061204145](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28061204145) |

## Primary references

- [Similar Efforts from AMD @haic0 PR Reference 1](https://github.com/SemiAnalysisAI/InferenceX/pull/1633)
- [Similar Efforts from AMD @haic0 Reference 2](https://github.com/SemiAnalysisAI/InferenceX/pull/1115#issuecomment-4295024377)
- [SPEED-Bench paper](https://arxiv.org/abs/2604.09557)
- [SPEED-Bench dataset and dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- [vLLM SPEED-Bench integration](https://github.com/vllm-project/vllm/pull/36029)
- [vLLM synthetic acceptance support](https://github.com/vllm-project/vllm/pull/40662)
- [InferenceX synthetic-acceptance tracking issue](https://github.com/SemiAnalysisAI/InferenceX/issues/1651)
- [InferenceX SPEED-Bench workflow](../.github/workflows/speedbench-al.yml)
- [InferenceX early reference-alignment PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1592)
- [InferenceX initial AL collector PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1650)
- [InferenceX multi-model AL collectors PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1706)
- [InferenceX multi-node synthetic-acceptance bring-up](https://github.com/SemiAnalysisAI/InferenceX/pull/1789)
