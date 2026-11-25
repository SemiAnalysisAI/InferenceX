# Hyperloom Qwen3.8-Flash-Next CI instructions

I use the existing InferenceX/srt-slurm CI path with a published base image, the complete package diffs applied inside each serving container, and one recipe covering C1 through C80. I follow the structure of [PR #4832](https://github.com/SemiAnalysisAI/InferenceX/pull/4832); I do not require its branch or apply its patches in addition to the package diffs.

## Image and required changes

I select this pullable base image in `configs/amd-master.yaml`:

```text
lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261118@sha256:723fbc92690378341609b5395055df1187cdac06f299554ac75e57cfde89c9f1
```

I apply exactly the assets in [the patch manifest](../benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/hyperloom/manifest.json). I retain their original bytes and SHA256 values.

I bind the manifest SHA256 `843c443145a81372265153a857734bf0107aa81863b5bed49ba202f0fef84648` in `base.environment.HYPERLOOM_MANIFEST_SHA256`. This global environment reaches setup before engine launch and is included in each bound recipe's fingerprint. I keep `manifest.json` in LF format across checkouts. When I change any package asset, I must update its manifest entry and the recipe's manifest pin; leaving either stale fails setup instead of producing a result under an old package identity.

| Asset | Where I use it |
| --- | --- |
| `hyperloom/sglang.diff` | I apply it to `/sgl-workspace/sglang` inside the base image. |
| `hyperloom/aiter.diff` | I apply it to `/sgl-workspace/aiter` inside the base image. |
| `hyperloom/qwen3_8_flash_next_mxfp4_tuned_fmoe.csv` | I install it under `/opt/stack/configs/tables` for the configured MoE tuning lookup. |
| `hyperloom/qwen3_8_flash_next_mxfp4_tuned_fmoe.r2c16.csv` | I install the second point-specific tuning table at the same location. |

I use [qwen3.8next-hyperloom-patches.sh](../benchmarks/multi_node/srt-slurm-recipes/configs/qwen3.8next-hyperloom-patches.sh) through the recipe's `setup_script` field. This is a serving-container setup script, not a cluster `host-setup` hook. It verifies asset hashes, checks both patches before applying either, removes the obsolete AITER JIT sorting build so it can rebuild, and installs the tables. A repeated invocation verifies the existing patch set. A missing asset, hash mismatch, unappliable patch or incompatible previous patch set fails the worker.

I use the image's source trees as patch bases: SGLang upstream `3a41a599192906eee6aace241798136a6db2b9ad` and AITER upstream `581579ab51ca26ddd4a08a6aa6aa983a78aaf276`, each with the nightly image's own changes. I do not replace those trees with clean upstream checkouts.

I require no separate SGLang/AITER PR checkout for this aggregate reproduction: the two included diffs supply the package changes. For upstream review I track those changes separately; the reported series of 38 SGLang and eight AITER commits is not yet independently recovered. I do not substitute a partial upstream PR for the complete package diff or import the nightly's own changes into an upstream code PR.

I request the required serving-stack exception in [waiver/5240.md](waiver/5240.md), covering both complete diffs, the JIT sorting rebuild and both external MoE table copies. The stock image can serve a baseline configuration but does not contain the complete optimized package selected here. The waiver records partial upstream links and outstanding coverage, and requires a Core decision. The eligible CODEOWNER must link it in their additional-detail section; filing the request does not approve it or exempt draft-precision and golden-identity rules.

## Recipe and matrix

I connect [agentic.yaml](../benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/agentic.yaml) to `qwen3.8next-fp4-mi355x-sglang-agentic-mtp` in [amd-master.yaml](../configs/amd-master.yaml), retaining `cluster:mi355x-amds`, TP1/EP1 and one GPU per worker.

| Client concurrency | Variant | KV offload | Admission | MTP steps / draft tokens | Prefill chunk |
| ---: | --- | --- | ---: | --- | ---: |
| 1 | `override_c1` | none | 1 | 3 / 4 | 16384 |
| 2 | `override_c2` | none | 2 | 3 / 4 | 16384 |
| 4 | `override_c4` | none | 4 | 3 / 4 | 16384 |
| 8 | `override_c8` | none | 8 | 1 / 2 | 16384 |
| 12 | `override_c12` | none | 24 | 3 / 4 | 8192 |
| 16 | `override_c16` | none | 32 | 3 / 4 | 16384 |
| 24 | `override_c24` | HiCache DRAM | 48 | 3 / 4 | 8192 |
| 48 | `override_c48` | HiCache DRAM | 128 | 3 / 4 | 16384 |
| 80 | `override_c80` | HiCache DRAM | 128 | 3 / 4 | 16384 |

I retain the package's per-point engine flags, breakable prefill graphs, checkpoint settings and explicit decode graph sizes. C24 uses `write_through_selective`, C48/C80 use `write_back`; C80 has `max-mamba-cache-size: 760`. I do not copy these settings from #4832, whose point set and settings differ.

I incorporate Ravi's November 25 clarification of the effective reference-run settings: C80 uses `mem-fraction-static: 0.87`, and C24 uses `hicache-size: 298`. The supplied `configs/C80.json` and `configs/C24.json`, preserved in provenance, instead contain `0.88` and `200`; the first CI recipe retained those values. I record the later clarification rather than rewrite original inputs. C48 remains at `mem-fraction-static: 0.88` and `hicache-size: 298`. Increasing C24's host cache changes the benchmark, so its earlier successful CI result does not validate the corrected cache setting.

The original C80 eval failed in `_short_conv` with an explicit GPU OOM: an additional 1.48 GiB allocation failed after PyTorch had allocated 281.66 GiB on the 287.98 GiB GPU. I use the clarified 0.87 fraction to leave more headroom, but require a new run to verify that this resolves the failure. I do not classify the cancelled C80 performance job as an observed OOM.

## C48 coverage and interpreting results

I verified the C48 failure criterion from AIPerf's `metadata.metric_duration_coverage`. In attempt 2 of [run #45597647764](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/45597647764), TTFT covered 2,920.68 seconds (81.13%) and inter-token latency 2,923.16 seconds (81.20%) of the requested 3,600 seconds; each requires at least 95%, or 3,420 seconds. This is temporal coverage of latency signals, not the percentage of completed requests. The request count was 1,364 completed out of 1,420 sent (96.06%), with 56 cancelled at teardown and zero errors among completed requests. The request-error check passed; metric coverage failed, including no latency signal during the last 180 seconds.

Attempt 1 had only about 13.7% temporal coverage. I retain both attempts as failed/partial evidence rather than present either as a validated C48 performance point, and I keep the 95% requirement intact. The first server log shows prefill stalling while `/metrics` stays responsive; it does not establish a specific kernel or OOM cause for C48.

## Expectations for an unpatched image

I distinguish this patched-package recipe from a stock-image benchmark. The `SGLANG_INFX_*` switches used here select paths introduced by the package changes, so I do not reuse its reported gains as expected stock-image gains. A plain upstream flag can still have been measured only in the patched context: Ravi cites `hybrid-cache-checkpoints-level: 3` (F1) and a reported 43–50% gain at C16/C80. I treat that as a report from the patched configuration, not an independently measured gain on the unmodified image. Before quoting any stock-image improvement I run a fresh comparison on the pinned unpatched image with the effective flags, workload and baseline recorded.

I use public image and repository references in public descriptions and replies. I do not link inaccessible internal hand-off repositories as reproduction instructions. I preserve contributor attribution separately from the links needed to run the benchmark.

I pin the model with the engine's `revision` argument to `bd6e49d9abfe0ce1da53a45593daaee83580035b`. I let the InferenceX binder supply the model location, network endpoint and workload, rather than reuse the reproduction host's absolute cache paths. I use the served model repository for the tokenizer.

## Acceptance and accuracy

I remove all hard-coded `SGLANG_SIMULATE_ACC_*` values from the active recipe. The current InferenceX connector selects the committed golden curve after resolving each variant. At `THINKING_MODE=thinking_on`, it selects 1.65 for C8's one step and 2.32 for the other points' three steps. I inspect the generated role environment to confirm the actual selection.

I do not fix `default-chat-template-kwargs` to the original package's `enable_thinking: false`: the current replay/eval client supplies the mode used by the golden collection. Consequently, I distinguish this CI integration from an exact rerun of the separately pinned original client.

I keep natural evals enabled. Eval-only launches remove synthetic acceptance and use real verification. Before trusting results I check the candidate image's effective draft precision, target/draft KV dtype, and the selected golden identity against current [contribution rules](../../CONTRIBUTING.md#draft-model-precision). The existing Qwen golden identity records default draft KV, whereas C24/C48/C80 explicitly request `fp8_e4m3`; I have not changed the recorded collection facts or claimed a new matching golden collection. I resolve that identity requirement with measured evidence or a compliant recipe adjustment before sign-off and reuse; I distinguish this approval requirement from launching a diagnostic sweep.

## Inspect the generated configuration on Linux

I check out the PR branch and initialize its pinned srt-slurm dependency. From the repository root I run:

```bash
git submodule update --init inferencex-e2e/utils/srt-slurm
cd inferencex-e2e
uv sync --locked --extra recipes
uv run --locked --extra recipes python -m infx.matrix.generate test-config \
  --config-files configs/amd-master.yaml \
  --config-keys qwen3.8next-fp4-mi355x-sglang-agentic-mtp \
  --scenario-type agentic-coding > hyperloom-matrix.json
uv run --locked --extra recipes infx generate \
  --config-key qwen3.8next-fp4-mi355x-sglang-agentic-mtp \
  --output-dir hyperloom-bound-recipes
uv run --locked --extra recipes python -m infx.srt_slurm.spec_identity snapshot \
  hyperloom-matrix.json --root . > hyperloom-spec-identity.json
```

I use a new output directory for `infx generate`. I expect the matrix to contain exactly concurrency 1/2/4/8/12/16/24/48/80, duration 3600, the November 18 image, the intended KV modes, and enabled eval selection. I inspect the bound files for the setup script, resolved variant, one GPU per worker, model revision and automatically selected acceptance variables. I treat successful generation as configuration validation, not GPU or numerical validation.

## Start the same-repository sweep

I use [PR #5240](https://github.com/SemiAnalysisAI/InferenceX/pull/5240), whose `hyperloom/qwen3.8-final-mainline-ci` branch lives in `SemiAnalysisAI/InferenceX`. I use the normal [Run Sweep workflow](../../.github/workflows/run-sweep.yml); I add no new benchmark workflow or standalone launcher.

I keep one appended `perf-changelog.yaml` block for the affected config key. After checking that the PR is mergeable and recording the candidate head, I use repository write access to apply exactly one primary sweep label. I can keep the PR Draft or mark it Ready for review: the same-repository workflow permits either state, and `gh pr ready` is not a launch prerequisite or a dispatch trigger.

```bash
gh pr view 5240 --repo SemiAnalysisAI/InferenceX \
  --json headRefOid,mergeable,isDraft,labels
gh pr edit 5240 --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast
```

I follow [PR sweep labels](../../.github/workflows/README.md#pr-sweep-labels): `run-sweep.yml` validates and plans the appended changelog, runs the selected canary, then fans out the benchmark and eval jobs through the existing reusable workflows. `full-sweep-fail-fast` enables the canary and per-matrix fail-fast. I record the candidate and checkout SHAs used by the run. I finish branch updates before applying the label; a later push with the label retained can start a new sweep and cancel the active one. To retrigger a settled labeled PR deliberately, I remove and reapply its primary label. I do not use `agentx-fast`, `trim-conc`, `evals-only` or an eval smoke limit for the complete sweep. I verify the runner's physical-node priority label is `nodes:1` and let the cluster scheduler allocate GPUs.

I observe the runs with:

```bash
gh run list --repo SemiAnalysisAI/InferenceX --workflow run-sweep.yml --limit 10
```

I open the matching run and record all nine point outcomes and the eval/acceptance gates. I preserve failed jobs as failures, collect server/client/configuration/image fingerprints and raw score artifacts, and verify the workflow returns failure for real startup, replay or eval errors. I do not treat my local package scoring script's acceptance/window checks as a substitute for the current CI's result and eval validators.

I use one complete sweep to validate CI operation. If I compare a two-replay mean against the supplied reference, I run the complete sweep a second time at the same source/image/model identity and record both run URLs. I request `/use <run_id>` only after the applicable full sweep and eval requirements pass, following the [same-repository sweep reuse procedure](../../.github/workflows/README.md#reusing-an-approved-pr-full-sweep).

## Verification status

I have prepared the image reference, exact package assets, container setup, nine-point recipe and changelog. I generated all nine resolved recipes, ran the real setup twice on the pinned base image without GPU devices, checked that a corrupted asset fails, and passed 396 focused existing configuration tests. I record this limited validation in [ci-preflight.json](reference/hyperloom-qwen38/ci-preflight.json). These local checks did not use the complete locked CI dependency environment. I retain the five earlier local MI350X scores in [the reproduction record](hyperloom-qwen38-reproduction.md).

I started [MI355X CI run #45597647764](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/45597647764) on head `2613600914cc9ab18413f9a36acb797d1331c94c`; at the November 25 review check its C1 canary was executing. That head predates the waiver and manifest-pin correction. I verified the correction with focused fingerprint/matrix/binder tests and repeated CPU-only setup in the pinned image: missing or mismatched pins fail before patching, the correct pin passes first and repeated setup, and a corrupted CSV fails. I finish these branch changes before starting a new candidate sweep; a push while the primary label is retained can cancel the active run. I require a complete sweep and natural evals on the final candidate identity before sign-off or reuse.
