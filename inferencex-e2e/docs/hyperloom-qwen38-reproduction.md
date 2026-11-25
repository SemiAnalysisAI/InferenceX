# Hyperloom Qwen3.8-Flash-Next reproduction draft

I document the pinned inputs, the nine-point replay contract and the available reproduction evidence for Ravi's final-mainline package. I provide the active image/patch/recipe integration in [the CI instructions](hyperloom-qwen38-ci.md). The reference JSON in this directory remains provenance data and is not consumed by the benchmark launcher.

I have five scored full replays on MI350X. In MI355X CI, C1/C2/C4/C8/C12/C16/C24 completed successfully at the earlier recipe revision; C48 failed temporal metric coverage in both attempts, the C80 performance job was cancelled, and C80 eval failed with GPU OOM. I retain these outcomes separately from the local replays and from validation of the corrected recipe.

## Inputs and attribution

I credit Chang Liu (@ChangLiu0709), Nikolai Protasov (@amd-nprotaso) and Christian Gilli (@amd-cgilli) as coauthors. I record Ravi as the source of the reproduction package and clarification of its bases.

I record all original file hashes, all nine original configurations and replay contracts in [provenance.json](reference/hyperloom-qwen38/provenance.json). I distinguish the base OCI image digest from the locally built image ID: a local image ID is not a registry digest that a CI worker can pull.

I use the November 18 ROCm nightly pinned in the manifest. Its SGLang tree is upstream `3a41a599192906eee6aace241798136a6db2b9ad` plus the image's changes; its AITER tree is upstream `581579ab51ca26ddd4a08a6aa6aa983a78aaf276` plus the image's changes. The package patches apply to those image trees, not directly to clean upstream. I exclude the nightly's changes from upstream SGLang/AITER proposals.

I pin model revision `bd6e49d9abfe0ce1da53a45593daaee83580035b`, InferenceX client `5a1f81029fe7421f2f6e870cae4af9c5ff2f2290`, lm-evaluation-harness `b315ef3b05176acc9732bb7fdec116abe1ecc476`, and AgentX dataset revision `8fecd2fc56694469f758f0afbbb6335ad3043740`. These are the reproduction pins; I do not claim they describe the current InferenceX runtime on this branch.

## Effective reference-run clarification

I incorporate Ravi's November 25 clarification that the final reference run used C80 `mem-fraction-static: 0.87` and C24 `hicache-size: 298`. The supplied JSON inputs contain C80 `0.88` and C24 `200`, which the first CI conversion retained; `reproduce.sh` passes those files directly and does not supply the clarified overrides. I preserve those original bytes and record the discrepancy as later source clarification. I apply the corrected values in the active CI recipe and require new GPU evidence before treating them as validated.

I verified that the C48 attempt-2 figure of about 81% means latency-metric duration coverage, not request completion: TTFT/ITL cover 81.13%/81.20% of the hour against a 95% requirement, while 1,364 of 1,420 requests completed. I retain both C48 attempts as failures. I also distinguish all package gains from stock-image claims: even F1's reported 43–50% at C16/C80 was measured in the patched context, so I require a new unpatched comparison before quoting a stock-image gain.

I keep public reproduction links limited to accessible image, engine and InferenceX resources; an inaccessible internal hand-off repository is not a public prerequisite.

## Replay contract

I reproduce C1/C2/C4/C8/C12/C16 without KV offload and C24/C48/C80 with HiCache DRAM offload, at TP1/EP1. I run a fresh server and a 3,600-second replay per attempt. The point name denotes client concurrency; admission is 1/2/4/8/24/32/48/128/128 respectively, so I do not equate concurrency with max-running-requests. C8 uses one speculative step and two draft tokens, while the other points use three steps and four draft tokens.

I retain the package's scoring checks: measured window at least 90% of the requested duration and median speculative acceptance within 0.15 of its golden value (1.65 at C8, 2.32 otherwise). I also require successful client and scoring exit codes. Passing these checks establishes a valid replay under this contract; it does not establish natural accuracy or performance equivalence across hardware.

I preserve the package's synthetic acceptance variables in the review JSON as historical inputs. For an active InferenceX submission I must use the measured golden distribution and identity registered in `infx/golden_al_distribution/`, including the effective draft KV dtype, rather than copy hard-coded synthetic acceptance into the serving recipe. I evaluate natural accuracy with synthetic acceptance removed from inherited and explicit environments.

## Measured results and limitations

I record actual final replay scores, run identifiers and pins in [mi350x-replays.json](reference/hyperloom-qwen38/mi350x-replays.json). These are local reproduction measurements, not results from a GitHub Actions run or validation of the current candidate branch.

| Point | MI350X P90 tokens/s/user | MI350X total tokens/s/GPU | MI355X reported P90 | MI355X reported total |
| --- | ---: | ---: | ---: | ---: |
| C1 | 393.92 | 16914.09 | 431.37 | 18145.00 |
| C2 | 313.53 | 21138.51 | 345.53 | 21253.00 |
| C4 | 218.43 | 25169.20 | 248.33 | 26631.00 |
| C12 | 146.55 | 64205.45 | 168.18 | 67630.00 |
| C16 | 104.30 | 83800.00 | 130.80 | 87444.00 |

I compare one MI350X replay per measured point against Ravi's reported MI355X mean of two replays. I do not treat MI350X as performance-equivalent to MI355X or attribute the differences to a particular patch.

I also observed a valid parallel C1 control at P90 144.91 tokens/s/user and total throughput 8,387.04 tokens/s/GPU, versus the initial sequential 393.92 and 16,914.09. I reduced parallelism to one replay at a time, excluded the parallel control from the principal curve and repeated C2 sequentially. I have not determined the cause of the control regression.

I could not validate C8 because server graph preparation exceeded the one-hour readiness budget. C24/C48/C80 encountered GPU availability failures; later retries found the local model cache missing. I restored the same model revision successfully, but the next attempts again found occupied GPUs. I am waiting for the shared workload to finish; I report no measurements for those points. I retain the failed attempts separately from successful scores.

## InferenceX integration and remaining validation

I update `qwen3.8next-fp4-mi355x-sglang-agentic-mtp` and its `qwen3.8next/sglang/mi355x-fp4-mtp/agentic.yaml` recipe to the November 18 base image plus the included package diffs. I cover all nine concurrency values and document the current InferenceX golden/client adaptations separately from exact original-package reproduction.

1. I complete the remaining local replay evidence and natural accuracy checks, and investigate the C1 control regression before assigning any final performance claim.
2. I use the published digest-pinned base image with the included diffs applied through the container setup script. I still verify draft precision and target/draft KV storage against current contribution rules and the golden identity; I do not invent a new measured identity.
3. I connect the converted package configurations to the existing recipe and master config, and use the existing launcher `python -m infx.launch`. I apply the supplied aggregate diffs once to their pinned image, without adding #4832 patches on top or patching a cluster host-setup hook.
4. I append one performance changelog block for the active recipe/image change, preserving all historical entries. I document the normal same-repository sweep path for [PR #5240](https://github.com/SemiAnalysisAI/InferenceX/pull/5240), whose `hyperloom/qwen3.8-final-mainline-ci` branch lives in `SemiAnalysisAI/InferenceX`.
5. I run the current recipe through the existing `cluster:mi355x-amds` integration, with explicit GPU allocation, bounded readiness, artifact collection on failure, the full point set and natural eval/acceptance gates. I use repository write access to apply exactly one primary sweep label, which starts `run-sweep.yml` with its canary and benchmark/eval jobs. I can launch this same-repository sweep with the PR Draft or Ready for review; marking Ready alone does not dispatch it.
6. I record the resulting workflow URL, hardware, candidate head SHA, image digest, all point outcomes and eval evidence. I obtain the accepted `/use <run_id>` only after a green complete sweep when the active benchmark change is ready for merge.

I retain the original package hashes as provenance, not as a substitute for recovering the declared 38 SGLang and eight AITER commits. I keep those source-series and upstream patch boundaries separate from this InferenceX reproduction draft.
