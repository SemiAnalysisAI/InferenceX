#  InferenceX™, Open Source Continuous Inference Standard and Research Platform / 开源持续推理标准与研究平台
<p align="center">
  <a href="https://github.com/SemiAnalysisAI/InferenceX/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/SemiAnalysisAI/InferenceX/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://inferencex.semianalysis.com/"><img alt="Dashboard" src="https://img.shields.io/badge/Performance-Dashboard-blue"></a>
  <a href="https://deepwiki.com/SemiAnalysisAI/InferenceX"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
  <a href="https://github.com/SemiAnalysisAI/InferenceX"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/SemiAnalysisAI/InferenceX?style=social"></a>
</p>
<div align="center">

**English** | [中文](./README_zh.md)

</div>

Trusted by Operators of Trillion Dollar Token Factories such as OpenAI, Meta, Microsoft, Oracle, etc, & ML Community such as PyTorch Foundation, vLLM, SGLang, Tri Dao

## News

- **[2026/06]** 🔥 MiniMax M3: continuous benchmarks live since Day 0 [dashboard](https://inferencex.semianalysis.com/inference?preset=minimax-m3-launch)
- **[2026/04]** 🔥 DeepSeek V4 Pro 1.6T: continuous benchmarks live since Day 0 [article](https://newsletter.semianalysis.com/p/deepseekv4-16t-day-0-to-day-43-performance), [dashboard](https://inferencex.semianalysis.com/inference?preset=dsv4-launch)
- **[2026/03]** 🔥 Qwen3.5 397B: continuous benchmarks live since Day 0 [dashboard](https://inferencex.semianalysis.com/)
- **[2026/03]** Added Kimi K2.5 (same architecture as Kimi 2.7-Code), GLM5 (same arch as GLM5.1), and MiniMax M2.5 (same arch as MiniMax M2.7) [dashboard](https://inferencex.semianalysis.com/)
- **[2026/02]** GB300 NVL72: added to InferenceX & continuously benchmarked [SGLang Maintainer Lmsys Blog](https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/)
- **[2026/02]** 🔥 InferenceX v2 launch — NVIDIA Blackwell vs AMD vs Hopper [article](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)
- **[2025/10]** 🔥 InferenceX (formerly InferenceMAX) v1 launch [article](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)

## Introduction

InferenceX™ (formerly InferenceMAX) is an inference performance research platform dedicated to continually analyzing & benchmarking the world’s most popular open-source inference frameworks used by major token factories and models to track real performance in real time. As these software stacks improve, InferenceX™ captures that progress in near real-time, providing a live indicator of inference performance progress. An [open-sourced](https://github.com/SemiAnalysisAI/InferenceX-app) live dashboard is available for free publicly at https://inferencex.com/. 

> [!IMPORTANT]
> Only [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo contains the Official InferenceX™ result, all other forks & repos are Unofficial. The benchmark setup & quality of machines/clouds in unofficial repos may be differ leading to subpar benchmarking. Unofficial must be explicitly labelled as Unofficial.
> Forks may not remove this disclaimer

<img width="2544" height="1424" alt="InferenceX DeepSeekv4 MXFP4 Performance Curve" src="https://github.com/user-attachments/assets/cc50b671-0a54-40b6-b184-19d5a59590cb" />


## Why?

InferenceX™, an open-source, under Apache2 license, automated benchmark designed to move at the same rapid speed as the software ecosystem itself, is built to address this challenge.

LLM Inference performance is driven by two pillars, hardware and software. While hardware innovation drives step jumps in performance every year through the release of new GPUs/XPUs and new systems, software evolves every single day, delivering continuous performance gains on top of these step jumps. Speed is the Moat 🚀
 
AI software like SGLang, vLLM, TensorRT-LLM, CUDA, and ROCm achieve this continuous improvement in performance through kernel-level optimizations, distributed inference strategies, and scheduling innovations that increase the pareto frontier of performance in incremental releases that can be just days apart.
 
This pace of software advancement creates a challenge: benchmarks conducted at a fixed point in time quickly go stale and do not represent the performance that can be achieved with the latest software packages.


## Officially Supported Hardware

| SKU | Status |
| --- | --- |
| GB300 NVL72 | ✅ |
| GB200 NVL72 | ✅ |
| MI355X | ✅ |
| B300 | ✅ |
| B200 | ✅ |
| MI325X | ✅ |
| MI300X | ✅ |
| H200 | ✅ |
| H100 | ✅ |
| TPUv7x Ironwood Ghostfish | Coming Soon 🔜 |
| MI455 UALoE72 | Coming Soon 🔜 |
| Vera Rubin NVL72 | Coming Soon 🔜 |
| Rubin NVL8 | Coming Soon 🔜 |
| Chip #1 from Hardware Vendor #1 | Coming Soon 🔜 |
| Chip #2 from Hardware Vendor #1 | Coming Soon 🔜 |
| Chip #1 from Hardware Vendor #2 | Coming Soon 🔜 |
| Chip #1 from Hardware Vendor #3 | Coming Soon 🔜 |
| Chip #1 from Hardware Vendor #4 | Coming Soon 🔜 |


## Contributing

PRs are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details on the PR review flow, the [PR Review Checklist](./docs/PR_REVIEW_CHECKLIST.md), and the merge process.

## Acknowledgements & Supporters
Thank you to Lisa Su and Anush Elangovan for providing the MI355X and CDNA3 GPUs for this free and open-source project. We want to recognize the many AMD contributors for their responsiveness and for debugging, optimizing, and validating performance across AMD GPUs. 
We’re also grateful to Jensen Huang and Ian Buck for supporting this open source with access to a GB200 NVL72 rack (through OCI) and B200 GPUs. Thank you to the many NVIDIA contributors from the NVIDIA inference team, NVIDIA Dynamo team.

We also want to recognize the SGLang, vLLM, and TensorRT-LLM maintainers for building a world-class software stack and open sourcing it to the entire world.
Finally, we’re grateful to Crusoe, CoreWeave, Nebius, TensorWave, Oracle and TogetherAI for supporting open-source innovation through compute resources, enabling this.

Full list of supporters & quotes: https://inferencex.semianalysis.com/quotes

<img width="938" height="487" alt="image" src="https://github.com/user-attachments/assets/aa9b8257-fa7d-4691-97c3-dada8db05cb3" />

