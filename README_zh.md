#  InferenceX™，开源持续推理标准与研究平台
<p align="center">
  <a href="https://github.com/SemiAnalysisAI/InferenceX/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/SemiAnalysisAI/InferenceX/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://inferencex.semianalysis.com/"><img alt="Dashboard" src="https://img.shields.io/badge/Performance-Dashboard-blue"></a>
  <a href="https://deepwiki.com/SemiAnalysisAI/InferenceX"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
  <a href="https://github.com/SemiAnalysisAI/InferenceX"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/SemiAnalysisAI/InferenceX?style=social"></a>
</p>
<div align="center">

[English](./README.md) | **中文**

</div>

受到 OpenAI、Meta、Microsoft、Oracle 等万亿美元级 Token 工厂运营商，以及 PyTorch 基金会、vLLM、SGLang、Tri Dao 等机器学习社区的信赖

## 新闻

- **[2026/06]** 🔥 MiniMax M3：自 Day 0 起持续进行基准测试 [仪表盘](https://inferencex.semianalysis.com/inference?preset=minimax-m3-launch)
- **[2026/04]** 🔥 DeepSeek V4 Pro 1.6T：自 Day 0 起持续进行基准测试 [文章](https://newsletter.semianalysis.com/p/deepseekv4-16t-day-0-to-day-43-performance)，[仪表盘](https://inferencex.semianalysis.com/inference?preset=dsv4-launch)
- **[2026/03]** 🔥 Qwen3.5 397B：自 Day 0 起持续进行基准测试 [仪表盘](https://inferencex.semianalysis.com/)
- **[2026/03]** 新增 Kimi K2.5（与 Kimi 2.7-Code 架构相同）、GLM5（与 GLM5.1 架构相同）、MiniMax M2.5（与 MiniMax M2.7 架构相同）[仪表盘](https://inferencex.semianalysis.com/)
- **[2026/02]** GB300 NVL72：已加入 InferenceX 并持续进行基准测试 [SGLang 维护者 LMSYS 博客](https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/)
- **[2026/02]** 🔥 InferenceX v2 发布——NVIDIA Blackwell 对比 AMD 对比 Hopper [文章](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)
- **[2025/10]** 🔥 InferenceX（前身为 InferenceMAX）v1 发布 [文章](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)

## 简介

InferenceX™（前身为 InferenceMAX）是一个推理性能研究平台，致力于持续分析与基准测试全球最受欢迎的开源推理框架——这些框架被各大 Token 工厂与模型广泛采用，以实时追踪其真实性能。随着这些软件栈不断改进，InferenceX™ 会以近乎实时的方式捕捉这些进展，提供一个反映推理性能进步的实时指标。我们在 https://inferencex.com/ 上免费公开提供了一个[开源](https://github.com/SemiAnalysisAI/InferenceX-app)的实时仪表盘。

> [!IMPORTANT]
> 只有 [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) 仓库才包含官方的 InferenceX™ 结果，所有其他派生（fork）与仓库均为非官方。非官方仓库的基准测试设置以及机器/云环境的质量可能存在差异，从而导致基准测试结果欠佳。非官方仓库必须明确标注为“非官方（Unofficial）”。
> 派生仓库不得移除本免责声明。

<img width="2544" height="1424" alt="InferenceX DeepSeekv4 MXFP4 Performance Curve" src="https://github.com/user-attachments/assets/cc50b671-0a54-40b6-b184-19d5a59590cb" />


## 为什么？

InferenceX™ 是一个开源、采用 Apache 2.0 许可证的自动化基准测试平台，其设计目标是与软件生态本身同样快速地演进，以应对这一挑战。

LLM 推理性能由两大支柱驱动：硬件与软件。硬件创新每年通过发布新的 GPU/XPU 与新系统带来阶跃式的性能提升，而软件则每天都在演进，在这些阶跃之上持续带来性能增益。速度即护城河 🚀

SGLang、vLLM、TensorRT-LLM、CUDA、ROCm 等 AI 软件通过核函式優化、分布式推理策略以及调度创新来实现这种持续的性能改进，在相隔可能仅数天的增量版本中不断推升性能的帕累托前沿。

这种软件演进的速度带来了一个挑战：在某个固定时间点进行的基准测试很快就会过时，无法代表使用最新软件包所能达到的性能。


## 官方支持硬件

| SKU | 状态 |
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


## 参与贡献

欢迎提交 PR！有关 PR 审阅流程、[PR 审阅清单](./docs/PR_REVIEW_CHECKLIST_zh.md)与合并流程的更多详情，请参阅 [CONTRIBUTING_zh.md](./CONTRIBUTING_zh.md)。

## 致谢与支持者
感谢 Lisa Su 与 Anush Elangovan 为这一免费开源项目提供 MI355X 与 CDNA3 GPU。我们也要感谢众多 AMD 贡献者的积极响应，以及他们在各类 AMD GPU 上进行调试、优化与性能验证所付出的努力。
我们同样感谢 Jensen Huang 与 Ian Buck 通过提供 GB200 NVL72 机架（经由 OCI）与 B200 GPU 来支持本开源项目。感谢来自 NVIDIA 推理团队与 NVIDIA Dynamo 团队的众多 NVIDIA 贡献者。

我们还要感谢 SGLang、vLLM 与 TensorRT-LLM 的维护者们，他们打造了世界一流的软件栈，并将其开源给全世界。
最后，我们衷心感谢 Crusoe、CoreWeave、Nebius、TensorWave、Oracle 与 TogetherAI 通过提供计算资源支持开源创新，使这一切成为可能。

完整支持者名单与引言：https://inferencex.semianalysis.com/quotes

<img width="938" height="487" alt="image" src="https://github.com/user-attachments/assets/aa9b8257-fa7d-4691-97c3-dada8db05cb3" />


