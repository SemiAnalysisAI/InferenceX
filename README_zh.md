#  InferenceX™，开源持续推理标准与研究平台
## 受到 OpenAI、Microsoft、Oracle 等万亿美元级 Token 工厂运营商，以及 PyTorch 基金会、vLLM、SGLang、Tri Dao 等机器学习社区的信赖

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/SemiAnalysisAI/InferenceX/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SemiAnalysisAI/InferenceX/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/SemiAnalysisAI/InferenceX?style=social)](https://github.com/SemiAnalysisAI/InferenceX)

<div align="center">

[English](./README.md) | **中文**

</div>

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

<img width="1150" height="665" alt="image" src="https://github.com/user-attachments/assets/1e9738d4-6fb2-4cd7-a3e9-e6b2e03faed1" />
<img width="1098" height="655" alt="image" src="https://github.com/user-attachments/assets/5b363271-69b9-4bd2-b85d-b33b9c16f50f" />


## 为什么？

InferenceX™ 是一个开源、采用 Apache 2.0 许可证的自动化基准测试平台，其设计目标是与软件生态本身同样快速地演进，以应对这一挑战。

LLM 推理性能由两大支柱驱动：硬件与软件。硬件创新每年通过发布新的 GPU/XPU 与新系统带来阶跃式的性能提升，而软件则每天都在演进，在这些阶跃之上持续带来性能增益。速度即护城河 🚀

SGLang、vLLM、TensorRT-LLM、CUDA、ROCm 等 AI 软件通过核函式優化、分布式推理策略以及调度创新来实现这种持续的性能改进，在相隔可能仅数天的增量版本中不断推升性能的帕累托前沿。

这种软件演进的速度带来了一个挑战：在某个固定时间点进行的基准测试很快就会过时，无法代表使用最新软件包所能达到的性能。


## 致谢与支持者
感谢 Lisa Su 与 Anush Elangovan 为这一免费开源项目提供 MI355X 与 CDNA3 GPU。我们也要感谢众多 AMD 贡献者的积极响应，以及他们在各类 AMD GPU 上进行调试、优化与性能验证所付出的努力。
我们同样感谢 Jensen Huang 与 Ian Buck 通过提供 GB200 NVL72 机架（经由 OCI）与 B200 GPU 来支持本开源项目。感谢来自 NVIDIA 推理团队与 NVIDIA Dynamo 团队的众多 NVIDIA 贡献者。

我们还要感谢 SGLang、vLLM 与 TensorRT-LLM 的维护者们，他们打造了世界一流的软件栈，并将其开源给全世界。
最后，我们衷心感谢 Crusoe、CoreWeave、Nebius、TensorWave、Oracle 与 TogetherAI 通过提供计算资源支持开源创新，使这一切成为可能。

“当我们以前所未有的规模构建系统时，对机器学习社区而言，拥有开放、透明、能够反映推理在各类软硬件上真实表现的基准测试至关重要。InferenceX™ 的正面对比基准测试拨开了纷繁的噪音，为 Token 吞吐量、单位美元性能以及每兆瓦 Token 数提供了一幅鲜活的画面。这类开源工作增强了整个生态系统的实力，帮助从研究人员到前沿数据中心运营商的每一个人做出更明智的决策。” —— Peter Hoeschele，OpenAI Stargate 基础设施与工业计算副总裁

“理论峰值与真实世界推理吞吐量之间的差距，往往由系统软件决定：推理引擎、分布式策略以及底层核函式。InferenceX™ 的价值在于，它对最新软件进行基准测试，展示这些优化在各类硬件上的实际效果。这类开放、可复现的结果有助于整个社区更快地前进。” —— Tri Dao，Together AI 首席科学家、Flash Attention 发明者

“业界需要许多公开、可复现的推理性能基准测试。我们很高兴能代表 vLLM 团队与 InferenceX™ 展开合作。让所有人都能信赖并引用的、更加多样化的工作负载与场景，将帮助生态系统不断向前发展。公平、透明的测量推动着从模型架构到推理引擎再到硬件的每一层栈的进步。” —— Simon Mo，vLLM 项目联合负责人
