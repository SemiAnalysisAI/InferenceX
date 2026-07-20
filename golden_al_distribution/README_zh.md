# 黄金接受长度分布

[English](README.md) | 中文

本目录包含今后用于在 InferenceX 中标准化推测解码的黄金接受长度（Acceptance Length，AL）曲线。每个 YAML 按模型、思考模式和草稿长度（`num_speculative_tokens`），记录在 SPEED-Bench Qualitative split 的 **coding** 类别上测得的平均 AL。

## 为什么选择 SPEED-Bench

[SPEED-Bench](https://arxiv.org/abs/2604.09557) 是一个统一的推测解码基准，覆盖多样的语义领域和真实的服务场景。其 Qualitative split 包含 880 个语义多样的提示词——11 个类别各 80 个——用于测量接受率（AR）和接受长度（AL）。其 Throughput splits 覆盖固定的 1K–32K 输入长度和多种熵区间，用于系统级评估。该基准使用真实提示词，因为随机 token 输入会扭曲接受行为、专家路由和吞吐量测量结果。

SPEED-Bench 是一个实用的跨引擎标准，而不是 InferenceX 独有的工作负载：

- vLLM 维护者已将 [SPEED-Bench 原生支持](https://github.com/vllm-project/vllm/pull/36029) 合入 `vllm bench serve`，用于 Qualitative AR/AL 测量和 Throughput 评估；相关用法已记录在 [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.22.0/benchmarking/cli/) 中。
- SGLang 提供原生的 [`speed-bench` 数据集适配器](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/speed_bench.py)，并在其[服务基准指南](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/developer_guide/bench_serving.mdx)中记录了相关用法。
- SPEED-Bench 论文评估了包括 vLLM、SGLang 和 TensorRT-LLM 在内的生产级引擎，因此其方法适合跨运行时比较。

正是由于这些上游维护者的采用，InferenceX 才使用 SPEED-Bench 作为黄金 AL 收集的共同测量基础。

## 为什么选择 coding 类别

AL 取决于工作负载：草稿模型的预测在某些领域比其他领域更容易被接受。智能体编程占其 token 总量的最大份额。因此，我们使用 SPEED-Bench 的 `coding` 类别来校准合成接受，而不是对角色扮演、翻译或创意写作等无关领域求平均。

## AgentX 公平性指南

根据 AgentX 指南，每个模型、思考模式和草稿长度都有一个已提交的黄金 AL。当某个基准场景启用合成接受后，提交可以选择任意受支持的草稿长度，但不能替换为其他接受目标。不同模型保留各自基于 SPEED-Bench 测得的曲线；所有评估同一模型和模式的提交都使用同一条曲线。

vLLM 通过合成拒绝采样支持这一策略。例如，EAGLE3 运行可以通过 `synthetic_acceptance_length` 注入所选 YAML 值：

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

该选项通过 [vllm-project/vllm#40662](https://github.com/vllm-project/vllm/pull/40662) 在 vLLM 的不同模型运行器中实现统一。

这一策略遵循与 MLPerf Inference 相同的广义原则：规定可比较系统测量所需的工作负载规则。InferenceX 评估的是推理系统性能，而不是微调特定基准推测头的能力。

## 黄金 AL 曲线如何收集

一键触发的 [`speedbench-al.yml`](../.github/workflows/speedbench-al.yml) 工作流最初由 [InferenceX#1650](https://github.com/SemiAnalysisAI/InferenceX/pull/1650) 引入，随后在 [InferenceX#1706](https://github.com/SemiAnalysisAI/InferenceX/pull/1706) 中扩展到更多 MTP 和 EAGLE3 模型。它取代了 [InferenceX#1592](https://github.com/SemiAnalysisAI/InferenceX/pull/1592) 中早期手工整理的参考值，使精确命令、日志、输出和生成的 YAML 都可以从同一次运行中审计。其流程如下：

1. 维护者触发工作流，指定模型、模型前缀、vLLM 镜像、草稿长度（通常为 1–8）、思考模式、`category=coding` 和 `output-len=4096`。
2. 工作流在 B300 runner 上启动模型，并选择 [`benchmarks/single_node/speedbench/`](../benchmarks/single_node/speedbench/) 下对应的收集脚本。
3. 对每个“思考模式 × 草稿长度”组合，收集脚本使用真实 MTP 或 EAGLE3 解码以及该模型的生产采样和聊天模板设置，启动一个干净的 vLLM 服务。
4. 收集脚本读取 vLLM 累计的已接受 token 和验证草稿计数器，通过 `vllm bench serve` 运行 SPEED-Bench Qualitative `coding` 类别中的全部提示词，然后再次读取计数器。
5. 按以下公式计算平均接受长度：

   ```text
   AL = 1 + (delta accepted draft tokens / delta verification drafts)
   ```

   其中 `1` 是目标模型保证生成的验证 token。结果四舍五入到小数点后两位。
6. 收集脚本生成 YAML 矩阵。工作流将其发布到 GitHub Actions step summary，并上传为 `speedbench-reference-al-<model-prefix>` artifact。
7. 工作流保留服务日志和逐请求详细结果，以便审阅者确认输出合理、思考模式正确，并且没有静默的服务或聊天模板故障。
8. 审阅完成后，将矩阵连同准确的采样元数据和源 Actions run URL 一并提交到本目录。

收集过程仅通过测量真实草稿头质量来建立黄金曲线。之后，AgentX 对所有可比较提交使用该已提交曲线作为合成接受目标。

## 复现一次收集

首先测试模型专用收集脚本和镜像，然后从包含该收集脚本的分支触发工作流：

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

在接受更新后的曲线之前，审阅者应验证：

- 所有请求的草稿长度和思考模式均已完成；
- 详细输出内容连贯，并使用预期的思考模式；
- 服务日志中没有回退、禁用草稿或聊天模板错误；
- YAML 元数据与触发时使用的镜像、采样设置、模型和推测方法一致；
- YAML 第一行链接了源 Actions run；并且
- 提交的数值与工作流 artifact 完全一致。

## 当前黄金曲线

| 模型 | 方法 | 黄金 YAML | 源 run |
| --- | --- | --- | --- |
| DeepSeek V4 Pro | MTP | [`dsv4_mtp.yaml`](dsv4_mtp.yaml) | [27180633016](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27180633016) |
| Qwen3.5 397B-A17B | MTP | [`qwen3.5_mtp.yaml`](qwen3.5_mtp.yaml) | [27317114007](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27317114007) |
| Kimi K2.5 | EAGLE3 | [`kimik2.5_eagle3.yaml`](kimik2.5_eagle3.yaml) | [28122195822](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28122195822) |
| MiniMax-M3 | EAGLE3 | [`minimaxm3_eagle3.yaml`](minimaxm3_eagle3.yaml) | [28061204145](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28061204145) |

## 主要参考资料

- [AMD @haic0 的类似工作：PR 参考 1](https://github.com/SemiAnalysisAI/InferenceX/pull/1633)
- [AMD @haic0 的类似工作：参考 2](https://github.com/SemiAnalysisAI/InferenceX/pull/1115#issuecomment-4295024377)
- [SPEED-Bench 论文](https://arxiv.org/abs/2604.09557)
- [SPEED-Bench 数据集和数据集卡](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- [vLLM SPEED-Bench 集成](https://github.com/vllm-project/vllm/pull/36029)
- [vLLM 合成接受支持](https://github.com/vllm-project/vllm/pull/40662)
- [InferenceX 合成接受跟踪 issue](https://github.com/SemiAnalysisAI/InferenceX/issues/1651)
- [InferenceX SPEED-Bench 工作流](../.github/workflows/speedbench-al.yml)
- [InferenceX 早期参考值对齐 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1592)
- [InferenceX 初始 AL 收集器 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1650)
- [InferenceX 多模型 AL 收集器 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1706)
- [InferenceX 多节点合成接受验证](https://github.com/SemiAnalysisAI/InferenceX/pull/1789)
