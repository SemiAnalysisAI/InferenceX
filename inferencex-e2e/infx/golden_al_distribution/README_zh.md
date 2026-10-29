# 黄金接受长度分布

[English](README.md) | 中文

本目录包含今后用于在 InferenceX 中标准化推测解码的黄金接受长度（Acceptance Length，AL）曲线。每个 YAML 按模型、思考模式和草稿长度（`num_speculative_tokens`），记录在 SPEED-Bench Qualitative split 的 **coding** 类别上测得的平均 AL。

## 为什么选择 SPEED-Bench

[SPEED-Bench](https://arxiv.org/abs/2604.09557) 是一个统一的推测解码基准，覆盖多样的语义领域和真实的服务场景。其 Qualitative split 包含 880 个语义多样的提示词，分为 11 个类别，每类 80 个。该 split 用于测量接受率（AR）和接受长度（AL）。其 Throughput splits 覆盖固定的 1K–32K 输入长度和多种熵区间，用于系统级评估。该基准使用真实提示词，因为随机 token 输入会扭曲接受行为、专家路由和吞吐量测量结果。

SPEED-Bench 是一个实用的跨引擎标准，而不是 InferenceX 独有的工作负载：

- vLLM 维护者已将 [SPEED-Bench 原生支持](https://github.com/vllm-project/vllm/pull/36029) 合入 `vllm bench serve`，用于 Qualitative AR/AL 测量和 Throughput 评估；相关用法已记录在 [vLLM Benchmark CLI](https://docs.vllm.ai/en/v0.22.0/benchmarking/cli/) 中。
- SGLang 提供原生的 [`speed-bench` 数据集适配器](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/datasets/speed_bench.py)，并在其[服务基准指南](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/developer_guide/bench_serving.mdx)中记录了相关用法。
- SPEED-Bench 论文评估了包括 vLLM、SGLang 和 TensorRT-LLM 在内的生产级引擎，因此其方法适合跨运行时比较。

正是由于这些上游维护者的采用，InferenceX 才使用 SPEED-Bench 作为黄金 AL 收集的共同测量基础。

## 为什么选择 coding 类别

AL 取决于工作负载：草稿模型的预测在某些领域比其他领域更容易被接受。智能体编程占其 token 总量的最大份额。因此，我们使用 SPEED-Bench 的 `coding` 类别来校准合成接受，而不是对角色扮演、翻译或创意写作等无关领域求平均。

## AgentX 公平性指南

根据 AgentX 指南，每个模型、思考模式和草稿长度都有一个已提交的黄金 AL。当某个基准场景启用合成接受后，提交可以选择任意受支持的草稿长度，但不能替换为其他接受目标。不同模型保留各自基于 SPEED-Bench 测得的曲线。所有评估同一模型和模式的提交都使用同一条曲线。

对于 Qwen3.5 的 srt-slurm 启动，srt 驱动会运行 `infx/srt_slurm/synthetic_acceptance.py` 连接器，在应用调用方覆盖设置后，根据生成角色的 `speculative-num-steps` 和 `THINKING_MODE` 选择已提交的曲线。验证 token 不计为额外的 MTP 步数。对于采用合成接受的 SGLang MTP 吞吐测试，如果生成角色缺少 `speculative-num-draft-tokens`，或者其值不是等于 `speculative-num-steps + 1` 的整数，连接器会拒绝启动。无需手动设置接受长度。仅评测运行会移除合成设置并使用真实验证。

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

SGLang 通过其模拟接受环境变量支持同一策略。这些变量设置在服务环境中（srt-slurm YAML 的 `aggregated_environment` / `decode_environment` 部分，或在基准脚本中于启动前导出）：

```yaml
SGLANG_SIMULATE_ACC_LEN: '3.24'   # 来自已提交黄金 YAML 的 AL
SGLANG_SIMULATE_ACC_METHOD: match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE: real-draft-token
```

TensorRT-LLM 通过 [`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`](https://github.com/NVIDIA/TensorRT-LLM/blob/2cbdaa0ffa36fbef7960a0ad9f0458373025fa9f/tensorrt_llm/_torch/speculative/interface.py#L1065-L1078) 支持该策略。**注意存在差一（off-by-one）：** 该变量只统计被接受的*草稿* token，不包括 bonus/验证 token，因此应设置为黄金 AL **减 1**。小数值也受支持。整数部分在每次迭代中始终被接受，小数部分是额外接受一个草稿 token 的概率。例如，黄金 AL 为 `3.5` 时应设置：

```bash
TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=2.5
```

ATOM 通过 `--spec-decode-acceptance-length` 支持该策略，该参数直接接受黄金 AL：

```bash
python -m atom.entrypoints.openai_server \
  --model MODEL \
  --draft-model DRAFT_MODEL \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 3.78
```

接受长度包含目标模型保证产出的验证 token，因此黄金 AL 可以原样填入，**不存在差一问题**，与 vLLM 的 `synthetic_acceptance_length` 和 SGLang 的 `SGLANG_SIMULATE_ACC_LEN` 单位一致。取值范围为 `[1, num_speculative_tokens + 1]`，并会解析为与 vLLM 相同的最小方差逐位置概率表，因此匹配的是接受长度的**分布**，而不仅仅是均值。被接受的位置输出真实的草稿 token，等价于 SGLang 的 `real-draft-token`。该参数还有一种等价的接受率写法 `--spec-decode-acceptance-rate`，取值为 `(AL - 1) / num_speculative_tokens`；两者互斥。相关支持见 [ROCm/ATOM#1948](https://github.com/ROCm/ATOM/pull/1948)。

实测值可从服务端 `/debug/mtp_stats` 的 `average_tokens_per_forward` 字段读取，或从 `atom:mtp_average_tokens_per_forward` Prometheus 指标读取。ATOM 按 `1 + 已接受草稿 token 数 / 验证步数` 计算，与黄金曲线的收集公式相同，可直接比较。

**DSpark 注意事项：** ATOM 禁止将强制接受与 DSpark 置信度调度器（`--dspark-config '{"confidence_schedule": true}'`）同时使用——后者在运行时决定每个请求的验证长度，可能把接受长度压到目标值以下，且无法提前检测。黄金 DSpark 曲线本身也是在未开启自适应验证的条件下收集的，因此关闭它同样是可比较的配置。

这一策略遵循与 MLPerf Inference 相同的广义原则：规定可比较系统测量所需的工作负载规则。InferenceX 评估的是推理系统性能，而不是微调特定基准推测头的能力。

## 黄金 AL 曲线如何收集

一键触发的 [`speedbench-al.yml`](../../../.github/workflows/speedbench-al.yml) 工作流最初由 [InferenceX#1650](https://github.com/SemiAnalysisAI/InferenceX/pull/1650) 引入，随后在 [InferenceX#1706](https://github.com/SemiAnalysisAI/InferenceX/pull/1706) 中扩展到更多 MTP 和 EAGLE3 模型。它取代了 [InferenceX#1592](https://github.com/SemiAnalysisAI/InferenceX/pull/1592) 中早期手工整理的参考值，使精确命令、日志、输出和生成的 YAML 都可以从同一次运行中审计。其流程如下：

1. 维护者触发工作流，指定模型、模型前缀、服务框架（`vllm` 或 `sglang`）、镜像、草稿长度（通常为 1–8）、思考模式、`category=coding` 和 `output-len=4096`。
2. 工作流在 B300 runner 上启动模型，并选择 [`benchmarks/single_node/speedbench/`](../../benchmarks/single_node/speedbench) 下对应的 `<model-prefix>_fp4_b300_<framework>.sh` 收集脚本。若 `collector-variant` 输入非空，则改为选择 `<model-prefix>_fp4_b300_<framework>_<collector-variant>.sh`。例如 `model-prefix=kimik3` 加 `collector-variant=probabilistic_sample_method_block_rejection_sample_method` 会运行 Kimi K3 概率草稿收集脚本。所选脚本不存在时，工作流会在启动前失败。
3. 对每个“思考模式 × 草稿长度”组合，收集脚本使用真实投机解码以及该模型的生产采样和聊天模板设置，启动一个干净的 vLLM 或 SGLang 服务。
4. vLLM 收集脚本读取 vLLM 累计的已接受 token 和验证草稿计数器，通过 `vllm bench serve` 运行 SPEED-Bench Qualitative `coding` 类别中的全部提示词，然后再次读取计数器。SGLang 收集脚本通过 `sglang.bench_serving` 运行相同的提示词，并记录其报告的 `accept_length`。
5. vLLM 收集脚本按以下公式计算平均接受长度：

   ```text
   AL = 1 + (delta accepted draft tokens / delta verification drafts)
   ```

   其中 `1` 是目标模型保证生成的验证 token。结果四舍五入到小数点后两位。
6. 收集脚本生成 YAML 矩阵。工作流将其发布到 GitHub Actions step summary，并上传为 `speedbench-reference-al-<model-prefix>` artifact。
7. 工作流保留服务日志和逐请求详细结果，以便审阅者确认输出合理、思考模式正确，并且没有静默的服务或聊天模板故障。
8. 审阅完成后，将矩阵连同准确的采样元数据和源 Actions run URL 一并提交到本目录。设置 `open-pr=true` 和 `golden-curve=<曲线文件名>`（例如 `deepseek-v4-pro__mtp__native__default-default`）时，工作流会写入本目录下的 `<曲线文件名>.yaml`，首行为源 run URL，再通过 `python -m infx.golden_al_distribution show` 加载校验，然后开启 PR。它会替换整个文件，因此本次未收集的格子会被删除。

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

- 所有请求的草稿长度和思考模式均已完成。
- 详细输出内容连贯，并使用预期的思考模式。
- 服务日志中没有回退、禁用草稿或聊天模板错误。
- YAML 元数据与触发时使用的镜像、采样设置、模型和推测方法一致。
- YAML 第一行链接了源 Actions run。
- 提交的数值与工作流 artifact 完全一致。

## 当前黄金曲线

| 模型 | 方法 | 黄金 YAML | 源 run |
| --- | --- | --- | --- |
| DeepSeek V4 Pro | MTP | [`deepseek-v4-pro__mtp__native__default-default.yaml`](deepseek-v4-pro__mtp__native__default-default.yaml) | [27180633016](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27180633016) |
| DeepSeek V4 Pro 0813 | DSpark（概率采样草稿） | [`deepseek-v4-pro-0813__dspark__native__probabilistic-default.yaml`](deepseek-v4-pro-0813__dspark__native__probabilistic-default.yaml) | [31742838308](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31742838308) |
| DeepSeek V4.1 Flash | DSpark（概率采样草稿 + 块验证） | [`deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml`](deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml) | [34493175056](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34493175056) |
| Qwen3.5 397B-A17B | MTP | [`qwen3.5-397b-a17b-nvfp4__mtp__native__default-default.yaml`](qwen3.5-397b-a17b-nvfp4__mtp__native__default-default.yaml) | [27317114007](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27317114007) |
| Kimi K3 | DSpark | [`kimi-k3__dspark__kimi-k3-dspark__default-default.yaml`](kimi-k3__dspark__kimi-k3-dspark__default-default.yaml) | [30304797750](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/30304797750) |
| Kimi K3 | DSpark（概率采样草稿 + 块验证） | [`kimi-k3__dspark__kimi-k3-dspark__probabilistic-block.yaml`](kimi-k3__dspark__kimi-k3-dspark__probabilistic-block.yaml) | [30316471205](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/30316471205) |
| MiniMax-M3 | EAGLE3 | [`minimax-m3__eagle3__minimax-m3-eagle3__default-default.yaml`](minimax-m3__eagle3__minimax-m3-eagle3__default-default.yaml) | [28061204145](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28061204145) |
| MiniMax-M3 | EAGLE3（GQA） | [`minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default.yaml`](minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default.yaml) | [29784780049](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/29784780049) |
| GLM-5.2 | MTP | [`glm-5.2-fp8__mtp__native__default-default.yaml`](glm-5.2-fp8__mtp__native__default-default.yaml) | [28058352479](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28058352479) |
| GLM-5.3 | MTP（临时，K=3 取自 GLM-5.2） | [`glm-5.3-fp8__mtp__native__default-default.yaml`](glm-5.3-fp8__mtp__native__default-default.yaml) | [28058352479](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28058352479)（GLM-5.2） |
| Qwen3.8-Flash-Next | MTP (native) | [`qwen3.8-flash-next-fp8__mtp__native__default-default.yaml`](qwen3.8-flash-next-fp8__mtp__native__default-default.yaml) | [33034290269](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33034290269) |
| Muse Spark Open | DFlash2 | [`muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml`](muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default.yaml) | [38808046215](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38808046215) |

## 在 srt-slurm 运行中使用黄金曲线

- 不要在 SRT 方案、主配置或启动器中硬编码合成接受长度，也不要手动添加 `SYNTHETIC_ACCEPTANCE_LENGTH`、vLLM `synthetic_acceptance_length`、SGLang `SGLANG_SIMULATE_ACC_LEN` 或 TRT-LLM `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` 设置。上文的引擎示例展示的是连接器注入的内容，而不是方案应设置的内容。
- 通过 srt 驱动（`infx/launch/drivers/srt/`）提交方案。它会运行 [`infx/srt_slurm` 连接器](../srt_slurm/synthetic_acceptance.py)，以原生 SRT `--set` / `--unset` 覆盖应用设置。直接调用上游 `srtctl` 会跳过这一选择。
- 在方案中显式保留投机方法、draft 模型、draft token 数和相关采样设置。连接器会在调用方覆盖之后，把生成角色（decode，否则为 aggregated）的设置与 `MODEL_PREFIX` 和 `THINKING_MODE` 组合来选择曲线。Kimi DSpark 需将 `draft_sample_method` 设为 `greedy` 或 `probabilistic`。
- 仅评测和非 AgentX 运行使用真实验证，连接器会移除陈旧的合成设置。非投机角色不会收到模拟设置。`RUN_EVAL` 不会关闭吞吐部分的模拟。
- 缺少黄金曲线或 draft 长度未测量时，会在提交前失败。新组合应补充实测黄金数据，而不是使用猜测或硬编码的长度。

## 曲线身份与命名

每个曲线文件命名为 `<model>__<method>__<draft>__<draft_sample>-<verify>.yaml`：曲线的模型键、方法（模型原生头为 `mtp`，以及 `eagle3`、`dspark`、`dflash`）、草稿模型（`native` 或草稿模型名），以及收集脚本设置的草稿采样器和验证器（沿用引擎默认值时为 `default`）。

[`identities.yaml`](identities.yaml) 为每条曲线提供两个块。新曲线合并前必须同时具备二者；测试会据此检查每个文件名。

`select` 决定哪些运行使用该曲线。`curve_name` 返回最具体的匹配曲线（声明 `draft_family`、`draft_sample`、`verify` 最多的那条），没有匹配或最具体的多条并列时报错。

| 键 | 含义 |
| --- | --- |
| `model_prefixes` | 使用该曲线的 InferenceX 模型前缀 |
| `method` | `mtp`、`eagle3`、`dspark` 或 `dflash`。SGLang 的 `EAGLE` 和 `NEXTN` 表示模型原生 `mtp` 头，MiniMax-M3 的 EAGLE 除外，其为 `eagle3` |
| `draft_family` | 仅在同一模型的曲线因草稿模型而不同时设置：草稿模型名含 `gqa` 时为 `gqa`，否则为 `default` |
| `draft_sample` | 仅在同一模型的曲线因草稿采样器而不同时设置：运行的 `draft_sample_method` 必须与之相等 |
| `verify` | 仅在同一模型的曲线因验证器而不同时设置：`standard`、`block` 或 `relaxed`。运行的验证器按审计相同的方式读取；未知验证器（ATOM、TileRT）不会匹配设置了 `verify` 的曲线 |

`collection` 记录曲线的测量方式：`engine`、`draft`、`draft_sample`（`greedy`、`probabilistic`）、`verify`（`standard`、`block`、`relaxed`）、`adaptive`、`topology`（`chain`、`tree-topk<N>`）、`draft_precision`（草稿权重的量化方式）、`draft_kv_cache_dtype` 和 `kv_cache_dtype`（目标模型的）。`provisional: true` 表示该曲线并非针对该模型测量。每项为 `{value, source}`：

| Source | 含义 |
| --- | --- |
| `explicit` | 由收集脚本或配方显式设置 |
| `documented` | 由 connector 或引擎确定：配方中的 synthetic 采样器会变为 `block`；block、relaxed 和 adaptive 验证均需显式开启，未设置即为 `standard` 且非自适应 |
| `assumed` | connector 自身的判断，例如 ATOM DSpark 采用 probabilistic 草稿 |
| `default` | 未设置，沿用引擎默认值；`value` 为 null |
| `unknown` | 无法读取，例如 ATOM 的验证器；`value` 为 null |

[`infx/srt_slurm/spec_identity.py`](../srt_slurm/spec_identity.py) 用同一词汇描述运行的生成角色，并对照所选曲线进行审计，但不改变选择结果。KV cache dtype 会做归一化：`fp8_e4m3` 即 `fp8`，`bfloat16` 即 `bf16`，`auto` 为引擎默认值。结果类型包括 `mismatch`（双方值均已知且不同）、`unverified`（任一方为引擎默认值或未知，双方同为同一引擎默认值时除外）、`adaptive`（不适用固定 k 曲线）、`provisional` 和 `engine`。审计扫描矩阵中的每个推测解码角色：

```bash
python -m infx.matrix.generate full-sweep --config-files configs/nvidia-master.yaml configs/amd-master.yaml --scenario-type agentic-coding > matrix.json
python -m infx.srt_slurm.spec_identity snapshot matrix.json --root .
```

## 查询曲线

本目录即 `infx.golden_al_distribution` Python 包。`golden_length(model_prefix, spec, thinking)` 返回 AgentX 合成接受所使用的数值，`curve_name` 给出某个投机解码配置对应的 YAML。也可以在 shell 中执行相同的查询：

```bash
python -m infx.golden_al_distribution list                          # 所有曲线及其已测量的草稿长度
python -m infx.golden_al_distribution show qwen3.5-397b-a17b-nvfp4__mtp__native__default-default              # 单条曲线的全部 AL
python -m infx.golden_al_distribution lookup qwen3.5 mtp 3          # 3.39（默认 thinking_on）
python -m infx.golden_al_distribution lookup kimik3 dspark 7 --draft-sample-method probabilistic --json
```

`lookup` 接受 InferenceX 模型前缀、引擎的方法名（`mtp`、`eagle`、`nextn`、`eagle3`、`dspark`、`dflash`）以及 `num_speculative_tokens`。`--draft-model` 用于选择 MiniMax GQA 曲线，`--draft-sample-method` 用于在两条 Kimi DSpark 曲线之间选择。缺少曲线或数值时以非零状态退出，与启动时的失败方式一致。

## 主要参考资料

- [AMD @haic0 的类似工作：PR 参考 1](https://github.com/SemiAnalysisAI/InferenceX/pull/1633)
- [AMD @haic0 的类似工作：参考 2](https://github.com/SemiAnalysisAI/InferenceX/pull/1115#issuecomment-4295024377)
- [SPEED-Bench 论文](https://arxiv.org/abs/2604.09557)
- [SPEED-Bench 数据集和数据集卡](https://huggingface.co/datasets/nvidia/SPEED-Bench)
- [vLLM SPEED-Bench 集成](https://github.com/vllm-project/vllm/pull/36029)
- [vLLM 合成接受支持](https://github.com/vllm-project/vllm/pull/40662)
- [ATOM 强制接受长度支持](https://github.com/ROCm/ATOM/pull/1948)
- [InferenceX 合成接受跟踪 issue](https://github.com/SemiAnalysisAI/InferenceX/issues/1651)
- [InferenceX SPEED-Bench 工作流](../../../.github/workflows/speedbench-al.yml)
- [InferenceX 早期参考值对齐 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1592)
- [InferenceX 初始 AL 收集器 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1650)
- [InferenceX 多模型 AL 收集器 PR](https://github.com/SemiAnalysisAI/InferenceX/pull/1706)
- [InferenceX 多节点合成接受验证](https://github.com/SemiAnalysisAI/InferenceX/pull/1789)

DeepSeek V4.1 Flash 包含草稿长度 1–5 在 thinking 开关两种模式下的实测值。同一镜像在[运行 34494319147](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34494319147) 中于服务启动前拒绝长度 6–8，因此未为这些长度填写 AL 数值。
