[English](2026-08-24-h200-fp4-fp8-power-ab-design.md) | [中文](2026-08-24-h200-fp4-fp8-power-ab-design_zh.md)

# H200 FP4 与 FP8 8K/1K 功耗 A/B 设计

## 目标

在现有 H200 集群上，对比 Qwen3.5 NVFP4 checkpoint 与对应 FP8 checkpoint 的交互性和能耗。工作负载固定为 8,192 个输入 token 和 1,024 个输出 token，采用 SGLang 单节点推理，不启用投机解码。

这是 Hopper 部署实验。H200 不执行原生 NVFP4 Tensor Core 计算；SGLang 通过 Marlin weight-only W4A16 fallback 运行 NVFP4 checkpoint，而 FP8 实验组使用 Hopper 原生 FP8 路径。结果必须按实际执行路径标注，不能表述为原生 FP4 与 FP8 的硬件能效对比。

## 冻结的对比契约

- 模型架构：Qwen3.5-397B-A17B。
- Checkpoint：`nvidia/Qwen3.5-397B-A17B-NVFP4-V2` 与 `Qwen/Qwen3.5-397B-A17B-FP8`。
- 硬件：单台 H200 节点；成对实验使用同一 runner cohort。
- 推理引擎：`lmsysorg/sglang:v0.5.14-cu130`。
- 工作负载：固定序列长度，ISL 8192、OSL 1024；random-range ratio 沿用 InferenceX 标准固定序列基准测试配置。
- 推理模式：仅 STP；关闭 radix cache；KV cache 使用 FP8 E4M3；Mamba state 使用 BF16。
- 主要成对拓扑：TP8/EP8。
- 主要并发点：1、2、4、8、16、32、64。
- 重复次数：通过 canary gate 后，每个精度和并发组合独立重复三次。

## 运行时实现

在 `configs/nvidia-master.yaml` 中增加 `qwen3.5-fp4-h200-sglang`。初始搜索空间与现有 H200 FP8 的 TP8/EP8 拓扑保持一致，并使用主要并发列表。

增加 `benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh`。脚本沿用现有 H200 FP8 harness，但显式选择兼容 Hopper 的 NVFP4 fallback：

```text
--quantization modelopt_fp4
--fp4-gemm-backend marlin
--moe-runner-backend marlin
--attention-backend flashinfer
--kv-cache-dtype fp8_e4m3
--disable-radix-cache
```

B200 FP4 脚本中的 `trtllm_mha` attention 和 `flashinfer_trtllm` MoE 设置仅适用于 Blackwell，不能复制到 H200。

## 分阶段执行与 gate

1. 从同一分支生成并检查 FP4 c4 TP8/EP8 job 和 FP8 c4 TP8/EP8 control。
2. 首先只 dispatch 这两个 canary。
3. Canary 只有在全部请求完成、workload metrics 有效、功耗采集有效、FP4 server log 明确确认两个 Marlin backend，且不存在持续无进展或功耗跌至 idle 的情况时才算通过。
4. 两个 canary 均通过后，对 TP8/EP8 并发矩阵执行每个点三次的完整扫描。
5. 随后从低并发到高并发逐级探测两种精度的 TP4/EP1；每个实验组遇到第一个可复现的显存或运行时 cliff 后停止。FP4 TP2/EP1 作为独立的 GPU consolidation 部署实验。
6. 只有 c64 TP8/EP8 健康且显存余量充足时，才尝试 c96 和 c128。

本实验不包含 MTP。此前 FP4 与 MTP 组合曾在 SGLang 内部 stall，而且加入 MTP 会干扰精度路径的对比。

## 测量指标与解释边界

文章主图：

- measured average board W/GPU 与单用户交互性（`1 / mean TPOT`）的关系；
- measured J/output-token 与单用户交互性的关系。

辅助指标包括 total board J/query、总吞吐量和单卡输出吞吐量、mean/tail TTFT、mean TPOT 以及端到端延迟。功耗积分只使用通过验收的 workload window。

只有拓扑相同的 FP4/FP8 成对实验才能支持精度执行路径对比。TP2、TP4 和 TP8 可以用总能耗、吞吐量和 GPU 数量比较部署经济性，但 GPU 数量不同的对比不能描述为纯精度效应。

## 验证

- 增加 matrix generation test，确保 H200 FP4/FP8 的 TP8/EP8 8K/1K rows 成对且平衡。
- 运行完整 `utils/matrix_logic/` 测试套件。
- 在本地生成精确的 c4 命令并检查输出 JSON，不能只依据 filter 行为推断。
- 对新增基准测试脚本执行 shell syntax check。
- 远程 dispatch 前核对已推送的 SHA 和生成的矩阵数量。
- 只有 workflow 成功且 artifact 中 workload 与功耗窗口都有效时，数据才能验收。

## 交付物

- 一个隔离分支，包含配置、Hopper FP4 launcher、测试和双语设计文档。
- 两个 c4 canary job 及其基于 artifact 的验收结果。
- Canary 通过后，完成 TP8/EP8 重复扫描，再分阶段执行 TP4/TP2 和 c96/c128 探测。
- 一份结果 ledger，明确区分原生 FP8、NVFP4 Marlin fallback、同拓扑因果对比和 GPU consolidation 部署对比。
