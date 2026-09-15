# Kimi-K3 TP8×PP2 运行时补丁

[English](README.md) | 中文

本目录包含在固定 vLLM ROCm 镜像 commit
`7c5dc571cbd1064ecc8a9b1045637ff647aa22cb` 上运行及优化
Kimi-K3 FP4 TP8×PP2 所需的运行时补丁。

## 流水线传输

[`pp_async_activation/`](pp_async_activation/) 包含双槽 PP activation ring、
PP1 receive 预投递、调度器 batch cap、上游发送缓冲区生命周期修复，以及
16-rank 传输回归测试。

v5 prepost 补丁会先应用 v2。将本目录挂载为 `/ppasync` 并设置：

```bash
K3_PP_ASYNC_ACTIVATION=1
K3_PP_PREPOST_RECV=1
python3 /ppasync/apply_pp_async_activation_v5.py
```

## 流水线并行下的 DSpark

[`apply_vllm_50514_pp_spec.sh`](apply_vllm_50514_pp_spec.sh) 及 vendored
`vllm_pr50514_*` 文件将 drafter 放在最后一个 PP stage，并跨 PP 传输
Kimi-K3 五个有序 auxiliary hidden states。

[`dspark_cache_reuse/`](dspark_cache_reuse/) 包含以下兼容性修复：

- PP 边界层被重复采集；
- 全局 draft KV-cache group 标记在 PP projection 中丢失；
- 在 EAGLE replay boundary 保留 Mamba state；
- 可选的 EAGLE 尾部 block no-drop 配置。

在 PP speculative-decoding 补丁之后应用 cache 修复，顺序如下：

```bash
python3 patch_dspark_aux_contract_v028.py
python3 patch_dspark_pp_scheduler_eagle_groups.py
python3 apply_dspark_replay_boundary_fix.py
python3 patch_disable_eagle_block_drop_v028.py  # 仅用于 no-drop 实验
```

## 验证证据

v5 传输回归测试通过了 100 个 TP8×PP2 step，覆盖动态 tensor shape 和
sender-side gather policy。

DSpark K=4 的受控 dummy-weight 测试连续发送两次相同的 18,089-token
prompt：

- replay-boundary retention：第二次请求命中 15,360 tokens；
- replay-boundary retention 加 no-drop：命中 16,896 tokens（93.41%）。

真实权重 AgentX 筛选中，仅修复 draft-group 标记时 prefix-cache hit rate
为 6.53%，加入 replay-boundary retention 后提升到 33.95%，再加入
no-drop 后提升到 37.46%。这些短时筛选仅作为实现证据，不能作为正式发布的
benchmark 结果。
