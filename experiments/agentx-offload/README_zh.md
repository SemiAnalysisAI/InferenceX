# B200 AgentX KV 卸载交叉点实验

[English](README.md) | **中文**

本实验从零开始，不导入之前的 MXFP8 结果、运行 ID、诊断数据、图表或验收结论。
运行记录初始为空。代码维护在已推送的 `experiment/agentx-b200-offload` 分支，
不创建 PR，也不合并。

## 目标与对照

在相同 AgentX 并发和运行条件下，测量增加主机 DRAM、本地 NVMe，或同时增加两者，
何时能改善 HBM-only 的吞吐量与延迟表现。还需将三级缓存分别与两个双级缓存方案比较。
先确定交叉点区间，再补点并重复测量；不能将孤立的优势视为阈值，也不能用平滑曲线填补缺失数据。

| 配置后缀 | GPU KV | 主机 KV | 本地 NVMe KV | 实现 |
| --- | --- | --- | --- | --- |
| `none` | 80 GiB/GPU | 0 | 0 | HBM 前缀缓存 |
| `dram` | 80 GiB/GPU | 739 GB/节点 | 0 | SimpleCPU，lazy |
| `nvme` | 80 GiB/GPU | 无常驻 KV 缓存，仍有传输缓冲区 | 4 TiB/节点 | SimpleCPU disk，lazy，直接 I/O |
| `dram-nvme` | 80 GiB/GPU | 739 GB/节点 | 2 TiB 停止保护阈值 | 原生 CPU + 文件系统分层缓存 |

DRAM 使用 fresh main 根据 B200、TP4 和 `dram-utilization: 0.683` 实际生成的预算，
按十进制 GB 换算，并在启动前检查。HBM 固定为每 GPU 80 GiB、总计 320 GiB，
避免不同连接器的内存布局改变公共预算。所有阈值都依赖这些容量，不是通用并发阈值。

NVMe-only 方案现在使用 4 TiB 有界容量，使单个运行能够使用更多本地阵列空间。
此前 1 TiB 预算生成的运行在记录中保持独立标识，不作为与 4 TiB 方案容量匹配的重复测量。
连接器会预分配已配置的磁盘文件，因此请求并发不会决定磁盘占用：准入要求
4,398,046,511,104 字节容量以及额外 128 GiB 预留空间。c256 完成了全部规范流程，
外部缓存命中率为 61.680%。c384 中点也成功完成，但与 c256 相比，外部缓存命中率
降至 5.299%，每 GPU 总吞吐量下降 46.716%，P90 端到端延迟上升 162.274%。
c512 探针以零请求错误完成了 5,677 个规范 warmup 请求中的 5,676 个，但最后一个请求
超过 1,800 秒 drain 时限，随后分配达到运行时限且未进入 profiling。c320 中点成功完成，
其外部缓存命中率为 24.540%，每 GPU 总吞吐量为 13,193.777 tokens/s，P90 交互性为
11.998 tokens/s/user，P90 TTFT 为 1,331.284 秒。c256、c320 和 c384 形成了陡峭且有序的
缓存复用与性能下降区间。匹配的 HBM-only c320 对照完成了 1,010 个 profiling 响应，
而 NVMe 完成了 1,697 个。在相同 3,600 秒 profiling 窗口内，NVMe 的每 GPU 总吞吐量
为 13,193.777 tokens/s，HBM-only 为 7,997.601 tokens/s（提升 64.972%）；P90 交互性
从 8.047 提升到 11.998 tokens/s/user（提升 49.097%）；P90 TTFT 从 1,990.296 秒降至
1,331.284 秒（降低 33.111%）。NVMe 的平均功耗也低 4.633%，每个成功查询的能耗低
43.241%。这一次匹配运行表明，在本实验条件下 4 TiB NVMe 在 c320 有帮助，但仍需重复运行
才能确认优势幅度是否稳定。下一步运行匹配的 HBM-only c384 对照，以判断越过缓存陡降区后
NVMe 是否仍有优势。该对照完成了 1,048 个 profiling 响应，而 NVMe 完成了 1,157 个。
NVMe 在 c384 仍然胜出，但优势明显小于 c320：每 GPU 总吞吐量为 9,198.026 对
8,277.940 tokens/s（提升 11.115%），P90 交互性为 8.721 对 8.149 tokens/s/user
（提升 7.028%），P90 TTFT 为 2,154.172 对 2,333.924 秒（降低 7.702%）。NVMe 的
平均功耗低 2.710%。匹配的 c448 也已完成：NVMe 的每 GPU 总吞吐量为 8,839.499，
HBM-only 为 8,251.805 tokens/s（提升 7.122%）；P90 交互性从 8.212 提升到
8.973 tokens/s/user（提升 9.273%）；完成响应数为 1,204 对 1,120。P90 TTFT
仅改善 0.427%，P90 端到端归一化交互性低 0.376%；平均功耗低 0.416%，每个成功查询
的能耗低 7.364%。优势仍为正，但继续收窄。匹配的 c480 随后也显示 NVMe 的混合优势：
每 GPU 总吞吐量为 8,642.782 对 8,104.248 tokens/s（提升 6.645%），每 GPU 输出吞吐量
为 87.219 对 74.494 tokens/s（提升 17.083%），P90 交互性为 8.547 对 8.119
tokens/s/user（提升 5.274%），P90 TTFT 为 2,815.145 对 2,947.946 秒（降低 4.505%）。
NVMe 完成了 1,186 对 1,097 个响应，每个成功查询的能耗低 6.940%；但 P90 端到端归一化
交互性低 2.010%，平均功耗高 0.610%。随后 c496 探针在可复用作业的 500 分钟执行时限内
始终处于规范 warmup 阶段。最后一次进度记录显示：5,500 个响应中已返回 5,091 个，已发送
5,459 个，仍有 368 个在途请求；经过 22,834.1 秒后请求错误仍为零。保留的 5,093 条请求
记录全部属于 warmup，profiling 从未开始。该结果仅表示在当前执行预算下 c496 的规范
warmup 局部不可行，不能视为零吞吐量结果。随后，位于两者中点的 c488 在 24,559.47 秒后
以零请求错误完成全部 5,401 个规范 warmup 请求，并完成完整的 3,600 秒 profiling 窗口。
该运行保留 1,168 个成功响应，每 GPU 总吞吐量为 8,541.085 tokens/s，每 GPU 输出吞吐量
为 81.795 tokens/s，P90 交互性为 8.904 tokens/s/user，P90 TTFT 为 2,904.031 秒，
每个成功查询耗能 11,170.326 焦耳；外部缓存服务了 1.443% 的提示词元。全部结果、原始
AgentX、服务日志、GPU 和功耗制品上传完成后，启动步骤因退出码 143 导致工作流最终显示失败，
因此 c488 仍是有效的局部性能点。匹配的 HBM-only c488 对照也完成规范 warmup 和完整
profiling，随后因相同的导出后时限终止而失败。NVMe 的每 GPU 总吞吐量为 8,541.085 对
8,052.723 tokens/s（提升 6.065%），每 GPU 输出吞吐量为 81.795 对 71.868 tokens/s
（提升 13.813%），P90 交互性为 8.904 对 8.153 tokens/s/user（提升 9.206%），
P90 TTFT 为 2,904.031 对 2,998.848 秒（降低 3.162%）。NVMe 完成 1,168 对 1,071
个响应，每个成功查询的能耗低 8.685%，在外部服务 1.443% 提示词元的同时少计算 0.847%
的提示词元。平均功耗低 0.415%，但 P90 端到端归一化交互性低 3.729%。因此 c488 是另一
个局部混合 NVMe 优势点。随后，原生 DRAM c488 对照发送了全部 5,401 个规范 warmup 请求，
并以零请求错误返回 5,395 个；但经过 25,174.7 秒后仍有 6 个 wire 请求未完成，固定的
1,800 秒加速 drain 时限因此到期。保留的 5,395 条记录全部属于 warmup，其中 8 条被归类为
InvalidInferenceResultError；profiling 从未开始，所以该运行是局部 warmup 可运行性失败，
不是性能结果。专属 scratch 已删除并被独立确认不存在。随后，HBM+DRAM+NVMe c488 对照
以零请求错误完成全部 5,401 个 warmup 请求和完整 3,600 秒 profiling，保留 1,904 个成功响应。
其每 GPU 总吞吐量为 13,026.732 tokens/s，每 GPU 输出吞吐量为 144.643 tokens/s，P90
交互性为 12.898 tokens/s/user，P90 TTFT 为 1,806.766 秒，每个成功查询耗能 5,738.145
焦耳。相对于 HBM-only，总吞吐量、输出吞吐量和 P90 交互性分别提高 61.768%、101.263% 和
58.193%，P90 TTFT 降低 39.751%，完成响应数增加 77.778%，每个成功查询的能耗降低
53.092%。但是，原生文件系统层在 2,234,874,994,223 逻辑字节时越过 2 TiB 停止保护阈值，
并继续增长，直到清理时达到 13,255,648,681,519 逻辑字节和 1,117,488 个文件；profiling
后段写入随后出现 ENOSPC。完整性能导出仍可作为原生分层和更大完成队列的局部证据，但不能视为
干净的 2 TiB 容量匹配对照，也不能单独归因于存储层。专属 scratch 已删除并被独立确认不存在。
在文件系统层具备强制容量边界，或监控器能在达到声明保护阈值时立即停止工作负载之前，不再提交
新的三级缓存运行。

监控器现在会在越过保护阈值时解析 shell 的实时进程树，先按最深层优先顺序向工作负载后代进程
发送信号，再向 shell 发送信号，使其执行专属清理 trap。保护记录会保存计划终止的进程列表。
修复后的 c488 复验运行 `37064957726` 随后在规范 warmup 阶段于 2,295,157,759,535 逻辑
字节触发保护。清理时记录 2,299,783,945,775 逻辑字节，仅比保护采样多 4.626 GB，并删除
专属 scratch；独立检查确认该目录不存在。该运行保留 153 条 warmup 记录且没有取消记录，
profiling 从未开始。这验证了及时、确定性的终止与清理，但不会把原生文件系统层变成有界 LRU
缓存；三级缓存指标仍不具备容量匹配比较资格。

下一自适应 NVMe-only 点为 c492，它位于已完成且取得局部混合优势的 c488 与规范 warmup
不可行的 c496 之间。该点保持 MiniMax-M3 NVFP4 模型、vLLM 引擎、TP4 布局、4 TiB
NVMe 预算、AgentX 语料、规范 warmup 和 3,600 秒 profiling 窗口不变。其结果仅作为
c488 到 c496 狭窄边界的局部证据。

三级缓存使用不同连接器和存储策略。固定版本的 FS 层没有有界 LRU 容量参数，
2 TiB 是停止保护阈值而非淘汰配额，并与 NVMe-only 容量独立。运行
`35456989669` 以零请求错误完成规范 profiling，但文件系统逻辑占用达到
1,413,881,142,831 字节并被正确判为无效，因此提高了该保护阈值。只有未触发保护的运行
可以参与比较；触发保护属于失败测量。
必须从日志和指标确认真实缓存容量及后端活动。如果三级缓存胜出，应增加原生 DRAM 对照，
再判断改善是否确实来自存储层，而不是连接器差异。

## 完整规范运行

沿用 main 的 `nvidia/MiniMax-M3-NVFP4` TP4 配方，搭配不可变的 vLLM
`nightly-dee37d89115db4c94a820a79a78a7828e141c910`，以及 EAGLE3-GQA 和
2.78 合成接受长度。四组复用同一启动脚本和固定的 AIPerf 子模块。
保留每条轨迹额外十次预热、全部必需快照预热、seed 42、录制助手响应回放、相同语料和空闲间隔策略，
并运行 **3,600 秒 profiling**。禁止 `agentx-fast`、缩短时长、unsafe 模式、合成工作负载、
自定义客户端或直接提交 Slurm 作业。确认真实语料版本、完整命令及实际 KV 分配后，才能认定对照匹配。

先运行并发 16 的四组对照，然后为四个方案构建完整曲线。首批曲线点为
1、4、8、16、32、64、128、256、320、384、448、480、488、492、496、512、1,024、4,096、8,192、16,384。
根据观测结果在变化区间补任意正整数并发，并跨节点重复区间两侧。
上限保持 16,384。失败或有效完成样本不足只能记录为可运行性结果，不能伪造零吞吐量点。
若规范预热超过工作流时限，需要明确调整执行预算，不能截断预热。

main 原镜像标签从 registry 返回 404，因此四组统一使用上述已发布的替代版本。
[image-provenance.json](image-provenance.json) 记录 registry 和 AMD64 digest、准确引擎提交及兼容性检查。
规范异构 KV 布局补丁可干净地应用，并通过幂等性验证。原镜像未产生任何完成的性能测量。
替代镜像在 Blackwell 上默认让 EAGLE3-GQA 草稿头使用 FA4；运行 `35401459251`
和 `35442679581` 均已进入引擎 warmup，但 FA4 拒绝了广播后的 descale 张量
（`strides[1] == 0`）。后一次运行确认固定版本的引擎会在 Blackwell 上将请求的
FA3 明确重定向回 FA4。因此本实验为该 FLASH_ATTN 草稿头选择 FA2 和未量化的草稿
KV 缓存，同时主模型继续使用 FlashInfer 和 FP8 KV。四组方案使用完全相同的设置。
这是相对 fresh main 的引擎兼容性差异，不是卸载优化。

## 仅使用 InferenceX 基础设施

每次提交前通过实际生成器预览单个配置点：

```bash
uv run --no-project --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate test-config \
  --config-keys agentx-offload-none --conc 16 --no-evals \
  --config-files experiments/agentx-offload/configs.yaml
```

通过 InferenceX 状态 API 和优先级调度器检查容量后，每次提交一个方案和一个并发，
确保四组拥有各自独立的 GitHub workflow run ID：

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref experiment/agentx-b200-offload \
  -f generate-cli-command='test-config --config-keys agentx-offload-none --conc 16 --no-evals --config-files experiments/agentx-offload/configs.yaml' \
  -f test-name='offload-v1-none-c16-r1'
```

其他后缀为 `dram`、`nvme`、`dram-nvme`。在 [runs.json](runs.json) 中记录提交 SHA、运行 ID、
方案、并发、重复编号、节点、状态和产物链接。工作流及结果校验完成前，不能称为完成结果。
失败和重试应保留各自 attempt ID。

每批提交前，查询当前 B200 节点分配情况和所有待处理 B200 作业；统计范围包括本研究、
旧作业及其他用户的作业。可按需要使用空闲且符合条件的节点，但每批提交后，整个 B200 集群最多
只允许一个作业处于排队状态。若已有一个或更多 B200 作业待处理，则暂停新提交，直到待处理数量
降到一个以下。仍需服从明确的优先级预约及工作流或调度器安全限制。
只使用工作流日志、产物及 InferenceX 状态 API，不由操作代理直接执行 SSH、`salloc`、
`sbatch`、`srun` 或 `scancel`。现有 InferenceX runner 内部仍可使用 Slurm。
标准作业申请 `nodes:1`。

任何时刻最多运行一个包含 NVMe 的工作流（`nvme` 或 `dram-nvme`）。提交下一个前，
上一个包含 NVMe 的工作流必须已结束，且其专属 scratch 必须验证 `deleted=true`。
预检要求 4 TiB 声明预算之外再保留 128 GiB 空闲空间。若旧的任务专属 scratch 尚未处理，
则不得提交；也不得在已知空间不足的节点上原样反复重试。此单运行限制独立于上述整个集群
最多一个排队作业的限制。
工作流目前将包含 NVMe 的分配固定到 `study.json` 的节点允许列表；该列表先只包含近期
c256 运行所在节点，其预检空闲空间为 17,763,847,192,576 字节且已验证清理完成。
只有获得新的逐节点存储证据后，才能扩展该列表。

若工作流时限导致常规 `finish` trap 未执行，应在 `cleanup_stale.py` 中登记准确的运行身份和
scratch 名称。c1 和 c4 HBM 维护探针固定到存在已登记 scratch 的节点；在正常实验设置前，它们会验证
`owner.json`，仅删除对应 scratch 子目录，并在新运行产物中保留清理凭据。清理工具不能接受
任意路径，随后产生的 HBM 测量仍执行完整规范流程。

保留 fresh main 的 `$/` 同仓库工作流引用。此语法要求 Actions runner 2.336.0 或更新版本；
actionlint 1.7.12 尚不识别该语法。
仅 `experiment: agentx-offload` 启用实验配置，普通配方保留原行为。

实验的自托管 checkout 暂时使用固定提交的 `actions/checkout@v5.0.1`，仍设置
`persist-credentials: false`。首批 NVMe 和 DRAM 作业在 v7.0.1 的条件式凭据文件
配置后获取仓库失败，尚未申请 GPU。这是针对 [上游路径匹配问题](https://github.com/actions/checkout/issues/2393)
的限定范围兼容性验证，不是卸载性能证据。普通工作流仍使用 v7.0.1。
确认 runner 路径或上游新版本兼容后，应移除此回退。

首批 checkout 失败后，先只重试 NVMe 方案。确认 checkout 和实验启动步骤成功后，
再提交其余对照组。集群和调度器观测均不得超过 90 秒；即使 API 可访问，
过期的调度器快照也不能作为允许提交的依据。

## 证据与可视化

保留标准 `bmk_agentic_*`、`agentic_*`、服务端日志和 GPU 指标产物。
`results/offload_config.json`、`offload-telemetry.jsonl` 和 `offload_cleanup.json`
记录准确容量、存储及直接 I/O 验证、文件占用、节点内存与磁盘计数、以及专属缓存清理结果。
磁盘计数属于整个节点，不等同于本进程 I/O。FS 回退到缓冲 I/O 的运行不能按声明的 NVMe 方案验收。
存储验证通过 `/sys/class/block` 解析挂载设备及其底层叶设备；容器无需仅为证明本地文件系统位于
非旋转 NVMe 设备之上而访问宿主机的 `/dev/md0` 设备节点。

每组对照完成后，比较成功输出吞吐量/GPU、P90 延迟、失败及取消数量、缓存来源和 I/O 证据。
原始 E2E-normalized X 轴需要从同一定义的 profiling 完成请求集合重新计算
`1 / P90(request E2E seconds / output tokens)`；不得把 inverse TPOT 标成此指标。
报告阈值前应重复胜出区间并量化波动。给定延迟 SLO 时，比较各方案满足相同 SLO 的最高吞吐量；
吞吐量和延迟存在取舍时，应同时报告，不能虚构唯一赢家。

应用支持逗号分隔的 GitHub workflow run ID：

- 图表：`https://inferencex.semianalysis.com/inference/minimax-m3?i_seq=agentic-traces&i_prec=fp4&i_pctl=p90&i_metric=y_tpPerGpu&unofficialruns=ID1,ID2,ID3,ID4`
- API：`https://inferencex.semianalysis.com/api/unofficial-run?runId=ID1,ID2,ID3,ID4`

在 `runs.json` 中保留每个 workflow ID，但只将成功完成的运行加入同一个已打开的图表标签页，
并将 URL 保存到台账。新成功结果出现后刷新，等待 unofficial overlay 加载完成，关闭 `Optimal Only`，
并确认图表中可以看到数据点。必须使用本次新实验的真实 GitHub ID，不能使用 Slurm ID。应用当前把 offload 元数据简化为 on/off，
因此还应保持各方案的运行身份可追溯。当前 E2E-normalized 图表明确禁用 unofficial overlay，
原因是缺少持久化的逐请求轨迹。标准 AgentX overlay 可在产物上传后使用；原始 E2E-normalized
对比需要增加应用支持，或单独分析本次保留的新轨迹。不能声称现有 API 已支持该视图。

服务停止后，只删除当前工作流唯一 scratch 子目录中的临时 KV 数据。
不要删除共享模型、镜像或语料缓存。分析后清理临时下载，只保留有价值的新原始证据和来源信息。
旧数据不属于本实验。
