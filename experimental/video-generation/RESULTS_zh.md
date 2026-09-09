# H3 后端结果契约

[English](./RESULTS.md) | **中文**

`result.json` 是下载后的 H3 CI 产物面向前端的入口。其 `schema_version`
为 **1.0.0**，[result.schema.json](./result.schema.json) 定义对外结构。
消费者必须拒绝未知版本。此契约不负责写入 InferenceX 数据库，也不代表发布验收。

产物保留原始媒体、请求记录、遥测、运行时及客户端日志、模型与运行时身份、
Slurm 回执，以及原有便携报告 `report/index.html`。导出器新增 `result.json`、
带时间戳的 `power/baseline.json`、`power/candidate.json` 和 `power-report.html`。
原文件不会被重写。发布方在导出后更新 `SHA256SUMS`；重处理时另存原始校验文件。
媒体及报告路径相对产物根目录，并附 SHA256。打开报告前需解压整个产物。

## 读取结果

1. 先检查 `status` 和 `invalid_reasons`。无效产物会写入失败结果并抛出错误，
   让 CI 保留日志，同时使导出任务失败。
2. 分开读取 `workload_status` 与 `regression_status`。A/A 视频有效且负载执行
   成功时，未校准策略仍可令回归结论为 `inconclusive`。此 MVP 的
   `release_qualified` 始终为 false。
3. 检查 `roles.<role>.metrics.status` 和各功率阶段的 `valid`。
   导出完成不代表所有功率数据都有效。无效功率及能量为 null，不能当作零。
   一个阶段通过不能使另一个阶段有效。
4. `execution.ci` 标识**原始 GPU 执行**，包括提交、运行及尝试编号。
   `producer` 标识导出器提交、当前 CI 和源码哈希。重处理不会产生新的 GPU 测量。
5. `hardware.selected_gpu_count` 是实际测量设备数；`reserved_gpu_count` 来自
   Slurm `AllocTRES`。保留八张卡、使用四张卡时，功率只覆盖四张卡，计算额度
   则按八张卡的保留时间计费。

`workload.plan` 固定提示词、随机种子、视频尺寸/时长/音频格式、步数、重复次数
和预热次数，`workload.server` 保留运行时设置。`execution` 关联 CI、Slurm、
GPU UUID、模型清单及实测运行时源码。导出器检查现有完整校验清单，拒绝不安全
路径及符号链接，复用 `verify_measurement_job`，并检查原报告的本地引用。
这是对可信运行器记录和媒体哈希的验证；导出器不会重新解码媒体，也不提供独立
硬件认证。此前完整解码分析仍绑定原始媒体字节。若提供独立获取的 GitHub 元数据，
必须核对成功的 H3 作业、运行编号、尝试编号、URL 及执行提交。同次运行导出时，
只有与导出器运行、尝试、仓库及提交完全一致的工作流才允许仍为 `in_progress`；
`workflow_status_at_export` 保留此状态，同时仍核对已经完成的 H3 作业和下载产物。

## 指标定义

| 指标 | 单位与边界 | 有效性及限制 |
| --- | --- | --- |
| 请求延迟 | 从提交到下载完成并通过技术校验的秒数 | 只统计有效测量视频，不含启动和预热；保留终态、下载及校验分段时间。 |
| 有效视频数/秒 | 有效测量视频数 / 串行测量区间墙钟秒数 | 墙钟时间包含失败尝试；不是并发饱和吞吐。 |
| 完成计数 | 计划、尝试、完成、有效、失败、未启动的视频数 | 完成的视频仍可能技术校验失败；预热记录独立保留。 |
| GPU 显存 | 每个选中 UUID 的设备已用 MiB 观测峰值 | 保留原有整个角色及含预热客户端区间，不是精确分配器峰值。 |
| 技术完整性 | 完整视频/音频解码检查，各检查自带单位 | 覆盖解码、几何、时长、时间戳/帧间隔、运动及声音缺陷；不代表语义或感知质量。 |
| 配对输出保真度 | 视频 PSNR dB、音频频谱余弦及 RMS 绝对比例误差 | 比较原始解码输出；视频完全一致时，有限 PSNR 为 null，`exact_match=true`。 |
| GPU 功率 | 每卡带时间戳的 W，及选中 GPU 的功率总和 | 板级传感器读数含设备显存，不含主机及未选中的 GPU。 |
| 平均/观测峰值功率 | 积分能量 J / 阶段秒数；区间内传感器 W 最大值 | 平均值按时间加权；峰值是采样观测值，不是瞬时电气峰值。 |
| 每个有效视频的 GPU 能量 | 梯形积分 J / 技术校验有效的测量视频数 | 分子包括所有已尝试生成区间，即使输出失败或无效；不含下载/本地解码。任一区间无效或有效视频数为零时为 null。 |

每个功率文件具有版本化的 `sample_series`、`windows`、`phases`、`semantics`
和 `clock_alignment`。分别记录启动、每次预热，以及每次测量从提交到观测到
提供方终态的区间。区间保留时钟来源及不确定性、精确单调时钟边界、每卡采样数
和间隔、覆盖秒数/比例、边界包围情况及无效原因。只有所有必需区间有效时，才
汇总该阶段。即使派生指标被保留为空，完整时间序列仍可下载。

积分复用 InferenceX 的梯形功率积分器，边界做线性插值，不做外推。UUID/进程
归属、有限读数、时间戳顺序、阶段重叠、时钟一致性及最大间隔共同决定有效性。
允许间隔为 `3 × 请求采样间隔`（本次运行为 3 秒）。旧版 UTC 事件重建必须与记录的单调
时钟耗时一致；启动边界未被覆盖时，该阶段可能无效。H200 NVML 功率读数有向后
平均的窗口，阶段边缘还受传感器平均窗口影响。不能从瓦数采样推断能量计数器。

## TDP 与架构结论

仅凭 `NVIDIA H200` 名称不能确定 SXM 形态或 700 W TDP。只有明确验证、附来源
且绑定相同物理 UUID 的硬件记录，才能使 `hardware.tdp` 可用。此时导出器提供
平均功率及观测峰值相对于所选 GPU 总规格 TDP 的比例。这些是描述性比例，
不是配置功率上限，也不是已校准的判定门槛。

后续只读硬件盘点保存在 `later_hardware_observation`，保留自身 CI/Slurm 身份和
时间。其配置、默认、实际执行及最大功率上限**不能回填历史生成设置**。
原始运行没有记录的历史上限仍为不可用。新运行会在
`configured_power_limits.by_role` 中保留各角色执行前后的配置、实际执行、默认
及最大功率上限（W）。每个快照具有观测时间和有效性；UUID、读数或时间不匹配
时保留该快照为空。`same_observed_values` 只比较两个端点，不能证明中间功率上限
始终不变。

观察到 H3 板级功率较高，只能说明该负载及硬件配置。声称其与 LLM 存在架构差异，
仍需匹配硬件/拓扑、精度、功率上限、采样与测量窗口、预热及负载，区分 LLM
prefill/decode，并进行重复可比测量。这些 A/A 点不证明统计显著性、通用视频
质量、性能提升或发布就绪。

## 导出 API

从仓库根目录执行，将仓库根目录及 `experimental/video-generation` 加入
`PYTHONPATH`，然后调用：

```python
from pathlib import Path
from evaluator.mvp_result import write_result

write_result(
    Path("/absolute/path/to/copied-source-artifact"),
    producer={"git_commit": "<40-character exporter commit>", "ci": {"run_id": "<export CI>"}},
    source_ci=trusted_github_run_metadata,
    hardware_profile=optional_later_inventory,
)
```

使用新的副本。`source_ci` 使用 GitHub CLI 字段 `databaseId`、`runAttempt`、
`headSha`、`url`、`status`、`conclusion`、`jobs`，必须独立于产物获取。
可选硬件记录必须关联相同 GPU UUID；已验证的 TDP 需包含 `status`、
`watts_per_gpu`、`hardware_variant`、`source_url` 和证据。
发布方负责最终校验和、上传及下载验收。不得修改旧 CI 或测量身份，将导出伪装
成新基准运行。

## 已观测的 A/A 证据

以下结果复用已保留的八秒机械狐狸运行：
[CI run 34293342829](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34293342829)，
提交 `65699f7c6`，Slurm **82261.0**。
[原始媒体、遥测及报告](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34293342829/artifacts/10082823150)
可下载。四张选中的 H200 GPU 按顺序运行相同运行时版本，每个角色各执行一次预热
和一次视频测量。

| 八秒视频测量 | Baseline | Candidate |
| --- | ---: | ---: |
| 从提交到媒体校验完成的延迟（s） | 149.768568 | 149.355090 |
| GPU 板级平均总功率（W） | 2737.371816 | 2731.735261 |
| 平均功率 / 已验证的总规格 TDP | 97.7633% | 97.5620% |
| 每个有效视频的 GPU 能量（J） | 406794.062298 | 404620.528709 |
| 采样覆盖比例 | 1.0 | 1.0 |
| 最大观测采样间隔（s） | 1.3303 | 1.3567 |

每张 GPU 的平均功率约为 **680–688 W**。两个测量生成区间的起止边界均被遥测
包围，最大采样间隔均低于 3 秒有效性上限。功率和能量覆盖提交到观测到提供方
完成的区间；延迟还包含媒体传输及校验，两者测量边界不同。

后续[硬件盘点运行 34297499754](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754)
（[原始盘点产物](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754/artifacts/10083702100)）
在 `worker-10` 上完成 Slurm **82290.0**，于 `2026-09-09T01:02:44Z` 观测原来的
四个 UUID。NVIDIA H200 PCI 设备/子系统编号 `233510DE` / `18BE10DE` 对应 SXM
板卡，厂家最大可配置 TDP 为[每张 700 W](https://www.nvidia.com/en-us/data-center/h200/)，
四张所选 GPU 合计 **2800 W**。实测平均功率分别为此规格值的
**97.7633% / 97.5620%**。按描述性的 90%“接近 TDP”标准，两者均满足，因此这些
H3 生成测量区间支持 Oren 的假设。该标准不是经过校准的基准门槛，也不能证明
与 LLM 存在架构差异。

后续盘点中，四张卡的配置、实际执行、默认及最大功率上限均为 700 W；原始生成
时的设置仍未知。原始盘点 profile 因 PCI 编号未带 `0x` 而将形态记为 unknown；
导出器现用自身 producer 提交重新解析保留的 XML，保留源 profile，不发起新的
GPU 查询。可使用[仅 CPU 的重放命令](README_zh.md#结果与本地检查)，将此次已验证
盘点与原始 H3 产物一起复用。

此前四秒 A/A 运行保存在
[CI run 34291306687](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34291306687)，
Slurm **82260.0**，并有
[原始产物](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34291306687/artifacts/10081961245)。
其 baseline 测量区间的结束边界未被遥测包围，因此功率及能量保留为空。
这不会使已保留的负载执行、延迟和媒体结果失效。

每个时长都是单独固定的负载。不能合并这些时长的数据，也不能将两者延迟差异
解释为回归。每个角色只有一个测量视频，只能提供点估计；负载执行通过，回归
仍未校准，结论为 inconclusive。
