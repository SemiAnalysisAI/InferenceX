# 视频生成测试工具

[English](README.md) | **中文**

本目录是 `video-generation` 基准场景的客户端与服务端：MiniMax-H3 由一个
diffusion 引擎在单节点上提供服务，选定的冻结 YAML 工作负载按每个配置的并发度发送，每段生成的视频都会被解码并校验，每个单元的
测量窗口会发布给 srt-slurm 的原生 GPU 功耗遥测。

调度与结果由 InferenceX 负责。`inferencex-e2e/configs/nvidia-master.yaml`
与 `amd-master.yaml` 声明部署（`h3-bf16-<sku>-<engine>-video`），`infx.matrix` 根据
`perf-changelog.yaml` 生成计划，`run-sweep.yml` 通过 `benchmark-tmpl.yml`
运行 `video` 桶，`python -m infx.launch run` 走 `srt-video` 路径：
`benchmarks/single_node/srt-slurm-recipes/h3/` 下的 SKU 配方通过
`python3 -m infx.bench video serve` 把引擎作为 generic service 启动，
并以 `benchmark.type: custom` 运行 `python3 -m infx.bench video client`。
本目录以 `/video-package` 挂载进任务。master 配置中的每个 `*-video` 条目都写明
所用的 spec 与集合。产物身份、复用校验与合并后发布见
[原生视频交付](../inferencex-e2e/docs/video-generation.md)。

## VideoGenX 标准 v0.1

草案。范围：MiniMax-H3（[`specs/h3-v2.yaml`](specs/h3-v2.yaml) 的 `release-v1` 集合）
与 Wan2.2-T2V-A14B（[`specs/wan22.yaml`](specs/wan22.yaml) 的 `release-v1` 集合），
只含无损档，单节点 1、2、4 或 8 卡，并发 1，引擎为 SGLang、vLLM-Omni 与
Sol-Engine。范围负责人：`@edwingao28`。

结果须满足下表每一条规则，才计入 VideoGenX v0.1。规则改动写进本表并升级版本号。
“执行位置”链接到提交 `89ad295` 中阻断或计算该规则的代码；“缺口”表示目前没有代码检查。

| # | 规则 | 原因 | 执行位置 | 效力 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 每个测量部署按其模型的 spec 与集合原样运行。H3：20 个 prompt，各用一个固定 seed，49 次去噪更新，124 帧、24 fps、1344×768，另有一个独立的 warmup prompt。Wan2.2：4 个 prompt × 5 个 seed，40 步，81 帧、16 fps、832×480，一次 warmup。 | 计划哈希相同的部署跑的是同一份工作。 | [`srt_selected.py:190`][plan-bind] 绑定计划哈希；[`video_artifacts.py:163`][plan-pub] 在发布前复核 | 阻断 | `@functionstackx` | 生效 |
| 2 | 镜像按 SHA256 digest 固定，引擎按 40 位 commit（`VIDEO_RUNTIME_REVISION`）固定，模型按 revision 固定。 | 任何人都能精确复跑同一个点。 | [`srt_selected.py:266`][image]、[`srt_slurm/video.py:336`][engine]、[`srt_slurm/video.py:149`][model] | 阻断 | recipe CODEOWNERS | 生效 |
| 3 | 每个部署只用一个节点。每种 GPU 数量是独立部署，各有自己的 `video/agg-<layout>.yaml` 配方。 | 每种并行布局各自通过 sweep 认证。 | [`validation.py:918`][single-node]（视频只有单节点场景）；[`validation.py:452`][naming] | 阻断 | `@edwingao28` | 生效 |
| 4 | 并发 1。 | 规则 12 的失败容忍度与样本总数都以一次一个请求为前提。 | 缺口：[`validation.py:488`][conc] 接受任意不重复的正并发；目前每一行都是 `[1]` | 阻断 | `@edwingao28` | 生效 |
| 5 | 所有引擎都用 libx264 `ultrafast`、CRF 18、yuv420p 编码；H3 另有 32 kHz 立体声 AAC 音频。 | 编码时间计入测量延迟。 | H3：[`h3-v2.yaml:157`][encoding] 把编码参数计入计划哈希，[`media.py:581`][sei] 从 x264 SEI 头读取 CRF 与 preset。Wan2.2 有缺口：`wan22.yaml` 是 schema 1，[`runner.py:183`][encoding-v2] 既不下发也不检查编码参数；Wan2.2 在 v0.2 迁到 schema 2，计划哈希会变，需要重新 sweep | 阻断 | `@functionstackx` | 生效 |
| 6 | 请求之间不复用工作：prompt embedding（conditioning）缓存关闭。 | 每个请求都完成全部计算。 | H3：[`srt_slurm/video.py:329`][cache] 要求各引擎的关闭开关。Wan2.2 有缺口：该检查只作用于 schema 2 的 spec，待规则 5 所述的 v0.2 迁移 | 阻断 | `@functionstackx` | 生效 |
| 7 | 只允许无损档，见[下方列表](#无损档)。 | 有损加速会改变输出，需要配对质量分，而 v0.1 还没有。 | SGLang：引擎日志 attestation [`attestation.py:36`][policy]，由 [`srt_bundle.py:378`][attest-gate] 把关。vLLM-Omni 与 Sol-Engine 有缺口：[`attestation.py:66`][policies] 只有 SGLang 的策略，且 [`video_artifacts.py:146`][attest-presence] 会放行没有 attestation 的 bundle | 阻断 | `@functionstackx` | 生效 |
| 8 | 未登记在 registry 中的优化一律按有损处理；由 registry 而不是提交者判定。 | 新技术未经审查不能进入无损排名。 | App [`optimizations.ts:26`][registry]。缺口：InferenceX 没有代码记录一次运行用了哪些优化，因此所有已发布运行都显示为无损 | 阻断 | `@SemiAnalysisAI/core` | 生效 |
| 9 | 所有配置都写在配方和 master 配置行里：不改 launcher 或客户端，不修补缓存，不在配方之外加启动参数，设置中不放密钥。 | 配方 fingerprint 能完整描述部署。 | [`srt_slurm/video.py:102`][env] 拒绝 launcher 专用和疑似凭据的环境变量名。缺口：检测不到打过补丁的镜像 | 阻断 | recipe CODEOWNERS | 生效 |
| 10 | 每段视频都通过技术检查：完整解码；视频为 H.264 yuv420p 且分辨率稳定；帧数一致，fps 误差在 0.1% 以内且帧间隔均匀；时长误差在一帧以内；画面不是全黑或全白；需要动作的 prompt 相邻帧有变化。H3 音频为 32 kHz 立体声 AAC，时间戳连续，需要声音的 prompt 电平高于 −80 dBFS。 | 不合格的视频不能计入速度。 | [`runner.py:946`][expected] 从计划生成检查项；[`media.py:806`][clip-valid] 只有全部检查通过才判为有效 | 阻断 | `@functionstackx` | 生效 |
| 11 | 冻结帧比例只记录，不检查。 | 还没有校准过的阈值。 | [`media.py:753`][frozen] 只在计划设置了 `max_frozen_fraction` 时检查；两个 spec 都没有设置 | 参考 | `@functionstackx` | 生效 |
| 12 | 部署须 warmup 视频有效、20 段测量视频全部有效（不容忍失败）、原生功耗有效，才算合格。 | 中位数基于完整的样本总体。 | [`runner.py:1124`][warmup] 在 warmup 失败后不进入测量；[`bench/video.py:28`][complete]；发布时由 [`video_artifacts.py:140`][power] 与 [`video_artifacts.py:183`][population] 检查 | 阻断 | `@functionstackx` | 生效 |
| 13 | 单段延迟从提交请求开始，到下载完生成的 MP4 为止，包含编码和轮询。图表使用有效测量视频的中位数，不计 warmup。 | 所有引擎用同一个定义。 | [`runner.py:1073`][latency]；[`serving.py:91`][median] | 定义 | `@functionstackx` | 生效 |
| 14 | 认证运行是在 PR 某个提交上对最终 changelog 矩阵做的完整常规 sweep，且其 `validate-video-results` 任务通过。合并后发布的就是这次运行，`main` 上不重跑。 | 发布的数字就是审过的数字。 | [`plan.py:178`][append-only] 拒绝 append-only；[`video_artifacts.py:76`][full-sweep] 要求完整视频 sweep；[`run-sweep.yml:910`][validate-job] | 阻断 | `@SemiAnalysisAI/core` | 生效 |
| 15 | 一位合格的 CODEOWNER 提交 PR Review Checklist，再由一位 core 维护者 approve。 | 每次规则改动和结果都有第二个人审。 | `main` 的 ruleset 要求 `@SemiAnalysisAI/core` 的一次 approve。缺口：不要求 code-owner review，admin 合并可以绕过 ruleset | 阻断 | `@SemiAnalysisAI/core` | 生效 |
| 16 | 参考 PSNR 只报告，从不卡复用或合并；`quality_status` 保持 `unjudged`。Wan2.2 没有参考视频，其视频保持未评分。 | 无损引擎和并行布局相对同一参考的 PSNR 本来就差异很大。 | [`quality.py:192`][unjudged]；没有 workflow 把该分数当作门槛 | 参考 | `@functionstackx` | 生效 |
| 17 | VLM judge 为每段测量视频评分，分数显示在视频旁边。它不阻止发布。 | 技术检查判断不了视频是否符合 prompt。 | 缺口：本仓库还没有 judge | 参考 | `@edwingao28` | 生效 |
| 18 | 所有引擎（SGLang、vLLM-Omni 与 Sol-Engine 一样）遵守同一套规则：PR rebase 到 `main`，使用按 digest 固定的公开镜像，并通过引擎日志 attestation。 | 所有引擎用同一个证据标准。 | digest 见 [`srt_selected.py:266`][image]；attestation 见 [`srt_bundle.py:378`][attest-gate]。缺口：[`srt_slurm/video.py:253`][runtime] 只接受 `sglang` 与 `vllm-omni`，attestation 策略也只有 SGLang（见规则 7）。Sol-Engine 的 binder 支持由其自己的 PR 加入；vLLM-Omni 与 Sol-Engine 的 attestation 策略由 InferenceX 添加 | 阻断 | `@edwingao28` | 生效 |
| 19 | 每个 workload 写明每条视频的规模：latent token 数、DiT 前向次数（去噪步数 × guidance 分支数），以及估算的 DiT 计算量。token 数按 DiT self-attention 的完整序列计算；模型把音频或文本 token 放进同一序列时也要计入。每次前向的计算量为 2 × token 数 × 作用于每个 token 的 DiT 权重量，加上 4 × token 数² × attention 宽度（head 数 × head 维度）× 层数；每条视频的总量再乘以前向次数。dashboard 在 workload 摘要中显示这三项。它们只描述 workload，不参与排名；不同 workload 的结果不合并成一条曲线。 | 视频秒数不代表工作量。H3（5.17 s）和 Wan2.2（5.06 s）的视频秒数几乎相同，但 H3 每条视频的 DiT 计算量约为 Wan2.2 的 1.29 倍：1.71e17 对 1.33e17 FLOPs。H3 在约 37,750 个 token（视频 37,296、音频 414、文本约 40）上做 49 次前向；Wan2.2 在 32,760 个 token 上做 80 次前向。full attention 的计算量随 token 数平方增长，每条视频的 token 数又取决于各模型的 VAE 与 patch 大小：H3 把 124 帧压成 37 个 latent 帧，Wan2.2 把 81 帧压成 21 个。 | 缺口：spec、artifact 与 App 都还没有记录这三项 | 定义 | `@edwingao28` | 生效 |

### 无损档

- 允许以下改动，因为它们只改变工作的调度方式，不改变计算内容：tensor、Ulysses、
  ring 与 CFG 并行；text encoder 的 tensor 并行；VAE tiling 与 patch 并行；
  attention 后端与 kernel 选择。
- 同样允许改变权重的存放位置。CPU offload、layerwise offload 与 FSDP 分片只移动
  权重，不改变结果，搬运耗时和其他工作一样计入延迟。SGLang 的 `--performance-mode`
  取 `auto`（默认值）或 `memory` 时，会按 GPU 空闲显存自动决定这些设置；取 `speed`
  时，2 张及以上 GPU 也可能开启 FSDP。因此配方须显式写明该模式。缺口（[#5181](https://github.com/SemiAnalysisAI/InferenceX/pull/5181)
  合并前）：目前没有代码检查配方是否写明该模式；SGLang attestation 只记录
  `performance_mode`，不记录它最终决定的 offload 设置（[`attestation.py:36`][policy]）。
- 不允许：改动任何工作负载或编码参数；BF16 以外的任何精度，包括 FP8、FP4
  及其他量化；跨请求复用工作；TeaCache、FirstBlockCache 等步间缓存；稀疏
  attention；LoRA；配方之外的启动参数或补丁。

[plan-bind]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_selected.py#L190-L191
[plan-pub]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L163-L168
[image]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_selected.py#L266-L268
[engine]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L336-L338
[model]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L149-L150
[single-node]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L918-L920
[naming]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L452-L467
[conc]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L488
[encoding]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/specs/h3-v2.yaml#L157-L164
[sei]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L581-L595
[encoding-v2]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L182-L183
[cache]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L329-L335
[policy]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/attestation.py#L36-L65
[attest-gate]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_bundle.py#L378-L389
[policies]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/attestation.py#L66
[attest-presence]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L146-L150
[registry]: https://github.com/SemiAnalysisAI/InferenceX-app/blob/d4f50964689aaf4beb97c40e0f7476f6b83e0a86/packages/app/src/components/video-benchmark/optimizations.ts#L26-L27
[env]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L102-L108
[expected]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L946-L955
[clip-valid]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L806
[frozen]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L753-L763
[warmup]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L1124-L1128
[complete]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/bench/video.py#L28-L34
[power]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L140-L145
[population]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L183-L191
[latency]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L1073
[median]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/serving.py#L91
[append-only]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/plan.py#L178
[full-sweep]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L75-L76
[validate-job]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/.github/workflows/run-sweep.yml#L910
[unjudged]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/quality.py#L192-L193
[runtime]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L246-L253

## 入口

原生入口 `python3 -m infx.bench video` 直接调用 `evaluator.srt_selected`。
两个动作都会先加载 `specs/` 中选定的 spec 与集合，并核对其 `plan_sha256`
与绑定值一致。

- `serve` 记录封存的服务端 spec 与启动信息，然后在任务的 GPU 上 exec
  引擎（`sglang` 或 `vllm-omni`）。
- `client` 对服务端点按 `CONC_LIST` 的每个并发度运行同一计划，每个并发度一个
  `gpu/c<N>/` 单元，并发布各单元的测量窗口。

任务结束后，launcher 运行 `python -m evaluator.cli srt-finalize`，把各单元、
其功耗校验边车与服务端记录合并为仪表盘摄入的
`h3-results-<run>-<attempt>-<recipe-fingerprint>` 包。

`specs/video_spec.py` 加载 YAML 工作负载、校验并对选定计划做哈希；
`infx.srt_slurm.video` 在绑定测试点时导入它。

离线 Diffusers 参考媒体使用[固定依赖与生成命令](reference/README.md)。
参考生成与服务延迟测试分开，跨引擎质量仍为未评估。

[`specs/h3-v2.yaml`](specs/h3-v2.yaml) 提供独立的 H3 5 秒、20 个不同
prompt 合同，集合选择器为 `release-v1`。请求使用 49 次去噪更新，预期输出
124 帧，warmup prompt 与测量集不重叠。runner 按引擎发送显式 H.264 参数，
并把编码策略计入计划哈希。媒体解码验证可观察的流属性，并从 x264 的 SEI
头读取 CRF 和 preset 固定的参数加以核对。计划只能使用这项检查能核对的
preset。实际编码器实现和跨引擎质量仍需单独证据。比较器将跨引擎质量标为未评估；
同实现数值检查则要求封存的镜像、启动命令、环境、GPU 和算法身份一致。

## 参考 PSNR

[Video Quality](../.github/workflows/video-quality.yml) 在 Run Sweep 结束后，
以 Diffusers 参考视频为基准，为每段测量视频计算 PSNR。它在 GitHub 托管的
runner 上运行 main 分支的代码，sweep 的 `h3-results-*` 产物只作为数据读取。
[`reference/h3-v2.json`](reference/h3-v2.json) 记录参考视频所在的 release、
生成参数与实现身份，以及每个 case 的 prompt、seed、MP4 的 SHA256 和字节数。
任务先下载 release 资源，哈希与描述文件不一致的资源直接拒绝，然后运行：

```bash
python -m evaluator.cli score-quality <artifact-id>=<bundle.zip|dir> ... \
  --descriptor reference/h3-v2.json --references <release-mp4-dir> \
  --run-id <run> --source-revision <git-sha> --output <new-dir>
```

只有记录的 prompt、seed、模型与生成参数都与描述文件一致时才评分，随后由
`compare_media` 比较全部解码帧。其余视频列在 `unscored` 中并注明原因：
`invalid_candidate`、`checksum_mismatch`、`no_reference`，或 `settings_mismatch`
及不一致的字段。无法读取的 bundle 列在 `unreadable` 中，其余 bundle 照常评分。
每个部署只取不晚于该 attempt 的最新 bundle。`h3-quality-<run>-<attempt>` 产物包含 `quality.json`、描述文件
副本和 `SHA256SUMS`。PSNR 只衡量相对单一参考实现的技术保真度，因此
`quality_status` 仍为 `unjudged`。如需为较早的运行补算，按其 `run_id` 与
`run_attempt` 手动触发该 workflow。

仪表板上的[质量指南](https://inferencex.semianalysis.com/zh/vgx/quality)说明如何用这些分数
评估有损加速，并按 GPU 展示一次已发布运行中每段视频的 PSNR。

## 本地检查

```bash
cd video-generation
PYTHONPATH=../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest==9.2.0' python -m pytest -q
```

[Test Video Harness](../.github/workflows/test-video-harness.yml) 会在相关
变更上运行这些 CPU 检查。它们不调用模型、调度器或 GPU；sweep 产物才是验收
证据。
