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
本目录以 `/video-package` 挂载进任务。H100、H200、B200 与 MI355X 选用
`h3-v2.yaml/release-v1`，在 C1 测量 20 个不同 prompt。
产物身份、复用校验与合并后发布见
[原生视频交付](../inferencex-e2e/docs/video-generation.md)。

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

## 与无损部署配对评分

在 `run.json` 的 `configuration.optimizations` 中声明了有损优化（如 `cache-dit`）
的部署，以其无损对照部署为基准评分，而不是 Diffusers 参考视频。手动触发
Video Quality 时，将 `baseline_run_id` 设为包含无损部署的运行。也可以是同一次运行，
但部署不会与自身配对。任务下载两次运行各自最新的 `h3-results-*` bundle，然后运行：

```bash
python -m evaluator.cli score-quality <artifact-id>=<bundle.zip|dir> ... \
  --baseline <artifact-id>=<bundle.zip|dir> ... --baseline-run-id <run> \
  --run-id <run> --source-revision <git-sha> --output <new-dir>
```

只有未声明任何优化的部署才能作为基准。部署与硬件、引擎、镜像 digest、runtime
revision、GPU 数、精度、并发数以及 `spec.json` 中的 `server` 参数都相同的无损部署
配对。只有该部署所声明的优化自身设置的参数可以不同：
[`evaluator/quality.py`](evaluator/quality.py) 中的 `OPTIMIZATION_SERVER_KEYS`
允许 `subblock-sparse-attn` 改变 `attention_backend`。每段视频与同一 case、同一
重复轮次的视频配对，且只有 prompt、seed、生成参数和封存请求的哈希都相同时才评分。
其余视频列在 `unscored` 中，原因为 `no_baseline`、`ambiguous_baseline`、
`invalid_baseline`、`settings_mismatch`（附不一致的字段）、`invalid_candidate` 或
`checksum_mismatch`。

每一对视频逐帧计算 PSNR、SSIM（BT.601 亮度，σ 1.5 的高斯窗口）和 LPIPS（AlexNet，
v0.1，权重哈希固定），并给出各指标的整段均值与最差帧。每个部署给出视频级的中位数与
最差值。完全相同的视频记为 `video_identical` 为 true，PSNR 为 `null`，SSIM 为 1，
LPIPS 为 0。`h3-paired-quality-<run>-<attempt>-<baseline-run>` 产物包含
`mode: paired_lossless` 的 `quality.json`，`quality_status` 仍为 `unjudged`。
参考评分仍只计算 PSNR，产物名称不变。

## 本地检查

```bash
cd video-generation
PYTHONPATH=../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest -q
```

[Test Video Harness](../.github/workflows/test-video-harness.yml) 会在相关
变更上运行这些 CPU 检查。它们不调用模型、调度器或 GPU；sweep 产物才是验收
证据。
