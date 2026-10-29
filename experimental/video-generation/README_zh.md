# H3 视频生成测试工具

[English](README.md) | **中文**

本目录是 `video-generation` 基准场景的客户端与服务端：MiniMax-H3 由一个
diffusion 引擎在单节点上提供服务，[`specs/h3.yaml`](specs/h3.yaml) 中冻结的
工作负载按每个配置的并发度发送，每段生成的视频都会被解码并校验，每个单元的
测量窗口会发布给 srt-slurm 的原生 GPU 功耗遥测。

调度与结果由 InferenceX 负责。`inferencex-e2e/configs/nvidia-master.yaml`
声明部署（`h3-bf16-<sku>-<engine>-video`），`infx.matrix` 根据
`perf-changelog.yaml` 生成计划，`run-sweep.yml` 通过 `benchmark-tmpl.yml`
运行 `video` 桶，`python -m infx.launch run` 走 `srt-video` 路径：
`benchmarks/single_node/srt-slurm-recipes/h3/` 下的 SKU 配方通过
`python3 -m infx.bench video serve` 把引擎作为 generic service 启动，
并以 `benchmark.type: custom` 运行 `python3 -m infx.bench video client`。
本目录以 `/h3-package` 挂载进任务。H200 选用 20 条视频的
`release-v1` 集合；B200 在 C1/C2/C4 选用 4 条视频的 `frozen-c124` 集合。
产物身份、复用校验与合并后发布见
[原生视频交付](../../inferencex-e2e/docs/video-generation.md)。

## 入口

`python -m evaluator.cli`（`vgbench`）提供原生入口调用的命令：

- `srt-yaml-serve`：加载 `specs/h3.yaml` 中选定的集合，核对其 `plan_sha256`
  与绑定值一致，记录封存的服务端 spec 与启动信息，然后在任务的 GPU 上 exec
  引擎（`sglang` 或 `vllm-omni`）。
- `srt-yaml-client`：对服务端点按 `CONC_LIST` 的每个并发度运行同一计划，
  每个并发度一个 `gpu/c<N>/` 单元，并发布各单元的测量窗口。
- `srt-finalize`：把各单元、其功耗校验边车与服务端记录合并为仪表盘摄入的
  `h3-results-<run>-<attempt>-<recipe-fingerprint>` 包。

`specs/h3_spec.py` 加载 YAML 工作负载、校验并对选定计划做哈希；
`infx.srt_slurm.video` 在绑定测试点时导入它。

## 本地检查

```bash
cd experimental/video-generation
PYTHONPATH=../../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest -q
```

[Test H3 Video Harness](../../.github/workflows/test-h3-spec.yml) 会在相关
变更上运行这些 CPU 检查。它们不调用模型、调度器或 GPU；sweep 产物才是验收
证据。
