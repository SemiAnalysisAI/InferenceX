# 视频工作负载规格

[English](README.md) | **中文**

## MiniMax-H3

`h3.yaml` 定义 SGLang 与 vLLM-Omni 共用的 MiniMax-H3 文生视频及音频工作负载。
`h3_spec.py` 加载、验证规格，并生成 #2916 runner 执行及计算哈希的 plan 对象。
此处包含 H3 规格，因为共享 loader 的默认命令使用它。

## YAML 字段

- `model` 和每个 `generation` 字段都有 `value` 与 `<ref>@<commit>:<path>:<line>` 格式的 `source` 引用。
- `sent.sglang` 和 `sent.vllm-omni` 表示相应引擎的请求是否携带该字段。SGLang 不接受调用方设置的 FPS 和帧数，固定以 24 FPS 运行，并根据 `target` 推导帧数和分辨率。只有 vLLM-Omni 接收 FPS、帧数、宽度和高度。
- `checked_on_clip` 表示是否用该字段核对生成的视频。runner 使用 `frame_count / fps` 检查时长，因此不直接检查 `duration_seconds`。H3 的 107 帧、24 FPS 对应 4.458 秒。
- `prompts` 定义提示词及动态、声音要求。`sets` 中的 case 将提示词与 seed 配对，`seed_sources` 为每个 seed 提供引用。
- `reference` 记录 SGLang 参考运行，只供查阅，不参与读取或哈希。该运行使用 workflow commit `f7c7f58`；其中的 spec、`mvp_runner.py` 和 `mvp_media.py` 与 `0cb4c847d` 逐字节一致。

## 哈希约定

`canonical_json_bytes` 与 #2916 `evaluator/mvp_runner.py` 中的实现逐字节一致。它按键排序、使用紧凑分隔符和 UTF-8，不允许 NaN。

- `plan_sha256(plan_for(spec, set_id))` 与 #2916 runner 对同一 plan 记录的 `plan_sha256` 相同。`frozen-c124` 集合还原 #2916 的 `h3-serving-2gpu-c124-v1` plan，哈希为 `8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68`。
- `spec_sha256(spec)` 覆盖模型、generation 值，以及每个集合解析后的 case 与计数。
- 引用、`sent`、`checked_on_clip`、`reference`、YAML 键顺序、注释和排版均不进入两个哈希。GPU UUID、job ID、端口和主机名也不进入哈希。
- `plan_for(spec, set_id, repetitions=..., warmup_runs=...)` 可覆盖计数。如果计数与集合原值不同，plan ID 会增加 `-r<repetitions>-w<warmup_runs>` 后缀，避免缩短运行与完整运行共用 ID 或哈希。

## 运行测试并打印哈希

从仓库根目录运行：

```bash
uv run --no-project --python 3.12 --with 'pyyaml==6.0.3' --with 'pytest>=8,<9' python -m pytest experimental/video-generation/tests/test_wan_spec.py -q
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' python experimental/video-generation/specs/h3_spec.py
```

第二条命令以 JSON 输出 `spec_sha256`，以及每个集合的 `plan_id` 与 `plan_sha256`。传入另一个规格文件路径可以计算它的哈希。

## Wan2.2-T2V-A14B

`wan.yaml` 使用相同 schema 定义 Wan2.2-T2V-A14B 纯视频工作负载，并由同一个 `h3_spec.py` 加载。`frozen-3733` 集合还原 #3733 的 `wan22-t2v-a14b-serving-smoke-v1` plan，哈希为 `1f1a093b391bfd94f64d6e492948989850508ac4dad472eeaa98785fd26327d6`。`shared-v1` 为 #3733 的每条提示词保留原 seed，并增加一个高 100 的 seed；每个新 seed 引用其原 seed 的来源行。`release-v1` 将四条已有提示词与五个已记录的 seed 交叉组合，每个引擎运行 20 条测量视频和一次预热。

两个引擎接收全部 generation 字段，其中 `frame_count` 作为 `num_frames` 发送。runner 检查视频宽度、高度、帧数和 FPS；纯视频模型约定还要求没有音频流。
[Wan vLLM-Omni 服务记录](wan-vllm-omni-serving_zh.md)保留八卡 H100 上的固定镜像、服务命令和并行配置。`../tests/fixtures/wan-vllm-omni-a8576ccb/` 保留该次请求、服务记录和 ffprobe 输出。Wan 测试位于 `tests/test_wan_spec.py` 和 `tests/test_wan_runner.py`；`.github/workflows/test-h3-video.yml` 在草稿及正式 PR 上运行整个 `experimental/video-generation` 测试集。
