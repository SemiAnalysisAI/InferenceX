# 独立运行 AgentX 测试工具

<div align="center">

[English](agentx-standalone.md) | **中文**

</div>

使用下方的 `aiperf profile` 命令，对已运行的 OpenAI 兼容服务执行 AgentX。
客户端不需要 GPU、Slurm、GitHub Actions 或 InferenceX 服务启动器。
推理服务需单独启动和配置，可参考
[ATOM 的 AgentX recipe](https://github.com/ROCm/ATOM/blob/main/recipes/Agentic-Kimi-K3.md)。

本页回放参数依据 InferenceX 的
[`build_replay_cmd()`](../benchmarks/benchmark_lib.sh) 和
[运行时设置](../benchmarks/runtime_settings.sh)。
若参数发生变化，以当前检出的源码为准。通过工作流执行测试和评估时，请参阅
[评估与 AgentX 操作流程](eval-agentx-procedures_zh.md)。

## 安装仓库锁定版本的客户端

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)，
以及用于安装依赖和下载公开轨迹数据集的网络连接，还需能够访问所测模型的 tokenizer。
客户端可以运行在推理服务所在主机，也可以运行在能访问该服务的另一台机器上。

在准备存放仓库的目录中执行：

```bash
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git
git -C InferenceX submodule update --init --depth 1 inferencex-e2e/utils/aiperf
cd InferenceX/inferencex-e2e

uv venv --python 3.11 .venv-agentx
uv pip install --python .venv-agentx/bin/python \
  -e ./utils/aiperf 'datasets>=4.7.0'
source .venv-agentx/bin/activate

git rev-parse HEAD
git -C utils/aiperf rev-parse HEAD
aiperf --version
```

若已有仓库，在仓库根目录更新子模块，然后从 `cd inferencex-e2e` 开始执行。
复现特定 InferenceX 运行时，先检出该运行的版本，再更新子模块。
使用该版本锁定的 AIPerf 提交；直接 `pip install aiperf` 或使用持续更新的
AIPerf 分支不能提供相同的版本保证。本步骤仅安装回放客户端及其依赖，
不安装 InferenceX 编排工具，也不改变推理服务环境。

后续命令均在同一已激活的 shell 中执行，工作目录为 `inferencex-e2e/`。
若 tokenizer 仓库有访问限制，请先完成 Hugging Face 身份认证。
`--tokenizer-trust-remote-code` 允许执行 tokenizer 仓库中的代码，请使用可信来源。

## 指定已有的推理服务

修改以下值，使其与已启动的服务一致：

```bash
export SERVER_URL="http://127.0.0.1:8000"
export SERVED_MODEL_NAME="moonshotai/Kimi-K3"
export TOKENIZER="moonshotai/Kimi-K3"
export DATASET="semianalysis_cc_traces_weka_062126"
export CONC=8
export DURATION=3600
export OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

curl --fail --silent --show-error "${SERVER_URL}/v1/models"
```

`SERVER_URL` 是不带 `/v1` 或 `/v1/chat/completions` 的基础 URL。
远程服务需将 `127.0.0.1` 替换为可达地址。
`SERVED_MODEL_NAME` 必须与 `/v1/models` 返回的模型 ID 一致；
`TOKENIZER` 是 Hugging Face ID 或客户端上的 tokenizer 目录，
不必与服务端文件系统路径相同。若端点要求认证，为 AIPerf 添加
`--api-key "$OPENAI_API_KEY"`，并为 curl 添加对应的
`Authorization: Bearer` 请求头，密钥需通过安全方式提供。

运行前选择语料。当前
[`resolve_trace_source()`](../benchmarks/benchmark_lib.sh) 的映射如下：

| InferenceX 模型前缀 | `DATASET` |
| --- | --- |
| `dsv4*`, `glm5.2*`, `glm5.3*`, `minimaxm3*`, `kimik3*` | `semianalysis_cc_traces_weka_062126` |
| 其他前缀，包括 Qwen3.5 和 Qwen3.8-Flash-Next | `semianalysis_cc_traces_weka_062126_256k` |

客户端会下载并缓存所选的公开 Hugging Face 数据集。
`_256k` 语料是预先过滤后的另一种工作负载，并非服务端配置。
比较测试结果时，应使用相同的日期锁定语料。若服务明确配置了更小的上下文上限，
添加 `--max-context-length <TOKENS>` 与其匹配，并说明过滤条件；
不要默默截断提示词，也不要将过滤后的结果当作完整语料的结果。

服务必须支持流式聊天补全、场景要求的 `ignore_eos=true` 请求字段，
以及所选上下文长度。前缀缓存、KV offload、并行配置和投机解码
均由服务自身的 recipe 配置，客户端命令不会设置这些选项。
进行带投机解码的 InferenceX 对比时，请遵循
[golden acceptance-length 规则](../infx/golden_al_distribution/README_zh.md)；
合成接受长度仅用于吞吐量测试，不用于正确性评估。

## 运行一个并发点

下方示例使用 InferenceX 的共享回放参数，测量窗口为一小时。
复现特定运行时，应使用其 recipe 规定的测量时长和预热 grace period。
数据集准备、缓存预热和在途请求排空会增加总运行时间。

```bash
set -eo pipefail

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/inferencex-revision.txt"
git -C utils/aiperf rev-parse HEAD > "$OUTPUT_DIR/aiperf-revision.txt"
uv pip freeze --python .venv-agentx/bin/python > "$OUTPUT_DIR/client-packages.txt"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$SERVER_URL" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "$SERVED_MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --benchmark-duration "$DURATION" \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

`--concurrency` 控制同时存活的会话树数量，每棵树包含其子 agent，
而不是对同时进行的 HTTP 请求数设置固定上限。
上述环境设置使用录制的助手回复构造后续轮次；实时回复参与测量，
但不会成为下一轮提示词的一部分。

场景会强制启用流式响应、首轮前缀 cache busting、
`ignore_eos=true`，并将全系统空闲间隔限制为 10 秒。
显式设置的 `0.25`/`0.75` 轨迹起点比例、随机种子 `42`
和每棵树 300 秒的空闲间隔上限与 InferenceX 包装脚本一致，
并不完全等同于通用 AIPerf 教程的默认值。复现 InferenceX 时不要替换成教程默认值。

可向同一命令添加以下选项：

- **Prometheus 指标：** 若服务提供指标端点，添加
  `--server-metrics "${SERVER_URL}/metrics"`。使用路由器时，在一个
  `--server-metrics` 参数后提供各 worker 可达的指标 URL。
  `--no-gpu-telemetry` 仅关闭 AIPerf GPU 遥测，不关闭引擎指标；
  独立命令不会启动 InferenceX 的单独功耗采集器。
- **Chat-template token 统计：** 若对应服务 recipe 使用此选项，
  如 ATOM 示例，添加 `--apply-chat-template`。
  它启用基于 chat template 的客户端 token 统计，不会启动或配置服务。
- **路由器会话亲和性：** 与路由器的会话亲和配置保持一致。
  使用多副本前请阅读锁定版本的
  [AgentX 路由指南](../utils/aiperf/docs/benchmark-modes/semianalysis-agentx-faq.md)。

执行并发扫描时，对每个 `CONC` 重跑一次上述测量代码块，
每次设置新的 `OUTPUT_DIR`。等待服务端在途请求排空后再开始下一个点。
若不同并发点的服务 recipe 有变化，需重启服务。
保持语料、客户端版本、随机种子及回放参数不变，并为每个结果记录服务端变更。

仅做短时诊断时，可将 `DURATION` 改为 `60` 并添加
`--unsafe-override`，此类运行会被标记为 `submission_valid=false`。
场景正常要求的最短时长为 900 秒；仅达到这一时长并不意味着
冒烟测试与配置规定的对比运行等价。

## 检查并保留结果

AIPerf 完成后会打印导出路径。请保留整个输出目录，包括 `aiperf.log`、
`profile_export_aiperf.json`、`profile_export.jsonl`
以及生成的 `server_metrics_export.*` 文件。
导出文件可能直接位于 `aiperf_artifacts/`，也可能位于各次运行的子目录；
请以日志报告的实际路径为准。

执行 InferenceX 使用的请求错误率检查：

```bash
python -m infx.results.agentic.validate_agentic_result \
  "$OUTPUT_DIR/aiperf_artifacts" \
  --failed-request-threshold 0.10
```

该检查验证已完成请求和失败比例，不验证场景有效性。
还需检查 `profile_export_aiperf.json` 中的 `metadata.submission_valid`
及相关有效性详情，并查看错误、token 数、吞吐量、TTFT 和请求延迟。
有效性标记为 `true` 并不能证明服务 recipe 或所有回放参数与另一次运行一致。

记录服务镜像 digest/版本、完整启动命令、模型/tokenizer 版本、硬件、
并行配置、KV cache/offload 和投机解码设置。
原始客户端导出并非 InferenceX dashboard 提交产物：
此路径不执行评估、不生成包装脚本的标准化结果 JSON，也不上传结果。
相应流程请参阅[结果与入库指南](results-and-ingestion_zh.md)。

## 运行失败时

- **未知场景或参数：** 检查当前 `aiperf` 是否来自 `.venv-agentx`，
  并确认子模块与仓库锁定版本一致。
- **HTTP 400/404：** 核对服务模型 ID、端点、上下文上限
  以及对 `ignore_eos` 的支持，保留服务端返回的错误内容。
- **数据集配置超时：** 检查 Hugging Face 连接、tokenizer 访问权限、
  可写缓存空间和客户端可用 CPU/RAM。上述两个配置超时均为 1,800 秒。
- **预热超时：** 查看服务日志和请求进度。grace period 是请求排空的
  截止时间，不是固定等待时间；部分高并发 recipe 会显式设置更长时间。
- **非零退出码或无效提交：** 保留日志和有效性详情，修复原因后重跑。
  不要把部分导出的指标文件当作成功的基准测试结果。
