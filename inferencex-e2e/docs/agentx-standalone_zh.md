# 独立运行 AgentX-Harness

<div align="center">

[English](agentx-standalone.md) | **中文**

</div>

对已运行的 OpenAI 兼容服务执行 AgentX 客户端。

## 安装

安装 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，
克隆测试工具并检出锁定的 `agentx-v1.0.6` 发布提交。其 CLI 名为 `aiperf`。

```bash
git clone --filter=blob:none --no-checkout https://github.com/SemiAnalysisAI/agentx-harness.git
cd agentx-harness
git checkout --detach 89b21867872a5bbc4b0676bf5005c404da5e9f94
uv venv --python 3.11 .venv-agentx
uv pip install --python .venv-agentx/bin/python -e . 'datasets>=4.7.0'
source .venv-agentx/bin/activate
```

## 运行

```bash
set -eo pipefail
SERVER_URL="http://127.0.0.1:8000"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
TOKENIZER="moonshotai/Kimi-K3"
DATASET="semianalysis_cc_traces_weka_062126"
CONC=8
OUTPUT_DIR="$PWD/results/agentx-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUTPUT_DIR"
aiperf profile \
  --scenario agentx \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL_NAME" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --public-dataset "$DATASET" \
  --output-artifact-dir "$OUTPUT_DIR/aiperf_artifacts" \
  2>&1 | tee "$OUTPUT_DIR/aiperf.log"
```

`agentx` 预设提供一小时测量窗口、每条 lane 十个预热请求，以及共享的报告和运行时设置。
显式 CLI 参数和 `AIPERF_*` 环境变量优先于预设默认值。
模型、tokenizer、数据集、并发和输出路径仍由调用方指定；只有 tokenizer 需要自定义代码时，才需使用 `--tokenizer-trust-remote-code`。

测量之外还需数据集准备、预热和请求排空时间。
扫描并发时，每个 `CONC` 值都必须重启服务并使用新的输出目录。
不要清空缓存或在不同并发点之间复用正在运行的服务。复现结果时，使用目标 recipe 的测量时长和预热设置。

服务提供指标端点时，可添加 `--server-metrics "${SERVER_URL}/metrics"`。

## 可选会话亲和性参数

DP-attention 运行需要路由器会话亲和性时，在 `aiperf profile` 前导出以下变量。
它们添加请求头，不替换 `X-Correlation-ID`。

```bash
# Adds X-Dynamo-Session-ID; subagents also get X-Dynamo-Parent-Session-ID.
# Only this option sends the parent ID, preserving forked-agent lineage.
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
# Also sends X-Session-ID, if a front-end router needs it.
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
```
