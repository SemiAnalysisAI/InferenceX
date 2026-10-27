#!/usr/bin/env bash

# Workflow-owned common settings, loaded before recipe-specific overrides.
# Benchmark receivers validate these inputs; they do not choose missing values.
# Jobs inherit these exports. Post-eval gets only WORKLOAD_ENV names (infx/launch/drivers/srt/submit.py).
export OPENAI_API_KEY='EMPTY'
export EVAL_ENDPOINT_READY_TIMEOUT_SECONDS='1800'
export EVAL_MODEL_STABILIZATION_SECONDS='30'
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING='1'
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID='false'
export AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS='3600'
export ENABLE_AGENTX_POWER='1'
export VLLM_ENGINE_READY_TIMEOUT_S='3600'
export SGLANG_TORCH_PROFILER_DIR='/workspace'
export VLLM_TORCH_PROFILER_DIR='/workspace'
