#!/bin/bash
# Dual-Engine Disaggregated Server Dispatcher
# =============================================================================
# Dispatches to the engine-specific server launcher based on ENGINE env var.
#   ENGINE=sglang-disagg (default) -> server_sglang.sh (SGLang + MoRI)
#   ENGINE=vllm-disagg             -> server_vllm.sh  (vLLM + Nixl/MoRI-IO)
# =============================================================================

ENGINE="${ENGINE:-sglang-disagg}"
WS_PATH="${WS_PATH:-${SGLANG_WS_PATH:-${VLLM_WS_PATH:-$(dirname "${BASH_SOURCE[0]}")}}}"
export WS_PATH ENGINE

echo "[DISPATCHER] ENGINE=$ENGINE  WS_PATH=$WS_PATH"

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    source "$WS_PATH/server_vllm.sh"
else
    source "$WS_PATH/server_sglang.sh"
fi
