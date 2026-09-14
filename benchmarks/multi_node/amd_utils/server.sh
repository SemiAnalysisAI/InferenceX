#!/bin/bash
# Multi-Engine Disaggregated Server Dispatcher
# =============================================================================
# Dispatches to the engine-specific server launcher based on ENGINE env var.
#   ENGINE=sglang-disagg (default) -> server_sglang.sh (SGLang + MoRI)
#   ENGINE=vllm-disagg             -> server_vllm.sh  (vLLM + Nixl/MoRI-IO)
#   ENGINE=atom-disagg             -> server_atom.sh  (ATOM + mooncake)
# =============================================================================

ENGINE="${ENGINE:-sglang-disagg}"
WS_PATH="${WS_PATH:-${SGLANG_WS_PATH:-${VLLM_WS_PATH:-${ATOM_WS_PATH:-$(dirname "${BASH_SOURCE[0]}")}}}}"
export WS_PATH ENGINE

source "$WS_PATH/power.sh"
if ! start_amd_multinode_power; then
    case "${REQUIRE_POWER:-0}" in
        1|true|TRUE|yes|YES) exit 1 ;;
    esac
    echo 'PowerX: continuing without optional worker telemetry' >&2
fi

echo "[DISPATCHER] ENGINE=$ENGINE  WS_PATH=$WS_PATH"

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    source "$WS_PATH/server_vllm.sh"
elif [[ "$ENGINE" == "atom-disagg" ]]; then
    export ATOM_WS_PATH="$WS_PATH"
    source "$WS_PATH/server_atom.sh"
else
    source "$WS_PATH/server_sglang.sh"
fi
