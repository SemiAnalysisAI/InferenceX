#!/usr/bin/env bash
# Prepare a TensorRT-LLM 1.3 worker for MiniMax-M3 AgentX: accept OpenAI's
# store=false chat field. Every rank runs this; the lock serializes ranks that
# share a container, and the patch is a no-op once applied.
set -euo pipefail
exec 9>/tmp/minimaxm3-trtllm-agentx.lock
flock 9
python3 /infmax-workspace/runners/patch_trtllm_chat_store.py
