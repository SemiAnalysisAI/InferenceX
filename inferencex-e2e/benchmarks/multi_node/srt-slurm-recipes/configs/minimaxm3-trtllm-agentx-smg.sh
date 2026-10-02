#!/usr/bin/env bash
# MiniMax-M3 TensorRT-LLM AgentX workers behind SMG: the worker preparation,
# then the SMG install. srt-slurm runs one setup script per recipe, in every
# worker rank and in SMG's own container. The ranks of a TensorRT-LLM worker
# share a container, so the lock serializes their concurrent pip installs.
set -euo pipefail
here=$(dirname "${BASH_SOURCE[0]}")
exec 8>/tmp/minimaxm3-trtllm-agentx-smg.lock
flock 8
bash "$here/minimaxm3-trtllm-agentx.sh"
bash "$here/smg-1.11.0.sh"
