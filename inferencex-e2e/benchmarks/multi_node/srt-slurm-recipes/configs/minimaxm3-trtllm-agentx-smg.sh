#!/usr/bin/env bash
# MiniMax-M3 TensorRT-LLM AgentX workers behind SMG: the worker preparation,
# then the SMG install. srt-slurm runs one setup script per recipe, in every
# worker container and in SMG's own container.
set -euo pipefail
here=$(dirname "${BASH_SOURCE[0]}")
bash "$here/minimaxm3-trtllm-agentx.sh"
bash "$here/smg-1.11.0.sh"
