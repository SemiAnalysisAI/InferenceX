#!/usr/bin/env bash
# Prepare a TensorRT-LLM 1.3 worker for MiniMax-M3 AgentX: keep request and
# iteration metrics without rc23's per-step timing collector, and accept
# OpenAI's store=false chat field. Every rank runs this; the lock serializes
# ranks that share a container, and each step is a no-op once applied.
set -euo pipefail
exec 9>/tmp/minimaxm3-trtllm-agentx.lock
flock 9
root=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("tensorrt_llm").origin).parent)')
executor="$root/_torch/pyexecutor/py_executor.py"
gate="enabled=getattr(self.llm_args, 'return_perf_metrics', False))"
if grep -Fq "$gate" "$executor"; then
    sed -i "s/enabled=getattr(self.llm_args, 'return_perf_metrics', False))/enabled=False)/" "$executor"
fi
grep -Fq "self.perf_manager = PerfMetricsManager(" "$executor"
grep -Fq "enabled=False)" "$executor"
python3 /infmax-workspace/runners/patch_trtllm_chat_store.py
