#!/usr/bin/env bash
# Patch the worker's TensorRT-LLM 1.3.0rc23 for MiniMax-M3 AgentX: keep
# Prometheus request/iteration metrics without the per-step timing collector,
# and accept BFCL's standard store=false chat field. Every MPI rank runs this
# in the node's container, so serialize and skip what is already applied.
set -eo pipefail
ws=/infmax-workspace
exec 9>/tmp/minimaxm3-trtllm-rc23.lock
flock 9
IS_AGENTIC=0 SCENARIO_TYPE='' source "$ws/benchmarks/benchmark_lib.sh"
py_executor="$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("tensorrt_llm").origin).parent)')/_torch/pyexecutor/py_executor.py"
if ! grep -Fq "enabled=False)" "$py_executor"; then
    disable_trtllm_detailed_perf_metrics
fi
python3 "$ws/runners/patch_trtllm_chat_store.py"
