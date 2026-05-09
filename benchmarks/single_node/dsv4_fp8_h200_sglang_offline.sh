#!/usr/bin/env bash

# DeepSeek-V4-Pro H200 single-node SGLang **offline** benchmark via sgl.Engine.
# H200 must use the FP8 routed-expert layout. MXFP4/FP4 expert kernels are
# Blackwell-only and must not be forced on Hopper.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

# SGLang's tokenizer manager still routes through Transformers AutoTokenizer,
# which rejects DeepSeek-V4-Pro's `model_type: deepseek_v4` in this image.
# Keep the V4 architecture so SGLang dispatches to its native DSv4 model.
python3 - "$MODEL" <<'PYEOF'
import json
import sys
from pathlib import Path

model = sys.argv[1]
if model.startswith("/"):
    path = Path(model) / "config.json"
else:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=model, filename="config.json"))

with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF

nvidia-smi

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
EP_SIZE="${EP_SIZE:-1}"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
SGLANG_CUDA_GRAPH_FLAG=()
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-true}" == "true" ]]; then
    SGLANG_CUDA_GRAPH_FLAG=(--disable-cuda-graph)
fi
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-32768}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-$CONC}"
DPA_ENGINE_ARGS=()
MOE_RUNNER_ARGS=(--moe-runner-backend triton)

patch_sglang_dsv4_empty_attn_allreduce() {
    PYTHONNOUSERSITE=1 python3 - <<'PY'
from pathlib import Path

import sglang.srt.models.deepseek_v4 as deepseek_v4

path = Path(deepseek_v4.__file__)
text = path.read_text()
old = """        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x
"""
new = """        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            if self.wo_b.reduce_results:
                from sglang.srt.layers.dp_attention import get_attention_tp_group

                output = x.new_empty((0, self.hidden_size))
                torch.distributed.all_reduce(
                    output, group=get_attention_tp_group().device_group
                )
                return output
            return x
"""

if new in text:
    print(f"[dsv4-sglang-patch] Already patched {path}")
elif old in text:
    path.write_text(text.replace(old, new))
    print(f"[dsv4-sglang-patch] Patched {path}")
else:
    raise RuntimeError(f"Unable to patch DSV4 empty attention allreduce in {path}")
PY
}

# Keep DP-attn at size 2 so the dense replicated state is smaller on H200. Use
# the FP8 Triton MoE path with A2A disabled while we isolate the EP+DPA failure.
# Patch SGLang's empty attention shards so zero-token DPA ranks still enter the
# all-reduce instead of hanging.
if [[ "${DP_ATTENTION}" == "true" ]]; then
    SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
    SGLANG_CPU_OFFLOAD_GB="${SGLANG_CPU_OFFLOAD_GB:-0}"
    DPA_ENGINE_ARGS=(
        --dpa-size 2
        --dpa-moe-a2a-backend none
        --dpa-moe-runner-backend triton
        --moe-dense-tp-size 1
        --enable-dp-lm-head
        --sglang-dpa-env-preset fp8
    )
    MOE_RUNNER_ARGS=()
    patch_sglang_dsv4_empty_attn_allreduce
    export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    export SGLANG_DSV4_FP4_EXPERTS=false
else
    SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
    SGLANG_CPU_OFFLOAD_GB="${SGLANG_CPU_OFFLOAD_GB:-0}"
    export SGLANG_DSV4_FP4_EXPERTS=false
fi
start_gpu_monitor --output "$PWD/gpu_metrics.csv"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

set -x
PYTHONNOUSERSITE=1 python3 utils/bench_offline/run_offline.py \
    --engine sglang \
    --model "$MODEL" \
    --tp "$TP" \
    --ep "$EP_SIZE" \
    --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mtp "$NUM_SPEC_TOKENS" \
    "${MOE_RUNNER_ARGS[@]}" \
    "${SGLANG_CUDA_GRAPH_FLAG[@]}" \
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE" \
    --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" \
    --cpu-offload-gb "$SGLANG_CPU_OFFLOAD_GB" \
    --kv-cache-dtype fp8_e4m3 \
    --temperature 1.0 \
    --infinitebench-input-len "$ISL" \
    --infinitebench-output-len 256 \
    --batch-size "$CONC" \
    --result-dir "$PWD/" \
    --result-filename "$RESULT_FILENAME" \
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=256" \
    "${DPA_ENGINE_ARGS[@]}" \
    "${DPA_FLAG[@]}"
set +x

stop_gpu_monitor
