#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI325X (gfx942) single-node vLLM recipe with EAGLE3
# speculative decoding — the spec-decoding=mtp variant of
# minimaxm3_fp8_mi325x.sh. Adds the Inferact/MiniMax-M3-EAGLE3 draft head via
# --speculative-config with 3 speculative tokens. Everything else mirrors the
# non-MTP MI325X recipe: mandatory --block-size 128, --language-model-only for
# the text-only benchmark, --attention-backend TRITON_ATTN, and
# --no-enable-prefix-caching. Runs with CUDA graphs (no --enforce-eager);
# VLLM_USE_BREAKABLE_CUDAGRAPH=0 avoids the M3-decode breakable-cudagraph path.
# The default BF16 KV cache is retained (unlike the MI355X recipe's FP8 KV
# cache): gfx942 has no calibrated q/prob scales for ROCm FP8 attention and
# vLLM's fallback scale of 1.0 corrupts accuracy.
#
# Unlike the CUDA recipes, the drafter needs no attention_backend override:
# the FlashInfer "page size 128 requires GQA/MQA" limitation that forced
# FLASH_ATTN for the EAGLE3 MHA head on Blackwell is FlashInfer/CUDA-specific.
# Here the whole server runs on TRITON_ATTN (set globally below), which serves
# the MHA draft fine.
#
# [AI generated draft test] The shipped vllm/vllm-openai-rocm:minimax-m3 image
# does NOT implement SupportsEagle3 on the AMD MiniMax-M3 model, so EAGLE3
# engine init fails with "Model does not support EAGLE3 interface but
# aux_hidden_state_outputs was requested". This recipe applies that fix
# (functionstackx/vllm#1 — ported from nvidia/model.py, upstreamed as
# vllm-project/vllm#45546) in-place to the installed vllm before serving, so we
# can validate EAGLE3 on real MI325X hardware ahead of an image rebuild. The
# same patch is validated green on MI355X. It is idempotent and fails the job
# loudly if the installed amd/model.py has drifted from the expected base.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# MODEL is a bare HF id on the mi325x single-node runner (a fast cache hit when
# pre-staged). The EAGLE3 draft is not staged; fetch it into the same cache.
if [[ "$MODEL" != /* ]]; then
  hf download "$MODEL"
  hf download "$DRAFT_MODEL"
fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

# [AI generated draft test] Patch the installed AMD MiniMax-M3 model to add the
# SupportsEagle3 interface (functionstackx/vllm#1, upstream vllm-project/vllm#45546).
# Mirrors nvidia/model.py: adds EagleModelMixin to the inner model +
# aux-hidden-state emission, and SupportsEagle3 to the two outer classes.
# Idempotent; hard-fails if the installed file has drifted from the expected
# base (so we never silently run unpatched and mislabel the result).
python3 - <<'PYEOF' || { echo "EAGLE3 in-place patch failed" >&2; exit 1; }
import ast, importlib.util, pathlib, sys

spec = importlib.util.find_spec("vllm")
root = pathlib.Path(spec.submodule_search_locations[0])
target = root / "models" / "minimax_m3" / "amd" / "model.py"
src = target.read_text()

if "EagleModelMixin" in src and "class MiniMaxM3Model(nn.Module, EagleModelMixin):" in src:
    print(f"[eagle3-patch] already applied: {target}")
    sys.exit(0)

edits = [
    (
        "from vllm.model_executor.models.interfaces import (\n"
        "    MultiModalEmbeddings,\n"
        "    SupportsMultiModal,\n"
        ")",
        "from vllm.model_executor.models.interfaces import (\n"
        "    EagleModelMixin,\n"
        "    MultiModalEmbeddings,\n"
        "    SupportsEagle3,\n"
        "    SupportsMultiModal,\n"
        ")",
    ),
    (
        "class MiniMaxM3Model(nn.Module):",
        "class MiniMaxM3Model(nn.Module, EagleModelMixin):",
    ),
    (
        "        inputs_embeds: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor:\n"
        "        if inputs_embeds is not None:",
        "        inputs_embeds: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:\n"
        "        if inputs_embeds is not None:",
    ),
    (
        "        residual = None\n\n"
        "        for layer in self.layers[self.start_layer : self.end_layer]:\n"
        "            hidden_states, residual = layer(positions, hidden_states, residual)\n\n"
        "        hidden_states, _ = self.norm(hidden_states, residual)\n"
        "        return hidden_states",
        "        residual = None\n\n"
        "        # EAGLE3 is not yet compatible with pipeline parallel\n"
        "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n"
        "        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):\n"
        "            hidden_states, residual = layer(positions, hidden_states, residual)\n"
        "            self._maybe_add_hidden_state(\n"
        "                aux_hidden_states, idx + 1, hidden_states, residual\n"
        "            )\n\n"
        "        hidden_states, _ = self.norm(hidden_states, residual)\n\n"
        "        if len(aux_hidden_states) > 0:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states",
    ),
    (
        "class MiniMaxM3SparseForCausalLM(nn.Module):",
        "class MiniMaxM3SparseForCausalLM(nn.Module, SupportsEagle3):",
    ),
    (
        "class MiniMaxM3SparseForConditionalGeneration(nn.Module, SupportsMultiModal):",
        "class MiniMaxM3SparseForConditionalGeneration(\n"
        "    nn.Module, SupportsMultiModal, SupportsEagle3\n"
        "):",
    ),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        sys.exit(
            f"[eagle3-patch] anchor matched {count} times (expected 1); "
            f"installed {target} has drifted from the expected base — aborting"
        )
    src = src.replace(old, new)

ast.parse(src)
target.write_text(src)
print(f"[eagle3-patch] applied EAGLE3 support to {target}")
PYEOF

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# Spec-decode acceptance rate degrades on raw random tokens; route prompts
# through the chat template as the other MTP recipes do.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
