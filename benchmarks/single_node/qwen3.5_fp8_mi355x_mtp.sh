#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

# PR #22908: Qwen3.5 MoE + EAGLE MTP conflicts with radix cache under no_buffer mamba
# scheduling on ROCm unless server_args auto-disables radix (or CUDA extra_buffer path).
# Embedded unified diff for server_args.py (line anchors match SGLang v0.5.10.post1).
# Upstream: https://github.com/sgl-project/sglang/pull/22908/changes
SGLANG_PKG_ROOT="$(python3 -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent.parent)")"
if command -v patch >/dev/null 2>&1; then
    _pr22908_patch_tmp=$(mktemp)
    cat <<'PATCH22908' >"$_pr22908_patch_tmp"
--- a/sglang/srt/server_args.py
+++ b/sglang/srt/server_args.py
@@ -2188,6 +2188,13 @@
                     == 0
                 ), f"For SSM models with extra buffer, either FLA_CHUNK_SIZE or page_size must be divisible by the other, got {FLA_CHUNK_SIZE=}, {self.page_size=}"
         elif not self.disable_radix_cache:  # no_buffer
+            if self.page_size is not None and self.page_size != 1:
+                logger.warning(
+                    f"{model_arch} with radix cache requires page_size=1 in the current "
+                    f"Mamba scheduling mode (no_buffer), but got {self.page_size}. "
+                    "Automatically setting page_size=1."
+                )
+                self.page_size = 1
             if self.speculative_algorithm is None:
                 logger.warning(
                     "Disabling overlap schedule since mamba no_buffer is not compatible with "
@@ -2203,10 +2210,26 @@
                     self.disable_overlap_schedule = False
             else:
                 if not self.disable_radix_cache:
-                    raise ValueError(
-                        f"Speculative decoding for {model_arch} is not compatible with radix cache when using --mamba-scheduler-strategy no_buffer."
-                        "To use radix cache with speculative decoding, please use --mamba-scheduler-strategy extra_buffer and set SGLANG_ENABLE_SPEC_V2=1."
-                    )
+                    if is_cuda():
+                        # Automatically switch to extra_buffer + SPEC_V2 on CUDA
+                        # to support Qwen3.5 MoE speculative decoding out of the box.
+                        logger.warning(
+                            f"Speculative decoding for {model_arch} requires "
+                            "--mamba-scheduler-strategy extra_buffer on CUDA. "
+                            "Automatically switching to extra_buffer and enabling SGLANG_ENABLE_SPEC_V2."
+                        )
+                        self.mamba_scheduler_strategy = "extra_buffer"
+                        if not envs.SGLANG_ENABLE_SPEC_V2.get():
+                            envs.SGLANG_ENABLE_SPEC_V2.set(True)
+                    else:
+                        # On ROCm/non-CUDA, extra_buffer is unsupported.
+                        # Automatically disable radix cache instead.
+                        logger.warning(
+                            f"Speculative decoding for {model_arch} is not compatible "
+                            "with radix cache on non-CUDA devices. "
+                            "Automatically disabling radix cache."
+                        )
+                        self.disable_radix_cache = True
 
     def _handle_sampling_backend(self):
         if self.sampling_backend is None:
PATCH22908
    if patch -N --dry-run -p1 -d "$SGLANG_PKG_ROOT" <"$_pr22908_patch_tmp" >/tmp/sglang_pr22908_patch.log 2>&1; then
        patch -N -p1 -d "$SGLANG_PKG_ROOT" <"$_pr22908_patch_tmp" | tee /tmp/sglang_pr22908_patch_apply.log >/dev/null
        echo "[sglang] Applied PR #22908 server_args patch under $SGLANG_PKG_ROOT"
    elif grep -qiE 'reversed|previously applied|Skipping patch|ignored' /tmp/sglang_pr22908_patch.log 2>/dev/null; then
        echo "[sglang] PR #22908 server_args patch already present; skipping apply."
    else
        echo "[sglang][WARN] PR #22908 patch does not apply (SGLang version mismatch?). Log:" >&2
        cat /tmp/sglang_pr22908_patch.log >&2
    fi
    rm -f "$_pr22908_patch_tmp"
else
    echo "[sglang][WARN] patch(1) not found; cannot apply PR #22908 server_args fix." >&2
fi

# Fix vLLM get_cached_tokenizer crash: transformers >= 5 removed
# all_special_tokens_extended from TokenizersBackend. Replace the bare
# attribute access with a safe getattr fallback so benchmark_serving.py
# can load the tokenizer without error.
python3 -c "
import pathlib, re, sys
vllm_tok = pathlib.Path('/opt/venv/lib/python3.10/site-packages/vllm/transformers_utils/tokenizer.py')
if not vllm_tok.exists():
    sys.exit(0)
src = vllm_tok.read_text()
old = 'tokenizer.all_special_tokens_extended'
new = 'getattr(tokenizer, \"all_special_tokens_extended\", getattr(tokenizer, \"all_special_tokens\", []))'
if old in src:
    vllm_tok.write_text(src.replace(old, new))
    print('[vllm] Patched get_cached_tokenizer: all_special_tokens_extended -> safe getattr')
else:
    print('[vllm] get_cached_tokenizer already safe or attribute not found; skipping.')
"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# MTP (EAGLE speculative decoding) — aligned with validated ROCm MI355X serve recipe
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_DRAFT_TOKENS=4
SPECULATIVE_EAGLE_TOPK=1

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host=0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --expert-parallel-size "$EP_SIZE" \
    --trust-remote-code \
    --attention-backend triton \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --enable-flashinfer-allreduce-fusion \
    --speculative-algorithm EAGLE \
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS" \
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK" \
    --speculative-num-draft-tokens "$SPECULATIVE_DRAFT_TOKENS" \
    --cuda-graph-max-bs "$CONC" \
    --max-running-requests "$CONC" \
    --chunked-prefill-size 4096 \
    $EVAL_CONTEXT_ARGS > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend sglang \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
