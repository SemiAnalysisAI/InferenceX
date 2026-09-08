# Sourced after the canonical profile; the parent owns server cleanup.
stop_background_process_tree "$SERVER_PID" "performance server" 60
SERVER_PID=""

K3_EVAL_PYTHON=/tmp/k3-eval/.venv/bin/python
"$AIPERF_UV_BIN" venv --system-site-packages --python /usr/bin/python3.12 /tmp/k3-eval/.venv
"$AIPERF_UV_BIN" pip install --python "$K3_EVAL_PYTHON" \
    'lm-eval[api] @ https://github.com/EleutherAI/lm-evaluation-harness/archive/b315ef3b05176acc9732bb7fdec116abe1ecc476.tar.gz' \
    'numpy==2.3.5'

for index in "${!VLLM_CMD[@]}"; do
    if [[ "${VLLM_CMD[$index]}" == --speculative-config ]]; then
        spec_index=$((index + 1))
        VLLM_CMD[$spec_index]=$("$K3_PYTHON" -c '
import json, sys
config = json.loads(sys.argv[1])
config["rejection_sample_method"] = "block"
config.pop("synthetic_acceptance_length", None)
print(json.dumps(config))
' "${VLLM_CMD[$spec_index]}")
    fi
done
mkdir -p "$RESULT_DIR/accuracy"
printf '%q ' "${VLLM_CMD[@]}" > "$RESULT_DIR/accuracy/vllm_command.txt"
printf '\n' >> "$RESULT_DIR/accuracy/vllm_command.txt"
"${VLLM_CMD[@]}" > "$RESULT_DIR/accuracy/server.log" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" \
    --server-log "$RESULT_DIR/accuracy/server.log" --server-pid "$SERVER_PID"

k3_eval_patch_dir=$(mktemp -d /tmp/k3-eval-patch-XXXXXX)
cp "$(_eval_patches_dir)/lm_eval_sitecustomize.py" "$k3_eval_patch_dir/sitecustomize.py"
PYTHONPATH="$k3_eval_patch_dir${PYTHONPATH:+:$PYTHONPATH}" OPENAI_API_KEY=EMPTY \
    "$K3_EVAL_PYTHON" -m lm_eval \
    --model local-chat-completions --apply_chat_template \
    --tasks "$INFERENCEX_REPO_ROOT/utils/evals/gsm8k.yaml" \
    --output_path "$RESULT_DIR/accuracy" --log_samples --limit 32 --seed 42 \
    --model_args "model=$MODEL,base_url=http://127.0.0.1:$PORT/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=3,num_concurrent=1,timeout=1800,tokenized_requests=False,max_length=16384" \
    --gen_kwargs 'max_tokens=8192,temperature=0,top_p=1' \
    > "$RESULT_DIR/accuracy/lm-eval.log" 2>&1
cat "$RESULT_DIR/accuracy/lm-eval.log"
echo 'K3_DENSE_FP8_GSM8K_EVAL_COMPLETED (32 samples, five-shot, real block acceptance)'
