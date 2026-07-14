#!/bin/bash
# Generate the random synthetic dataset for the DeepSeek-V4-Pro gen-only SOL sweep.
#   isl=8192  osl=1024  random_ratio=1  num_prompts=16384  tokenizer=deepseek_v4
# random_ratio=1 pins every prompt to exactly ISL/OSL (no length variance), for a clean
# fixed-length gen SOL. (The reference sweep_config.yaml used random_ratio=0.8, num_samples=200000.)
# num_prompts must be >= the largest benchmarked concurrency (8192 here); 16384 gives warmup margin.
#
# IMPORTANT: --custom_tokenizer deepseek_v4 imports tensorrt_llm.tokenizer.deepseek_v4, so this
# must run in an environment that has TensorRT-LLM installed (e.g. inside the trtllm container,
# or a venv/conda env with the trtllm wheel). transformers/numpy/tqdm are also required.
#
# Usage:
#   MODEL_PATH=/path/to/DeepSeek-V4-Pro OUT_DIR=/path/to/dataset bash gen_dataset.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Parameters (override via env) ----
MODEL_PATH="${MODEL_PATH:-<path_to>/DeepSeek-V4-Pro}"   # tokenizer/model dir (fp4 checkpoint)
OUT_DIR="${OUT_DIR:-$HERE/data}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-16384}"   # must be >= max benchmarked concurrency (8192)
RANDOM_RATIO="${RANDOM_RATIO:-1}"
NUM_WORKERS="${NUM_WORKERS:-100}"

if [[ "$MODEL_PATH" == *"<path_to>"* ]]; then
  echo "ERROR: set MODEL_PATH to your DeepSeek-V4-Pro checkpoint directory." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
# Filename pattern: DeepSeek-V4-<isl>-<osl>-<num>-ratio-<ratio>_for_serve.json
OUT_PREFIX="$OUT_DIR/DeepSeek-V4-${ISL}-${OSL}-${NUM_PROMPTS}-ratio-${RANDOM_RATIO}"

echo "Generating dataset -> ${OUT_PREFIX}_for_serve.json"
python3 "$HERE/random_generator.py" \
    --num_prompts "$NUM_PROMPTS" \
    --num_tokens "$ISL" \
    --max_tokens "$OSL" \
    --random_ratio "$RANDOM_RATIO" \
    --tokenizer_name "$MODEL_PATH" \
    --custom_tokenizer deepseek_v4 \
    --output_file_serve "${OUT_PREFIX}_for_serve.json" \
    --num_workers "$NUM_WORKERS" \
    --use_parallel

echo
echo "Done. Ensure 'dataset_file' in the benchmark/configs/*.yaml points here:"
echo "  ${OUT_PREFIX}_for_serve.json"
