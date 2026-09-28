#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/benchmark_lib.sh --validation-only
check_env_vars TILERT_VERSION TILERT_ROLE
case "$TILERT_ROLE" in
    prefill|decode|router) ;;
    *) echo "Unknown TileRT role: $TILERT_ROLE" >&2; exit 1 ;;
esac

install_args=(--quiet --no-cache-dir)
if [[ "$TILERT_ROLE" == prefill ]]; then
    # Preserve the prefill image's vLLM/Torch dependency set.
    install_args+=(--no-deps)
fi
python3 -m pip install "${install_args[@]}" "tilert==$TILERT_VERSION"

if ! python3 -c 'import mooncake.engine' >/dev/null 2>&1; then
    python3 -m pip install --quiet --no-cache-dir 'mooncake-transfer-engine-rocm>=0.3.13'
fi
if [[ "$TILERT_ROLE" != prefill ]]; then
    if ! python3 -c 'import uvicorn' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir fastapi uvicorn httpx
    fi
    if ! python3 -c 'import transformers' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir 'transformers>=4.56'
    fi
fi

if [[ "$TILERT_ROLE" != decode ]]; then
    exit 0
fi
check_env_vars TILERT_WEIGHTS_DIR TILERT_CONVERT_LOCK_WAIT

weights_ready() {
    local rank
    for rank in {0..7}; do
        [[ -f "$TILERT_WEIGHTS_DIR/rank$rank/model.safetensors.index.json" ]] || return 1
    done
    [[ -f "$TILERT_WEIGHTS_DIR/tilert_meta.json" ]] || return 1
    python3 - "$TILERT_WEIGHTS_DIR/tilert_meta.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    metadata = json.load(source)
sys.exit(0 if int(metadata.get("num_mtp", 0)) >= 1 else 1)
PY
}

if ! weights_ready; then
    mkdir -p "$TILERT_WEIGHTS_DIR"
    exec 9>"$TILERT_WEIGHTS_DIR/.convert.lock"
    flock -w "$TILERT_CONVERT_LOCK_WAIT" 9
    if ! weights_ready; then
        python3 -m tilert.models.glm_5_2_rocm.weight_converter \
            --model_dir /model --save_dir "$TILERT_WEIGHTS_DIR" --device cuda:7 --num_mtp 3
        weights_ready
    fi
    exec 9>&-
fi

for file in /model/*; do
    [[ -f "$file" ]] || continue
    name="${file##*/}"
    [[ "$name" == *.safetensors || "$name" == model.safetensors.index.json ]] && continue
    [[ -e "$TILERT_WEIGHTS_DIR/$name" ]] || cp -p "$file" "$TILERT_WEIGHTS_DIR/$name"
done
[[ -f "$TILERT_WEIGHTS_DIR/chat_template.jinja" ]]
[[ -f "$TILERT_WEIGHTS_DIR/tokenizer_config.json" || -f "$TILERT_WEIGHTS_DIR/tokenizer.json" ]]
echo "TileRT TP8/MTP weights ready: $TILERT_WEIGHTS_DIR"
