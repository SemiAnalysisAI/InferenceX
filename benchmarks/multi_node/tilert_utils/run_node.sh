#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL_NAME=${MODEL_NAME:-glm5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-202752}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
RESULT_DIR=${RESULT_DIR:-/workspace}
BENCHMARK_LOGS_DIR=${BENCHMARK_LOGS_DIR:-/workspace}

DECODE_CTRL_PORT=${DECODE_CTRL_PORT:-5556}
DECODE_HTTP_PORT=${DECODE_HTTP_PORT:-5557}
PREFILL_PORT=${PREFILL_PORT:-8000}
ROUTER_PORT=${PORT:-8888}
DECODE_WAIT=${DECODE_WAIT:-3600}
KV_P2P_TRANSFER=${KV_P2P_TRANSFER:-nixl}
TILERT_WEIGHTS_DIR=${TILERT_WEIGHTS_DIR:-/workspace/GLM-5-FP8-TileRT}
TILERT_MODEL_TYPE=${TILERT_MODEL_TYPE:-glm-5}
TILERT_PARSER=${TILERT_PARSER:-none}
DECODE_KV_DTYPE=${DECODE_KV_DTYPE:-fp8}
PREFILL_KV_DTYPE=${PREFILL_KV_DTYPE:-fp8_ds_mla}

PREFILL_SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
DECODE_MTP=(--with-mtp)

TILERT_IS_AGENTIC=0
if [[ "${IS_AGENTIC:-0}" == "1" || "${SCENARIO_TYPE:-}" == "agentic-coding" ]]; then
    TILERT_IS_AGENTIC=1
fi

if [[ "$TILERT_IS_AGENTIC" == "1" ]]; then
    TILERT_QUEUE_TIMEOUT=${TILERT_QUEUE_TIMEOUT:-1800}
fi
TILERT_QUEUE_TIMEOUT=${TILERT_QUEUE_TIMEOUT:-0}

AGENTIC_LOGS_DIR=${AGENTIC_LOGS_DIR:-$RESULT_DIR/LOGS/agentic}

: "${DECODE_HOST:?DECODE_HOST is unset -- submit.sh must export it}"
: "${PREFILL_HOST:?PREFILL_HOST is unset -- submit.sh must export it}"
: "${TILERT_ROLE:?TILERT_ROLE is unset -- submit.sh must set it to decode or prefill}"

mkdir -p "$BENCHMARK_LOGS_DIR"

DONE_SENTINEL="$BENCHMARK_LOGS_DIR/.tilert_done.${SLURM_JOB_ID:-local}"
echo "[tilert-run_node] ROLE=$TILERT_ROLE host=$(hostname) DECODE_HOST=$DECODE_HOST PREFILL_HOST=$PREFILL_HOST"

log_and_run_bg() {
    local label="$1" logfile="$2"; shift 2
    local _xtrace=0; [[ $- == *x* ]] && _xtrace=1
    { set +x; } 2>/dev/null
    { printf '===== [%s] %s =====\n' "$label" "$(date '+%F %T')"
      printf '[cmd]'; printf ' %q' "$@"; printf '\n'
      printf '[cwd] %s\n[host] %s\n\n' "$PWD" "$(hostname)"
    } | tee -a "$logfile"
    "$@" >>"$logfile" 2>&1 &
    LAST_BG_PID=$!
    echo "[$label] pid=$LAST_BG_PID log=$logfile"
    (( _xtrace )) && set -x
    return 0
}

bench_result_stem() {
    local conc="$1"
    local pg=$(( ${PREFILL_TP:-8} * ${PREFILL_NUM_WORKERS:-1} ))
    local dg=$(( ${DECODE_TP:-8} * ${DECODE_NUM_WORKERS:-1} ))
    printf '%s_c%s_gpus_%s_ctx_%s_gen_%s' \
        "${RESULT_FILENAME}" "$conc" "$(( pg + dg ))" "$pg" "$dg"
}

rdma_preflight() {
    local warn=0
    echo "[rdma] role=$TILERT_ROLE UCX_NET_DEVICES=${UCX_NET_DEVICES:-<unset>} UCX_MEMTYPE_CACHE=${UCX_MEMTYPE_CACHE:-<unset>} UCX_MEMTYPE_REG_WHOLE=${UCX_MEMTYPE_REG_WHOLE:-<unset>}"

    local uverbs=(/dev/infiniband/uverbs*)
    if [[ -e "${uverbs[0]}" ]]; then
        echo "[rdma] verbs devices: ${uverbs[*]}"
    else
        echo "[rdma] WARNING: /dev/infiniband/uverbs* missing -- the container has no RDMA device nodes." >&2
        echo "[rdma]    docker: add --device /dev/infiniband; pyxis: needs cluster-side passthrough" >&2
        warn=1
    fi

    local ml; ml="$(ulimit -l 2>/dev/null)"
    if [[ "$ml" == "unlimited" ]]; then
        echo "[rdma] memlock: unlimited"
    else
        echo "[rdma] WARNING: memlock=$ml (not unlimited) -- pinning memory for RDMA may fail." >&2
        echo "[rdma]    docker: add --cap-add CAP_IPC_LOCK (or --ulimit memlock=-1)" >&2
        warn=1
    fi

    if command -v ibv_devices >/dev/null 2>&1; then
        echo "[rdma] ibv_devices:"; ibv_devices 2>&1 | sed 's/^/[rdma]   /'
    fi

    if (( warn )) && [[ "${TILERT_RDMA_STRICT:-0}" == "1" ]]; then
        echo "[rdma] TILERT_RDMA_STRICT=1 and preflight did not fully pass -- aborting" >&2
        return 1
    fi
    return 0
}

stage_tokenizer_files() {
    local staged=0 f b
    for f in "$MODEL_PATH"/*; do
        [[ -f "$f" ]] || continue
        b="$(basename "$f")"
        [[ "$b" == *.safetensors ]] && continue
        [[ "$b" == "model.safetensors.index.json" ]] && continue
        [[ -e "$TILERT_WEIGHTS_DIR/$b" ]] && continue
        cp -p "$f" "$TILERT_WEIGHTS_DIR/$b" && staged=$((staged+1))
    done
    echo "[stage_tokenizer] staged $staged auxiliary file(s) from $MODEL_PATH"
    local missing=()
    [[ -f "$TILERT_WEIGHTS_DIR/chat_template.jinja" ]] || missing+=(chat_template.jinja)
    [[ -f "$TILERT_WEIGHTS_DIR/tokenizer_config.json" || -f "$TILERT_WEIGHTS_DIR/tokenizer.json" ]] \
        || missing+=("tokenizer.json/tokenizer_config.json")
    if (( ${#missing[@]} )); then
        echo "[stage_tokenizer] ERROR: $TILERT_WEIGHTS_DIR is missing ${missing[*]}"
        echo "[stage_tokenizer]        decode_server loads the tokenizer and chat template from that directory."
        echo "[stage_tokenizer]        Check that MODEL_PATH=$MODEL_PATH is an HF directory containing the tokenizer."
        return 1
    fi
    return 0
}

convert_weights() {
    local index_json="$TILERT_WEIGHTS_DIR/model.safetensors.index.json"
    if [[ -f "$index_json" ]]; then
        echo "[weight_converter] cache hit (index.json present), skipping conversion: $TILERT_WEIGHTS_DIR"
        return 0
    fi

    mkdir -p "$TILERT_WEIGHTS_DIR"
    exec 9>"$TILERT_WEIGHTS_DIR/.convert.lock"
    flock -w "${TILERT_CONVERT_LOCK_WAIT:-21600}" 9 || {
        echo "[weight_converter] timed out waiting for the conversion lock (another job still converting?)"; return 1; }
    if [[ -f "$index_json" ]]; then
        echo "[weight_converter] cache produced by a concurrent job, skipping conversion"; exec 9>&-; return 0
    fi
    if [[ -n "$(ls -A "$TILERT_WEIGHTS_DIR" 2>/dev/null | grep -v '^\.convert\.lock$')" ]]; then
        echo "[weight_converter] found leftovers without index.json (previous conversion incomplete), cleaning and re-converting"
        find "$TILERT_WEIGHTS_DIR" -mindepth 1 ! -name '.convert.lock' -delete
    fi

    echo "[weight_converter] $MODEL_PATH -> $TILERT_WEIGHTS_DIR (model_type=$TILERT_MODEL_TYPE)"
    "${PY:-python}" -m tilert.models.preprocess.weight_converter \
        --model_type "$TILERT_MODEL_TYPE" --model_dir "$MODEL_PATH" --save_dir "$TILERT_WEIGHTS_DIR"
    local rc=$?
    exec 9>&-
    if [[ $rc -ne 0 || ! -f "$index_json" ]]; then
        echo "[weight_converter] conversion failed (rc=$rc, no index.json produced): $TILERT_WEIGHTS_DIR"
        return 1
    fi
    echo "[weight_converter] conversion done and cached: $TILERT_WEIGHTS_DIR"
}

start_decode() {
    local cmd=("${PY:-python}" -m tilert.pd_vllm.decode_server
        --engine tilert --model "$MODEL_NAME"
        --model-weights-dir "$TILERT_WEIGHTS_DIR"
        --max-seq-len "$MAX_MODEL_LEN"
        --kv-cache-dtype "$DECODE_KV_DTYPE" --transport "$KV_P2P_TRANSFER"
        --ctrl-port "$DECODE_CTRL_PORT" --http-port "$DECODE_HTTP_PORT"
        "${DECODE_MTP[@]}")
    log_and_run_bg decode "$BENCHMARK_LOGS_DIR/tilert_decode.log" "${cmd[@]}"
    DECODE_PID=$LAST_BG_PID
}

start_prefill() {
    local served=("$MODEL_NAME")
    [[ -n "${MODEL:-}" && "$MODEL" != "$MODEL_NAME" ]] && served+=("$MODEL")
    local cmd=(vllm serve "$MODEL_PATH"
        --served-model-name "${served[@]}" --port "$PREFILL_PORT"
        --tensor-parallel-size "$PREFILL_TP" --max-model-len "$MAX_MODEL_LEN"
        --enforce-eager --trust-remote-code --return-tokens-as-token-ids
        --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype "$PREFILL_KV_DTYPE"
        "${PREFILL_SPEC[@]}"
        --kv-transfer-config "{\"kv_connector\":\"TileRTConnector\",\"kv_connector_module_path\":\"tilert.pd_vllm.prefill_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"tilert_host\":\"$DECODE_HOST\",\"tilert_ctrl_port\":$DECODE_CTRL_PORT,\"tilert_model\":\"$MODEL_NAME\",\"tilert_max_seq_len\":$MAX_MODEL_LEN,\"tilert_transport\":\"$KV_P2P_TRANSFER\"}}")
    log_and_run_bg prefill "$BENCHMARK_LOGS_DIR/tilert_prefill.log" "${cmd[@]}"
    PREFILL_PID=$LAST_BG_PID
}

start_router() {
    local cmd=(env CUDA_VISIBLE_DEVICES= "${PY:-python}" -m tilert.pd_vllm.pd_router
        --vllm-url "http://$PREFILL_HOST:$PREFILL_PORT"
        --decode "$DECODE_HOST:$DECODE_CTRL_PORT:$DECODE_HTTP_PORT"
        --port "$ROUTER_PORT" --model-path "$MODEL_PATH" --parser "$TILERT_PARSER"
        --queue-timeout "$TILERT_QUEUE_TIMEOUT")
    log_and_run_bg router "$BENCHMARK_LOGS_DIR/tilert_router.log" "${cmd[@]}"
    ROUTER_PID=$LAST_BG_PID
}

wait_for_tcp() {
    local host="$1" port="$2" deadline=$(( SECONDS + ${3:-600} ))
    local _xtrace=0; [[ $- == *x* ]] && _xtrace=1
    { set +x; } 2>/dev/null
    local rc=0
    until (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; do
        if [[ $SECONDS -ge $deadline ]]; then
            echo "[wait_for_tcp] timeout $host:$port after ${3:-600}s"; rc=1; break
        fi
        sleep 5
    done
    [[ $rc -eq 0 ]] && { exec 3>&- 2>/dev/null || true; echo "[wait_for_tcp] $host:$port ready"; }
    (( _xtrace )) && set -x
    return $rc
}

run_bench_and_eval() {
    wait_for_server_ready --port "$ROUTER_PORT" \
        --server-log "$BENCHMARK_LOGS_DIR/tilert_router.log" --server-pid "$ROUTER_PID"
    local rc=0 conc np
    for conc in $CONC_LIST; do
        np=$(( conc * 10 ))
        [[ "$np" -lt 16 ]] && np=16
        run_benchmark_serving \
            --bench-serving-dir /workspace \
            --model "$MODEL_NAME" --port "$ROUTER_PORT" \
            --backend openai-chat --endpoint /v1/chat/completions \
            --input-len "$ISL" --output-len "$OSL" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$np" --max-concurrency "$conc" \
            --use-chat-template --server-pid "$ROUTER_PID" \
            --tokenizer "$MODEL_PATH" --trust-remote-code \
            --result-filename "$(bench_result_stem "$conc")" --result-dir "$RESULT_DIR" \
            || { rc=$?; echo "[bench] WARNING: conc=$conc failed/timed out (rc=$rc)"; }
    done
    run_lm_eval
    return $rc
}

run_lm_eval() {
    [[ "${RUN_EVAL}" = "true" ]] || return 0
    if [[ -n "${EVAL_CONC:-}" ]]; then
        export EVAL_CONCURRENT_REQUESTS="$EVAL_CONC"
    else
        export EVAL_CONCURRENT_REQUESTS="$(tr ' ' '\n' <<< "$CONC_LIST" | sort -n | tail -1)"
    fi
    export CONC="$EVAL_CONCURRENT_REQUESTS"
    run_eval --framework lm-eval --port "$ROUTER_PORT"
    append_lm_eval_summary
}

run_agentic_replay() {
    wait_for_server_ready --port "$ROUTER_PORT" \
        --server-log "$BENCHMARK_LOGS_DIR/tilert_router.log" --server-pid "$ROUTER_PID"
    local rc=0 conc conc_result_dir
    local result_filename_base="$RESULT_FILENAME"
    for conc in $CONC_LIST; do
        conc_result_dir="$AGENTIC_LOGS_DIR/conc_${conc}"
        mkdir -p "$conc_result_dir"
        export CONC="$conc"
        export RESULT_FILENAME="${result_filename_base}_conc${conc}"
        build_replay_cmd "$conc_result_dir"
        run_agentic_replay_and_write_outputs "$conc_result_dir" \
            || { rc=$?; echo "[agentic] WARNING: conc=$conc failed/timed out (rc=$rc)"; }
    done
    export RESULT_FILENAME="$result_filename_base"
    return $rc
}

# shellcheck source=./setup_deps.sh
source "$(dirname "$0")/setup_deps.sh"

set -x
case "$TILERT_ROLE" in
    decode)
        rdma_preflight || exit 1
        convert_weights || exit 1
        stage_tokenizer_files || exit 1
        start_decode
        { set +x; } 2>/dev/null
        while kill -0 "$DECODE_PID" 2>/dev/null; do
            [[ -f "$DONE_SENTINEL" ]] && break
            sleep 5
        done
        if [[ -f "$DONE_SENTINEL" ]]; then
            echo "[decode] done sentinel received, shutting down"; kill "$DECODE_PID" 2>/dev/null || true; exit 0
        fi
        echo "[decode] decode_server exited early (see $BENCHMARK_LOGS_DIR/tilert_decode.log)"; exit 1
        ;;
    prefill)
        rdma_preflight || exit 1
        rm -f "$DONE_SENTINEL"
        if [[ "$TILERT_IS_AGENTIC" == "1" ]]; then
            resolve_trace_source
            install_agentic_deps
        fi
        wait_for_tcp "$DECODE_HOST" "$DECODE_CTRL_PORT" "$DECODE_WAIT" \
            || echo "[prefill] WARNING: timed out waiting for the decode ctrl port ($DECODE_HOST:$DECODE_CTRL_PORT), starting anyway"
        start_prefill
        wait_for_tcp "$PREFILL_HOST" "$PREFILL_PORT" "${PREFILL_WAIT:-3600}" \
            || echo "[prefill] WARNING: timed out waiting for the vLLM port ($PREFILL_HOST:$PREFILL_PORT), continuing (see $BENCHMARK_LOGS_DIR/tilert_prefill.log)"
        start_router
        if [[ "$TILERT_IS_AGENTIC" == "1" ]]; then
            run_agentic_replay; BENCH_RC=$?
        else
            run_bench_and_eval; BENCH_RC=$?
        fi
        touch "$DONE_SENTINEL"
        kill "$ROUTER_PID" "$PREFILL_PID" 2>/dev/null || true
        exit $BENCH_RC
        ;;
    *)
        echo "unknown ROLE=$TILERT_ROLE"; exit 2 ;;
esac
