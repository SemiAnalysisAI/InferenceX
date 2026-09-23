#!/bin/bash
# TileRT disaggregated launcher: upstream vLLM ROCm prefill (TileRTConnector,
# kv_producer) + TileRT decode_server + the OpenAI-compatible pd_router.
# Every value below is supplied by the recipe through job.slurm; this script
# validates them and never invents a default for caller-owned configuration.

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only

check_env_vars \
    NODE0_ADDR NODE_RANK MODEL_DIR MODEL_NAME MODEL_PATH xP yD IPADDRS \
    DRY_RUN GPUS_PER_NODE PREFILL_TP_SIZE DECODE_TP_SIZE \
    BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_MAX_CONCURRENCY \
    RUN_EVAL EVAL_ONLY EVAL_FRAMEWORK BENCHMARK_LOGS_DIR WS_PATH \
    SLURM_JOB_ID SPEC_DECODING \
    TILERT_PROFILE TILERT_MODEL_TYPE TILERT_MODEL_PKG TILERT_MAX_MODEL_LEN \
    TILERT_TRANSPORT TILERT_PARSER TILERT_QUEUE_TIMEOUT TILERT_WEIGHTS_DIR \
    TILERT_RDMA_STRICT TILERT_CONVERT_LOCK_WAIT TILERT_SIMULATE_ACC_METHOD \
    PREFILL_KV_DTYPE PREFILL_BLOCK_SIZE PREFILL_SPEC_TOKENS DECODE_KV_DTYPE \
    DECODE_MTP_SIZE GPU_MEM_UTIL SERVED_MODEL_NAME \
    DECODE_CTRL_PORT DECODE_HTTP_PORT PREFILL_PORT ROUTER_PORT \
    DECODE_WAIT PREFILL_WAIT ROUTER_WAIT SKIP_CONTAINER_BARRIER \
    CONTAINER_BARRIER_TIMEOUT

LOG_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}"
SHARED_LOG_DIR="${BENCHMARK_LOGS_DIR}/logs/slurm_job-${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

if [[ "$xP" -ne 1 || "$yD" -ne 1 ]]; then
    echo "ERROR: tilert supports exactly 1 prefill + 1 decode worker (got xP=$xP yD=$yD)" >&2
    exit 1
fi
if [[ "$NODE_RANK" -lt "$xP" ]]; then
    TILERT_ROLE=prefill
else
    TILERT_ROLE=decode
fi
export TILERT_ROLE

source "$WS_PATH/setup_deps.sh"
source "$WS_PATH/env.sh"
# benchmark_lib.sh derives AGENTIC_DIR/AIPERF_DIR from this at source time, so
# it must be set before the library is loaded, not in run_agentic_replay. The
# AgentX replay runs in this container, where the repo is mounted at /workspace.
export INFMAX_CONTAINER_WORKSPACE=/workspace
source /workspace/benchmarks/benchmark_lib.sh

# Model-specific engine environment (not caller configuration): the prefill
# vLLM env block lives with the model, exactly as models_atom.yaml carries the
# ATOM `env` string. Everything else is passed in by the recipe.
MODELS_YAML="${WS_PATH}/models_tilert.yaml"
eval "$("$PY" - "$MODELS_YAML" "$MODEL_NAME" <<'PYEOF'
import shlex, sys, yaml
path, name = sys.argv[1], sys.argv[2]
with open(path) as f:
    models = yaml.safe_load(f) or {}
if name not in models:
    sys.exit(f"model '{name}' is not present in {path}")
m = models[name] or {}
for key, var in (("prefill_env", "TILERT_PREFILL_ENV"),
                 ("prefill_extra_flags", "TILERT_PREFILL_EXTRA_FLAGS"),
                 ("decode_extra_flags", "TILERT_DECODE_EXTRA_FLAGS")):
    print(f"{var}={shlex.quote(str(m.get(key) or ''))}")
PYEOF
)" || { echo "ERROR: cannot read the tilert model entry for '$MODEL_NAME' from $MODELS_YAML" >&2; exit 1; }
echo "[tilert] model entry '$MODEL_NAME' loaded from $MODELS_YAML"

export ROUTER_PORT
export SERVED_MODEL_NAME

PREFILL_SPEC=()
DECODE_MTP=()
if [[ "$SPEC_DECODING" == "mtp" ]]; then
    # The prefill rank only has to build the MTP layer's KV; TileRT decode owns
    # the draft depth (DECODE_MTP_SIZE), so the two counts differ by design.
    PREFILL_SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${PREFILL_SPEC_TOKENS}}")
    # decode_server only accepts depth 3 today, but pass it explicitly so the
    # converter's --num_mtp, the golden-curve key and the engine depth agree by
    # data flow rather than by coincidence of defaults.
    DECODE_MTP=(--with-mtp --num-mtp "$DECODE_MTP_SIZE")
fi

# The only TileRT recipe on this cluster is AgentX (agentic-coding).
if [[ "${IS_AGENTIC:-0}" != "1" && "${IS_AGENTIC:-}" != "true" && "${SCENARIO_TYPE:-}" != "agentic-coding" ]]; then
    echo "ERROR: server_tilert.sh only runs agentic-coding (IS_AGENTIC=${IS_AGENTIC:-} SCENARIO_TYPE=${SCENARIO_TYPE:-})" >&2
    exit 1
fi

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"
PREFILL_HOST="${IP_ARRAY[0]:-$NODE0_ADDR}"
DECODE_HOST="${IP_ARRAY[$xP]:-}"
if [[ -z "$DECODE_HOST" ]]; then
    echo "ERROR: cannot resolve the decode node IP from IPADDRS='$IPADDRS' (xP=$xP)" >&2
    exit 1
fi
host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
host_name=$(hostname)

echo "[tilert] ROLE=$TILERT_ROLE rank=$NODE_RANK host=$host_name ($host_ip)"
echo "[tilert] PREFILL_HOST=$PREFILL_HOST:$PREFILL_PORT  DECODE_HOST=$DECODE_HOST:$DECODE_CTRL_PORT/$DECODE_HTTP_PORT  ROUTER=:$ROUTER_PORT"
echo "[tilert] MODEL_PATH=$MODEL_PATH  profile=$TILERT_PROFILE  served=$SERVED_MODEL_NAME  max_len=$TILERT_MAX_MODEL_LEN  transport=$TILERT_TRANSPORT  kv=${PREFILL_KV_DTYPE}->${DECODE_KV_DTYPE}  mtp=${SPEC_DECODING}"

# Enable libibverbs fork safety on both ranks before any verbs context exists.
# Without it, ibv_fork_init() can fail in these containers while Mooncake
# initialization still reports success, silently falling back from RDMA to TCP.
# Slower prefill-to-decode KV transfer degrades TTFT; TPOT is unaffected.
export RDMAV_FORK_SAFE=1

for env_pair in ${TILERT_EXTRA_ENV}; do
    export "${env_pair?}"
    echo "[tilert][EXTRA_ENV] $env_pair"
done

log_and_run_bg() {
    local label="$1" logfile="$2"; shift 2
    { printf '===== [%s] %s =====\n' "$label" "$(date '+%F %T')"
      printf '[cmd]'; printf ' %q' "$@"; printf '\n'
      printf '[cwd] %s\n[host] %s\n\n' "$PWD" "$host_name"
    } | tee -a "$logfile"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: [$label] not started"
        LAST_BG_PID=""
        return 0
    fi
    "$@" >>"$logfile" 2>&1 &
    LAST_BG_PID=$!
    echo "[$label] pid=$LAST_BG_PID log=$logfile"
}

rdma_preflight() {
    local warn=0
    echo "[rdma] role=$TILERT_ROLE IBDEVICES=${IBDEVICES:-<unset>} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
    local uverbs=(/dev/infiniband/uverbs*)
    if [[ -e "${uverbs[0]}" ]]; then
        echo "[rdma] verbs devices: ${uverbs[*]}"
    else
        echo "[rdma] WARNING: /dev/infiniband/uverbs* missing -- the container has no RDMA device nodes (job.slurm passes --device /dev/infiniband)" >&2
        warn=1
    fi
    local ml; ml="$(ulimit -l 2>/dev/null)"
    if [[ "$ml" == "unlimited" ]]; then
        echo "[rdma] memlock: unlimited"
    else
        echo "[rdma] WARNING: memlock=$ml (not unlimited) -- pinning memory for RDMA may fail (job.slurm passes --ulimit memlock=-1)" >&2
        warn=1
    fi
    if command -v ibv_devices >/dev/null 2>&1; then
        echo "[rdma] ibv_devices:"; ibv_devices 2>&1 | sed 's/^/[rdma]   /'
    fi
    if (( warn )) && [[ "$TILERT_RDMA_STRICT" == "1" ]]; then
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
    echo "[stage_tokenizer] staged $staged auxiliary file(s) from $MODEL_PATH -> $TILERT_WEIGHTS_DIR"
    local missing=()
    [[ -f "$TILERT_WEIGHTS_DIR/chat_template.jinja" ]] || missing+=(chat_template.jinja)
    [[ -f "$TILERT_WEIGHTS_DIR/tokenizer_config.json" || -f "$TILERT_WEIGHTS_DIR/tokenizer.json" ]] \
        || missing+=("tokenizer.json/tokenizer_config.json")
    if (( ${#missing[@]} )); then
        echo "[stage_tokenizer] ERROR: $TILERT_WEIGHTS_DIR is missing ${missing[*]}; check that MODEL_PATH=$MODEL_PATH is an HF directory with the tokenizer" >&2
        return 1
    fi
    return 0
}

_tilert_weights_cached() {
    local r
    for r in $(seq 0 $((DECODE_TP_SIZE - 1))); do
        [[ -f "$TILERT_WEIGHTS_DIR/rank${r}/model.safetensors.index.json" ]] || return 1
    done
    # The engine refuses a cache converted without the MTP module (end2end.py
    # checks tilert_meta.json num_mtp), but only after loading ~90 GiB of
    # weights. Check the same field here so a stale non-MTP cache is
    # re-converted instead of failing late.
    [[ -f "$TILERT_WEIGHTS_DIR/tilert_meta.json" ]] || return 1
    if [[ "$SPEC_DECODING" == "mtp" ]]; then
        "$PY" - "$TILERT_WEIGHTS_DIR/tilert_meta.json" <<'PYEOF' || return 1
import json, sys
sys.exit(0 if int(json.load(open(sys.argv[1])).get("num_mtp", 0)) >= 1 else 1)
PYEOF
    fi
    return 0
}

convert_weights() {
    if _tilert_weights_cached; then
        echo "[weight_converter] cache hit (${DECODE_TP_SIZE}/${DECODE_TP_SIZE} rank index.json), skipping conversion: $TILERT_WEIGHTS_DIR"
        return 0
    fi
    mkdir -p "$TILERT_WEIGHTS_DIR" || { echo "[weight_converter] ERROR: cannot create $TILERT_WEIGHTS_DIR (set TILERT_WEIGHTS_DIR to a writable shared path)" >&2; return 1; }
    exec 9>"$TILERT_WEIGHTS_DIR/.convert.lock"
    flock -w "$TILERT_CONVERT_LOCK_WAIT" 9 || {
        echo "[weight_converter] timed out waiting for the conversion lock (another job still converting?)" >&2; return 1; }
    if _tilert_weights_cached; then
        echo "[weight_converter] cache produced by a concurrent job, skipping conversion"; exec 9>&-; return 0
    fi
    if [[ -n "$(ls -A "$TILERT_WEIGHTS_DIR" 2>/dev/null | grep -v '^\.convert\.lock$')" ]]; then
        echo "[weight_converter] leftovers without index.json (previous conversion incomplete); cleaning and re-converting"
        find "$TILERT_WEIGHTS_DIR" -mindepth 1 ! -name '.convert.lock' -delete
    fi
    echo "[weight_converter] $MODEL_PATH -> $TILERT_WEIGHTS_DIR (model_type=$TILERT_MODEL_TYPE)"
    local conv_mod conv_args
    if "$PY" -c "import tilert.models.${TILERT_MODEL_PKG}.weight_converter" 2>/dev/null; then
        conv_mod="tilert.models.${TILERT_MODEL_PKG}.weight_converter"
        conv_args=(--model_dir "$MODEL_PATH" --save_dir "$TILERT_WEIGHTS_DIR"
                   --device "cuda:$((GPUS_PER_NODE - 1))")
        [[ "$SPEC_DECODING" == "mtp" ]] && conv_args+=(--num_mtp "$DECODE_MTP_SIZE")
    else
        conv_mod="tilert.models.preprocess.weight_converter"
        conv_args=(--model_type "$TILERT_MODEL_TYPE" --model_dir "$MODEL_PATH" --save_dir "$TILERT_WEIGHTS_DIR")
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PY -m $conv_mod ${conv_args[*]}"
        exec 9>&-; return 0
    fi
    echo "[weight_converter] using $conv_mod"
    "$PY" -m "$conv_mod" "${conv_args[@]}" \
        2>&1 | tee "$LOG_DIR/tilert_weight_converter_${host_name}.log"
    local rc=${PIPESTATUS[0]}
    exec 9>&-
    if [[ $rc -ne 0 ]] || ! _tilert_weights_cached; then
        echo "[weight_converter] ERROR: conversion failed (rc=$rc, per-rank index.json complete: $(_tilert_weights_cached && echo yes || echo no))" >&2
        return 1
    fi
    echo "[weight_converter] conversion done and cached: $TILERT_WEIGHTS_DIR"
}

start_decode() {
    # shellcheck disable=SC2206
    local extra=( ${TILERT_DECODE_EXTRA_FLAGS} )
    if [[ "$SPEC_DECODING" == "mtp" && "$EVAL_ONLY" != "true" && "$RUN_EVAL" != "true" ]]; then
        check_env_vars MODEL_PREFIX THINKING_MODE
        local curve="${WS_PATH%/benchmarks/*}/golden_al_distribution/${MODEL_PREFIX}_mtp.yaml"
        TILERT_SIMULATE_ACC_LEN="$("$PY" - "$curve" "$THINKING_MODE" "$DECODE_MTP_SIZE" <<'PYEOF'
import sys, yaml
path, thinking, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = yaml.safe_load(open(path))
if not isinstance(data, dict) or len(data) != 1:
    sys.exit(f"golden curve {path} must hold exactly one model key")
model, modes = next(iter(data.items()))
try:
    value = float(modes[thinking][tokens])
except (KeyError, TypeError, ValueError):
    sys.exit(f"no golden acceptance for {model}/{thinking}/{tokens} draft tokens in {path}")
if not 1 <= value <= tokens + 1:
    sys.exit(f"golden acceptance {value} out of range for {tokens} draft tokens")
print(f"{value:g}")
PYEOF
)" || { echo "[tilert] ERROR: golden AL lookup failed (curve=$curve)" >&2; exit 1; }
        echo "[tilert] golden AL ${TILERT_SIMULATE_ACC_LEN} from $(basename "$curve") ($THINKING_MODE, K=${DECODE_MTP_SIZE})"
    fi

    if [[ -n "${TILERT_SIMULATE_ACC_LEN:-}" && "$EVAL_ONLY" != "true" ]]; then
        export TILERT_SIMULATE_ACC_LEN
        export TILERT_SIMULATE_ACC_METHOD
        echo "[decode] simulated acceptance: TILERT_SIMULATE_ACC_LEN=${TILERT_SIMULATE_ACC_LEN}" \
             "method=${TILERT_SIMULATE_ACC_METHOD} (output text is meaningless by design)"
    else
        unset TILERT_SIMULATE_ACC_LEN TILERT_SIMULATE_ACC_METHOD
        echo "[decode] real MTP verification (no simulated acceptance)"
    fi
    local cmd=("$PY" -m tilert.pd_vllm.decode_server
        --engine tilert --model "$TILERT_PROFILE"
        --model-weights-dir "$TILERT_WEIGHTS_DIR"
        --max-seq-len "$TILERT_MAX_MODEL_LEN"
        --kv-cache-dtype "$DECODE_KV_DTYPE" --transport "$TILERT_TRANSPORT"
        --ctrl-port "$DECODE_CTRL_PORT" --http-port "$DECODE_HTTP_PORT"
        "${DECODE_MTP[@]}" "${extra[@]}")
    log_and_run_bg decode "$LOG_DIR/decode_${host_name}.log" "${cmd[@]}"
    DECODE_PID=$LAST_BG_PID
}

start_prefill() {
    for env_pair in ${TILERT_PREFILL_ENV}; do
        export "${env_pair?}"
        echo "[PREFILL_ENV] $env_pair"
    done
    local served=("$SERVED_MODEL_NAME")
    [[ -n "$MODEL_NAME" && "$MODEL_NAME" != "$SERVED_MODEL_NAME" ]] && served+=("$MODEL_NAME")
    # shellcheck disable=SC2206
    local extra=( ${TILERT_PREFILL_EXTRA_FLAGS} )
    local kv_cfg
    kv_cfg=$(printf '{"kv_connector":"TileRTConnector","kv_connector_module_path":"tilert.pd_vllm.prefill_connector","kv_role":"kv_producer","kv_connector_extra_config":{"tilert_host":"%s","tilert_ctrl_port":%s,"tilert_model":"%s","tilert_max_seq_len":%s,"tilert_transport":"%s"}}' \
        "$DECODE_HOST" "$DECODE_CTRL_PORT" "$TILERT_PROFILE" "$TILERT_MAX_MODEL_LEN" "$TILERT_TRANSPORT")
    local cmd=(vllm serve "$MODEL_PATH"
        --served-model-name "${served[@]}" --port "$PREFILL_PORT"
        --tensor-parallel-size "$PREFILL_TP_SIZE" --max-model-len "$TILERT_MAX_MODEL_LEN"
        --enforce-eager --trust-remote-code --return-tokens-as-token-ids
        --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype "$PREFILL_KV_DTYPE"
        --block-size "$PREFILL_BLOCK_SIZE"
        "${PREFILL_SPEC[@]}"
        --kv-transfer-config "$kv_cfg"
        "${extra[@]}")
    log_and_run_bg prefill "$LOG_DIR/prefill_${host_name}.log" "${cmd[@]}"
    PREFILL_PID=$LAST_BG_PID
}

start_router() {
    local cmd=(env HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=
        "$PY" -m tilert.pd_vllm.pd_router
        --vllm-url "http://$PREFILL_HOST:$PREFILL_PORT"
        --decode "$DECODE_HOST:$DECODE_CTRL_PORT:$DECODE_HTTP_PORT"
        --host 0.0.0.0 --port "$ROUTER_PORT" --model-path "$MODEL_PATH" --parser "$TILERT_PARSER"
        --queue-timeout "$TILERT_QUEUE_TIMEOUT")
    log_and_run_bg router "$LOG_DIR/router_${host_name}.log" "${cmd[@]}"
    ROUTER_PID=$LAST_BG_PID
}

tcp_open() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }

wait_for_tcp() {
    local host="$1" port="$2" timeout="${3:-600}" pid="${4:-}"
    local deadline=$(( SECONDS + timeout ))
    until tcp_open "$host" "$port"; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[wait_for_tcp] process $pid exited before $host:$port opened" >&2; return 2
        fi
        if [[ $SECONDS -ge $deadline ]]; then
            echo "[wait_for_tcp] timeout: $host:$port not open after ${timeout}s" >&2; return 1
        fi
        sleep 5
    done
    echo "[wait_for_tcp] $host:$port ready"
}

wait_for_tcp_close() {
    local host="$1" port="$2" pid="${3:-}"
    while tcp_open "$host" "$port"; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[wait_for_tcp_close] process $pid exited while $host:$port is still open" >&2; return 2
        fi
        sleep 10
    done
    echo "[wait_for_tcp_close] $host:$port closed"
}

copy_logs_to_shared() {
    [[ "$DRY_RUN" -eq 0 ]] || return 0
    mkdir -p "$SHARED_LOG_DIR" && cp -r "$LOG_DIR"/. "$SHARED_LOG_DIR"/ \
        && echo "Copied $LOG_DIR -> $SHARED_LOG_DIR" \
        || echo "WARNING: failed to copy $LOG_DIR to $SHARED_LOG_DIR" >&2
}
trap copy_logs_to_shared EXIT

run_lm_eval_on_router() {
    echo "Running lm-eval evaluation on the router..."
    local ok=false _attempt
    for _attempt in 1 2 3; do
        if curl -sf --max-time 10 "http://0.0.0.0:${ROUTER_PORT}/health" >/dev/null 2>&1; then ok=true; break; fi
        echo "Eval health check attempt $_attempt failed, retrying in 10s..."; sleep 10
    done
    if [[ "$ok" != "true" ]]; then
        echo "ERROR: router health check failed after 3 attempts; skipping eval" >&2
        return 1
    fi
    local eval_failed=0
    pushd /workspace >/dev/null || return 1
    if [[ -n "${EVAL_CONC:-}" ]]; then
        export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
    else
        export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
    fi
    # run_lm_eval reads the endpoint from PORT (check_env_vars) and names the
    # model from MODEL_NAME, which the vLLM prefill also serves next to
    # SERVED_MODEL_NAME; MODEL stays the local HF dir for the context lookup.
    export PORT="$ROUTER_PORT"
    export MODEL="$MODEL_PATH"
    export MAX_MODEL_LEN="$TILERT_MAX_MODEL_LEN"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: run_eval --port $ROUTER_PORT (framework=${EVAL_FRAMEWORK}, conc=${EVAL_CONCURRENT_REQUESTS})"
    else
        run_eval --port "$ROUTER_PORT"
        local eval_rc=$?
        if [[ $eval_rc -ne 0 ]]; then
            echo "ERROR: run_eval exited rc=$eval_rc; preserving failure artifacts" >&2
            eval_failed=1
        else
            export TP="${PREFILL_TP_SIZE}" CONC="${EVAL_CONCURRENT_REQUESTS}" EP_SIZE=1
            export PREFILL_TP="${PREFILL_TP_SIZE}" PREFILL_EP=1 PREFILL_NUM_WORKERS="${xP}"
            export DECODE_TP="${DECODE_TP_SIZE}" DECODE_EP=1 DECODE_NUM_WORKERS="${yD}"
            export DP_ATTENTION=false PREFILL_DP_ATTENTION=false DECODE_DP_ATTENTION=false
            export ISL="${BENCH_INPUT_LEN}" OSL="${BENCH_OUTPUT_LEN}"
            # As on the SGLang path: rewrite meta_env.json from the exports above,
            # then stage unless run_eval already did (eval-only).
            rewrite_lm_eval_meta_env
            if [[ "$EVAL_ONLY" != "true" ]]; then
                append_lm_eval_summary
            fi
        fi
        local eval_copy_dir="$LOG_DIR/eval_results"
        if stage_eval_artifacts "$eval_copy_dir" /workspace "${EVAL_RESULT_DIR:-}"; then
            echo "Eval artifacts staged in $eval_copy_dir"
        else
            echo "ERROR: failed to stage eval artifacts in $eval_copy_dir" >&2
            eval_failed=1
        fi
    fi
    popd >/dev/null || true
    return $eval_failed
}

run_agentic_replay() {
    local rc=0
    wait_for_server_ready --port "$ROUTER_PORT" --server-log "$LOG_DIR/router_${host_name}.log" --server-pid "$ROUTER_PID"
    cd /workspace || return 1

    export PORT="$ROUTER_PORT"
    export MODEL="$MODEL_PATH"              # aiperf --tokenizer (local HF dir)
    export SERVED_MODEL_NAME                # aiperf --model (name the router/vLLM serve)
    check_env_vars DURATION RESULT_FILENAME INFMAX_CONTAINER_WORKSPACE
    export MAX_MODEL_LEN="$TILERT_MAX_MODEL_LEN"
    # TileRT decode exposes no /metrics route; only the vLLM prefill is scraped.
    export AIPERF_SERVER_METRICS_URLS="http://${PREFILL_HOST}:${PREFILL_PORT}/metrics"
    export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
    # Keep the trace corpus and aiperf's HF downloads on the node's /tmp mount
    # instead of the container's ephemeral ~/.cache, as the SGLang client does.
    export HF_HOME=/run_logs/hf_cache

    local result_dir="$LOG_DIR/agentic"
    local result_filename_base="$RESULT_FILENAME"
    mkdir -p "$result_dir"

    # Neither server.sh nor this script runs with errexit; a failed bootstrap
    # must not fall through into the replay loop and its misleading cascade.
    resolve_trace_source || return 1
    install_agentic_deps || return 1

    local conc conc_result_dir
    for conc in ${BENCH_MAX_CONCURRENCY//x/ }; do
        echo "=========================================="
        echo "Agentic trace replay: conc=$conc"
        echo "=========================================="
        conc_result_dir="$result_dir/conc_${conc}"
        mkdir -p "$conc_result_dir"
        export CONC="$conc" USERS="$conc"
        build_replay_cmd "$conc_result_dir"
        export RESULT_FILENAME="${result_filename_base}_conc${conc}"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $REPLAY_CMD"
        elif ! run_agentic_replay_and_write_outputs "$conc_result_dir"; then
            echo "WARNING: agentic trace replay for conc=$conc failed (replay or validation) after writing available results" >&2
            rc=1
        fi
        echo "-----------------------------------------"
    done
    export RESULT_FILENAME="$result_filename_base"
    return $rc
}

echo "Waiting at the container creation barrier on $host_name"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: skipping container creation barrier"
elif [[ "$SKIP_CONTAINER_BARRIER" == "1" ]]; then
    echo "SKIP_CONTAINER_BARRIER=1: caller asserts all containers are up"
else
    # --grace 60: after the barrier passes, sync.py keeps the port open for
    # max(60, timeout/2) seconds in the foreground so a peer one poll behind
    # still sees it. At CONTAINER_BARRIER_TIMEOUT=5400 that is a 45-minute idle
    # sleep on every rank (jobs 45373/45374 slept 06:28-07:13). Both ranks pass
    # within one 5 s poll of each other and the stages below have their own
    # readiness waits, so 60 s is plenty.
    "$PY" "$WS_PATH/sync.py" barrier \
        --local-ip "${host_ip}" --local-port 5000 --enable-port \
        --node-ips "${IPADDRS}" --node-ports 5000 \
        --wait-for-all-ports --timeout "$CONTAINER_BARRIER_TIMEOUT" --grace 60 \
        || { echo "ERROR: container creation barrier failed after ${CONTAINER_BARRIER_TIMEOUT}s -- the peer rank never opened port 5000." \
                  "A cold image pull is the usual cause: this recipe pulls two ~32 GB images, one per rank, and the rank that" \
                  "comes up first waits out the whole timeout while the other is still pulling." >&2; exit 1; }
fi

case "$TILERT_ROLE" in
    decode)
        echo "${host_name}:${host_ip} is the TileRT Decode Node (Model: ${MODEL_NAME}, profile: ${TILERT_PROFILE})"
        rdma_preflight || exit 1
        convert_weights || exit 1
        stage_tokenizer_files || exit 1
        start_decode
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: decode role complete"; exit 0
        fi
        echo "Waiting for the router port ${PREFILL_HOST}:${ROUTER_PORT} to open (timeout ${ROUTER_WAIT}s)..."
        wait_for_tcp "$PREFILL_HOST" "$ROUTER_PORT" "$ROUTER_WAIT" "$DECODE_PID"; wrc=$?
        if [[ $wrc -eq 2 ]]; then
            echo "ERROR: decode_server exited before the router came up (see $LOG_DIR/decode_${host_name}.log)" >&2
            tail -50 "$LOG_DIR/decode_${host_name}.log" >&2 || true
            copy_logs_to_shared; exit 1
        elif [[ $wrc -ne 0 ]]; then
            echo "WARNING: router never opened within ${ROUTER_WAIT}s; shutting down decode" >&2
            kill "$DECODE_PID" 2>/dev/null || true
            copy_logs_to_shared; exit 1
        fi
        echo "Waiting until the router port closes..."
        wait_for_tcp_close "$PREFILL_HOST" "$ROUTER_PORT" "$DECODE_PID"; wrc=$?
        if [[ $wrc -eq 2 ]]; then
            echo "ERROR: decode_server died while the benchmark was running (see $LOG_DIR/decode_${host_name}.log)" >&2
            tail -50 "$LOG_DIR/decode_${host_name}.log" >&2 || true
            copy_logs_to_shared; exit 1
        fi
        echo "Killing the decode server"
        kill "$DECODE_PID" 2>/dev/null || true
        sleep 2
        copy_logs_to_shared
        ;;
    prefill)
        echo "NODE INFO ======================================="
        echo "Node List : ${SLURM_JOB_NODELIST:-}"
        echo "Node IPs  : ${IPADDRS}"
        echo "Model     : ${MODEL_NAME}"
        echo "${host_name}:${host_ip} is the Prefill Node (vLLM + TileRTConnector) and Router Node"
        echo "================================================"
        rdma_preflight || exit 1
        echo "Waiting for the decode ctrl port ${DECODE_HOST}:${DECODE_CTRL_PORT} (timeout ${DECODE_WAIT}s)..."
        if [[ "$DRY_RUN" -eq 0 ]]; then
            wait_for_tcp "$DECODE_HOST" "$DECODE_CTRL_PORT" "$DECODE_WAIT" \
                || echo "WARNING: timed out waiting for the decode ctrl port; starting prefill anyway" >&2
        fi
        start_prefill
        if [[ "$DRY_RUN" -eq 0 ]]; then
            wait_for_tcp "$PREFILL_HOST" "$PREFILL_PORT" "$PREFILL_WAIT" "$PREFILL_PID"; wrc=$?
            if [[ $wrc -ne 0 ]]; then
                echo "ERROR: vLLM prefill did not open ${PREFILL_HOST}:${PREFILL_PORT} (rc=$wrc, see $LOG_DIR/prefill_${host_name}.log)" >&2
                tail -50 "$LOG_DIR/prefill_${host_name}.log" >&2 || true
                kill "$PREFILL_PID" 2>/dev/null || true
                copy_logs_to_shared; exit 1
            fi
        fi
        start_router
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: prefill/router role complete"; exit 0
        fi
        echo "Ready for benchmarking on ${host_name}:${host_ip}"
        cd "$WS_PATH" || exit 1
        # EVAL_ONLY skips the AgentX replay and runs GSM8K on the same router;
        # RUN_EVAL after a replay runs it once the replay has finished.
        if [[ "$EVAL_ONLY" == "true" ]]; then
            echo "EVAL_ONLY mode: skipping the AgentX replay"
            wait_for_server_ready --port "$ROUTER_PORT" --server-log "$LOG_DIR/router_${host_name}.log" --server-pid "$ROUTER_PID"
            export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
            run_lm_eval_on_router; BENCH_RC=$?
        else
            run_agentic_replay; BENCH_RC=$?
            if [[ "$RUN_EVAL" == "true" ]]; then
                run_lm_eval_on_router || BENCH_RC=1
            fi
        fi
        copy_logs_to_shared
        echo "Killing the router and the prefill server"
        kill "$ROUTER_PID" "$PREFILL_PID" 2>/dev/null || true
        sleep 2
        pkill -f "tilert.pd_vllm.pd_router" 2>/dev/null || true
        pkill -f "vllm serve" 2>/dev/null || true
        if [[ "$BENCH_RC" -ne 0 ]]; then
            echo "ERROR: benchmark/eval reported rc=$BENCH_RC" >&2
            exit "$BENCH_RC"
        fi
        ;;
    *)
        echo "ERROR: unknown TILERT_ROLE='$TILERT_ROLE'" >&2; exit 2 ;;
esac

echo "Script completed successfully"
exit 0
