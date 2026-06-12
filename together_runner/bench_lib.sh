#!/usr/bin/env bash
# together_runner shared library: env validation, staged timing + ETA, and a
# server-startup monitor that maps SGLang log markers to named stages.
#
# Source this AFTER config.env. It is side-effect free except for defining
# functions and a couple of module-level vars.

# Log line prefix — carries node + rank so the same format works unchanged when
# this is extended to multi-node (rank defaults to 0 on a single box).
_TR_HOST="$(hostname -s 2>/dev/null || hostname)"
log_prefix() { echo "[node=${_TR_HOST} rank=${RANK:-0}]"; }
trlog()  { echo "$(log_prefix) $*"; }
trwarn() { echo "$(log_prefix) WARN: $*" >&2; }
trerr()  { echo "$(log_prefix) ERROR: $*" >&2; }

# ---------------------------------------------------------------------------
# Env validation (mirrors benchmarks/benchmark_lib.sh:check_env_vars)
# ---------------------------------------------------------------------------
check_env_vars() {
    local missing=()
    local v
    for v in "$@"; do
        if [[ -z "${!v:-}" ]]; then missing+=("$v"); fi
    done
    if (( ${#missing[@]} > 0 )); then
        trerr "missing required environment variables:"
        for v in "${missing[@]}"; do echo "  - $v" >&2; done
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Staged timing + ETA
#
# Baseline per-stage durations (seconds), measured on this 8xB200 node
# (2026-06-11 DSR1-FP4 cold start). Used only to print a rough "next stage ~N
# min" hint — tune freely. autotune is ~0 when ENABLE_TUNING=0.
# ---------------------------------------------------------------------------
# Measured on this 8xB200 node (2026-06-11, DSR1-FP4, weights on local /scratch,
# ENABLE_TUNING=0). autotune ~12min only when ENABLE_TUNING=1.
declare -gA STAGE_ETA_BASELINE=(
    [container-start]=10
    [engine-init]=90
    [weight-load]=120
    [autotune]=720
    [graph-capture]=120
)

# Server process pattern per engine (for pgrep/pkill of the inference server).
server_proc_pat() { [[ "${ENGINE:-sglang}" == "vllm" ]] && echo "vllm serve" || echo "sglang.launch_server"; }

# Ordered startup plan — set by the launcher (run_2) so the monitor can
# announce the NEXT stage and a running "to ready" estimate. autotune is included
# only when tuning is on. Leave empty to disable the look-ahead hints.
STAGE_PLAN="${STAGE_PLAN:-}"

# echo the stage that follows $1 in STAGE_PLAN (empty if $1 is last/unknown)
_stage_after() {
    local cur="$1" s found=0
    for s in $STAGE_PLAN; do
        [[ $found == 1 ]] && { echo "$s"; return; }
        [[ "$s" == "$cur" ]] && found=1
    done
}
# sum of baseline ETAs of all stages strictly after $1 (rough "to ready" hint)
_eta_after() {
    local cur="$1" s total=0 found=0
    for s in $STAGE_PLAN; do
        [[ $found == 1 ]] && total=$(( total + ${STAGE_ETA_BASELINE[$s]:-0} ))
        [[ "$s" == "$cur" ]] && found=1
    done
    echo "$total"
}

_STAGE_NAME=""
_STAGE_T0=0
_RUN_T0=0

_now() { date +%s; }
_fmt_dur() { local s=$1; printf '%dm%02ds' $(( s/60 )) $(( s%60 )); }

run_timer_start() { _RUN_T0=$(_now); }
run_elapsed()     { echo $(( $(_now) - _RUN_T0 )); }

# stage_begin <name> [<eta-override-seconds>]
stage_begin() {
    local name="$1" eta="${2:-}"
    [[ -n "$_STAGE_NAME" ]] && stage_end
    _STAGE_NAME="$name"
    _STAGE_T0=$(_now)
    [[ -z "$eta" ]] && eta="${STAGE_ETA_BASELINE[$name]:-}"
    if [[ -n "$eta" && "$eta" -gt 0 ]]; then
        trlog "[STAGE ▶ ${name}] starting (this stage ~$(_fmt_dur "$eta"))"
    else
        trlog "[STAGE ▶ ${name}] starting"
    fi
    # Look-ahead: announce the next stage and a rough total time to ready.
    if [[ -n "$STAGE_PLAN" ]]; then
        local nxt; nxt=$(_stage_after "$name")
        if [[ -n "$nxt" ]]; then
            local neta="${STAGE_ETA_BASELINE[$nxt]:-0}" rem; rem=$(_eta_after "$name")
            trlog "         ⏭ next: ${nxt} (~$(_fmt_dur "$neta")) • ~$(_fmt_dur "$rem") to ready"
        else
            trlog "         ⏭ final stage — server should be ready shortly"
        fi
    fi
}

stage_end() {
    [[ -z "$_STAGE_NAME" ]] && return 0
    local d=$(( $(_now) - _STAGE_T0 ))
    trlog "[STAGE ✔ ${_STAGE_NAME}] done in $(_fmt_dur "$d")"
    _STAGE_NAME=""
}

# ---------------------------------------------------------------------------
# Server-startup monitor
#
# Polls the in-container server log + /health, detects stage transitions from
# log markers, prints progress with per-stage timing & ETA, and aborts (with a
# log dump) if the server process dies.
#
# Usage: monitor_server_until_ready <container> <port> <in_container_logpath> [timeout_s]
# Returns 0 when /health is up, non-zero on death/timeout.
# ---------------------------------------------------------------------------
_log_has() {  # _log_has <container> <logpath> <grep-regex>
    docker exec "$1" bash -c "tr '\r' '\n' < '$2' 2>/dev/null | grep -qE -- \"$3\""
}
_rank_count() {  # count "<marker>" occurrences -> "X/TP ranks"
    local c
    c=$(docker exec "$1" bash -c "tr '\r' '\n' < '$2' 2>/dev/null | grep -cE -- \"$3\"" 2>/dev/null || echo 0)
    echo "${c:-0}"
}

# print a per-stage detail line only when it differs from the last one
_LAST_DETAIL=""
_emit_detail() { [[ "$1" != "$_LAST_DETAIL" ]] && { trlog "  $1"; _LAST_DETAIL="$1"; }; }

monitor_server_until_ready() {
    local container="$1" port="$2" logpath="$3" timeout="${4:-3600}"
    local poll=5
    local stage="" t_start
    _LAST_DETAIL=""
    t_start=$(_now)

    # Engine-specific process pattern + log markers.
    local proc_pat m_weight m_capture engine="${ENGINE:-sglang}"
    if [[ "$engine" == "vllm" ]]; then
        proc_pat="vllm serve"
        m_weight="Loading model weights|Model loading took|Starting to load model"
        m_capture="Capturing CUDA graph|Capturing cudagraph"
    else
        proc_pat="sglang.launch_server"
        m_weight="Load weight begin"
        m_capture="Capture cuda graph begin"
    fi

    trlog "monitoring startup of '${container}' (engine=${engine}, log=${logpath}, timeout=$(_fmt_dur "$timeout"))"

    while :; do
        # Ready?
        if docker exec "$container" bash -c "curl -sf http://localhost:${port}/health >/dev/null 2>&1"; then
            [[ -n "$stage" ]] && stage_end
            trlog "[STAGE ✔ ready] server is up after $(_fmt_dur $(( $(_now) - t_start )) ) — total"
            return 0
        fi
        # Process dead?
        if ! docker exec "$container" pgrep -f "$proc_pat" >/dev/null 2>&1; then
            trerr "server process ('$proc_pat') exited before becoming healthy. Last log lines:"
            docker exec "$container" bash -c "tr '\r' '\n' < '$logpath' 2>/dev/null | tail -n 40" >&2
            return 1
        fi
        # Timeout?
        if (( $(_now) - t_start > timeout )); then
            trerr "timed out after $(_fmt_dur "$timeout") waiting for server. Last log lines:"
            docker exec "$container" bash -c "tr '\r' '\n' < '$logpath' 2>/dev/null | tail -n 20" >&2
            return 1
        fi

        # Detect the furthest stage reached (order matters: latest wins).
        # autotune is sglang-only.
        # autotune applies to both engines: sglang fp4_gemm, and vLLM's FlashInfer
        # MoE autotune (triggered by VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1).
        local newstage="$stage"
        if   _log_has "$container" "$logpath" "$m_capture"; then newstage="graph-capture"
        elif _log_has "$container" "$logpath" "AutoTuner|Tuning fp4_gemm|Tuning flashinfer"; then newstage="autotune"
        elif _log_has "$container" "$logpath" "$m_weight"; then newstage="weight-load"
        elif _log_has "$container" "$logpath" "."; then        newstage="${newstage:-engine-init}"
        fi

        if [[ "$newstage" != "$stage" && -n "$newstage" ]]; then
            stage_begin "$newstage"
            stage="$newstage"
        fi

        # Per-stage progress detail — printed only when it changes (no spam).
        case "$stage" in
            weight-load)
                local r; r=$(_rank_count "$container" "$logpath" "$m_weight")
                _emit_detail "weight-load: ${r} rank/worker load event(s) seen"
                ;;
            graph-capture)
                local r; r=$(_rank_count "$container" "$logpath" "$m_capture")
                _emit_detail "graph-capture: ${r} capture event(s) (cudagraph sweep)"
                ;;
            autotune)
                # Surface the live tqdm percentage so progress is visible.
                local p; p=$(docker exec "$container" bash -c \
                    "tr '\r' '\n' < '$logpath' 2>/dev/null | grep -oE 'Tuning [^:]+:[^]]*\]' | tail -1" 2>/dev/null)
                local hint; hint=$([[ "$engine" == "vllm" ]] && echo "FlashInfer MoE" || echo "ENABLE_TUNING=0 to skip")
                _emit_detail "autotune: ${p:-profiling kernels...} (${hint})"
                ;;
        esac

        sleep "$poll"
    done
}

# ---------------------------------------------------------------------------
# Preflight assertion helpers (used by run_0_preflight.sh).
# Each prints a PASS/FAIL line and returns 0/1. Never exit — caller tallies.
# ---------------------------------------------------------------------------
_PF_FAILS=0
pf_pass() { echo "  [PASS] $*"; }
pf_fail() { echo "  [FAIL] $*"; _PF_FAILS=$(( _PF_FAILS + 1 )); }
pf_info() { echo "  [info] $*"; }
pf_reset() { _PF_FAILS=0; }
pf_failures() { echo "$_PF_FAILS"; }
