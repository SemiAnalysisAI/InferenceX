#!/usr/bin/env bash
# Record golden baselines on this reference machine across several concurrency
# points, reusing a SINGLE warm server (autotune ON for peak numbers).
#
#   bash record_baselines.sh                 # conc = 16 64 128 (default)
#   CONC_LIST="16 32 64 128 256" bash record_baselines.sh
#
# For each conc it benchmarks the live server and copies the emitted result into
# baselines/<hw>/<profile>/<seqtag>/conc<N>.json. Commit baselines/ afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/bench_lib.sh"

CONC_LIST="${CONC_LIST:-16 64 128}"
# Baselines mean PEAK performance -> keep sglang autotune on. vLLM has no
# separable autotune, so leave ENABLE_TUNING at the config default (0) there;
# this keeps vLLM baselines under untuned/ to match routine --full runs.
[[ "$ENGINE" != "vllm" ]] && export ENABLE_TUNING=1
# Launch with the largest conc so scheduler-recv-interval is set for high load.
LAUNCH_CONC=$(echo "$CONC_LIST" | tr ' ' '\n' | sort -n | tail -1)

step() { echo; echo "######## $* ########"; }

# 1) Stop any server already bound to the port (e.g. a prior --full run), so the
#    fresh autotuned server can take the port. Keep the container.
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if docker exec "$CONTAINER_NAME" pgrep -f "$(server_proc_pat)" >/dev/null 2>&1; then
        trlog "stopping existing server (need a fresh autotuned one for baselines)..."
        docker exec "$CONTAINER_NAME" pkill -f "$(server_proc_pat)" 2>/dev/null || true
        for _ in $(seq 1 30); do
            docker exec "$CONTAINER_NAME" pgrep -f "$(server_proc_pat)" >/dev/null 2>&1 || break
            sleep 2
        done
    fi
fi

step "GATE-1 PREFLIGHT"
bash "$HERE/run_0_preflight.sh"

# 2) Stage weights if needed.
if [[ "$MODEL" != /* && ! -f "$MODELS_ROOT/$PROFILE/.ready" && -z "${MODEL_PATH:-}" ]]; then
    step "PRESTAGE WEIGHTS"; bash "$HERE/prestage_weights.sh"
fi

step "START CONTAINER"; bash "$HERE/run_1_start_container.sh"

if [[ "$ENGINE" == "vllm" ]]; then
    step "LAUNCH SERVER (vLLM; ~5min cold)"
else
    step "LAUNCH SERVER (sglang, ENABLE_TUNING=1 autotune ON — slow, ~15-20min cold)"
fi
CONC="$LAUNCH_CONC" bash "$HERE/run_2_launch_server.sh"

# 3) Sweep concurrency against the one warm server.
DATE="$(date +%F)"
TUNE=$([[ "$ENABLE_TUNING" == "1" ]] && echo tuned || echo untuned)
RESDIR="$HERE/results/$HW/$CLUSTER/$_TR_HOST/$DATE"
# hw-wide golden by default; set BASELINE_CLUSTER=1 to scope under <hw>/<cluster>.
if [[ "${BASELINE_CLUSTER:-0}" == "1" ]]; then
    BDIR="$HERE/baselines/$HW/$CLUSTER/$ENGINE/$PROFILE/$SEQ_TAG/$TUNE"
else
    BDIR="$HERE/baselines/$HW/$ENGINE/$PROFILE/$SEQ_TAG/$TUNE"
fi
mkdir -p "$BDIR"
declare -a SAVED=()

for conc in $CONC_LIST; do
    step "BENCHMARK conc=$conc"
    CONC="$conc" BENCH=1 bash "$HERE/run_3_test_client.sh"
    src="$RESDIR/${ENGINE}_${PROFILE}_${SEQ_TAG}_${TUNE}_conc${conc}.json"
    dst="$BDIR/conc${conc}.json"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        SAVED+=("$dst")
        trlog "baseline saved: $dst"
    else
        trwarn "result not found for conc=$conc ($src) — skipped"
    fi
done

step "SUMMARY"
trlog "recorded ${#SAVED[@]} baseline(s) under $BDIR :"
for f in "${SAVED[@]}"; do
    python3 -c "import json;d=json.load(open('$f'));m=d['metrics'];print('  conc%-4d total_tput=%.1f tok/s  out_tput=%.1f  med_tpot=%.2fms'%(d['conc'],m['total_token_throughput'],m['output_token_throughput'],m['median_tpot_ms']))"
done
echo
echo "Commit them:"
echo "  git -C $(dirname "$HERE") add together_runner/baselines && \\"
echo "    git -C $(dirname "$HERE") commit -m 'baseline: ${HW} ${PROFILE} ${SEQ_TAG} conc[$(echo $CONC_LIST | tr ' ' ,)] (tuned)'"
echo
echo "Teardown server when done:  docker exec $CONTAINER_NAME pkill -f "$(server_proc_pat)""
