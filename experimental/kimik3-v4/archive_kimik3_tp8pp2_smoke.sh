#!/usr/bin/env bash
# Archive a finished TP8×PP2 smoke into /it-share/charwu/k3-pp-result/
# Layout mirrors kimik3_mi355x_tp8pp2_conc10_20260815_055015.
#
#   TS=20260815_055015 CONC=10 bash experimental/kimik3-v4/archive_kimik3_tp8pp2_smoke.sh
#
set -euo pipefail

TS="${TS:?set TS=<launcher timestamp, e.g. 20260815_055015>}"
CONC="${CONC:?set CONC=<concurrency>}"
TP="${TP:-8}"
PP="${PP:-2}"
SKU="${SKU:-mi355x}"
MODEL_TAG="${MODEL_TAG:-kimik3}"
REPO="${REPO:-$HOME/InferenceX}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_tp8pp2_smoke_logs}"
DEST_ROOT="${DEST_ROOT:-/it-share/charwu/k3-pp-result}"
OUT="${OUT:-${DEST_ROOT}/${MODEL_TAG}_${SKU}_tp${TP}pp${PP}_conc${CONC}_${TS}}"

RESULT_HOST="${LOG_ROOT}/conc${CONC}_${TS}"
RANK1_HOST="${LOG_ROOT}/rank1_${TS}"
MAIN_LOG="${LOG_ROOT}/kimik3_tp${TP}pp${PP}_c${CONC}_${TS}.log"
# Prefer dedicated launcher_*.log; fall back to seq / main log.
LAUNCHER_LOG="${LOG_ROOT}/launcher_${TS}.log"
[[ -f "$LAUNCHER_LOG" ]] || LAUNCHER_LOG="$MAIN_LOG"

mkdir -p "$OUT"/{scripts,env,logs,results,aiperf,aiter_configs}

echo "=== archiving CONC=${CONC} TS=${TS} -> ${OUT} ==="

# --- scripts ---
cp -a "$REPO/experimental/kimik3-v4/run_kimik3_tp8pp2_smoke_g06_g17.sh" "$OUT/scripts/" 2>/dev/null || true
cp -a "$REPO/experimental/kimik3-v4/stop_kimik3_tp8pp2_smoke.sh" "$OUT/scripts/" 2>/dev/null || true
cp -a "$REPO/benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm_tp8pp2.sh" "$OUT/scripts/" 2>/dev/null || true
cp -a "$LOG_ROOT/rank0_launch_${TS}.sh" "$OUT/scripts/" 2>/dev/null || true
cp -a "$LOG_ROOT/rank1_launch_${TS}.sh" "$OUT/scripts/" 2>/dev/null || true
cp -a "$REPO/experimental/kimik3-v4/aiter/patch_gemm_n6288_chunk.py" "$OUT/scripts/" 2>/dev/null || true
cp -a "$REPO/experimental/kimik3-v4/aiter/patch_ca_graph_flush_sync.py" "$OUT/scripts/" 2>/dev/null || true
cp -a "$REPO/experimental/kimik3-v4/aiter/aiter_site/sitecustomize.py" "$OUT/scripts/" 2>/dev/null || true

# --- aiter configs ---
cp -a "$REPO/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.combined.csv" \
  "$OUT/aiter_configs/" 2>/dev/null || true

# --- logs ---
[[ -f "$LAUNCHER_LOG" ]] && cp -a "$LAUNCHER_LOG" "$OUT/logs/"
[[ -f "$MAIN_LOG" ]] && cp -a "$MAIN_LOG" "$OUT/logs/"
cp -a "$LOG_ROOT/rank1_srun_${TS}.log" "$OUT/logs/" 2>/dev/null || true
if [[ -d "$RESULT_HOST" ]]; then
  cp -a "$RESULT_HOST/server.log" "$OUT/logs/pp0_server.log" 2>/dev/null || true
  cp -a "$RESULT_HOST/benchmark.log" "$OUT/logs/" 2>/dev/null || true
  cp -a "$RESULT_HOST/benchmark_command.txt" "$OUT/logs/" 2>/dev/null || true
  cp -a "$RESULT_HOST/vllm_command.txt" "$OUT/logs/pp0_vllm_command.txt" 2>/dev/null || true
fi
if [[ -d "$RANK1_HOST" ]]; then
  cp -a "$RANK1_HOST/server.log" "$OUT/logs/pp1_server.log" 2>/dev/null || true
  cp -a "$RANK1_HOST/vllm_command.txt" "$OUT/logs/pp1_vllm_command.txt" 2>/dev/null || true
fi

# --- results + aiperf ---
if [[ -d "$RESULT_HOST" ]]; then
  rsync -a --exclude='server.log' --exclude='benchmark.log' \
    "$RESULT_HOST/" "$OUT/results/"
  if [[ -d "$RESULT_HOST/aiperf_artifacts" ]]; then
    rsync -a "$RESULT_HOST/aiperf_artifacts/" "$OUT/aiperf/"
  fi
fi

# --- env ---
DURATION_VAL="$(grep -oE 'dur=[0-9]+s' "$OUT/logs/"*.log 2>/dev/null | head -1 | sed 's/dur=//;s/s$//' || true)"
DURATION_VAL="${DURATION_VAL:-1800}"
{
  echo "# Kimi-K3 TP${TP}xPP${PP} CONC=${CONC} smoke env snapshot"
  echo "# timestamp=${TS}"
  echo "# hold=${SLURM_REUSE_JOBID:-16758} nodes=mia1-p01-g06+mia1-p02-g17"
  echo "# image=vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263"
  echo
  echo "SLURM_REUSE_JOBID=${SLURM_REUSE_JOBID:-16758}"
  echo "TP=${TP}"
  echo "PP=${PP}"
  echo "CONC=${CONC}"
  echo "DURATION=${DURATION_VAL}"
  echo "ENFORCE_EAGER=false"
  echo "DISABLE_CUSTOM_ALL_REDUCE=1"
  echo "AITER_N6288_CHUNK_PATCH=1"
  echo "AITER_CA_FLUSH_SYNC_PATCH=1"
  echo "KV_OFFLOADING=dram"
  echo "KV_OFFLOAD_BACKEND=vllm-simple"
  echo "TOTAL_CPU_DRAM_GB=1500"
  echo "AITER_GEMM_MERGE=auto"
  echo "AITER_GEMM_EXTRA_CSV=experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.combined.csv"
  echo "MODEL=moonshotai/Kimi-K3"
  echo "MODEL_PATH=/it-share/hf_cache/Kimi-K3"
  echo "PRECISION=fp4"
  echo "FRAMEWORK=vllm"
  echo "SPEC_DECODING=none"
  echo "IS_AGENTIC=1"
  echo "DEBUG_ROCM=minimal"
} > "$OUT/env/smoke_env.sh"
head -20 "$LAUNCHER_LOG" > "$OUT/env/launcher_header.txt" 2>/dev/null || true
cp -a "$OUT/logs/pp0_vllm_command.txt" "$OUT/env/" 2>/dev/null || true

# --- README with final tput if present ---
OUT_TPUT="$(grep -F 'Output Token Throughput (tokens/sec)' "$OUT/aiperf/profile_export_console.txt" 2>/dev/null \
  | tail -1 | awk -F'│' '{print $3}' | tr -d ' ,' || true)"
IN_TPUT="$(grep -F 'Input Token Throughput (tokens/sec)' "$OUT/aiperf/profile_export_console.txt" 2>/dev/null \
  | tail -1 | awk -F'│' '{print $3}' | tr -d ' ,' || true)"
TOT_TPUT="$(grep -F 'Effective Total Throughput (tokens/sec)' "$OUT/aiperf/profile_export_console.txt" 2>/dev/null \
  | tail -1 | awk -F'│' '{print $3}' | tr -d ' ,' || true)"
GPUS=$((TP * PP))

cat > "$OUT/README.md" <<EOF
# Kimi-K3 ${SKU} TP${TP}×PP${PP} CONC=${CONC} smoke

- **Topology**: TP=${TP}, PP=${PP} (${GPUS} GPUs, g06+g17)
- **Concurrency**: ${CONC}
- **Duration**: ${DURATION_VAL}s profiling
- **Timestamp**: ${TS}
- **Hold job**: ${SLURM_REUSE_JOBID:-16758}
- **Key flags**: DISABLE_CUSTOM_ALL_REDUCE=1, dram/vllm-simple offload, aiter N6288 + CA flush-sync patches

## Final AIPerf (client)

| Metric | Total | per GPU (÷${GPUS}) |
|---|---|---|
| Output tok/s | ${OUT_TPUT:-n/a} | $([[ -n "$OUT_TPUT" ]] && python3 -c "print(round(float('${OUT_TPUT}')/${GPUS}, 1))" || echo n/a) |
| Input tok/s | ${IN_TPUT:-n/a} | $([[ -n "$IN_TPUT" ]] && python3 -c "print(round(float('${IN_TPUT}')/${GPUS}, 1))" || echo n/a) |
| Total tok/s | ${TOT_TPUT:-n/a} | $([[ -n "$TOT_TPUT" ]] && python3 -c "print(round(float('${TOT_TPUT}')/${GPUS}, 1))" || echo n/a) |

## Layout

- \`scripts/\` — smoke launcher, multinode bench, patches, rank launch scripts
- \`env/\` — smoke_env.sh + vllm command
- \`logs/\` — launcher / pp0+pp1 server / benchmark logs
- \`results/\` — result host directory copy
- \`aiperf/\` — AIPerf artifacts (json/csv/console)
- \`aiter_configs/\` — combined GEMM tune CSV used for this run
EOF

du -sh "$OUT"
echo "DEST=${OUT}"
