#!/usr/bin/env bash
# Reroute aiter FlyDSL decode-range GEMMs off FlyDSL for Kimi-K3 FP4 DSpark on MI355X/gfx950.
#
# Runs INSIDE the container (awk-edits the merged tuned-GEMM CSV in place). The
# orchestrator apply_k3_container_patches.sh invokes this as the last patch step.
#
# WHY: At decode M=72 (conc-24, M=3*conc), the K3 dense BF16 GEMMs
#   (N,K) in {(1024,7168),(384,7168),(7168,512)} route to libtype=="flydsl" in
#   merged_bf16_tuned_gemm.csv. FlyDSL's per-call Python launcher + split-K
#   semaphore path is NOT vLLM-FULL-cudagraph-capturable, so the decode step
#   falls OFF the replayed FULL graph and runs kernel-by-kernel from the host.
#   That desyncs the 8 TP ranks -> the straggler starves -> the others spin in
#   cross_device_reduce_2stage (the "all-reduce" symptom). conc-16 (M=48) and
#   conc-48 (M=144) do NOT hit this because their FlyDSL kernelNames aren't
#   catalog-valid, so they already fall back to `torch` (torch.matmul ->
#   rocBLAS/hipBLASLt Cijk kernels, which ARE captured).
#
# FIX: convert the FlyDSL rows for those 3 dense (N,K) shapes at all decode-range
#   M (<=192 = max_num_seqs 64 * decode_query_len 3) to `torch` -- exactly the
#   capturable backend the working concurrencies already use. Prefill (M>192)
#   keeps FlyDSL.
#
# Idempotent. Backs up once to <csv>.pre_flydsl_fix.bak.
set -euo pipefail

CSV="${CSV:-/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv}"
[ -f "$CSV" ] || { echo "!! tuned GEMM CSV not found at $CSV" >&2; exit 1; }

cp -n "$CSV" "$CSV.pre_flydsl_fix.bak" || true
# Count BEFORE rewriting. "0 remaining" alone cannot distinguish "converted N"
# from "the table never had these shapes" — the latter means K3's dense decode
# GEMMs have no tuned entry at all and fall back to the untuned heuristic.
sel='$3<=192 && ( (($4==1024||$4==384)&&$5==7168) || ($4==7168&&$5==512) )'
before_flydsl=$(awk -F, "\$11==\"flydsl\" && $sel" "$CSV" | wc -l)
before_any=$(awk -F, "NR>1 && $sel" "$CSV" | wc -l)
echo "K3 dense decode shapes (M<=192) present in table: $before_any (of which flydsl: $before_flydsl)"
if [ "$before_any" -eq 0 ]; then
  echo "!! WARNING: no tuned rows for the K3 dense decode shapes — dense decode GEMMs will run UNTUNED" >&2
fi
awk -F, -v OFS="," '
NR==1 {print; next}
# NOTE: encode exactly like a native torch row -> splitK(col13)=0, kernelName(col15)=native.
# Leaving splitK EMPTY makes pandas.read_csv infer the whole column as float (NaN),
# which then feeds float splitK (e.g. 3.0) into aiter::_gemm_a16w16_asm (Optional[int]
# only) and crashes engine init. Keep the column integer-typed.
($11=="flydsl" && $3<=192 && ( (($4==1024||$4==384)&&$5==7168) || ($4==7168&&$5==512) )) {
  $11="torch"; $12="0"; $13="0"; $15="native"; print; next
}
{print}
' "$CSV" > "$CSV.new" && mv "$CSV.new" "$CSV"
after_flydsl=$(awk -F, "\$11==\"flydsl\" && $sel" "$CSV" | wc -l)
echo "flydsl->torch converted: $(( before_flydsl - after_flydsl )) rows; flydsl remaining (expect 0): $after_flydsl"
echo "flydsl->torch reroute applied to $CSV"
