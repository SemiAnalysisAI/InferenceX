#!/bin/bash
# One nccom-test run of paired ranks; runs inside the trn3-ubench container (own /tmp).
#   nccom/one.sh OUT NAME OP CORES RANKS XOR [MAXBYTES]
# OP: ag_hbm | ag_sbuf | cp_hbm (nccom-test all_gather, all_gather --sb2sb, permute). Ranks are the visible cores in
# order at LNC=1; rank r is paired with r ^ XOR, all pairs at once: 2-rank replica groups for all_gather, source-target
# pairs both ways for permute (permute ignores replica groups: with them it times the same ~11.6k on every mask). Sizes are nccom-test's:
# all_gather = output bytes (2 x the per-rank tile), permute = bytes per rank; tile 512 B upward, x4 per step.
# F = size step factor (default 4). XARGS = extra nccom-test arguments (multi-node script mode).
# GRP=all: one replica group of every rank (the tool's default); GRP=percore: one group per core index (rank % 8),
# so each group holds that core of every chip in the launch; default: the XOR pairs. With GRP set, the all_gather
# sizes start at 512 B x group size (512 B per rank). OP also: a2a_hbm (alltoall), ar_hbm (all_reduce), a2av_hbm.
# LNC=2 runs at logical core size 2 (CORES are then logical cores, 64 per node).
# Runs with output checking (-c); if that launch fails, once more without, and the JSON says which.
set -u
OUT=$1; NAME=$2; OP=$3; CORES=$4; R=$5; X=$6; MAXT=${7:-8388608}   # MAXT = largest per-rank tile, bytes
mkdir -p "$OUT"; cd /tmp
export NEURON_LOGICAL_NC_CONFIG=${LNC:-1} NEURON_RT_VISIBLE_CORES=$CORES
python3 -c "
import json; R, X = $R, $X
P = ${PAIRS:-8}   # PAIRS=n: only core indices below n of each chip are paired (the rest idle)
json.dump({'replica_groups': [[[r, r ^ X] for r in range(R) if r < r ^ X < R and r % 8 < P]]}, open('rg.json', 'w'))
json.dump({'src_target_pairs': [[[r, r ^ X] for r in range(R)]]}, open('stp.json', 'w'))"
# STP (permute only): oneway = only the lower rank of each XOR pair sends (core index < PAIRS); mixin = chips 0 mod 4
# only receive, cores 0-3 from the tray partner and 4-7 from the same-plane chip; mixout = the reverse.
[ -n "${STP:-}" ] && python3 -c "
import json; R, X, P, m = $R, $X, ${PAIRS:-8}, '${STP:-}'
if m == 'oneway': st = [[r, r ^ X] for r in range(R) if r < r ^ X < R and r % 8 < P]
else:
    st = [[8 * (c + (1 if k < 4 else 2)) + k, 8 * c + k] for c in range(0, R // 8, 4) for k in range(8)]
    if m == 'mixout': st = [[b, a] for a, b in st]
json.dump({'src_target_pairs': [st]}, open('stp.json', 'w'))"
G=(--custom-replica-groups rg.json); [ $OP = cp_hbm ] && G=(--custom-src-target-pairs stp.json)
GS=2
case ${GRP:-pair} in
  all) G=(); GS=$R ;;
  plane) python3 -c "
import json; R = $R   # one group per (core index, chip parity): that core of the 8 chips of one switch plane
json.dump({'replica_groups': [[[r for r in range(R) if r % 8 == k and (r // 8) % 2 == p] for k in range(8) for p in range(2)]]}, open('rg.json', 'w'))"; GS=$((R / 16)) ;;
  tray) python3 -c "
import json; R = $R   # one group per tray: all 16 cores of the two chips
json.dump({'replica_groups': [[[r for r in range(R) if r // 16 == t] for t in range(R // 16)]]}, open('rg.json', 'w'))"; GS=16 ;;
  swmix|swin|swx) python3 -c "
import json; R = $R; g = '${GRP}'   # switch plane only (4-node job): swin = core pairs to the same-plane chip 2 away in the node (r ^ 16),
# swx = to the same chip of the other node of the rack (r ^ 128), swmix = core index %4 < 2 in-node, else cross-node
m = lambda r: 16 if g == 'swin' or (g == 'swmix' and r % 4 < 2) else 128
json.dump({'replica_groups': [[[r, r ^ m(r)] for r in range(R) if r < r ^ m(r)]]}, open('rg.json', 'w'))"; GS=2 ;;
  quad|quadn|mix) python3 -c "
import json; R = $R; g = '${GRP}'
# quad: one group per (core index, 2 trays = 4 chips), ring order 0 1 3 2 so every ring neighbour is a direct link
# (tray, plane, tray, plane); quadn: natural order 0 1 2 3 (1-2 is two hops). mix: 2-rank pairs, core index < 4 with
# the tray partner, >= 4 with the same-plane chip 2 away: both links of a chip loaded at once.
if g == 'mix':
    rg = [[r, r ^ (8 if r % 8 < 4 else 16)] for r in range(R) if r < r ^ (8 if r % 8 < 4 else 16)]
else:
    o = [0, 1, 3, 2] if g == 'quad' else [0, 1, 2, 3]
    rg = [[32 * q + 8 * c + k for c in o] for q in range(R // 32) for k in range(8)]
json.dump({'replica_groups': [rg]}, open('rg.json', 'w'))"; GS=4; [ ${GRP} = mix ] && GS=2 ;;
  chips[0-9]*|slot|npl2|rackpl|rack|nodepl) python3 -c "
import json; R = $R; g = '${GRP}'; C = R // 8   # whole-chip groups; chip id = 16 * node + chip, nodes 0,1 = rack 0, nodes 3,2 = rack 1
ch = lambda c: list(range(8 * c, 8 * c + 8))
if g.startswith('chips'): n = int(g[5:]); gs = [sum((ch(c) for c in range(q, q + n)), []) for q in range(0, C, n)]   # n consecutive chips
elif g == 'slot': gs = [ch(c) + ch(c ^ 48) for c in range(C) if c < c ^ 48]         # same chip, same slot, other rack (the cable)
elif g == 'npl2': gs = [ch(c) + ch(c ^ 16) for c in range(C) if c < c ^ 16]         # same chip index on the two nodes of a rack (switch plane)
elif g == 'nodepl': gs = [[r for c in range(C) if c // 16 == n and c % 2 == p for r in ch(c)] for n in range(C // 16) for p in (0, 1)]   # one plane of one node
elif g == 'rackpl': gs = [sum((ch(c) for c in range(C) if (c // 16 in nn) and c % 2 == p), []) for nn in ((0, 1), (2, 3)) for p in (0, 1)]   # one plane of one rack
else: gs = [sum((ch(c) for c in range(C) if c // 16 in nn), []) for nn in ((0, 1), (2, 3))]   # one rack
json.dump({'replica_groups': [gs]}, open('rg.json', 'w')); open('gs', 'w').write(str(len(gs[0])))"; GS=$(cat gs) ;;
  percore) python3 -c "
import json; R = $R
json.dump({'replica_groups': [[[r for r in range(R) if r % 8 == k] for k in range(8)]]}, open('rg.json', 'w'))"; GS=$((R / 8)) ;;
esac
case $OP in
  ag_hbm)  A=(all_gather -b $((${MINT:-512} * GS)) -e $((GS * MAXT)) -f ${F:-4}) ;;
  a2a_hbm) A=(alltoall -b $((512 * GS)) -e $((GS * MAXT)) -f ${F:-4}) ;;
  ar_hbm)  A=(all_reduce -b 512 -e $MAXT -f ${F:-4}) ;;
  a2av_hbm) E=${ELEMS:-256}   # alltoallv, ELEMS bf16 elements from every rank to every rank (use with GRP=all); one size per run
    python3 -c "
import json; R, E = $R, $E
json.dump({'metadata': [{'send_counts': [E] * R, 'send_displs': [E * i for i in range(R)], 'recv_counts': [], 'recv_displs': []}]}, open('meta.json', 'w'))"
    A=(alltoallv -b $((2 * E * R)) -e $((2 * E * R)) --variable-size-op-metadata meta.json) ;;
  ag_sbuf) A=(all_gather -b 1024 -e $((2 * MAXT)) -f ${F:-4} --sb2sb --output-shape "[1,131072][4,128]" --cc-dim 0 --sweep-dim 0) ;;
  cp_hbm)  A=(permute -b ${MINT:-512} -e $MAXT -f ${F:-4}) ;;
esac
go() {  # check-flag...
  rm -f /tmp/nccom_test_log_*
  timeout 1500 nccom-test -r "$R" -d bf16 -n 50 -w 10 --non-interactive -t avg p1 p50 p99 --show-algorithm "$@" \
    "${G[@]}" "${A[@]}" --report-to-json-file "$OUT/$NAME.json" > "$OUT/$NAME.log" 2>&1
}
note() { for l in /tmp/nccom_test_log_*; do [ -f "$l" ] && grep -E 'level=(error|fatal)|ERROR|unable|Failed' "$l" | grep -v level=warning | cut -c1-600 | sort -u | head -8; done > "$OUT/$NAME.$1"; }
CHK=1
A+=(${NARGS:-})   # single-node extras, e.g. --channels 2
[ -n "${PCT:-}" ] && A+=(-t avg p5 p15 p25 p35 p45 p55 p65 p75 p85 p95)   # deciles: mean(1 / t) for the chip total when ranks differ
[ -n "${ITERS:-}" ] && A+=(-n $ITERS)   # more timed ops than the default 50: the mean then approaches the steady state
if [ -n "${NOCHECK:-}" ]; then A+=(${XARGS:-}); go || note err; CHK=0   # wall-clock probes: no output check
elif [ -n "${XARGS:-}" ]; then   # multi-node (nccom/xnode.sh): one attempt, every node must issue the same launch
  A+=($XARGS); go -c random || { note err; CHK=0; }
elif ! go -c random; then note err-check; CHK=0; go || note err; fi
# KEEPLOG=1 (nccom/xprobe.sh): keep the tool's own log, minus the bulk
[ -n "${KEEPLOG:-}" ] && cat /tmp/nccom_test_log_* 2>/dev/null | grep -v "Successfully generated\|debug_info\|Loading model" | head -c 400000 > "$OUT/$NAME.nblog"
echo "$NAME op=$OP cores=$CORES ranks=$R xor=$X grp=${GRP:-pair} checked=$CHK" > "$OUT/$NAME.txt"
echo "done $NAME checked=$CHK $(grep -c 'Failure running' "$OUT/$NAME.log") failed sizes"
