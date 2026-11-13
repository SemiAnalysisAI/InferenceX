#!/bin/bash
# Two (or more) all_gather streams in one launch, each with its own XOR pairing; runs inside the container.
#   nccom/streams.sh OUT NAME CORES RANKS TILE XOR0 XOR1 [XOR2 ...]
# Stream i pairs rank r with r ^ XORi, so every core sends on every stream at once (e.g. XOR 8 = tray partner,
# XOR 16 = a chip on the same switch plane). Sizes are all_gather output bytes = 2 x TILE.
set -u
OUT=$1; NAME=$2; CORES=$3; R=$4; T=$5; shift 5
mkdir -p "$OUT"; cd /tmp
export NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=$CORES
python3 -c "
import json, sys
R, T = $R, $T; xs = [int(x) for x in sys.argv[1:]]
st = []
for i, X in enumerate(xs):
    json.dump({'replica_groups': [[[r, r ^ X] for r in range(R) if r < r ^ X]]}, open(f'rg{i}.json', 'w'))
    st.append({'operation': 'allg', 'datatype': 'bf16', 'minbytes': str(2 * T), 'maxbytes': str(2 * T), 'custom_replica_groups': f'rg{i}.json'})
json.dump({'streams': st}, open('streams.json', 'w'))" "$@"
rm -f /tmp/nccom_test_log_*
timeout 1500 nccom-test -r "$R" -d bf16 -n 50 -w 10 --non-interactive -t avg p1 p10 p30 p50 p70 p90 p99 --show-algorithm ${CHECK--c random} \
  --streams-json streams.json ${NARGS:-} all_gather -b $((2 * T)) -e $((2 * T)) --custom-replica-groups rg0.json \
  --report-to-json-file "$OUT/$NAME.json" > "$OUT/$NAME.log" 2>&1
for l in /tmp/nccom_test_log_*; do [ -f "$l" ] && grep -E 'level=(error|fatal)|ERROR|unable|Failed|invalid' "$l" | cut -c1-500 | sort -u | head -8; done > "$OUT/$NAME.err"
echo "$NAME streams xors=$* tile=$T"; grep -E "bf16|size|Failure|rror" "$OUT/$NAME.log" | cut -c1-240; head -5 "$OUT/$NAME.err"
