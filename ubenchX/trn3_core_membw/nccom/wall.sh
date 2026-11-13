#!/bin/bash
# Wall-clock cross-check of nccom-test's per-op time; runs inside the container (no container start in the timing).
#   nccom/wall.sh XOR [TILE] [N2]     128 ranks, pairs (r, r ^ XOR), all_gather, no output check
# Two launches of the same NEFF, 50 and N2 timed iterations; (wall2 - wall1) / (N2 - 50) is the wall time of one
# iteration with every rank running. If it equals the tool's average op time, the ranks run back to back and at once,
# and the chip moves sum(tile / t_i) = 8 x tile x mean(1 / t).
X=$1; T=${2:-67108864}; N2=${3:-2050}; R=128
cd /tmp; export NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0-127
python3 -c "
import json; R, X = $R, $X
json.dump({'replica_groups': [[[r, r ^ X] for r in range(R) if r < r ^ X]]}, open('rg.json', 'w'))"
w=()
for n in 50 $N2; do
  s=$(date +%s.%N)
  nccom-test -r $R -d bf16 -n $n -w 10 --non-interactive -t avg p50 --show-algorithm --custom-replica-groups rg.json \
    all_gather -b $((2 * T)) -e $((2 * T)) --report-to-json-file /tmp/w$n.json > /tmp/w$n.log 2>&1
  w+=($(python3 -c "print($(date +%s.%N) - $s)"))
done
python3 -c "
import json
r = json.load(open('/tmp/w$N2.json'))['results'][-1]; t = $T
w = (${w[1]} - ${w[0]}) / ($N2 - 50) * 1e6
print('xor=$X tile=%d MiB %s: wall/iter %.0f us; tool avg %.0f, p50 %.0f us; per chip %.0f GB/s by wall, %.0f by avg, %.0f by p50'
      % (t >> 20, r.get('algorithm'), w, r['time:avg(us)'], r['time:p50(us)'], 8 * t / w / 1e3, 8 * t / r['time:avg(us)'] / 1e3, 8 * t / r['time:p50(us)'] / 1e3))"
