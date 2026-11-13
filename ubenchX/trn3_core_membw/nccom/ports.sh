#!/bin/bash
# inside the container: which hardware port, DMA engine and queue each core's ring channels use (debug log), one tiny run
#   ports.sh TAG GS RGFILE [XOR]
export NEURON_RT_LOG_LEVEL=DEBUG N=2
bash nccom/chain.sh ${4:-0} 1048576 INTRA_RING_ALG $1 $2 "${3:-}" 12 | tail -1
grep "encd_alg_ring_init_channel_internal" /tmp/nb12.log | sed -E 's/.*\[nec_dev ([0-9]+) channel ([0-9-]+)\] port ([0-9]+), alloc DMA eng ([0-9]+) qid ([0-9]+).*/\1 \2 \3 \4 \5/' | sort -u | sort -n > /tmp/p.txt
python3 - <<'PY'
import collections
rows = [l.split() for l in open('/tmp/p.txt')]
print("lines", len(rows), "| distinct ports", sorted({int(r[2]) for r in rows}), "| distinct engines", sorted({int(r[3]) for r in rows}), "| qids", sorted({int(r[4]) for r in rows}))
by = collections.defaultdict(list)
for d, ch, port, eng, q in rows: by[int(d)].append(f"ch{ch}:port{port}/eng{eng}/q{q}")
for d in list(range(0, 18)) + [24, 25]: print(" core", d, "(chip %d)" % (d // 8), " ".join(by.get(d, [])))
c = collections.Counter((int(r[0]) // 8 % 2, int(r[1].split('-')[0]) if '-' in r[1] else int(r[1]), int(r[2])) for r in rows)
print("count by (chip parity, channel, port):", dict(sorted(c.items())))
PY
