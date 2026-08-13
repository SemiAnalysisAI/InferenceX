#!/usr/bin/env bash
# Did the 12,244 c96 run actually sit under or over the 2^31 buffer_load cap?
# The handover records pool 47,210,624 -- work out the KV cache byte span from
# the server log's own numbers rather than guessing.
set -u
L=/home/jiacao/dep8-hiconc/c96_mtpr8192/server.log.gz
echo "=== KV cache / pool lines ==="
zcat "$L" | grep -aiE "GPU KV cache size|kv cache|num_gpu_blocks|maximum concurrency|block_size" | head -20
echo
echo "=== gluon kernel's own line ==="
zcat "$L" | grep -aiE "gluon|pa_decode_sparse|USE_BUFFER_LOAD|sparse.*mla|SPARSE" | head -20
echo
echo "=== was DP attention on ==="
zcat "$L" | grep -aiE "data_parallel|dp_size|enable_expert_parallel" | head -6
