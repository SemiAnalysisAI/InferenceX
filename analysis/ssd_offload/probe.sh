#!/usr/bin/env bash
set -eo pipefail

# THROWAWAY: one-off feasibility probe, not a benchmark.
#
# Question: can vLLM on B200 offload KV to local SSD at all? Nothing in this
# repo does -- every kv-offloading path is a host-DRAM tier and the matrix
# schema only knows {none, dram}. Before adding a real recipe, find out
# whether the plumbing works on this hardware with this model.
#
# Route: LMCache's disk tier (local_disk / max_local_disk_size). It is the
# only backend already vendored here that has one upstream. The DRAM tier is
# deliberately made tiny so a cache hit can only come from disk.
#
# Verdict is a JSON blob on stdout. Three things must hold to call it working:
#   1. the server comes up with the connector attached
#   2. LMCache reports bytes actually written to the disk path
#   3. a re-sent prompt, evicted from GPU and CPU, comes back faster than cold
source "$(dirname "$0")/../../benchmarks/benchmark_lib.sh"
check_env_vars MODEL TP RESULT_DIR

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export GPU_COUNT="$TP"

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi

# ---- 1. what disk is actually here ------------------------------------------
# The recipe question hinges on this: a "local NVMe" that turns out to be a
# network mount would make the whole measurement meaningless.
echo "=== block devices ==="
lsblk -o NAME,ROTA,SIZE,TYPE,MOUNTPOINT 2>&1 || true
echo "=== filesystems ==="
df -hT 2>&1 || df -h 2>&1 || true
echo "=== mounts ==="
mount 2>&1 | grep -vE ' (proc|sysfs|devpts|cgroup|tmpfs|overlay)' || true

# Pick a LOCAL disk. The first run of this probe picked /ix and measured NFS
# at 260 MB/s: the candidate list had no entry on the node's own storage, and
# NFS simply had the most free space. This node's 8x3.5T NVMe RAID0 (md0) is
# mounted at /, with 16T free, so prefer a directory there. tmpfs is excluded
# outright -- /tmp here is 1 TB of RAM, which would measure nothing at all.
pick_scratch() {
    if [[ -n "${SSD_PROBE_DIR:-}" ]]; then printf '%s' "$SSD_PROBE_DIR"; return; fi
    local best="" best_avail=0 d avail fstype
    for d in /raid /local /scratch /mnt/local /mnt/nvme /mnt/resource /ephemeral \
             /var/tmp/ssdprobe /ssdprobe "$RESULT_DIR"; do
        mkdir -p "$d" 2>/dev/null || continue
        [[ -w "$d" ]] || continue
        fstype=$(df -PT "$d" 2>/dev/null | awk 'NR==2{print $2}')
        # Neither RAM nor the network is the thing under test.
        case "$fstype" in tmpfs|ramfs|nfs|nfs4|cifs|fuse.*) continue;; esac
        avail=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{print $4}') || continue
        if [[ -n "$avail" && "$avail" -gt "$best_avail" ]]; then best_avail=$avail; best=$d; fi
    done
    printf '%s' "$best"
}
SCRATCH="$(pick_scratch)"
if [[ -z "$SCRATCH" ]]; then
    echo "No writable local (non-tmpfs, non-network) filesystem on this node." >&2
    echo "An SSD-offload benchmark here would be measuring NFS or RAM." >&2
    exit 1
fi
DISK_DIR="$SCRATCH/vllm_kv_tier"
mkdir -p "$DISK_DIR"
DISK_AVAIL_GB=$(( $(df -Pk "$DISK_DIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
DISK_SRC=$(df -P "$DISK_DIR" | awk 'NR==2{print $1}')
DISK_FSTYPE=$(df -PT "$DISK_DIR" 2>/dev/null | awk 'NR==2{print $2}' || echo unknown)
echo "scratch=$DISK_DIR device=$DISK_SRC fstype=$DISK_FSTYPE avail=${DISK_AVAIL_GB}GB"

# Raw bandwidth floor, so a slow cache hit later can be attributed correctly.
DD_W=$(dd if=/dev/zero of="$DISK_DIR/.bw" bs=1M count=4096 oflag=direct 2>&1 | tail -1 || \
       dd if=/dev/zero of="$DISK_DIR/.bw" bs=1M count=4096 conv=fdatasync 2>&1 | tail -1)
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
DD_R=$(dd if="$DISK_DIR/.bw" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1 || \
       dd if="$DISK_DIR/.bw" of=/dev/null bs=1M 2>&1 | tail -1)
rm -f "$DISK_DIR/.bw"
echo "dd write: $DD_W"
echo "dd read:  $DD_R"

# ---- 2. vLLM's own filesystem KV tier ---------------------------------------
# vLLM registers an "fs" secondary tier (FileSystemTierManager, Medium.STORAGE)
# behind TieringOffloadingSpec, driven by OffloadingConnector -- which, unlike
# LMCacheConnectorV1, does subclass SupportsHMA. That matters for this model:
# the connector keeps the hybrid KV cache manager on, so Flash's heterogeneous
# KV specs are never forced through unify_hybrid_kv_cache_specs.
#
# The tier writes <root_dir>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin and probes
# O_DIRECT once, falling back to buffered IO where the filesystem refuses it.
# No external server and no pip install: it is in the image already.

DISK_GB="${SSD_PROBE_DISK_GB:-200}"
if (( DISK_GB > DISK_AVAIL_GB - 20 )); then DISK_GB=$(( DISK_AVAIL_GB - 20 )); fi
if (( DISK_GB < 20 )); then
    echo "not enough free space on $DISK_DIR (${DISK_AVAIL_GB}GB) to probe" >&2; exit 1
fi

# Keep the CPU primary tier small so it cannot hold the working set: anything
# still retrievable after the flush phase has to come back off the fs tier.
CPU_BYTES="${SSD_PROBE_CPU_BYTES:-4294967296}"   # 4 GiB
OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": $CPU_BYTES,
    "secondary_tiers": [
      {"type": "fs", "root_dir": "$DISK_DIR/kv", "n_read_threads": 16, "n_write_threads": 16}
    ]
  }
}
EOF
)
echo "$OFFLOAD_CONFIG" | tee "$RESULT_DIR/offload_config.json"
# Identical prefixes must hash to identical block filenames.
export PYTHONHASHSEED=0

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONUNBUFFERED=1
# No expandable_segments: vLLM refuses LMCacheConnectorV1 alongside it, because
# the VMM allocator can remap KV virtual addresses out from under memory the
# connector has pinned. Carried in by copy from the engram driver; not needed here.
unset PYTORCH_CUDA_ALLOC_CONF

# benchmark_lib on this branch has no select_available_server_port, and pyxis
# shares the host network, so 8888 can already belong to a host service.
for candidate in $(seq "${PORT_FLOOR:-8890}" 8960); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$candidate") 2>/dev/null; then
        PORT="$candidate"; export PORT; break
    fi
done
if [[ -z "${PORT:-}" ]]; then echo "no free port in 8890-8960" >&2; exit 1; fi
echo "Using vLLM endpoint http://127.0.0.1:${PORT}"

# Serve flags copied from the shipped B200 Flash recipe, minus the agentic
# bits. max-model-len is cut to 256k: the indexer's startup buffer scales as
# batched-tokens x max-model-len x 2B and 1M is not needed to prove a disk hit.
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --engram-config '{"cpu_offload":true}'
    --max-model-len 262144
    --max-num-seqs 8
    --max-num-batched-tokens 8192
    --enable-prefix-caching
    --kv-transfer-config "$OFFLOAD_CONFIG"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' >> "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- 3. does a prompt actually come back off the disk ------------------------
PORT="$PORT" MODEL="$MODEL" DISK_DIR="$DISK_DIR" RESULT_DIR="$RESULT_DIR" \
DISK_SRC="$DISK_SRC" DISK_FSTYPE="$DISK_FSTYPE" DD_W="$DD_W" DD_R="$DD_R" \
    python3 "$(dirname "$0")/probe_client.py"
