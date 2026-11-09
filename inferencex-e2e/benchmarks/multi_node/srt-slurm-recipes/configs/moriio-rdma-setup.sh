#!/usr/bin/env bash
# MoRI-IO engines under pyxis --container-remap-root: log the node's GPU/RDMA kernel stack,
# raise the launching shell's memlock soft limit to its hard limit (the remapped root lacks
# host CAP_IPC_LOCK, so NIC memory registration counts against RLIMIT_MEMLOCK), then
# install SETUP_PIP_PACKAGES. The engine is exec'd by that shell and inherits the limit.
set -eo pipefail
log() { printf '[moriio-rdma-setup %s] %s\n' "$(hostname -s)" "$*"; }

log "kernel $(uname -r), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)"
log "modules: $(grep -E '^(ib_peer_mem|amdgpu|ionic|ionic_rdma|ib_uverbs) ' /proc/modules | cut -d' ' -f1,4 | tr '\n' ';' || true)"
log "memory_peers: $(ls /sys/kernel/mm/memory_peers 2>&1 | tr '\n' ' ')"
for dev in /sys/class/infiniband/*; do
    log "$(basename "$dev") fw $(cat "$dev/fw_ver" 2>/dev/null || echo unknown)"
done

limits() { prlimit --pid "$PPID" --memlock --raw --noheadings --output SOFT,HARD 2>&1 || true; }
log "memlock soft/hard before: $(limits)"
if hard=$(prlimit --pid "$PPID" --memlock --raw --noheadings --output HARD 2>/dev/null) && [[ -n "$hard" ]]; then
    prlimit --pid "$PPID" --memlock="${hard}:" || log "could not raise the memlock soft limit"
fi
log "memlock soft/hard after: $(limits)"

exec bash "$(dirname "${BASH_SOURCE[0]}")/pip-runtime-deps.sh"
