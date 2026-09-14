#!/usr/bin/env bash
#SBATCH --partition=batch_1
#SBATCH --account=benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=25
# sbatch rather than salloc: a queued salloc needs slurmctld to call back into
# the login node, which is blocked on DSXE (job 2549 died before any step).
set -uo pipefail
echo "node=$(hostname) job=$SLURM_JOB_ID $(date -u --iso-8601=seconds)"
lib=/usr/lib/x86_64-linux-gnu
abi() { readelf -V "$1" 2>/dev/null | grep -oE 'IBVERBS_PRIVATE_[0-9]+' | sort -u | tr '\n' ' '; }
echo "== host rdma packages"
dpkg-query -W -f='${Package} ${Version}\n' rdma-core libibverbs1 ibverbs-providers libmlx5-1 libefa1 2>&1
head -20 /opt/amazon/efa_installed_packages 2>&1
echo "== host libraries"
for f in "$lib/libibverbs.so.1" "$lib/libmlx5.so.1" "$lib/libefa.so.1"; do
    printf '%s -> %s : %s\n' "$f" "$(readlink -f "$f")" "$(abi "$f")"
done
ls -l "$lib/libibverbs/" /opt/amazon/efa/lib/libibverbs/ 2>&1
cat /etc/libibverbs.d/*.driver 2>&1
for p in "$lib"/libibverbs/*.so /opt/amazon/efa/lib/libibverbs/*.so; do
    [[ -f "$p" ]] && printf '%s : %s\n' "$p" "$(abi "$p")"
done
echo "== host ibv_devices"
ibv_devices 2>&1
echo "== host rdma ports"
for p in /sys/class/infiniband/*/ports/1; do
    d=${p#/sys/class/infiniband/}; d=${d%%/*}
    printf '%s state=%s phys=%s rate=%s link=%s lid=%s sm_lid=%s\n' "$d" \
        "$(cat "$p/state" 2>&1)" "$(cat "$p/phys_state" 2>&1)" "$(cat "$p/rate" 2>&1)" \
        "$(cat "$p/link_layer" 2>&1)" "$(cat "$p/lid" 2>&1)" "$(cat "$p/sm_lid" 2>&1)"
done
ibv_devinfo -v 2>&1 | grep -vE 'rdmap|efa' | grep -E 'hca_id|link_layer|active_mtu|max_msg|port_lid|sm_lid|state|rate|max_mr|dmabuf|odp' | head -40
echo "== host peer-memory / kernel modules"
ls /sys/kernel/mm/memory_peers/ 2>&1
lsmod | grep -E 'nvidia|peermem|peer_mem|ib_core|mlx5|efa|nv_' 2>&1
cat /proc/driver/nvidia/version 2>&1 | head -1
uname -r
command -v ib_write_bw && { ib_write_bw -d "$(ls /sys/class/infiniband | grep -v rdmap | head -1)" -x 0 -s 4194304 -n 200 -F >/tmp/ibw-srv.$$ 2>&1 & sleep 1; ib_write_bw -d "$(ls /sys/class/infiniband | grep -v rdmap | head -1)" -x 0 -s 4194304 -n 200 -F 127.0.0.1 2>&1 | tail -6; wait; } || echo "no ib_write_bw on host"
echo "== enroot hooks"
ls -l /etc/enroot/hooks.d/ 2>&1
for h in /etc/enroot/hooks.d/*efa* /etc/enroot/hooks.d/*mellanox* /etc/enroot/hooks.d/*aia*; do
    [[ -f "$h" ]] || continue
    echo "--- $h"; sed -n 1,150p "$h" 2>&1
done
grep -v '^#' /etc/enroot/enroot.conf 2>/dev/null | grep . || echo "enroot.conf not readable"
ls /etc/enroot/mounts.d/ 2>&1 && cat /etc/enroot/mounts.d/*.fstab 2>&1

mounts="$GITHUB_WORKSPACE:/workspace,$lib:/host-usr-lib:ro"
[[ -d /opt/amazon ]] && mounts+=",/opt/amazon:/host-opt-amazon:ro"
echo "== container probe with mounts $mounts"
# The node's Enroot PMI hook mounts PMIx step directories; MPI=none leaves them absent.
srun --mpi=pmix \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$mounts" \
    --no-container-mount-home \
    --container-remap-root \
    --container-workdir=/workspace \
    --no-container-entrypoint --export=ALL \
    bash runners/b300_runtime_diagnostic.sh
probe_rc=$?
echo "container probe rc=$probe_rc"
exit "$probe_rc"
