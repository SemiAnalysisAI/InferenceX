#!/usr/bin/env bash
#SBATCH --partition=batch_1
#SBATCH --account=benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10
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
