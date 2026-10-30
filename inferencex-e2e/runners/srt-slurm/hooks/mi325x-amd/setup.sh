#!/usr/bin/env bash
set -eo pipefail

# DDR5 power diagnostic: sample the Quanta BMC's node-level Power_Memory (RAPL does
# not cover DIMMs) and Power_CPU for the whole job. Slurm kills processes left in
# this srun step, so the sampler runs as a transient systemd unit and stops when
# the job's slurmstepd exits.
host=$(hostname -s)
out="${SRTCTL_OUTPUT_DIR:?srtctl sets SRTCTL_OUTPUT_DIR}/logs/bmc_power_${host}.csv"
sudo -n ipmitool sensor reading Power_Memory Power_CPU
sudo -n systemd-run --quiet --collect --unit "powerx-bmc-${SLURM_JOB_ID}" \
    --uid "$(id -u)" --gid "$(id -g)" --property RuntimeMaxSec=36000 \
    bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/bmc-power-sampler.sh" "$SLURM_JOB_ID" "$out"
echo "[$host] BMC sampler writing $out"

# hipHostMalloc pins the HiCache tier on the GPUs' NUMA node; drop page cache first so
# pinning does not stall ranks past the startup barrier (see the MI355X hook).
if [[ "${KV_OFFLOADING:-}" == dram ]]; then
    sync
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
fi

# Idle baseline before any worker starts.
sleep 40
echo "[$host] idle baseline ended at $(date +%s.%N)"
