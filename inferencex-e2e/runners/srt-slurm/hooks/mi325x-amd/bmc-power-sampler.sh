#!/usr/bin/env bash
# Usage: bmc-power-sampler.sh JOB_ID OUTPUT_CSV
# A cached SDR spares each read the ~10 s repository scan that sets the period of an
# uncached `sensor reading`.
job=$1
out=$2
sdr=$(mktemp)
sudo -n ipmitool sdr dump "$sdr" >/dev/null || sdr=""
echo "epoch_s,power_memory_w,power_cpu_w" > "$out"
while pgrep -f "slurmstepd: \[${job}\." >/dev/null; do
    start=$(date +%s.%N)
    if ! readings=$(sudo -n ipmitool ${sdr:+-S "$sdr"} sensor reading Power_Memory Power_CPU 2>&1); then
        echo "# $(date +%s.%N) ipmitool failed: ${readings//$'\n'/ }" >> "$out"
        sleep 2
        continue
    fi
    end=$(date +%s.%N)
    awk -F'|' -v start="$start" -v end="$end" '
        { gsub(/ /, "", $1); gsub(/ /, "", $2); value[$1] = $2 }
        END { printf "%.3f,%s,%s\n", (start + end) / 2, value["Power_Memory"], value["Power_CPU"] }
    ' <<< "$readings" >> "$out"
done
rm -f "$sdr"
