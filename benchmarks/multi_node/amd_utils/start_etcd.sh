#!/bin/bash
set -x

IPADDRS="${IPADDRS:-localhost}"

# Use management network IP (matching what the Slurm script resolved)
host_ip=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.*src \([^ ]*\).*/\1/p')
if [[ -z "$host_ip" ]]; then
    host_ip=$(hostname -I | awk '{print $1}')
fi

IFS=',' read -ra ADDR <<< "$IPADDRS"

# Determine node name based on position in the IPADDRS list
index=0
for ip in "${ADDR[@]}"; do
  if [[ "$ip" == "$host_ip" ]]; then
    break
  fi
  index=$((index + 1))
done
node_name="etcd-$((index+1))"

# Build initial cluster string
initial_cluster=""
for i in "${!ADDR[@]}"; do
  peer_name="etcd-$((i+1))"
  initial_cluster+="$peer_name=http://${ADDR[i]}:2380"
  if [[ $i -lt $((${#ADDR[@]} - 1)) ]]; then
    initial_cluster+=","
  fi
done

mkdir -p /var/lib/etcd
rm -rf /var/lib/etcd/*

/usr/local/bin/etcd/etcd \
  --name "$node_name" \
  --data-dir /var/lib/etcd \
  --initial-advertise-peer-urls http://$host_ip:2380 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://$host_ip:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster "$initial_cluster" \
  --initial-cluster-state new \
  2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/etcd_NODE${NODE_RANK}.log
