#!/bin/bash
# Cluster-specific RDMA device detection and SGLang/MoRI environment setup.
#
# Expandability:
#   - Set AMD_CLUSTER_TYPE=ionic or AMD_CLUSTER_TYPE=rdma to skip hostname detection.
#   - To add a new cluster, either add a hostname pattern below or use AMD_CLUSTER_TYPE.

set -x

if [[ -n "${AMD_CLUSTER_TYPE:-}" ]]; then
    # Explicit cluster type override — no hostname detection needed
    case "$AMD_CLUSTER_TYPE" in
        ionic)
            export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
            ;;
        rdma)
            export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
            ;;
        *)
            echo "[Error] Unknown AMD_CLUSTER_TYPE: $AMD_CLUSTER_TYPE (expected: ionic, rdma)"
            exit 1
            ;;
    esac
    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
    export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
else
    # Hostname-based auto-detection
    NODENAME=$(hostname)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
        export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
        export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}')
    elif [[ $NODENAME == mia1* ]]; then
        export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
        export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
        export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
    else
        echo "[Error] Unable to detect cluster from hostname '$NODENAME'."
        echo "        Set AMD_CLUSTER_TYPE=ionic or AMD_CLUSTER_TYPE=rdma explicitly."
        exit 1
    fi
fi

set +x

export NCCL_IB_HCA=$IBDEVICES

export SGLANG_USE_AITER=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# Disable allocating memory in one pass
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_FP8_DISP=True

export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

export MORI_APP_LOG_LEVEL=INFO

ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
    TC=$(( 4 * ND_DSCP ))
    export MORI_RDMA_SL=$ND_PRIO
    export MORI_RDMA_TC=$TC
else
    echo "[WARN] nicctl QoS data unavailable; MORI_RDMA_SL and MORI_RDMA_TC not set."
    echo "       This is expected outside the Docker container."
fi
