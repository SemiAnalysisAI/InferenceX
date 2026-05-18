#!/bin/bash
# ATOM environment setup for multi-node disaggregated serving (enabling phase).
#
# Sourced by server.sh. Mirrors amd_utils/env.sh structure (RDMA NIC detection,
# default gateway interface, QoS/DSCP detection via nicctl) but trims the
# SGLang/MoRI-specific knobs and adds the ATOM_* / AITER_* env vars used by
# the single-node atom launcher (benchmarks/single_node/dsv4_fp4_mi355x_atom.sh).
#
# REQUIRED env vars (typically set by runners/launch_mi355x-amds.sh):
#   IBDEVICES       RDMA device names, e.g. rdma0,rdma1,...,rdma7 on mia1
#
# OPTIONAL env vars:
#   MORI_RDMA_TC    RDMA traffic class (set by runner if cluster uses QoS).

set -x
export PYTHONDONTWRITEBYTECODE=1

# IBDEVICES: prefer runner-set, fall back to hostname detection.
if [[ -z "${IBDEVICES:-}" ]]; then
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    elif [[ $NODENAME == mia1* ]]; then
        export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    else
        echo "ERROR: Unable to detect cluster from hostname $NODENAME and IBDEVICES not set" >&2
        exit 1
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES from hostname $NODENAME"
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

# Auto-detect default network interface. atom-dev image may not have
# iproute2; fall back to /proc/net/route which is always present.
if command -v ip >/dev/null 2>&1; then
    _GW_IFACE=$(ip route | awk '/^default/ {print $5; exit}')
else
    _GW_IFACE=$(awk 'NR>1 && $2=="00000000" {print $1; exit}' /proc/net/route)
fi
export GLOO_SOCKET_IFNAME=$_GW_IFACE
export NCCL_SOCKET_IFNAME=$_GW_IFACE

export NCCL_IB_HCA=$IBDEVICES

# ATOM-specific env vars carried over verbatim from
# benchmarks/single_node/dsv4_fp4_mi355x_atom.sh. The DSv4 atom launcher
# sets these unconditionally; we replicate them here so the multi-node
# disagg server.sh launches under the same conditions.
export ATOM_DISABLE_MMAP=true
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

# Required by atom's KV-transfer (MoRI-IO) implementation per
# /app/ATOM/atom/kv_transfer/disaggregation/README.md inside the atom-dev
# image: separates MoRI (MoE all-to-all) and MORI-IO (KV transfer) symmetric
# heap pools. Without ISOLATION the two subsystems collide and OOM during
# warmup.
export MORI_SHMEM_MODE=ISOLATION

# atom's startup inserts /app/aiter-test paths at the front of sys.path
# and (apparently) displaces /opt/venv/lib/python3.12/site-packages from
# the inherited path of multiprocessing-spawned ModelRunner subprocesses.
# That breaks `from mooncake.engine import TransferEngine` inside the
# subprocess (where the mooncake package lives in the venv).
# Set PYTHONPATH so Python re-adds the venv site-packages to sys.path
# in every subprocess at interpreter startup, before atom's own sys.path
# mutations run.
if [[ -d /opt/venv/lib/python3.12/site-packages ]]; then
    export PYTHONPATH="/opt/venv/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
fi

# QoS/DSCP configuration (preserved from amd_utils/env.sh — same physical
# NICs, same nicctl path).
if [[ -n "${MORI_RDMA_TC:-}" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC (set by runner or environment)"
elif command -v nicctl &> /dev/null; then
    ND_PRIO=$(nicctl show qos 2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
    ND_DSCP=$(nicctl show qos 2>/dev/null | awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p { print $3; exit }')
    if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
        TC=$(( 4 * ND_DSCP ))
        export MORI_RDMA_SL=$ND_PRIO
        export MORI_IO_SL=$ND_PRIO
        export MORI_RDMA_TC=$TC
        export MORI_IO_TC=$TC
        echo "[INFO] Detected QoS config via nicctl: MORI_RDMA_TC=$MORI_RDMA_TC"
    else
        NODENAME=$(hostname -s)
        if [[ $NODENAME == mia1* ]]; then
            export MORI_RDMA_TC=104
            export MORI_IO_TC=104
        elif [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
            export MORI_RDMA_TC=96
            export MORI_IO_TC=96
        fi
    fi
fi

set +x
