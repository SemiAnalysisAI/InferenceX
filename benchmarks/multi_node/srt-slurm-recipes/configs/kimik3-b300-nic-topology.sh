#!/usr/bin/env bash
# Print this compute node's GPU-to-NIC topology, then verify the rails the
# Mooncake store is pinned to.
#
# Mooncake on b300-nv needs its peers on the same rail: the nodes are
# rail-isolated, and leaving device_name empty let the prefill node pick
# mlx5_1/mlx5_2 that the decode node did not have, which ended in "Active
# handshake RPC failed" and a segfault during bring-up. Naming rails instead
# of leaving them to auto-discovery means the set has to come from the real
# affinity map rather than a guess. The worker logs already list which devices
# are up; what they do not show is which NIC belongs to which GPU, which is
# what this prints.
#
# Everything here is diagnostic and best-effort; nothing fails the job. The
# rail check below reports a mismatch rather than refusing to run, because
# the pinned set was read off four nodes and the scheduler draws from more
# than four: a mismatch means this node was never inventoried, which is not
# the same as this node being broken.

set -uo pipefail

echo "=== [b300-topo] node: $(hostname) ==="

echo "--- nvidia-smi topo -m ---"
nvidia-smi topo -m 2>&1 | head -40 || true

echo "--- ibdev2netdev ---"
ibdev2netdev 2>&1 | head -40 || true

echo "--- ibv_devinfo -l ---"
ibv_devinfo -l 2>&1 | head -40 || true

echo "--- infiniband port state / rate ---"
for _d in /sys/class/infiniband/*; do
    [[ -e "$_d" ]] || continue
    _n="$(basename "$_d")"
    printf '%-10s state=%-14s rate=%s\n' \
        "$_n" \
        "$(cat "$_d/ports/1/state" 2>/dev/null || echo '?')" \
        "$(cat "$_d/ports/1/rate" 2>/dev/null || echo '?')"
done

echo "=== [b300-topo] end ==="

# ---------------------------------------------------------------------------
# Verify the rails the Mooncake store is pinned to.
#
# srtslurm writes one store config for the whole job, so RDMA devices can only
# be named by index -- and an index does not mean the same physical rail on
# every node here. b300-012 puts mlx5_0 on 172.16.0 where every other node puts
# it on 172.16.192, and b300-003 swaps mlx5_3 with mlx5_9. When two nodes
# disagree, the cross-node QP cannot come up: "Failed to modify QP to RTR ...
# Connection timed out", then Mooncake retries via another peer RNIC and the
# transfer hits "local access violation work queue error" against a memory
# region registered for a different device. One earlier bring-up died that
# way, 36 minutes in.
#
# The recipe therefore names only rails whose subnet was identical on every
# node inventoried so far -- b300-003, -011, -012 and -019. Check that here
# rather than trust it, so that if the QP setup does fail later the log
# already says which node and which rail disagreed.
EXPECTED_RAILS="mlx5_4:172.16.128 mlx5_8:172.16.64 mlx5_10:172.17.192 \
mlx5_16:172.17.0 mlx5_20:172.17.128 mlx5_22:172.17.64"

echo "=== [b300-rails] verifying pinned rails on $(hostname) ==="
_rail_bad=0
for _pair in $EXPECTED_RAILS; do
    _dev="${_pair%%:*}"
    _want="${_pair##*:}"
    _port="/sys/class/infiniband/${_dev}/ports/1"

    if [[ ! -d "$_port" ]]; then
        echo "[b300-rails] FAIL ${_dev}: device absent"; _rail_bad=1; continue
    fi
    _state="$(cat "${_port}/state" 2> /dev/null || echo '?')"
    if [[ "$_state" != *ACTIVE* ]]; then
        echo "[b300-rails] FAIL ${_dev}: port state '${_state}'"; _rail_bad=1; continue
    fi
    # GID index 3 is the RoCEv2 IPv4 entry; its last four bytes are the address.
    _gid="$(cat "${_port}/gids/3" 2> /dev/null || echo '')"
    _hex="$(printf '%s' "$_gid" | tr -d ':' | tail -c 8)"
    if [[ ${#_hex} -ne 8 ]]; then
        echo "[b300-rails] FAIL ${_dev}: cannot read GID index 3 (got '${_gid}')"; _rail_bad=1; continue
    fi
    _got="$((16#${_hex:0:2})).$((16#${_hex:2:2})).$((16#${_hex:4:2}))"
    if [[ "$_got" != "$_want" ]]; then
        echo "[b300-rails] FAIL ${_dev}: subnet ${_got}.x, expected ${_want}.x"
        _rail_bad=1
    else
        echo "[b300-rails] ok   ${_dev}: ${_got}.$((16#${_hex:6:2}))"
    fi
done

if [[ "$_rail_bad" -ne 0 ]]; then
    echo "[b300-rails] WARNING: $(hostname) does not match the pinned rail layout." >&2
    echo "[b300-rails] This node was not in the inventory the rails were read from." >&2
    echo "[b300-rails] If the Mooncake handshake fails on this job, that is the" >&2
    echo "[b300-rails] first thing to look at: either exclude the node or re-pin" >&2
    echo "[b300-rails] device_name in the b300 Kimi-K3 agentic recipes." >&2
else
    echo "=== [b300-rails] all pinned rails match ==="
fi

# Nothing to patch: the image carries the Kimi-K3 stack already. Print what it
# is, so the log still records which build served the numbers.
python3 - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("vllm")
if spec and spec.origin:
    ver = os.path.join(os.path.dirname(spec.origin), "_version.py")
    try:
        for line in open(ver):
            if "version" in line and "=" in line:
                print(f"[b300-rails] vllm {line.strip()}")
    except OSError:
        pass
PY
