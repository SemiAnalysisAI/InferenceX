# shellcheck shell=bash
# CollectiveX — cross-node PG bootstrap network fix + diagnostic (sourced per-rank/per-node).
#
# torch.distributed's gloo/NCCL TCP bootstrap advertises each rank's address from its hostname. On
# clusters whose /etc/hosts aliases the hostname to loopback 127.0.1.1 (MI355X) the per-rank gloo
# connectFullMesh then tries to connect to 127.0.1.1 and fails ("Gloo connectFullMesh ... Connection
# refused, remote=[127.0.1.1]"). Pinning GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME to the NIC that holds
# the cluster's routable address (the 10.x management/ethernet subnet) makes the mesh advertise the
# reachable interface. RDMA EP transports (UCCL/MoRI/IBGDA) use their own RDMA NICs; this only fixes
# the TCP control-plane rendezvous.
#
# NOTE this does NOT change the TCPStore *connect target* (that is MASTER_ADDR, fixed by the launcher):
# if the rank-0 MASTER_ADDR is unreachable from inside a peer's container network namespace, no iface
# pin helps — that is a cluster topology / container-net property, surfaced by the diagnostic below.
#
# The diagnostic ALWAYS prints what the container can see (hostname + every IPv4), so a cross-node GHA
# log is self-documenting even when auto-detection or reachability fails. Robust to a missing iproute2
# (`ip`) in minimal CUDA images: falls back to `hostname -I` / /proc parsing.

# ---- diagnostic: what does this container's network namespace actually see? ----
_cx_host="$(hostname 2>/dev/null || echo '?')"
if command -v ip >/dev/null 2>&1; then
  _cx_addrs="$(ip -o -4 addr show 2>/dev/null | awk '{print $2"="$4}' | tr '\n' ' ')"
else
  _cx_addrs="(no iproute2) hostname-I=[$(hostname -I 2>/dev/null)]"
fi
printf '[collectivex] xnode-net host=%s rank=%s addrs: %s\n' "$_cx_host" "${RANK:-?}" "$_cx_addrs" >&2

# ---- pin GLOO/NCCL bootstrap iface to the routable 10.x NIC (operator override respected) ----
if [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
  _cx_if=""
  if command -v ip >/dev/null 2>&1; then
    _cx_if="$(ip -o -4 addr show 2>/dev/null | awk '$4 ~ /^10\./ {print $2; exit}')"
  fi
  if [ -n "$_cx_if" ]; then
    export GLOO_SOCKET_IFNAME="$_cx_if" NCCL_SOCKET_IFNAME="$_cx_if"
    printf '[collectivex] cross-node PG iface: GLOO/NCCL_SOCKET_IFNAME=%s\n' "$_cx_if" >&2
  else
    printf '[collectivex] xnode-net: no routable 10.x iface auto-detected (ip present=%s); relying on MASTER_ADDR\n' \
      "$(command -v ip >/dev/null 2>&1 && echo yes || echo no)" >&2
  fi
fi
