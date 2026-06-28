# shellcheck shell=bash
# CollectiveX — cross-node PG bootstrap network fix (sourced per-rank/per-node).
#
# torch.distributed's gloo/NCCL TCP bootstrap (connectFullMesh / the rendezvous TCPStore) advertises
# each rank's address from its hostname. On clusters whose /etc/hosts aliases the hostname to the
# loopback 127.0.1.1 (MI355X) — or where the default iface isn't the inter-node-routable one — the
# mesh tries to connect to 127.0.1.1 and fails ("Gloo connectFullMesh ... Connection refused,
# remote=[127.0.1.1]"). Pinning GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME to the NIC that holds the
# cluster's routable address (the 10.x management/ethernet subnet on both the MI355X and H200-dgxc
# fleets) makes the bootstrap advertise the reachable interface. RDMA EP transports (UCCL/MoRI/IBGDA)
# still use their own RDMA NICs; this only fixes the TCP control-plane rendezvous.
#
# Respect an operator override; otherwise auto-detect the first iface with a 10.x IPv4.
if [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
  _cx_if="$(ip -o -4 addr show 2>/dev/null | awk '$4 ~ /^10\./ {print $2; exit}')"
  if [ -n "$_cx_if" ]; then
    export GLOO_SOCKET_IFNAME="$_cx_if" NCCL_SOCKET_IFNAME="$_cx_if"
    printf '[collectivex] cross-node PG iface: GLOO/NCCL_SOCKET_IFNAME=%s\n' "$_cx_if" >&2
  fi
fi
