#!/usr/bin/env bash
# Print comma-separated RDMA HCA names for Broadcom NetXtreme-E (bnxt_re*) ports in state ACTIVE.
# Uses `rdma link` (iproute2 / rdma-core). Prints nothing if no match; exit 0.
# Example line:   link bnxt_re0/1 state ACTIVE physical_state LINK_UP netdev ens26np0

if ! command -v rdma >/dev/null 2>&1; then
  exit 0
fi

rdma link 2>/dev/null | awk '
$1 == "link" && $2 ~ /^bnxt_re/ && $0 ~ /state ACTIVE/ {
  split($2, a, "/")
  d = a[1]
  if (!seen[d]++) printf "%s%s", (n++ ? "," : ""), d
}
END { if (n) print "" }'
