#!/bin/bash
# Remove every artifact of the SEngine-crossing experiments from a work node and confirm the host runtime
# library is the stock one. Usage (from the laptop): sendrecv/hack_cleanup.sh <node-ip>
# Nothing outside /opt/dlami/nvme/trn3ub/runs was ever written, so this is only housekeeping.
HERE=$(cd "$(dirname "$0")/.." && pwd)
"$HERE/cluster/node.sh" "${1:?node ip}" <<'R'
W=/opt/dlami/nvme/trn3ub
pgrep -fc "sendrecv/[x]seng.sh|p2p/[x]coll.sh" | grep -q "^0$" || { echo "an experiment is still running; not deleting"; exit 1; }
rm -rf $W/runs/hack $W/runs/xseng $W/runs/xseng-long $W/runs/xseng-chips $W/runs/xcoll
got=$(md5sum /opt/aws/neuron/lib/libnrt.so.2.35.30.0 | cut -d" " -f1)
[ "$got" = bb35a80472f94e825f9e97437fc241b0 ] && echo "$(hostname -i): cleaned; libnrt is stock ($got)" || echo "$(hostname -i): libnrt MD5 is $got, NOT the recorded stock value"
env | grep -E "^LD_PRELOAD|^NCMAP" && echo "WARNING: shim variables are set in this shell" || true
R
