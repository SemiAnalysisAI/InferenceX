#!/usr/bin/env bash
# Debug: probe GPU reads of registered host memory on this node, then apply
# the engram device-ptr patch. The probe never fails the job.
set -euo pipefail
python3 /configs/dsv41flash-engram-host-probe.py || echo "engram host probe exited $?"
bash /configs/dsv41flash-sglang-engram-device-ptr.sh
