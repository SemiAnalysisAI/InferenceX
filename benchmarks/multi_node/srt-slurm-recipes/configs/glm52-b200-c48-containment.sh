#!/usr/bin/env bash
set -eo pipefail

installer=/glm52-containment/install_native_containment.py
printf '%s  %s\n' \
    327f68b33957358b5a80c52817e2e4e924311a4abfc699af8093c9f48da311a0 \
    "$installer" | sha256sum --check --status

/opt/sglang/bin/python3 "$installer" \
    --inputs /glm52-containment/inputs \
    --wheels /glm52-containment/wheels \
    --receipt-sha256 27aa9a6eb223616d956dd7d507c0e26839cadcfb84339179898ff3a502ccbcba \
    --output /tmp/glm52-containment-install.json

cat /tmp/glm52-containment-install.json
