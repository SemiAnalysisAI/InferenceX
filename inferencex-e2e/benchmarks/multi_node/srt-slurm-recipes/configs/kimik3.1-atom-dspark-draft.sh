#!/usr/bin/env bash
# Stage the Kimi-K3.1 Pro DSpark draft before ATOM starts: with an uncached repo id
# every rank pulls the same 5.4 GB at once. Shared-cache downloads can hit
# transient stale handles, hence the retries.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
for attempt in 1 2 3 4 5; do
    hf download Inferact/Kimi-K3.1-Pro-DSpark-preview && exit 0
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
done
echo "hf download of Inferact/Kimi-K3.1-Pro-DSpark-preview failed after 5 attempts" >&2
exit 1
