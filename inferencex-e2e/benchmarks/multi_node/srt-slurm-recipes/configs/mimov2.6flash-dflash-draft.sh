#!/usr/bin/env bash
# Stage the pinned MiMo-V2.6-Flash-RL snapshot before vLLM starts. Its DFlash drafter
# ships in the checkpoint's dflash/ subfolder, and vLLM loads a draft model only from a
# concrete local path, so the recipe points the draft at this snapshot's dflash/ in the
# shared hub cache. Shared-cache downloads can hit transient stale handles, hence the
# retries.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
REPO=XiaomiMiMo/MiMo-V2.6-Flash-RL
REVISION=f657ba2f7d24c9adc5730675baaa2d34de878a49
# The recipe's speculative-config names this path; keep the two in sync.
DRAFT=/mnt/hf_hub_cache/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/$REVISION/dflash
for attempt in 1 2 3 4 5; do
    if HF_HUB_CACHE=/mnt/hf_hub_cache hf download "$REPO" --revision "$REVISION" >/dev/null; then
        if [[ -f "$DRAFT/config.json" ]]; then
            exit 0
        fi
        echo "hf download finished but $DRAFT/config.json is missing" >&2
        exit 1
    fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
done
echo "hf download of $REPO@$REVISION failed after 5 attempts" >&2
exit 1
