#!/usr/bin/env bash
set -eo pipefail

source ./benchmarks/check_env.sh
check_env_vars SGLANG_IMAGE_COMMIT SGLANG_MEGAKERNEL_COMMIT FLASHINFER_TACTIC_SET FLASHINFER_TACTIC_ROLE

for checkout in sglang sgl-nvidia-mega; do
    expected="$SGLANG_IMAGE_COMMIT"
    if [[ "$checkout" == sgl-nvidia-mega ]]; then
        expected="$SGLANG_MEGAKERNEL_COMMIT"
    fi
    actual=$(git -C "/sgl-workspace/$checkout" rev-parse HEAD)
    if [[ "$actual" != "$expected" ]]; then
        echo "$checkout is at $actual, expected $expected" >&2
        exit 1
    fi
done

pins="$(dirname "$0")/qwen35-gb300-flashinfer-tactics/$FLASHINFER_TACTIC_SET"
cache=$(python3 -c 'import os; print(os.environ.get("SGLANG_CACHE_DIR", os.path.expanduser("~/.cache/sglang")))')
flashinfer=$(python3 -c 'from importlib.metadata import version; print(version("flashinfer-python"))')
root="$cache/flashinfer/autotune/$flashinfer/sm103"

copied=0
while read -r role key rank source; do
    [[ "$role" == "$FLASHINFER_TACTIC_ROLE" ]] || continue
    mkdir -p "$root/$key"
    temporary=$(mktemp "$root/$key/.tactic.XXXXXX")
    cp "$pins/$source" "$temporary"
    mv -f "$temporary" "$root/$key/$rank"
    copied=$((copied + 1))
done < "$pins/pins.tsv"
if (( copied == 0 )); then
    echo "No FlashInfer tactics for $FLASHINFER_TACTIC_SET/$FLASHINFER_TACTIC_ROLE" >&2
    exit 1
fi
