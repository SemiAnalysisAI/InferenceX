#!/usr/bin/env bash
# Why does the target image's aiter lack PR #4269 when #4269 merged 2026-08-01,
# nine days before the image's vLLM commit (b22afe45, 2026-08-10)?
#
# Candidate explanations, in order of likelihood:
#   (a) the image installs aiter from a RELEASE TAG / pinned SHA, not main
#   (b) the image installs from a ROCm-vendored branch that lags main
#   (c) the wheel was built earlier and cached
# Distinguish them by reading the installed distribution's own metadata:
# dist-info/METADATA and direct_url.json record where pip got it, and
# aiter/_version.py records what the build tree called itself.
set -u
D="${1:-/home/jiacao/3way-20260812-2214}"

for side in ref target vendor; do
    R=$(find "$D/$side" -maxdepth 9 -type d -name aiter -path '*dist-packages*' 2>/dev/null | head -1)
    [ -z "$R" ] && { echo "=== $side <absent>"; continue; }
    P=$(dirname "$R")
    echo "=========================== $side"

    echo "--- _version.py ---"
    grep -hE "version|commit" "$R/_version.py" 2>/dev/null | grep -E "^\s*(__version__|version|__commit|commit)" | head -4 | sed 's/^/    /'

    echo "--- dist-info ---"
    for di in "$P"/aiter*.dist-info; do
        [ -d "$di" ] || continue
        echo "    $(basename "$di")"
        # direct_url.json is written by pip when installing from a VCS/URL and
        # carries the exact requested ref and resolved commit.
        if [ -f "$di/direct_url.json" ]; then
            echo "    direct_url.json:"
            cat "$di/direct_url.json" 2>/dev/null | sed 's/^/        /'
            echo
        else
            echo "    (no direct_url.json -- not a VCS install)"
        fi
        grep -iE "^(Version|Home-page|Download-URL|Project-URL)" "$di/METADATA" 2>/dev/null | head -6 | sed 's/^/        /'
    done

    echo "--- git metadata left in the tree? ---"
    for g in "$R/../.git" "$R/.git" "$R/../aiter/.git"; do
        [ -e "$g" ] && echo "    found: $g"
    done
    # Some builds stamp the SHA into a hidden file or the jit config.
    grep -rhoE "[0-9a-f]{40}" "$R/jit/optCompilerConfig.json" 2>/dev/null | head -2 | sed 's/^/    optCompilerConfig sha: /'
    echo
done
