#!/usr/bin/env bash
# Add pipeline parallelism to the Muse Spark decoder and DFLASH worker in the
# dev-muse-spark-open image, which run single-stage only. Each stage builds
# only its own layers, with the fp32 residual crossing stages as a bf16 hi/lo
# pair. Under SGLANG_ENABLE_PP_SPEC the last stage holds the draft: it verifies
# the relayed block and tail-drafts the next, while earlier stages verify the
# same block and relay their DFLASH taps. The patch is cut against the image's
# pinned fork, so a new image needs a new patch.
set -euo pipefail
root=${SGLANG_ROOT:-/sgl-workspace/sglang-muse-spark}
patch_file=${MUSESPARK_PP_PATCH:-/configs/patches/musespark-pp.patch}
if patch -d "$root" -p1 -F0 -R -f -s --dry-run < "$patch_file" > /dev/null 2>&1; then
    echo "musespark-pp: already applied"
    exit 0
fi
patch -d "$root" -p1 -F0 -f -s --dry-run < "$patch_file"
patch -d "$root" -p1 -F0 -f -s --no-backup-if-mismatch < "$patch_file"
echo "musespark-pp: applied"
