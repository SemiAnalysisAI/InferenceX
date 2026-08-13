#!/usr/bin/env bash
#SBATCH --job-name=dsv4-build
#SBATCH --account=amd-aifw-aim
#SBATCH --qos=amd-aifw-aim-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:10:00
#SBATCH --output=/home/jiacao/InferenceX/dsv4-build-%j.out
#SBATCH --exclude=crsuse2-m2m-203,crsuse2-m2m-074,crsuse2-m2m-240,crsuse2-m2m-029,crsuse2-m2m-061,crsuse2-m2m-071
#
# Build the DSv4-Pro FP4 measurement image the way CI would: take the pinned
# stock nightly base, run apply_dsv4_container_patches.sh *inside* it (the exact
# runtime-patch mechanism PR #2508 uses -- no aiter rebuild, no private vendor
# blobs), and `docker commit` the result to a local tag. The GPU is present so
# the script's import-verify actually loads aiter (catches a post2-python-over-
# v0.1.19-.so ABI break; a login node cannot, aiter probes rocminfo at import).
#
# The commit lives only on this node's docker daemon; we also `docker save` it to
# the shared FS so a later confirmation run (or a push you approve) can load it.
# We do NOT `docker push` -- publishing is your call.
set -uo pipefail

hostname
date -u

BASE="vllm/vllm-openai-rocm:nightly-3ee2df30337a301164c46ae444b76ee67e71c106"
OUT="dsv4-pro-fp4-mi355x:3ee2df30-patched"
PATCH="/home/jiacao/InferenceX/.claude/worktrees/dsv4-ci-sweep-pr/benchmarks/single_node/agentic/apply_dsv4_container_patches.sh"
SAVE_TAR="/home/jiacao/InferenceX/dsv4-pro-fp4-mi355x-f8d03e77-patched.${SLURM_JOB_ID:-manual}.tar"
CTR="dsv4_build_${SLURM_JOB_ID:-manual}"

[ -f "$PATCH" ] || { echo "FATAL: patch script not found: $PATCH" >&2; exit 1; }

# --- pull the pinned base -----------------------------------------------------
for _try in 1 2 3; do
    docker image inspect "$BASE" >/dev/null 2>&1 && break
    echo "pulling $BASE (attempt $_try/3) ..."
    docker pull "$BASE" || true
done
docker image inspect "$BASE" >/dev/null 2>&1 \
    || { echo "FATAL: base pull failed after 3 attempts" >&2; exit 1; }

echo "=== base identity ==="
docker image inspect "$BASE" \
    --format 'id={{.Id}}
created={{.Created}}
digest={{index .RepoDigests 0}}' 2>/dev/null

# --- start an idle GPU container ---------------------------------------------
# network=host so the patch script can git-clone aiter and curl the PR diffs;
# ROCm device flags so import-verify can load the JIT kernels.
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --network=host \
    --entrypoint sleep "$BASE" infinity \
    || { echo "FATAL: could not start container" >&2; exit 1; }

# --- run the patch script inside ---------------------------------------------
echo
echo "=== applying apply_dsv4_container_patches.sh in $CTR ==="
docker exec -i "$CTR" bash < "$PATCH"
rc=$?
echo "patch script rc=$rc"

# --- commit only on a clean apply + verify -----------------------------------
if [ "$rc" -eq 0 ]; then
    docker commit \
        --change 'LABEL dsv4.base=nightly-3ee2df30337a301164c46ae444b76ee67e71c106' \
        --change 'LABEL dsv4.aiter=v0.1.19.post2+main@97d0c6e4+pr4673' \
        --change 'LABEL dsv4.vllm=pr51473(native)+pr51714+pr51918;pr48728=DROPPED(base-drift)' \
        "$CTR" "$OUT"
    echo "committed local image: $OUT"
    docker images "$OUT"

    # docker save is OFF by default: the shared /home volume is at 100% and an
    # 11GB tar hits EDQUOT. The image lives on this node's daemon; verify pins
    # here, and distribution is a registry push (your call). DO_SAVE=1 to force.
    if [ "${DO_SAVE:-0}" = "1" ]; then
        echo "=== docker save -> $SAVE_TAR ==="
        if docker save "$OUT" -o "$SAVE_TAR"; then ls -lh "$SAVE_TAR"; echo "SAVED_OK $SAVE_TAR";
        else echo "SAVE_FAILED (image still present locally as $OUT)"; fi
    else
        echo "docker save skipped (DO_SAVE=0); image is on $(hostname)'s daemon as $OUT"
    fi
else
    echo "patch failed (rc=$rc) -- NOT committing, NOT saving."
fi

# --- cleanup ------------------------------------------------------------------
docker rm -f "$CTR" >/dev/null 2>&1 || true
echo
echo "BUILD_RESULT rc=$rc image=$OUT"
date -u
exit "$rc"
