#!/usr/bin/env bash
# CollectiveX Docker launcher for the Slurm-less "-tw" AMD clusters (mi325x-tw,
# mi300x-tw), single-node scale-up.
#
# Their GHA runners run as `gharunner` directly on an 8x CDNA (gfx942) node that has
# Docker (gharunner is in the docker/video/render groups) but NO Slurm and NO enroot.
# So unlike the Slurm+enroot mi-amds launcher, this launcher runs each case in a
# Docker container driven by torchrun. It is EP8 scale-up only: there is no scheduler
# or RDMA fabric on these clusters to build EP16 scale-out on.
set -euo pipefail

HERE="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity ---------------------------------------------------------------
RUNNER="${COLLX_SHARD_SKU:-}"
case "$RUNNER" in
  mi325x-tw | mi300x-tw) ;;
  *) collx_die "launch_mi-tw expects a Slurm-less -tw AMD SKU (mi325x-tw|mi300x-tw), got '${RUNNER}'" ;;
esac
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-mori}" COLLX_VENDOR=amd
case "$COLLX_BENCH" in
  mori | uccl-ep) ;;
  *) collx_die "the -tw AMD clusters support only the mori and uccl-ep backends, got '$COLLX_BENCH'" ;;
esac

# ---- setup: trimmed prologue (no Slurm stage-dir / enroot squash) -----------
# collx_launcher_prologue's collx_prepare_stage_dir requires COLLX_SQUASH_DIR (the
# enroot squash path); this cluster has neither, so run only the pieces a Docker
# launcher needs: the fail-safe trap (allocation cleanup no-ops without a JOB_ID)
# and the operator config, which supplies the Docker image tag.
collx_install_launcher_fail_safe
[ -n "${COLLX_SHARD_FILE:-}" ] || collx_die "COLLX_SHARD_FILE is required"
collx_load_operator_config
collx_require_vars COLLX_IMAGE

NODES="${COLLX_NODES:-1}"; GPN="${COLLX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-8}"
[ "$NODES" = 1 ] || collx_die "mi325x-tw is single-node scale-up only (NODES=$NODES); no Slurm/RDMA on this cluster"
NGPUS=$((NODES * GPN))
export COLLX_TRANSPORT=xgmi
IMAGE="$COLLX_IMAGE"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

command -v docker >/dev/null 2>&1 || collx_die "docker not found on the $RUNNER runner"
# -tw runner accounts differ: some are in the docker group (direct socket access,
# e.g. mi325x-tw), others only have passwordless sudo (e.g. mi300x-tw's `cam`).
# Pick whichever works so the launcher is portable across the -tw clusters.
DOCKER=(docker)
if ! docker ps >/dev/null 2>&1; then
  if sudo -n docker ps >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    collx_die "docker present but unusable by $(id -un): not in the docker group and no passwordless sudo"
  fi
fi

# The image is imported once per node and reused; pull only when absent.
"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 \
  || "${DOCKER[@]}" pull "$IMAGE" >&2 \
  || collx_die "docker pull failed for $IMAGE"

collx_log "runner=$RUNNER nodes=1 x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH image=$IMAGE (${DOCKER[*]}/torchrun)"

# ---- uccl-ep: prepare source + build (persisted node-local, outside the job root) --------
# UCCL is not in the image, so build from source. Cases each run in a throwaway `docker run
# --rm`, so build ONCE into a prefix every case container puts on PYTHONPATH. The prefix lives
# in a NODE-LOCAL /tmp path keyed on the pinned commit, NOT under the isolated job root: the
# build container writes it as root, and the workflow's cleanup step (non-root runner) cannot
# rm root-owned files under the job root (that reds an otherwise-green leg). /tmp is outside
# cleanup's scope and lets a second leg on the same node reuse the build. The AMD build applies
# the CDNA managed->pinned-host-memory patch (see prepare_backend.sh).
UCCL_PFX_HOST="/tmp/collx-uccl-pfx-$COLLX_UCCL_COMMIT"
UCCL_PFX_MOUNT=()
if [ "$COLLX_BENCH" = uccl-ep ]; then
  REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
  collx_prepare_uccl_source "$REPO_ROOT" || collx_die "UCCL source preparation failed"
  UCCL_ARCH="$(python3 - "$COLLX_DIR/configs/platform_config.json" "$RUNNER" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["platforms"][sys.argv[2]]["arch"])
PY
)"
  mkdir -p "$UCCL_PFX_HOST"
  UCCL_PFX_MOUNT=(-v "$UCCL_PFX_HOST:/uccl_pfx")
  if [ ! -d "$UCCL_PFX_HOST/deep_ep" ]; then
    collx_log "uccl-ep: one-time from-source build (arch=$UCCL_ARCH, USE_DMABUF, host-atomic path)"
    "${DOCKER[@]}" run --rm \
      --device /dev/kfd --device /dev/dri --group-add video --group-add render \
      --ipc host --shm-size 32g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      -v "$COLLX_DIR:/cx" -v "$UCCL_PFX_HOST:/uccl_pfx" -w /cx "$IMAGE" \
      bash -c '
        set -e
        { pip install -q nanobind || pip install -q --break-system-packages nanobind; } >&2
        rm -rf /tmp/ub && cp -R "/cx/.collx_sources/uccl-'"$COLLX_UCCL_COMMIT"'" /tmp/ub
        # gfx942/gfx950 lack usable managed memory; swap UCCL'"'"'s cudaMallocManaged CPU-proxy
        # handles to pinned host memory (coherent + device-accessible on CDNA).
        sed -i "s/cudaMallocManaged/cudaMallocHost/g" /tmp/ub/ep/src/uccl_ep.cc /tmp/ub/ep/src/uccl_proxy.cpp
        cd /tmp/ub/ep && env USE_DMABUF=1 PER_EXPERT_BATCHING=1 PYTORCH_ROCM_ARCH="'"$UCCL_ARCH"'" python3 setup.py install >&2
        # --no-deps: the wrapper install_requires=["uccl"] pulls the PyPI uccl->uccl-cu12 wheel
        # (absent on ROCm); our from-source ep build already provides uccl.ep in site-packages.
        cd /tmp/ub/ep/deep_ep_wrapper && { pip install -q --no-deps . || pip install -q --no-deps --break-system-packages . ; } >&2
        SP="$(python3 -c "import site;print(site.getsitepackages()[0])")"
        rm -rf /uccl_pfx/* && cp -R "$SP"/deep_ep* "$SP"/uccl* /uccl_pfx/
        python3 -c "import torch,sys; sys.path.insert(0,\"/uccl_pfx\"); import deep_ep; from deep_ep import Buffer; assert hasattr(Buffer,\"get_dispatch_layout\")" >&2
      ' >&2 \
      || collx_die "uccl-ep from-source build failed"
    collx_log "uccl-ep: build persisted to $UCCL_PFX_HOST"
  else
    collx_log "uccl-ep: reusing persisted build at $UCCL_PFX_HOST"
  fi
fi

# ---- execute: one Docker+torchrun invocation per case -----------------------
# The shard control and results dir live under the CX source tree the workflow
# checked out; mount that tree so run_ep.py's `results/*.json` land where the
# workflow's stage step collects them. Per-case run_ep.py argv is decoded from the
# shard by config.py case-args (same codec the Slurm launcher uses), passed to the
# container as a NUL-delimited argv file — never as env.
cd "$COLLX_DIR"
mkdir -p results

ncases="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$COLLX_SHARD_FILE")" \
  || collx_die "cannot count cases in $COLLX_SHARD_FILE"
[ "$ncases" -gt 0 ] || collx_die "shard $COLLX_SHARD_FILE declares no cases"

if [ "$COLLX_BENCH" = uccl-ep ]; then
  # uccl-ep imports the host-persisted build via PYTHONPATH; CDNA needs the aggressive
  # host-atomic EP path (matches prepare_backend.sh's uccl_prepare AMD branch).
  docker_env=(
    -e PYTHONPATH=/uccl_pfx
    -e UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC="${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC:-1}"
    -e HSA_NO_SCRATCH_RECLAIM=1
    -e COLLECTIVEX_SOURCE_SHA="${COLLECTIVEX_SOURCE_SHA:-}"
  )
else
  # MoRI's SDMA "anvil" transport (hsaKmtCreateQueueExt with HSA_QUEUE_SDMA_BY_ENG_ID)
  # fails at init on the mi300x-tw nodes' kernel thunk (anvil.cpp:193, both nodes), so
  # disable it there and let MoRI fall back to the hipIpc/P2P intra-node path (correct
  # results, normal latency). mi325x-tw's thunk accepts the SDMA queue, so keep it on.
  mori_sdma_default=1
  [ "$RUNNER" = mi300x-tw ] && mori_sdma_default=0
  docker_env=(
    -e MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-0}"
    -e MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-$mori_sdma_default}"
    -e MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-info}"
    -e HSA_NO_SCRATCH_RECLAIM=1
    -e COLLECTIVEX_SOURCE_SHA="${COLLECTIVEX_SOURCE_SHA:-}"
  )
fi

final_rc=0
for ((ci = 0; ci < ncases; ci++)); do
  argv_file="$(mktemp "${TMPDIR:-/tmp}/cx-argv.XXXXXX")"
  if ! python3 "$COLLX_RUNTIME_DIR/config.py" case-args \
      "$COLLX_SHARD_FILE" "$ci" "$RUNNER" "$TS" "$NGPUS" "$NODES" "$GPN" "$SCALE_UP_DOMAIN" \
      > "$argv_file"; then
    collx_log "case $ci: argv generation failed"
    final_rc=1; rm -f "$argv_file"; continue
  fi
  # A cold first torchrun on a freshly-imported image occasionally dies at worker
  # launch before run_ep.py even starts (no output, ~5s), while the same case runs
  # fine immediately after (verified: 3/3 standalone successes vs 1 first-invocation
  # flake). Retry once so a transient launch flake does not red an otherwise-good leg;
  # a real failure fails both attempts. The successful attempt overwrites --out.
  case_ok=0
  for attempt in 1 2; do
    collx_log "case $ci/$ncases attempt $attempt: docker torchrun --nproc-per-node=$NGPUS"
    if "${DOCKER[@]}" run --rm \
        --device /dev/kfd --device /dev/dri \
        --group-add video --group-add render \
        --ipc host --shm-size 32g \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --network host \
        "${docker_env[@]}" \
        -v "$COLLX_DIR:/cx" -v "$argv_file:/cx-argv:ro" ${UCCL_PFX_MOUNT[@]+"${UCCL_PFX_MOUNT[@]}"} -w /cx \
        "$IMAGE" \
        bash -c 'xargs -0 torchrun --standalone --nproc-per-node='"$NGPUS"' bench/run_ep.py < /cx-argv'; then
      case_ok=1; break
    fi
    collx_log "case $ci attempt $attempt returned nonzero"
  done
  [ "$case_ok" = 1 ] || { collx_log "case $ci failed after 2 attempts"; final_rc=1; }
  rm -f "$argv_file"
done

collx_log "done - result artifacts in results/ (rc=$final_rc)"
exit "$final_rc"
