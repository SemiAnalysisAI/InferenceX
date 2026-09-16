#!/usr/bin/env bash
# CollectiveX Docker launcher for the Slurm-less "-tw" AMD clusters (mi325x-tw, mi300x-tw).
# Their GHA runners run as `gharunner` on an 8x CDNA (gfx942) node with Docker but no Slurm or
# enroot, so each case runs in a Docker container driven by torchrun. EP8 scale-up only: there
# is no scheduler or RDMA fabric to build EP16 scale-out on.
set -eo pipefail

HERE="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

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

# collx_launcher_prologue requires COLLX_SQUASH_DIR (enroot squash), which this cluster lacks;
# run only the fail-safe trap and the operator config (source of the Docker image tag).
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
# -tw runner accounts differ: mi325x-tw's is in the docker group, mi300x-tw's `cam` only has
# passwordless sudo.
DOCKER=(docker)
if ! docker ps >/dev/null 2>&1; then
  if sudo -n docker ps >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    collx_die "docker present but unusable by $(id -un): not in the docker group and no passwordless sudo"
  fi
fi

# Reap containers left by an earlier leg. `docker run --rm` only removes on exit, so when Actions
# kills the runner's process tree the root-owned container survives the non-root cleanup step and
# pins every GPU; the symptom is hipIpcGetMemHandle "invalid argument" for jobs with >= 3 ranks.
# One runner per node, so any container from the pinned image that predates this launcher is stale.
COLLX_CX_LABEL="collectivex.leg"
for stray_id in $("${DOCKER[@]}" ps -q --filter "ancestor=$IMAGE" 2>/dev/null); do
  stray_started="$("${DOCKER[@]}" inspect -f '{{.State.StartedAt}}' "$stray_id" 2>/dev/null)" || continue
  collx_log "reaping stray container ${stray_id:0:12} from an earlier leg (started $stray_started)"
  "${DOCKER[@]}" rm -f "$stray_id" >/dev/null 2>&1 || true
done

"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 \
  || "${DOCKER[@]}" pull "$IMAGE" >&2 \
  || collx_die "docker pull failed for $IMAGE"

collx_log "runner=$RUNNER nodes=1 x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH image=$IMAGE (${DOCKER[*]}/torchrun)"

# UCCL is not in the image; build once into a prefix every case container puts on PYTHONPATH.
# The prefix is node-local /tmp keyed on the pinned commit, not the job root: the build writes
# as root and the non-root cleanup step cannot rm root-owned files under the job root.
UCCL_PFX_MOUNT=()
if [ "$COLLX_BENCH" = uccl-ep ]; then
  REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
  collx_prepare_uccl_source "$REPO_ROOT" || collx_die "UCCL source preparation failed"
  UCCL_ARCH="$(python3 - "$COLLX_DIR/configs/platform_config.json" "$RUNNER" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["platforms"][sys.argv[2]]["arch"])
PY
)"
  # Cache key = pinned commit + image content id + GPU arch: the content id (not the mutable tag)
  # catches a torch/ROCm ABI change under a re-pushed tag, and the arch a cross-SKU reuse.
  UCCL_IMAGE_ID="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null \
    || printf '%s' "$IMAGE")"
  UCCL_CACHE_KEY="$(printf '%s\0%s\0%s' "$COLLX_UCCL_COMMIT" "$UCCL_IMAGE_ID" "$UCCL_ARCH" \
    | sha1sum | cut -c1-16)"
  UCCL_PFX_HOST="/tmp/collx-uccl-pfx-$UCCL_CACHE_KEY"
  UCCL_PFX_MOUNT=(-v "$UCCL_PFX_HOST:/uccl_pfx")
  # `.ready` is written last, after the in-container import check, so an interrupted copy never
  # leaves a half-populated cache; build into a temp dir and publish atomically with `mv -T`.
  if [ ! -f "$UCCL_PFX_HOST/.ready" ]; then
    collx_log "uccl-ep: one-time from-source build (arch=$UCCL_ARCH, key=$UCCL_CACHE_KEY, USE_DMABUF, host-atomic path)"
    rm -rf "$UCCL_PFX_HOST"   # clear any partial/aborted prior attempt (no .ready)
    uccl_build_tmp="$(mktemp -d /tmp/collx-uccl-pfx.XXXXXX)" || collx_die "uccl-ep: mktemp failed"
    if "${DOCKER[@]}" run --rm \
        --device /dev/kfd --device /dev/dri --group-add video --group-add render \
        --ipc host --shm-size 32g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        -v "$COLLX_DIR:/cx" -v "$uccl_build_tmp:/uccl_pfx" -w /cx "$IMAGE" \
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
          touch /uccl_pfx/.ready   # publish gate: written only after the import check succeeds
        ' >&2; then
      mv -T "$uccl_build_tmp" "$UCCL_PFX_HOST" 2>/dev/null || rm -rf "$uccl_build_tmp"
    else
      rm -rf "$uccl_build_tmp"
      collx_die "uccl-ep from-source build failed"
    fi
    [ -f "$UCCL_PFX_HOST/.ready" ] || collx_die "uccl-ep: build did not publish a ready cache"
    collx_log "uccl-ep: build persisted to $UCCL_PFX_HOST"
  else
    collx_log "uccl-ep: reusing persisted build at $UCCL_PFX_HOST"
  fi
fi

# $COLLX_DIR is mounted so run_ep.py's results/*.json land where the workflow collects them.
# Per-case argv comes from config.py case-args (the Slurm launcher's codec) as a NUL-delimited
# argv file, never as env.
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
  # MoRI's SDMA "anvil" transport (hsaKmtCreateQueueExt with HSA_QUEUE_SDMA_BY_ENG_ID) fails at
  # init on the mi300x-tw kernel thunk (anvil.cpp:193); disable it there so MoRI falls back to
  # the hipIpc/P2P intra-node path. mi325x-tw's thunk accepts the SDMA queue.
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
  # A cold first torchrun on a freshly-imported image occasionally dies at worker launch before
  # run_ep.py starts (no output, ~5s) while the same case then runs fine; retry once. The
  # successful attempt overwrites --out.
  case_ok=0
  for attempt in 1 2; do
    collx_log "case $ci/$ncases attempt $attempt: docker torchrun --nproc-per-node=$NGPUS"
    if "${DOCKER[@]}" run --rm \
        --label "$COLLX_CX_LABEL=${COLLECTIVEX_EXECUTION_ID:-manual}" \
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
