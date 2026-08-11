#!/usr/bin/env bash
set -euo pipefail

# MI300X srt-slurm validation path. The existing launcher remains the default;
# matrix rows opt in by exporting CONFIG_FILE through additional-settings.
SRT_SLURM_REPOSITORY="https://github.com/SemiAnalysisAI/srt-slurm.git"
SRT_SLURM_COMMIT="141f035b5539fa8bbc1b4018ae4817283093092d"
INFERA_REPOSITORY="https://github.com/cquil11/Infera.git"
INFERA_COMMIT="8ed8f1728c745d4e91ba9eaa09ed81159aa57e41"
ATOM_REPOSITORY="https://github.com/cquil11/ATOM.git"
ATOM_COMMIT="2ab42bcd9473095206d2bd2df263c56a0b6430d9"
SLURM_PARTITION="compute"
EXCLUDED_NODES="chi-mi300x-049,chi-mi300x-121"
REMOTE_BASE="/raid/hf-hub-cache/inferencex/srt-slurm"
VLLM_IMAGE="vllm/vllm-openai-rocm:v0.26.0"
VLLM_ROUTER_IMAGE="vllm/vllm-router:nightly-20260809-d2ba586"
ATOM_IMAGE="rocm/infera:atom-v0.1.1"
VLLM_SQSH="${REMOTE_BASE}/containers/vllm-openai-rocm-v0.26.0.sqsh"
VLLM_ROUTER_SQSH="${REMOTE_BASE}/containers/vllm-router-nightly-20260809-d2ba586.sqsh"
ATOM_SQSH="${REMOTE_BASE}/containers/infera-atom-v0.1.1.sqsh"

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set by Actions}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set by the benchmark workflow}"

: "${CONFIG_FILE:?CONFIG_FILE must name an srt-slurm recipe}"

case "${IMAGE:?IMAGE must identify the recipe container}" in
    "$VLLM_IMAGE")
        ENGINE_IMAGE="$VLLM_IMAGE"
        ENGINE_SQSH="$VLLM_SQSH"
        AUX_IMAGE="$VLLM_ROUTER_IMAGE"
        AUX_SQSH="$VLLM_ROUTER_SQSH"
        ;;
    "$ATOM_IMAGE")
        ENGINE_IMAGE="$ATOM_IMAGE"
        ENGINE_SQSH="$ATOM_SQSH"
        AUX_IMAGE=""
        AUX_SQSH=""
        ;;
    *)
        echo "Unsupported MI300X srt-slurm image: $IMAGE" >&2
        exit 1
        ;;
esac

CONFIG_PATH="${CONFIG_FILE%%:*}"
LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
CLUSTER_PROFILE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi300x-amds.yaml"
[[ -f "$LOCAL_RECIPE" ]] || { echo "Missing recipe: $LOCAL_RECIPE" >&2; exit 1; }
[[ -f "$CLUSTER_PROFILE" ]] || { echo "Missing cluster profile: $CLUSTER_PROFILE" >&2; exit 1; }

RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-runner}"
REMOTE_RUNTIME="${REMOTE_BASE}/runtime/inferencex-${RUN_KEY}"
REMOTE_SRT_RUNTIME="${REMOTE_BASE}/runtime/srt-slurm-${SRT_SLURM_COMMIT}"
REMOTE_INFERA_RUNTIME="${REMOTE_BASE}/runtime/infera-${INFERA_COMMIT}"
REMOTE_ATOM_RUNTIME="${REMOTE_BASE}/runtime/atom-${ATOM_COMMIT}"
REMOTE_RESULTS="${REMOTE_BASE}/results"
WORK_DIR="${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}"
SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
mkdir -p "$WORK_DIR"

# The login and compute nodes do not share a filesystem. Stage only the
# unchanged InferenceX benchmark client and immutable public container images
# onto every eligible node. The cluster has nine nodes and this validation
# excludes two, so the staging allocation must cover all seven remaining nodes.
# The batch job exits normally; it does not cancel or preempt any allocation.
RUNTIME_ARCHIVE="${WORK_DIR}/inferencex-benchmark.tar.gz"
tar -C "$GITHUB_WORKSPACE" -czf "$RUNTIME_ARCHIVE" utils/bench_serving
RUNTIME_PAYLOAD=$(base64 -w0 "$RUNTIME_ARCHIVE")
STAGE_SCRIPT="${WORK_DIR}/stage-runtime.sbatch"
cat > "$STAGE_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --exclude=${EXCLUDED_NODES}
#SBATCH --job-name=${RUNNER_NAME:-mi300x-srt}-stage
set -euo pipefail
source_archive="/tmp/inferencex-benchmark-source-\${SLURM_JOB_ID}.tar.gz"
node_archive="/tmp/inferencex-benchmark-\${SLURM_JOB_ID}.tar.gz"
printf '%s' '${RUNTIME_PAYLOAD}' | base64 -d > "\$source_archive"
sbcast --force "\$source_archive" "\$node_archive"
srun --ntasks-per-node=1 bash -c '
  set -euo pipefail
  runtime="${REMOTE_RUNTIME}"
  srt_runtime="${REMOTE_SRT_RUNTIME}"
  infera_runtime="${REMOTE_INFERA_RUNTIME}"
  atom_runtime="${REMOTE_ATOM_RUNTIME}"
  export ENROOT_RUNTIME_PATH="\${TMPDIR:-/tmp}/enroot-runtime-\${UID}"
  mkdir -p "\$ENROOT_RUNTIME_PATH" "\$runtime" "${REMOTE_RESULTS}" "${REMOTE_BASE}/containers"
  chmod 700 "\$ENROOT_RUNTIME_PATH"
  ensure_container_image() {
    local target="\$1"
    local image="\$2"
    local lock_fd
    local tmp
    if unsquashfs -s "\$target" >/dev/null 2>&1; then
      return
    fi
    exec {lock_fd}>"\${target}.lock"
    flock -w 2400 "\$lock_fd"
    if ! unsquashfs -s "\$target" >/dev/null 2>&1; then
      tmp="\${target}.tmp.\${SLURM_JOB_ID}"
      for attempt in 1 2 3; do
        rm -f "\$tmp"
        if enroot import -o "\$tmp" "docker://\${image}"; then
          break
        fi
        if [[ "\$attempt" -eq 3 ]]; then
          echo "Failed to import \${image} after \${attempt} attempts" >&2
          return 1
        fi
        echo "Retrying \${image} import after attempt \${attempt}" >&2
        sleep "\$((attempt * 10))"
      done
      unsquashfs -s "\$tmp" >/dev/null
      mv "\$tmp" "\$target"
    fi
    flock -u "\$lock_fd"
    exec {lock_fd}>&-
  }
  ensure_git_checkout() {
    local target="\$1"
    local repository="\$2"
    local commit="\$3"
    local temporary="\${target}.tmp.\${SLURM_JOB_ID}.\${BASHPID}"
    local quarantine="\${target}.incomplete.\${SLURM_JOB_ID}.\${BASHPID}"
    if [[ ! -d "\$target/.git" ]]; then
      if [[ -e "\$target" ]]; then
        mv "\$target" "\$quarantine"
      fi
      git clone --quiet "\$repository" "\$temporary"
      git -C "\$temporary" fetch --quiet origin "\$commit"
      git -C "\$temporary" checkout --quiet --detach "\$commit"
      test "\$(git -C "\$temporary" rev-parse HEAD)" = "\$commit"
      mv "\$temporary" "\$target"
    else
      git -C "\$target" fetch --quiet origin "\$commit"
      git -C "\$target" checkout --quiet --detach "\$commit"
      test "\$(git -C "\$target" rev-parse HEAD)" = "\$commit"
    fi
  }
  ensure_container_image "${ENGINE_SQSH}" "${ENGINE_IMAGE}"
  if [[ -n "${AUX_IMAGE}" ]]; then
    ensure_container_image "${AUX_SQSH}" "${AUX_IMAGE}"
  fi
  ensure_git_checkout "\$srt_runtime" "${SRT_SLURM_REPOSITORY}" "${SRT_SLURM_COMMIT}"
  make -C "\$srt_runtime" --no-print-directory setup ARCH=x86_64
  ensure_git_checkout "\$infera_runtime" "${INFERA_REPOSITORY}" "${INFERA_COMMIT}"
  ensure_git_checkout "\$atom_runtime" "${ATOM_REPOSITORY}" "${ATOM_COMMIT}"
  tar -xzf "/tmp/inferencex-benchmark-\${SLURM_JOB_ID}.tar.gz" -C "\$runtime"
  printf "%s\\n" "${GITHUB_SHA:-unknown}" > "\$runtime/.inferencex-source-head"
'
EOF
STAGE_JOB_ID=$(sbatch --wait --parsable "$STAGE_SCRIPT")
echo "Staged InferenceX benchmark client with Slurm job ${STAGE_JOB_ID}"

git clone "$SRT_SLURM_REPOSITORY" "$SRT_REPO_DIR"
git -C "$SRT_REPO_DIR" checkout "$SRT_SLURM_COMMIT"
ACTUAL_SRT_COMMIT=$(git -C "$SRT_REPO_DIR" rev-parse HEAD)
[[ "$ACTUAL_SRT_COMMIT" == "$SRT_SLURM_COMMIT" ]] || {
    echo "srt-slurm checkout mismatch: $ACTUAL_SRT_COMMIT" >&2
    exit 1
}

mkdir -p "${SRT_REPO_DIR}/$(dirname "$CONFIG_PATH")"
cp "$LOCAL_RECIPE" "${SRT_REPO_DIR}/${CONFIG_PATH}"
cp "$CLUSTER_PROFILE" "${WORK_DIR}/srtslurm.yaml"
python3 - "${WORK_DIR}/srtslurm.yaml" "$REMOTE_RUNTIME" "$REMOTE_RESULTS" "$REMOTE_INFERA_RUNTIME" "$REMOTE_ATOM_RUNTIME" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
runtime, results, infera_runtime, atom_runtime = sys.argv[2:]
needle = "  /raid/hf-hub-cache: /hf_hub_cache\n"
text = path.read_text()
if text.count(needle) != 1:
    raise SystemExit("expected exactly one Hugging Face cache mount")
path.write_text(
    text.replace(
        needle,
        needle
        + f"  {runtime}: /infmax-workspace\n"
        + f"  {results}: /results\n"
        + f"  {infera_runtime}: /infera-source\n"
        + f"  {atom_runtime}: /atom-source\n",
    )
)
PY

export PATH="$HOME/.local/bin:$PATH"
cd "$SRT_REPO_DIR"
uv venv --python 3.12
uv pip install -e .
make setup ARCH=x86_64
source .venv/bin/activate
export SRTSLURM_CONFIG="${WORK_DIR}/srtslurm.yaml"
export SRTCTL_RUNTIME_SOURCE_DIR="$REMOTE_SRT_RUNTIME"
export INFMAX_WORKSPACE="$REMOTE_RUNTIME"

echo "Submitting ${CONFIG_PATH} with srt-slurm ${SRT_SLURM_COMMIT}"
set +e
SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" \
    --no-preflight \
    --tags "mi300x,inferencex,github-actions,${RUN_KEY}" 2>&1)
SRTCTL_RC=$?
set -e
echo "$SRTCTL_OUTPUT"
if [[ $SRTCTL_RC -ne 0 ]]; then
    echo "srtctl apply failed with exit code ${SRTCTL_RC}" >&2
    exit "$SRTCTL_RC"
fi
JOB_ID=$(grep -oE 'Job [0-9]+' <<< "$SRTCTL_OUTPUT" | awk '{print $2}' | tail -1)
[[ -n "$JOB_ID" ]] || { echo "Unable to parse srt-slurm job ID" >&2; exit 1; }
echo "SRT_SLURM_JOB_ID=$JOB_ID"

while squeue --noheader --jobs "$JOB_ID" | grep -q .; do
    squeue --noheader --jobs "$JOB_ID" --format='srt-slurm %i %T %M %R'
    sleep 15
done

read -r JOB_STATE JOB_EXIT JOB_NODELIST < <(
    sacct -X --noheader --parsable2 --jobs "$JOB_ID" \
        --format=State,ExitCode,NodeList | head -1 | tr '|' ' '
)
JOB_BATCH_HOST=$(scontrol show job "$JOB_ID" -dd | sed -n 's/.*BatchHost=\([^ ]*\).*/\1/p' | head -1)
[[ -n "$JOB_BATCH_HOST" ]] || {
    echo "Unable to resolve BatchHost for srt-slurm job ${JOB_ID}" >&2
    exit 1
}
echo "srt-slurm job ${JOB_ID}: state=${JOB_STATE} exit=${JOB_EXIT} nodes=${JOB_NODELIST} batch_host=${JOB_BATCH_HOST}"

# Results live on the allocation's node-local RAID. Retrieve the small result
# bundle with a separate completed Slurm job on the batch node.
RETRIEVE_DIR="${WORK_DIR}/retrieved"
mkdir -p "$RETRIEVE_DIR"
RESULT_PAYLOAD=$(srun --partition="$SLURM_PARTITION" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --time=00:05:00 --nodelist="$JOB_BATCH_HOST" \
    bash -c "tar -C '${REMOTE_RESULTS}/${JOB_ID}' -czf - . | base64 -w0")
printf '%s' "$RESULT_PAYLOAD" | base64 -d | tar -xzf - -C "$RETRIEVE_DIR"

mkdir -p "$GITHUB_WORKSPACE/LOGS"
if [[ -f "$RETRIEVE_DIR/runtime-logs.tar.gz" ]]; then
    cp "$RETRIEVE_DIR/runtime-logs.tar.gz" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
fi
cp -R "$RETRIEVE_DIR/." "$GITHUB_WORKSPACE/LOGS/"

if [[ "${DISAGG:-false}" == "true" ]]; then
    PREFILL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP))
    DECODE_GPUS=$((DECODE_NUM_WORKERS * DECODE_TP))
    TOTAL_GPUS=$((PREFILL_GPUS + DECODE_GPUS))
else
    # Aggregate srt-slurm rows intentionally use the multinode workflow so
    # this launcher owns orchestration. The aggregate worker is represented
    # by the prefill-shaped matrix fields; decode workers are zero.
    TOTAL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP * ${PREFILL_PP_SIZE:-1} * ${PREFILL_PCP_SIZE:-1}))
fi
shopt -s nullglob
RESULTS=("$RETRIEVE_DIR"/fixed-seq/*.json)
shopt -u nullglob
[[ ${#RESULTS[@]} -gt 0 ]] || { echo "No fixed-sequence results retrieved" >&2; exit 1; }
for result in "${RESULTS[@]}"; do
    concurrency=$(basename "$result" | sed -n 's/.*-c\([0-9][0-9]*\)\.json/\1/p')
    [[ -n "$concurrency" ]] || { echo "Cannot parse concurrency from $result" >&2; exit 1; }
    if [[ "${DISAGG:-false}" == "true" ]]; then
        output="${GITHUB_WORKSPACE}/${RESULT_FILENAME}_srt-${JOB_ID}_conc${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    else
        output="${GITHUB_WORKSPACE}/${RESULT_FILENAME}_srt-${JOB_ID}_conc${concurrency}_gpus_${TOTAL_GPUS}.json"
    fi
    cp "$result" "$output"
    echo "Collected $output"
done

if [[ "$JOB_STATE" != COMPLETED || "$JOB_EXIT" != 0:0 ]]; then
    echo "srt-slurm validation failed: ${JOB_STATE} (${JOB_EXIT})" >&2
    exit 1
fi

printf '%s\n' "$SRT_SLURM_COMMIT" > "$GITHUB_WORKSPACE/srt-slurm-producer-sha.txt"
echo "MI300X srt-slurm validation completed successfully"
