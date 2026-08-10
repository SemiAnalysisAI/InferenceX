#!/usr/bin/env bash
set -euo pipefail

# MI300X srt-slurm validation path. The existing launcher remains the default;
# matrix rows opt in by exporting CONFIG_FILE through additional-settings.
SRT_SLURM_REPOSITORY="https://github.com/SemiAnalysisAI/srt-slurm.git"
SRT_SLURM_COMMIT="105824810a67d52b58761077ad3b94d4a05eb3ac"
SLURM_PARTITION="compute"
EXCLUDED_NODES="chi-mi300x-049,chi-mi300x-121"
REMOTE_BASE="/raid/hf-hub-cache/inferencex/srt-slurm"

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set by Actions}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set by the benchmark workflow}"

# The existing MI300X orchestration-capable runner pool defaults to the
# aggregate recipe. Multinode rows continue to pass their recipe explicitly
# through the existing worker additional-settings contract.
if [[ -z "${CONFIG_FILE:-}" && "${RUNNER_TYPE:-}" == "mi300x-disagg" ]]; then
    CONFIG_FILE="recipes/vllm/qwen3-0.6b/mi300x/agg-fixed-seq.yaml"
fi
: "${CONFIG_FILE:?CONFIG_FILE must name an srt-slurm recipe}"

CONFIG_PATH="${CONFIG_FILE%%:*}"
LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
CLUSTER_PROFILE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi300x-amds.yaml"
[[ -f "$LOCAL_RECIPE" ]] || { echo "Missing recipe: $LOCAL_RECIPE" >&2; exit 1; }
[[ -f "$CLUSTER_PROFILE" ]] || { echo "Missing cluster profile: $CLUSTER_PROFILE" >&2; exit 1; }

RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-runner}"
REMOTE_RUNTIME="${REMOTE_BASE}/runtime/inferencex-${RUN_KEY}"
REMOTE_SRT_RUNTIME="${REMOTE_BASE}/runtime/srt-slurm-${SRT_SLURM_COMMIT}"
REMOTE_RESULTS="${REMOTE_BASE}/results"
WORK_DIR="${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}"
SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
mkdir -p "$WORK_DIR"

# The login and compute nodes do not share a filesystem. Stage only the
# unchanged InferenceX benchmark client onto every eligible node. The batch job
# exits normally; it does not cancel or preempt any allocation.
RUNTIME_ARCHIVE="${WORK_DIR}/inferencex-benchmark.tar.gz"
tar -C "$GITHUB_WORKSPACE" -czf "$RUNTIME_ARCHIVE" utils/bench_serving
RUNTIME_PAYLOAD=$(base64 -w0 "$RUNTIME_ARCHIVE")
STAGE_SCRIPT="${WORK_DIR}/stage-runtime.sbatch"
cat > "$STAGE_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
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
  mkdir -p "\$runtime" "${REMOTE_RESULTS}"
  test -r /raid/hf-hub-cache/inferencex/srt-slurm/containers/vllm-openai-rocm-v0.26.0.sqsh
  test -r /raid/hf-hub-cache/inferencex/srt-slurm/containers/vllm-router-nightly-20260809-d2ba586.sqsh
  if [[ ! -d "\$srt_runtime/.git" ]]; then
    git clone --quiet "${SRT_SLURM_REPOSITORY}" "\$srt_runtime"
  fi
  git -C "\$srt_runtime" fetch --quiet origin "${SRT_SLURM_COMMIT}"
  git -C "\$srt_runtime" checkout --quiet --detach "${SRT_SLURM_COMMIT}"
  test "\$(git -C "\$srt_runtime" rev-parse HEAD)" = "${SRT_SLURM_COMMIT}"
  make -C "\$srt_runtime" --no-print-directory setup-compute ARCH=x86_64
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
python3 - "${WORK_DIR}/srtslurm.yaml" "$REMOTE_RUNTIME" "$REMOTE_RESULTS" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
runtime, results = sys.argv[2:]
needle = "  /raid/hf-hub-cache: /hf_hub_cache\n"
text = path.read_text()
if text.count(needle) != 1:
    raise SystemExit("expected exactly one Hugging Face cache mount")
path.write_text(
    text.replace(
        needle,
        needle + f"  {runtime}: /infmax-workspace\n  {results}: /results\n",
    )
)
PY

export PATH="$HOME/.local/bin:$PATH"
cd "$SRT_REPO_DIR"
uv venv --python 3.12
uv pip install -e .
make setup-compute ARCH=x86_64
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
    TOTAL_GPUS=$((TP * ${PP_SIZE:-1} * ${PCP_SIZE:-1}))
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
