#!/usr/bin/env bash
set -euo pipefail

# MI355X validation path for the AMD-capable srt-slurm branch. Matrix rows opt
# in explicitly with CONFIG_FILE; all existing MI355X launch behavior remains
# unchanged for every other row.
SRT_SLURM_REPOSITORY="https://github.com/SemiAnalysisAI/srt-slurm.git"
SRT_SLURM_COMMIT="ad63a66c72404691bdda98c656d6e211156fb582"
SLURM_PARTITION="compute"
SGLANG_IMAGE="lmsysorg/sglang:v0.5.16-rocm720-mi35x"
SHARED_BASE="/it-share/gharunners2/srt-slurm"
SHARED_IMAGE="${SHARED_BASE}/containers/sglang-v0.5.16-mi35x.sqsh"
SHARED_HF_CACHE="/it-share/hf-hub-cache"
SHARED_RESULTS="${SHARED_BASE}/results"

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set by Actions}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set by the benchmark workflow}"
: "${CONFIG_FILE:?CONFIG_FILE must name an srt-slurm recipe}"
: "${MODEL:?MODEL must identify the Hugging Face model}"

CONFIG_PATH="${CONFIG_FILE%%:*}"
LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
CLUSTER_PROFILE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi355x-amds.yaml"
[[ -f "$LOCAL_RECIPE" ]] || { echo "Missing recipe: $LOCAL_RECIPE" >&2; exit 1; }
[[ -f "$CLUSTER_PROFILE" ]] || { echo "Missing cluster profile: $CLUSTER_PROFILE" >&2; exit 1; }

RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-runner}"
WORK_DIR="${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}"
SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
mkdir -p "$WORK_DIR" "$SHARED_RESULTS"

# Materialize one immutable, shared squashfs and the small public validation
# model. This job exits normally and never cancels or preempts another job.
# The shared lock makes concurrent aggregate/disaggregated validations safe.
STAGE_SCRIPT="${WORK_DIR}/stage-mi355x-runtime.sbatch"
cat > "$STAGE_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=${RUNNER_NAME:-mi355x-srt}-stage
set -euo pipefail
mkdir -p "$(dirname "$SHARED_IMAGE")" "$SHARED_HF_CACHE"
exec 9>"${SHARED_IMAGE}.lock"
flock -w 2400 9
if ! unsquashfs -s "$SHARED_IMAGE" >/dev/null 2>&1; then
    tmp="${SHARED_IMAGE}.tmp.\${SLURM_JOB_ID}"
    rm -f "\$tmp"
    local_image="/var/lib/squash/lmsysorg_sglang_v0.5.16-rocm720-mi35x.sqsh"
    if unsquashfs -s "\$local_image" >/dev/null 2>&1; then
        cp --sparse=always "\$local_image" "\$tmp"
    else
        enroot import -o "\$tmp" "docker://${SGLANG_IMAGE}"
    fi
    unsquashfs -s "\$tmp" >/dev/null
    mv "\$tmp" "$SHARED_IMAGE"
fi
flock -u 9
srun --nodes=1 --ntasks=1 \
    --container-image="$SHARED_IMAGE" \
    --container-mounts="$SHARED_HF_CACHE:/hf_hub_cache" \
    --container-writable --container-remap-root --no-container-entrypoint \
    --export=ALL,HF_HOME=/hf_hub_cache,HF_HUB_CACHE=/hf_hub_cache/hub,HUGGINGFACE_HUB_CACHE=/hf_hub_cache/hub,MODEL_REPO=${MODEL} \
    python3 -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["MODEL_REPO"])'
EOF
STAGE_JOB_ID=$(sbatch --wait --parsable "$STAGE_SCRIPT")
STAGE_JOB_ID="${STAGE_JOB_ID%%;*}"
echo "MI355X runtime prerequisites verified with Slurm job ${STAGE_JOB_ID}"

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
python3 - "${WORK_DIR}/srtslurm.yaml" "$GITHUB_WORKSPACE" "$SHARED_RESULTS" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
workspace, results = sys.argv[2:]
needle = "  /it-share/hf-hub-cache: /hf_hub_cache\n"
text = path.read_text()
if text.count(needle) != 1:
    raise SystemExit("expected exactly one Hugging Face cache mount")
path.write_text(
    text.replace(
        needle,
        needle + f"  {workspace}: /infmax-workspace\n  {results}: /results\n",
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
export SRTCTL_RUNTIME_SOURCE_DIR="$SRT_REPO_DIR"

echo "Submitting ${CONFIG_PATH} with srt-slurm ${SRT_SLURM_COMMIT}"
set +e
SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" \
    --tags "mi355x,inferencex,github-actions,${RUN_KEY}" 2>&1)
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
echo "srt-slurm job ${JOB_ID}: state=${JOB_STATE} exit=${JOB_EXIT} nodes=${JOB_NODELIST}"

RESULT_DIR="${SHARED_RESULTS}/${JOB_ID}"
mkdir -p "$GITHUB_WORKSPACE/LOGS"
if [[ -f "$RESULT_DIR/runtime-logs.tar.gz" ]]; then
    cp "$RESULT_DIR/runtime-logs.tar.gz" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
fi
cp -R "$RESULT_DIR/." "$GITHUB_WORKSPACE/LOGS/"

if [[ "${DISAGG:-false}" == "true" ]]; then
    PREFILL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP))
    DECODE_GPUS=$((DECODE_NUM_WORKERS * DECODE_TP))
    TOTAL_GPUS=$((PREFILL_GPUS + DECODE_GPUS))
else
    TOTAL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP * ${PREFILL_PP_SIZE:-1} * ${PREFILL_PCP_SIZE:-1}))
fi

shopt -s nullglob
RESULTS=("$RESULT_DIR"/fixed-seq/*.json)
shopt -u nullglob
[[ ${#RESULTS[@]} -gt 0 ]] || { echo "No fixed-sequence results found in $RESULT_DIR" >&2; exit 1; }
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
echo "MI355X srt-slurm validation completed successfully"
