#!/usr/bin/env bash
set -euo pipefail

# Shared MI355X entry point for AMD-capable srt-slurm recipes. Matrix rows opt
# in explicitly with CONFIG_FILE; recipe files own model- and topology-specific
# behavior while this launcher owns staging, submission, logs, and results.
# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

SRT_SLURM_REPOSITORY="https://github.com/SemiAnalysisAI/srt-slurm.git"
SRT_SLURM_COMMIT="c87d7b34b009be920896126013ad6dc74c5a99d5"
SLURM_PARTITION="compute"
SHARED_BASE="/it-share/gharunners2/srt-slurm"
SHARED_HF_CACHE="/it-share/hf-hub-cache"
LEGACY_HF_CACHE="/it-share/hf_home"
SHARED_AIPERF_CACHE="/it-share/aiperf-cache"
SHARED_RESULTS="${SHARED_BASE}/results"

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set by Actions}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set by the benchmark workflow}"
: "${CONFIG_FILE:?CONFIG_FILE must name an srt-slurm recipe}"
: "${IMAGE:?IMAGE must identify the SGLang container image}"
: "${MODEL:?MODEL must identify the Hugging Face model}"

SGLANG_IMAGE="$IMAGE"
IMAGE_KEY="${SGLANG_IMAGE//\//_}"
IMAGE_KEY="${IMAGE_KEY//:/_}"
SHARED_IMAGE="${SHARED_BASE}/containers/${IMAGE_KEY}.sqsh"
LOCAL_IMAGE="/var/lib/squash/${IMAGE_KEY}.sqsh"
SRT_MODEL_LOCAL_PATH="${SRT_MODEL_LOCAL_PATH:-}"
SRT_DRAFT_MODEL="${SRT_DRAFT_MODEL:-}"

CONFIG_PATH="${CONFIG_FILE%%:*}"
LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
CLUSTER_PROFILE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi355x-amds.yaml"
[[ -f "$LOCAL_RECIPE" ]] || { echo "Missing recipe: $LOCAL_RECIPE" >&2; exit 1; }
[[ -f "$CLUSTER_PROFILE" ]] || { echo "Missing cluster profile: $CLUSTER_PROFILE" >&2; exit 1; }

RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-runner}"
WORK_DIR="${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}"
SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
mkdir -p "$WORK_DIR" "$SHARED_RESULTS" "$SHARED_AIPERF_CACHE"

# Materialize one immutable shared squashfs and the requested model inputs.
# Production checkpoints may be Hugging Face repositories or an existing
# shared local directory. Legacy cache layouts are linked into the canonical
# HF_HOME/hub layout so large checkpoints are reused without copies.
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
    local_image="${LOCAL_IMAGE}"
    if unsquashfs -s "\$local_image" >/dev/null 2>&1; then
        cp --sparse=always "\$local_image" "\$tmp"
    else
        enroot import -o "\$tmp" "docker://${SGLANG_IMAGE}"
    fi
    unsquashfs -s "\$tmp" >/dev/null
    mv "\$tmp" "$SHARED_IMAGE"
fi
flock -u 9
mkdir -p "$SHARED_HF_CACHE/hub"
seed_legacy_cache() {
    local repo="\$1"
    local cache_key="models--\${repo//\//--}"
    local legacy_model_dir="$SHARED_HF_CACHE/\${cache_key}"
    local canonical_model_dir="$SHARED_HF_CACHE/hub/\${cache_key}"
    exec 8>"$SHARED_HF_CACHE/.\${cache_key}.stage.lock"
    flock -w 2400 8
    if [[ ! -e "\$canonical_model_dir" ]]; then
        if [[ -f "\$legacy_model_dir/refs/main" && -d "\$legacy_model_dir/snapshots" ]]; then
            ln -s "../\${cache_key}" "\$canonical_model_dir"
        elif [[ -f "$LEGACY_HF_CACHE/\${cache_key}/refs/main" && -d "$LEGACY_HF_CACHE/\${cache_key}/snapshots" ]]; then
            ln -s "$LEGACY_HF_CACHE/\${cache_key}" "\$canonical_model_dir"
        fi
    fi
    flock -u 8
}

model_repos=()
if [[ -n "$SRT_MODEL_LOCAL_PATH" ]]; then
    python3 - "$SRT_MODEL_LOCAL_PATH" <<'PYMODEL'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"local model directory does not exist: {root}")
for required in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
    if not (root / required).is_file():
        raise SystemExit(f"local model is missing {required}: {root}")
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(f"local model index has no shards: {root}")
missing = [shard for shard in shards if not (root / shard).is_file()]
if missing:
    raise SystemExit(f"local model is missing {len(missing)} indexed shards: {missing[:5]}")
print(f"validated local model {root}: {len(shards)} indexed shards")
PYMODEL
else
    model_repos+=("$MODEL")
fi
if [[ -n "$SRT_DRAFT_MODEL" ]]; then
    model_repos+=("$SRT_DRAFT_MODEL")
fi
for repo in "\${model_repos[@]}"; do
    seed_legacy_cache "\$repo"
done

if (( \${#model_repos[@]} == 0 )); then
    exit 0
fi
model_repo_list="\$(IFS=,; echo "\${model_repos[*]}")"
srun --nodes=1 --ntasks=1 \
    --container-image="$SHARED_IMAGE" \
    --container-mounts="$SHARED_HF_CACHE:/hf_hub_cache,$LEGACY_HF_CACHE:$LEGACY_HF_CACHE" \
    --container-writable --container-remap-root --no-container-entrypoint \
    --export=ALL,HF_HOME=/hf_hub_cache,HF_HUB_CACHE=/hf_hub_cache/hub,HUGGINGFACE_HUB_CACHE=/hf_hub_cache/hub,MODEL_REPOS="\$model_repo_list" \
    python3 -c 'import os; from huggingface_hub import snapshot_download; [snapshot_download(repo) for repo in os.environ["MODEL_REPOS"].split(",") if repo]'
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
python3 - "${WORK_DIR}/srtslurm.yaml" "${SRT_REPO_DIR}/${CONFIG_PATH}" \
    "$GITHUB_WORKSPACE" "$SHARED_RESULTS" "$SHARED_AIPERF_CACHE" "$SHARED_IMAGE" <<'PY'
import os
import sys
from pathlib import Path

import yaml

profile_path = Path(sys.argv[1])
recipe_path = Path(sys.argv[2])
workspace, results, aiperf_cache, image_path = sys.argv[3:]
needle = "  /it-share/hf-hub-cache: /hf_hub_cache\n"
text = profile_path.read_text()
if text.count(needle) != 1:
    raise SystemExit("expected exactly one Hugging Face cache mount")
profile_path.write_text(
    text.replace(
        needle,
        needle
        + f"  {aiperf_cache}: /aiperf_mmap_cache\n"
        + f"  {workspace}: /infmax-workspace\n"
        + f"  {results}: /results\n",
    )
)

recipe = yaml.safe_load(recipe_path.read_text())
container_alias = recipe["model"]["container"]
profile = yaml.safe_load(profile_path.read_text())
profile.setdefault("containers", {})[container_alias] = image_path
profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))

benchmark_env = recipe.setdefault("benchmark", {}).setdefault("env", {})
forwarded = (
    "AIPERF_EXPERIMENTAL_FAST",
    "CONC",
    "CONC_LIST",
    "DECODE_DP_ATTN",
    "DECODE_EP",
    "DECODE_NUM_WORKERS",
    "DECODE_PCP_SIZE",
    "DECODE_PP_SIZE",
    "DECODE_TP",
    "DURATION",
    "EVAL_CONC",
    "EVAL_LIMIT",
    "EVAL_ONLY",
    "FRAMEWORK",
    "IS_AGENTIC",
    "ISL",
    "KV_OFFLOADING",
    "MAX_MODEL_LEN",
    "MODEL",
    "MODEL_PREFIX",
    "PREFILL_DP_ATTN",
    "PREFILL_EP",
    "PREFILL_NUM_WORKERS",
    "PREFILL_PCP_SIZE",
    "PREFILL_PP_SIZE",
    "PREFILL_TP",
    "PRECISION",
    "RESULT_FILENAME",
    "RUN_EVAL",
    "RUNNER_TYPE",
    "OSL",
    "SPEC_DECODING",
    "TOTAL_CPU_DRAM_GB",
)
for key in forwarded:
    value = os.environ.get(key)
    if value:
        benchmark_env[key] = value

# The legacy MI355X launcher sized DP+EP admission from the largest
# concurrency exercised by a recipe. It also honored a model-specific MoRI
# dispatch pin when present; only the inter-kernel switch threshold was
# derived per topology. Preserve those semantics rather than treating MTP
# draft tokens as additional independent requests.
if (
    os.environ.get("PREFILL_DP_ATTN", "false").lower() == "true"
    and int(os.environ.get("PREFILL_EP", "1")) > 1
):
    concurrency_text = os.environ.get("CONC_LIST") or os.environ.get("CONC")
    if not concurrency_text:
        raise SystemExit("DP+EP recipe requires CONC_LIST or CONC")
    concurrency_values = concurrency_text.split()
    concurrency = max(int(value) for value in concurrency_values)
    prefill = recipe["backend"]["sglang_config"]["prefill"]
    decode = recipe["backend"]["sglang_config"]["decode"]
    prefill["max-running-requests"] = concurrency
    decode["max-running-requests"] = concurrency

    decode_tp = int(os.environ["DECODE_TP"])
    decode_environment = recipe["backend"]["decode_environment"]
    dispatch_tokens = max(1, concurrency // decode_tp)
    decode_environment.setdefault(
        "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", str(dispatch_tokens)
    )
    decode_environment["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] = str(
        2 * dispatch_tokens
    )

eval_only = os.environ.get("EVAL_ONLY", "false").lower() == "true"
run_eval = os.environ.get("RUN_EVAL", "false").lower() == "true"
if eval_only or run_eval:
    decode_env = recipe.get("backend", {}).get("decode_environment", {})
    for key in (
        "SGLANG_SIMULATE_ACC_LEN",
        "SGLANG_SIMULATE_ACC_METHOD",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE",
    ):
        decode_env.pop(key, None)
    server_config = recipe.get("backend", {}).get("sglang_config", {})
    for mode in ("prefill", "decode"):
        server_config.get(mode, {}).pop("ep-dispatch-algorithm", None)

    resources = recipe.get("resources", {})
    prefill = server_config.get("prefill", server_config.get("aggregated", {}))
    decode = server_config.get("decode", prefill)

    def topology_value(config, *keys, default=1):
        for key in keys:
            if key in config:
                return int(config[key])
        return default

    topology_defaults = {
        "IS_MULTINODE": "true",
        "MODEL_NAME": os.environ["MODEL"],
        "EVAL_MAX_MODEL_LEN": str(
            prefill.get(
                "context-length", os.environ.get("MAX_MODEL_LEN", "16384")
            )
        ),
        "PREFILL_TP": str(
            topology_value(prefill, "tp-size", "tensor-parallel-size")
        ),
        "PREFILL_EP": str(
            topology_value(prefill, "ep-size", "expert-parallel-size")
        ),
        "PREFILL_NUM_WORKERS": str(
            resources.get("prefill_workers", resources.get("agg_workers", 1))
        ),
        "DECODE_TP": str(
            topology_value(decode, "tp-size", "tensor-parallel-size")
        ),
        "DECODE_EP": str(
            topology_value(decode, "ep-size", "expert-parallel-size")
        ),
        "DECODE_NUM_WORKERS": str(
            resources.get("decode_workers", resources.get("agg_workers", 1))
        ),
        "PREFILL_DP_ATTN": str(prefill.get("enable-dp-attention", False)).lower(),
        "DECODE_DP_ATTN": str(decode.get("enable-dp-attention", False)).lower(),
    }
    for key, value in topology_defaults.items():
        benchmark_env.setdefault(key, value)

    eval_command = r'''
set -euo pipefail
eval_root="/results/${SLURM_JOB_ID}/eval"
mkdir -p "${eval_root}"
cd "${eval_root}"
source /infmax-workspace/benchmarks/benchmark_lib.sh
export EVAL_SERVER_HOST="${SRT_FRONTEND_HOST}"
if [[ -n "${EVAL_CONC:-}" ]]; then
  export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
else
  export EVAL_CONCURRENT_REQUESTS="$(printf '%s\n' "${CONC_LIST:-${CONC:-1}}" | tr ' ' '\n' | sort -n | tail -1)"
fi
export CONC="${EVAL_CONCURRENT_REQUESTS}"
bridge_disagg_eval_metadata
run_eval --framework lm-eval --port "${SRT_FRONTEND_PORT}"
append_lm_eval_summary
'''.strip()
    if eval_only:
        recipe["benchmark"]["command"] = eval_command
    else:
        recipe["benchmark"]["command"] = (
            recipe["benchmark"]["command"].rstrip() + "\n" + eval_command
        )
recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
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

OUTPUT_LOG_DIR="${SHARED_BASE}/outputs/${JOB_ID}/logs"
LOG_FILE="${OUTPUT_LOG_DIR}/sweep_${JOB_ID}.log"
stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || exit 1

read -r JOB_STATE JOB_EXIT JOB_NODELIST < <(
    sacct -X --noheader --parsable2 --jobs "$JOB_ID" \
        --format=State,ExitCode,NodeList | head -1 | tr '|' ' '
)
echo "srt-slurm job ${JOB_ID}: state=${JOB_STATE} exit=${JOB_EXIT} nodes=${JOB_NODELIST}"

RESULT_DIR="${SHARED_RESULTS}/${JOB_ID}"
mkdir -p "$GITHUB_WORKSPACE/LOGS"
if [[ -d "$OUTPUT_LOG_DIR" ]]; then
    tar -C "$OUTPUT_LOG_DIR" -czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" .
fi
if [[ -d "$RESULT_DIR" ]]; then
    cp -R "$RESULT_DIR/." "$GITHUB_WORKSPACE/LOGS/"
fi

if [[ "${DISAGG:-false}" == "true" ]]; then
    PREFILL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP))
    DECODE_GPUS=$((DECODE_NUM_WORKERS * DECODE_TP))
    TOTAL_GPUS=$((PREFILL_GPUS + DECODE_GPUS))
else
    TOTAL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP * ${PREFILL_PP_SIZE:-1} * ${PREFILL_PCP_SIZE:-1}))
fi

if [[ "${EVAL_ONLY:-false}" != "true" && "${IS_AGENTIC:-0}" == "1" ]]; then
    shopt -s nullglob
    RESULTS=("$GITHUB_WORKSPACE/${RESULT_FILENAME}"_conc*.json)
    shopt -u nullglob
    [[ ${#RESULTS[@]} -gt 0 ]] || {
        echo "No AgentX aggregate results found for ${RESULT_FILENAME}" >&2
        exit 1
    }
    printf 'Collected %s\n' "${RESULTS[@]}"
elif [[ "${EVAL_ONLY:-false}" != "true" ]]; then
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
fi

if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
    if [[ "${EVAL_ONLY:-false}" == "true" && ! -f "$RESULT_DIR/eval/meta_env.json" ]]; then
        echo "No eval metadata found in $RESULT_DIR/eval" >&2
        exit 1
    fi
    copy_eval_artifacts "$RESULT_DIR/eval" "$GITHUB_WORKSPACE" || exit 1
fi

if [[ "$JOB_STATE" != COMPLETED || "$JOB_EXIT" != 0:0 ]]; then
    echo "srt-slurm validation failed: ${JOB_STATE} (${JOB_EXIT})" >&2
    exit 1
fi

printf '%s\n' "$SRT_SLURM_COMMIT" > "$GITHUB_WORKSPACE/srt-slurm-producer-sha.txt"
echo "MI355X srt-slurm validation completed successfully"
