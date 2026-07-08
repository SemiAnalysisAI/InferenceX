#!/usr/bin/env bash
# Portable srt-slurm multinode orchestrator.
#
# This file is CLUSTER-AGNOSTIC. It must never contain a cluster-specific
# path, partition name, or model location. Cluster facts arrive exclusively
# through the environment contract documented in README.md (same directory)
# and are declared by the multinode branch of a runners/launch_<cluster>.sh
# script, which `source`s this file after setting them.
#
# Flow:
#   1. Validate the contract.
#   2. Resolve the srt-slurm pin (env override > sources.tsv > main).
#   3. Clone srt-slurm and overlay benchmarks/multi_node/srt-slurm-recipes/.
#   4. Install srtctl into a per-run venv.
#   5. Render srtslurm.yaml from contract variables.
#   6. srtctl apply the recipe named by CONFIG_FILE.
#   7. Tail the sweep log until the Slurm job finishes.
#   8. Copy result/eval artifacts to $GITHUB_WORKSPACE using the
#      conventions benchmark-multinode-tmpl.yml expects.
#
# INFX_DRY_RUN=1 prints the resolved plan (pin, workdir, srtslurm.yaml,
# srtctl arguments) and returns before any network or filesystem side
# effects beyond the resolution itself. Use it to sanity-check a cluster
# profile from a login node or in CI.

set -o pipefail

_INFX_SRT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_infx_die() {
    echo "ERROR: $*" >&2
    # This file is sourced by the runner; `return` unwinds to the runner,
    # the trailing `exit` covers direct execution.
    return 1 2>/dev/null || exit 1
}

_infx_require_env() {
    local var missing=""
    for var in "$@"; do
        [[ -n "${!var:-}" ]] || missing="$missing $var"
    done
    if [[ -n "$missing" ]]; then
        _infx_die "missing required contract variable(s):$missing (see benchmarks/multi_node/srt_slurm/README.md)"
    fi
}

# ---------------------------------------------------------------------------
# 1. Contract
# ---------------------------------------------------------------------------

# Cluster facts (set by the runner profile).
_infx_require_env \
    INFX_CLUSTER \
    SLURM_ACCOUNT \
    SLURM_PARTITION \
    INFX_GPUS_PER_NODE \
    INFX_ARCH \
    SQUASH_FILE \
    NGINX_SQUASH_FILE \
    MODEL_PATH \
    SRT_SLURM_MODEL_PREFIX \
    || return 1 2>/dev/null || exit 1

# Benchmark identity (set by benchmark-multinode-tmpl.yml).
_infx_require_env \
    IMAGE \
    MODEL_PREFIX \
    PRECISION \
    FRAMEWORK \
    ISL \
    OSL \
    RUNNER_NAME \
    GITHUB_WORKSPACE \
    || return 1 2>/dev/null || exit 1

if [[ -z "${CONFIG_FILE:-}" ]]; then
    _infx_die "CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings (MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK})" \
        || return 1 2>/dev/null || exit 1
fi

# Optional contract variables and their defaults.
INFX_SLURM_TIME_LIMIT="${INFX_SLURM_TIME_LIMIT:-4:00:00}"
INFX_NETWORK_INTERFACE="${INFX_NETWORK_INTERFACE:-}"
INFX_SRT_WORK_DIR="${INFX_SRT_WORK_DIR:-$GITHUB_WORKSPACE}"
INFX_CONTAINER_KEY="${INFX_CONTAINER_KEY:-$IMAGE}"
INFX_EXTRA_CONTAINER_ALIASES="${INFX_EXTRA_CONTAINER_ALIASES:-}"
INFX_EXTRA_MODEL_ALIASES="${INFX_EXTRA_MODEL_ALIASES:-}"
INFX_SRTSLURM_EXTRA="${INFX_SRTSLURM_EXTRA:-}"
INFX_HEALTH_CHECK_MAX_ATTEMPTS="${INFX_HEALTH_CHECK_MAX_ATTEMPTS:-}"
INFX_NO_PREFLIGHT="${INFX_NO_PREFLIGHT:-0}"
INFX_VENV_PYTHON="${INFX_VENV_PYTHON:-}"
export INFMAX_WORKSPACE="${INFMAX_WORKSPACE:-$GITHUB_WORKSPACE}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
RUN_EVAL="${RUN_EVAL:-false}"
IS_AGENTIC="${IS_AGENTIC:-0}"

# ---------------------------------------------------------------------------
# 2. srt-slurm pin
# ---------------------------------------------------------------------------
# The standard pin is NVIDIA/srt-slurm @ main. sources.tsv is the burn-down
# registry of exceptions; SRT_SLURM_REPO / SRT_SLURM_REF env vars (e.g. from
# a config entry's additional-settings) override both.

_infx_resolve_pin() {
    local table="$_INFX_SRT_SLURM_DIR/sources.tsv"
    local scenario="fixed"
    [[ "$IS_AGENTIC" == "1" ]] && scenario="agentic"

    local cluster_pat scenario_pat framework_pat model_pat precision_pat repo ref
    if [[ -f "$table" ]]; then
        while read -r cluster_pat scenario_pat framework_pat model_pat precision_pat repo ref; do
            [[ -z "$cluster_pat" || "$cluster_pat" == \#* ]] && continue
            # shellcheck disable=SC2254
            case "$INFX_CLUSTER" in $cluster_pat) ;; *) continue ;; esac
            case "$scenario" in $scenario_pat) ;; *) continue ;; esac
            case "$FRAMEWORK" in $framework_pat) ;; *) continue ;; esac
            case "$MODEL_PREFIX" in $model_pat) ;; *) continue ;; esac
            case "$PRECISION" in $precision_pat) ;; *) continue ;; esac
            _INFX_PIN_REPO="$repo"
            _INFX_PIN_REF="$ref"
            return 0
        done < "$table"
    fi
    _INFX_PIN_REPO="https://github.com/NVIDIA/srt-slurm.git"
    _INFX_PIN_REF="main"
}

_INFX_PIN_REPO=""
_INFX_PIN_REF=""
_infx_resolve_pin
SRT_SLURM_REPO="${SRT_SLURM_REPO:-$_INFX_PIN_REPO}"
SRT_SLURM_REF="${SRT_SLURM_REF:-$_INFX_PIN_REF}"

# Unique per-run workspace so parallel matrix jobs sharing a filesystem
# never collide, and a cancelled run never poisons the next one.
# (shasum fallback keeps dry runs working on macOS.)
_infx_sha1() { if command -v sha1sum >/dev/null 2>&1; then sha1sum; else shasum -a 1; fi; }
_INFX_RUN_KEY="$(printf '%s' "${RESULT_FILENAME:-$RUNNER_NAME}" | _infx_sha1 | cut -c1-12)"
SRT_REPO_DIR="${INFX_SRT_WORK_DIR}/srt-slurm-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${_INFX_RUN_KEY}"

# ---------------------------------------------------------------------------
# 3. srtslurm.yaml (rendered up front so dry-run can show it)
# ---------------------------------------------------------------------------

_infx_yaml_model_paths() {
    echo "model_paths:"
    echo "  \"${SRT_SLURM_MODEL_PREFIX}\": \"${MODEL_PATH}\""
    local alias seen="|${SRT_SLURM_MODEL_PREFIX}|"
    for alias in "$MODEL_PREFIX" $INFX_EXTRA_MODEL_ALIASES; do
        case "$seen" in *"|${alias}|"*) continue ;; esac
        seen="${seen}${alias}|"
        echo "  \"${alias}\": \"${MODEL_PATH}\""
    done
}

_infx_yaml_containers() {
    echo "containers:"
    # Recipes across frameworks reference these alias keys; mapping every
    # alias to the one squash file imported for $IMAGE is what each cluster
    # did individually before, unioned here.
    local alias seen="|"
    for alias in dynamo-trtllm dynamo-sglang dynamo-vllm latest v0.5.11 sglang-v0.5.11-cu130 \
                 "$INFX_CONTAINER_KEY" $INFX_EXTRA_CONTAINER_ALIASES; do
        case "$seen" in *"|${alias}|"*) continue ;; esac
        seen="${seen}${alias}|"
        echo "  \"${alias}\": \"${SQUASH_FILE}\""
    done
    echo "  nginx-sqsh: \"${NGINX_SQUASH_FILE}\""
}

_INFX_SRTSLURM_YAML="$(cat <<EOF
# Rendered by benchmarks/multi_node/srt_slurm/run.sh for ${INFX_CLUSTER}
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "${INFX_SLURM_TIME_LIMIT}"
gpus_per_node: ${INFX_GPUS_PER_NODE}
network_interface: "${INFX_NETWORK_INTERFACE}"
srtctl_root: "${SRT_REPO_DIR}"
$(_infx_yaml_model_paths)
$(_infx_yaml_containers)
${INFX_SRTSLURM_EXTRA}
EOF
)"

_INFX_TAGS="${INFX_CLUSTER},${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"

if [[ "${INFX_DRY_RUN:-0}" == "1" ]]; then
    echo "=== INFX multinode dry run ==="
    echo "cluster:        $INFX_CLUSTER"
    echo "srt-slurm repo: $SRT_SLURM_REPO"
    echo "srt-slurm ref:  $SRT_SLURM_REF"
    echo "work dir:       $SRT_REPO_DIR"
    echo "config file:    $CONFIG_FILE"
    echo "setup script:   ${SRTCTL_SETUP_SCRIPT:-<none>}"
    echo "no-preflight:   $([[ "$IS_AGENTIC" == "1" || "$INFX_NO_PREFLIGHT" == "1" ]] && echo yes || echo no)"
    echo "tags:           $_INFX_TAGS"
    echo "--- srtslurm.yaml ---"
    echo "$_INFX_SRTSLURM_YAML"
    echo "=== end dry run ==="
    return 0 2>/dev/null || exit 0
fi

set -x

# ---------------------------------------------------------------------------
# 4. Clone + recipe overlay
# ---------------------------------------------------------------------------

rm -rf "$SRT_REPO_DIR"
git clone "$SRT_SLURM_REPO" "$SRT_REPO_DIR" || _infx_die "failed to clone $SRT_SLURM_REPO"
cd "$SRT_REPO_DIR" || _infx_die "failed to enter $SRT_REPO_DIR"
git checkout "$SRT_SLURM_REF" || _infx_die "failed to checkout $SRT_SLURM_REF"

# Cluster-specific escape hatch (e.g. apply a local patch while an upstream
# fix is in flight). Defined by the runner profile when needed.
if declare -F infx_hook_post_clone >/dev/null; then
    infx_hook_post_clone || _infx_die "infx_hook_post_clone failed"
fi

# Overlay every version-controlled recipe tree onto the clone. CONFIG_FILE
# selects exactly one recipe at apply time, so unused recipes are inert.
_INFX_RECIPES_SRC="$INFMAX_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes"
if [[ -d "$_INFX_RECIPES_SRC" ]]; then
    for _infx_dir in "$_INFX_RECIPES_SRC"/*/; do
        _infx_name="$(basename "$_infx_dir")"
        case "$_infx_name" in
            configs)
                # srtctl setup scripts referenced via --setup-script or a
                # recipe's setup_script field must live in configs/.
                cp "$_infx_dir"/*.sh configs/ 2>/dev/null || true
                ;;
            *)
                mkdir -p "recipes/$_infx_name"
                cp -rT "$_infx_dir" "recipes/$_infx_name"
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# 5. srtctl install
# ---------------------------------------------------------------------------

export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$GITHUB_WORKSPACE/.local/bin}"
if ! command -v uv >/dev/null 2>&1 && ! [ -x "$UV_INSTALL_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# --seed installs pip/setuptools/wheel; srt-slurm's prefetch-ai-dynamo-wheel
# path needs pip inside the venv. Pinning the interpreter matters on clusters
# where the venv lives on shared FS and uv's default python is a
# login-node-only path (the venv symlink would be broken on compute nodes).
if [[ -n "$INFX_VENV_PYTHON" && -x "$INFX_VENV_PYTHON" ]]; then
    uv venv --seed --python "$INFX_VENV_PYTHON" || _infx_die "uv venv failed"
else
    uv venv --seed || _infx_die "uv venv failed"
fi
source .venv/bin/activate
uv pip install -e . || _infx_die "srtctl install failed"
command -v srtctl >/dev/null 2>&1 || _infx_die "srtctl not on PATH after install"

# ---------------------------------------------------------------------------
# 6. Configure + submit
# ---------------------------------------------------------------------------

echo "$_INFX_SRTSLURM_YAML" > srtslurm.yaml
echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

make setup ARCH="$INFX_ARCH" || _infx_die "make setup failed"

# CONFIG_FILE may carry a :override[N] suffix understood by srtctl; file
# edits must target the bare path.
_INFX_CONFIG_PATH="${CONFIG_FILE%%:*}"
[[ -f "$_INFX_CONFIG_PATH" ]] || _infx_die "CONFIG_FILE does not exist after recipe overlay: $_INFX_CONFIG_PATH"

# Keep the Slurm job name aligned with the GitHub runner name so the
# workflow's pre/post scancel cleanup can find stale jobs.
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$_INFX_CONFIG_PATH"

# Optional cluster-wide health-check budget (e.g. slow shared FS needs more
# time to load a ~700 GB checkpoint than the recipe default allows).
if [[ -n "$INFX_HEALTH_CHECK_MAX_ATTEMPTS" ]]; then
    sed -i '/^health_check:/,/^[^ ]/{ /^health_check:/d; /^  /d; }' "$_INFX_CONFIG_PATH"
    printf '\nhealth_check:\n  max_attempts: %s\n  interval_seconds: 10\n' \
        "$INFX_HEALTH_CHECK_MAX_ATTEMPTS" >> "$_INFX_CONFIG_PATH"
fi

# Cluster-specific recipe edits (e.g. excluding a known-bad node).
if declare -F infx_hook_edit_recipe >/dev/null; then
    infx_hook_edit_recipe "$_INFX_CONFIG_PATH" || _infx_die "infx_hook_edit_recipe failed"
fi

# sbatch's default --export=ALL would propagate VIRTUAL_ENV (set by the
# activate above) into the compute-node orchestrator, whose `uv run` then
# inspects a venv whose interpreter path may not exist there. srtctl itself
# resolves through PATH.
unset VIRTUAL_ENV

SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "$_INFX_TAGS"
)
# Agentic recipes resolve model.path to compute-node-local storage the
# login node can't stat; skip srtctl's in-process filesystem preflight
# there (the server still fails loudly at runtime if the path is missing).
if [[ "$IS_AGENTIC" == "1" || "$INFX_NO_PREFLIGHT" == "1" ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi
if [[ -n "${SRTCTL_SETUP_SCRIPT:-}" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script "$SRTCTL_SETUP_SCRIPT")
fi

SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

[[ -n "$JOB_ID" ]] || _infx_die "failed to extract JOB_ID from srtctl output"
echo "Extracted JOB_ID: $JOB_ID"

if declare -F infx_hook_post_submit >/dev/null; then
    infx_hook_post_submit "$JOB_ID" || _infx_die "infx_hook_post_submit failed"
fi

# ---------------------------------------------------------------------------
# 7. Wait for completion
# ---------------------------------------------------------------------------

LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

# Snapshot worker logs on any exit path — normal completion, error, or the
# SIGTERM a `gh run cancel` sends — so the Upload server logs workflow step
# always has something to collect.
_infx_snapshot_server_logs() {
    if [[ -n "${LOGS_DIR:-}" && -d "$LOGS_DIR" && -n "${GITHUB_WORKSPACE:-}" ]]; then
        cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
}
trap _infx_snapshot_server_logs EXIT

while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        return 1 2>/dev/null || exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"
# -F follows by name and polls instead of inotify, which NFS needs.
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

set -x
echo "Job $JOB_ID completed!"

# ---------------------------------------------------------------------------
# 8. Collect artifacts
# ---------------------------------------------------------------------------

if [[ ! -d "$LOGS_DIR" ]]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    [[ "${EVAL_ONLY}" == "true" ]] || return 1 2>/dev/null || exit 1
fi

if [[ "${EVAL_ONLY}" != "true" ]]; then
    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)
    if [[ -z "$RESULT_SUBDIRS" ]]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR"
    else
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"
            CONFIG_NAME=$(basename "$result_subdir")
            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)
            for result_file in $RESULT_FILES; do
                [[ -f "$result_file" ]] || continue
                # Files are results_concurrency_N_gpus_G_ctx_C_gen_D.json
                # (disagg) or results_concurrency_N_gpus_G.json (agg).
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')
                echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"
                if [[ -n "$ctx" && -n "$gen" ]]; then
                    WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                else
                    WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}.json"
                fi
                cp "$result_file" "$WORKSPACE_RESULT_FILE"
                echo "Copied result file to: $WORKSPACE_RESULT_FILE"
            done
        done
    fi
    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

if [[ "${RUN_EVAL}" == "true" || "${EVAL_ONLY}" == "true" ]]; then
    EVAL_DIR="$LOGS_DIR/eval_results"
    if [[ -d "$EVAL_DIR" ]]; then
        echo "Extracting eval results from $EVAL_DIR"
        shopt -s nullglob
        for eval_file in "$EVAL_DIR"/*; do
            [[ -f "$eval_file" ]] || continue
            eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"
            rm -f "$eval_dest"
            if cp "$eval_file" "$eval_dest"; then
                echo "Copied eval artifact: $(basename "$eval_file")"
            else
                echo "WARNING: Failed to copy eval artifact, continuing: $(basename "$eval_file")"
            fi
        done
        shopt -u nullglob
    else
        echo "WARNING: eval requested but no eval results found at $EVAL_DIR"
    fi
fi

# Snapshot before cleanup so the EXIT trap's directory guard still sees
# $LOGS_DIR when it fires after the rm below.
_infx_snapshot_server_logs

if declare -F infx_hook_cleanup >/dev/null; then
    infx_hook_cleanup || true
fi

# Clean up srt-slurm outputs to keep NFS silly-rename lock files from
# blocking the next job's checkout on this runner.
echo "Cleaning up srt-slurm outputs..."
for _infx_i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $_infx_i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
