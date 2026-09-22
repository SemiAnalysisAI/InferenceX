#!/usr/bin/env bash

# Shared native srt-slurm execution; cluster paths and defaults come from callers.
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || return 1

run_srt_recipe_job() (
    set -eo pipefail
    check_env_vars CONFIG_FILE EVAL_ONLY IS_AGENTIC IS_MULTINODE KEEP_LOGS RUN_EVAL \
        GITHUB_WORKSPACE RESULT_FILENAME IMAGE MODEL FRAMEWORK \
        SRT_SLURM_REPOSITORY SRT_SLURM_COMMIT SRT_SLURM_SHARED_BASE \
        SRT_SLURM_CLUSTER_CONFIG SRT_SLURM_COMPUTE_ARCH AIPERF_MMAP_CACHE_HOST_PATH \
        GITHUB_RUN_ID GITHUB_RUN_ATTEMPT RUNNER_NAME HOME

    SHARED_BASE="$SRT_SLURM_SHARED_BASE"
    SHARED_AIPERF_CACHE="$AIPERF_MMAP_CACHE_HOST_PATH"
    SHARED_RESULTS="${SHARED_BASE}/results"
    RUN_KEY="${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${RUNNER_NAME}"
    WORK_DIR="${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}"
    SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
    CONFIG_PATH="${CONFIG_FILE%%:*}"
    LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
    CLUSTER_PROFILE="$SRT_SLURM_CLUSTER_CONFIG"
    ADAPTER=infx.workflows.srt_slurm

    # Only shared, user-owned directories are prepared here. Pyxis manages
    # container imports/lifetime; engines use the cache mounted by the profile.
    mkdir -p "$WORK_DIR" "$SHARED_RESULTS" "$SHARED_AIPERF_CACHE"
    setup_srt_slurm "$SRT_REPO_DIR" "$FRAMEWORK" 0

    export PATH="$HOME/.local/bin:$PATH"
    cd "$SRT_REPO_DIR"
    uv venv --python 3.12
    uv pip install -e .
    source .venv/bin/activate

    PYTHONPATH="$GITHUB_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" python -m "$ADAPTER" prepare \
        --recipe "$LOCAL_RECIPE" --profile "$CLUSTER_PROFILE" \
        --work-dir "$WORK_DIR" --workspace "$GITHUB_WORKSPACE" \
        --results-root "$SHARED_RESULTS" --aiperf-cache "$SHARED_AIPERF_CACHE" \
        --image-cache "${SHARED_BASE}/containers"
    export SRTSLURM_CONFIG="${WORK_DIR}/srtslurm.yaml"
    # Use upstream's normal setup with the prepared cluster configuration;
    # providing it first also keeps setup non-interactive in Actions.
    cp "$SRTSLURM_CONFIG" "$SRT_REPO_DIR/srtslurm.yaml"
    make setup ARCH="$SRT_SLURM_COMPUTE_ARCH"
    PREPARED_RECIPE="${WORK_DIR}/recipe.yaml"

    # Do not leak the CI host's bytecode-cache location into the containers.
    # Recipes may still explicitly set their own container cache environment.
    unset PYTHONPYCACHEPREFIX
    apply_srt_recipe "$PREPARED_RECIPE" "$FRAMEWORK" -f "$PREPARED_RECIPE" \
        --tags "inferencex,github-actions,${RUN_KEY}" --json \
        > "${WORK_DIR}/submission.json"
    JOB_ID=$(jq -er '.slurm_job_id' "${WORK_DIR}/submission.json")
    OUTPUT_DIR=$(jq -er '.output_dir' "${WORK_DIR}/submission.json")
    echo "SRT_SLURM_JOB_ID=$JOB_ID"

    # srt-slurm handles queue/accounting transitions and propagates failures.
    # Always collect available diagnostics, including on an unsuccessful job.
    job_rc=0
    srtctl wait "$JOB_ID" --log-file "${OUTPUT_DIR}/logs/sweep_${JOB_ID}.log" || job_rc=$?
    collect_rc=0
    PYTHONPATH="$GITHUB_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" python -m "$ADAPTER" collect --submission "${WORK_DIR}/submission.json" \
        --workspace "$GITHUB_WORKSPACE" --results-root "$SHARED_RESULTS" || collect_rc=$?
    [[ "$job_rc" -eq 0 ]] || exit "$job_rc"
    [[ "$collect_rc" -eq 0 ]] || exit "$collect_rc"
    printf '%s\n' "$SRT_SLURM_COMMIT" > "$GITHUB_WORKSPACE/srt-slurm-producer-sha.txt"
    echo "srt-slurm validation completed successfully"
    exit 0
)
