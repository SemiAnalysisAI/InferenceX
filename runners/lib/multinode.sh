#!/usr/bin/env bash
# Shared helpers for the multinode branches of runners/launch_*.sh.
#
# A runner's multinode branch is a thin "cluster profile": it declares
# cluster facts (Slurm settings, squash cache locations, model registry
# key), resolves the model path from configs/runners.yaml, imports the
# container squash files however this cluster requires, then sources
# benchmarks/multi_node/srt_slurm/run.sh — the cluster-agnostic
# orchestrator. Anything cluster-specific belongs in the profile; nothing
# cluster-specific belongs in benchmarks/.

_INFX_RUNNERS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve MODEL_PATH / SRT_SLURM_MODEL_PREFIX / SERVED_MODEL_NAME from the
# model-paths registry in configs/runners.yaml.
#
#   infx_resolve_model_paths cluster:gb300-nv [--fallback-to-model "$MODEL"]
#
# Honors a pre-set MODEL_PATH env var when it points at an existing
# directory (dispatch-time staging overrides).
infx_resolve_model_paths() {
    local cluster_key="$1"
    shift
    local resolved
    if ! python3 -c 'import yaml' 2>/dev/null; then
        python3 -m pip install --user --quiet pyyaml \
            || { echo "ERROR: PyYAML unavailable and pip bootstrap failed" >&2; return 1; }
    fi
    resolved=$(python3 "$_INFX_RUNNERS_LIB_DIR/resolve_model_path.py" \
        --runners-yaml "${INFX_RUNNERS_YAML:-$GITHUB_WORKSPACE/configs/runners.yaml}" \
        --cluster "$cluster_key" \
        --model-prefix "$MODEL_PREFIX" \
        --precision "$PRECISION" \
        --framework "$FRAMEWORK" \
        --env-model-path "${MODEL_PATH:-}" \
        "$@") || return 1
    eval "$resolved"
}

# Canonical squash-file path for an image inside a cache dir.
#   infx_squash_path /data/squash vllm/vllm-openai:v0.21.0 [separator]
infx_squash_path() {
    local dir="$1" image="$2" sep="${3:-_}"
    echo "${dir%/}/$(echo "$image" | sed "s/[\\/:@#]/${sep}/g").sqsh"
}

# Import an image to a squash file on the local/login node. Concurrent
# matrix jobs import to the same shared-FS path, so imports are serialized
# under flock and the file is replaced atomically — readers never observe a
# partially written squash.
#   infx_import_squash <squash-file> <image>
infx_import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    mkdir -p "$(dirname "$squash")"
    (
        exec 9>"$lock"
        flock -w 1800 9 || { echo "Failed to acquire lock for $squash" >&2; exit 1; }
        if unsquashfs -l "$squash" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash"
        else
            rm -f "$squash" "$squash".tmp.*
            enroot import -o "${squash}.tmp.$$" "docker://$image"
            mv -f "${squash}.tmp.$$" "$squash"
        fi
    ) || return 1
}

# Same as infx_import_squash, but the import runs on a compute node via
# srun — required when the login node's architecture differs from the
# compute nodes' (e.g. x86_64 login in front of an aarch64 fleet).
#   infx_import_squash_srun <squash-file> <image> [srun args...]
infx_import_squash_srun() {
    local squash="$1" image="$2"
    shift 2
    local lock="${squash}.lock"
    srun "$@" bash -c "
        exec 9>\"$lock\"
        flock -w 1800 9 || { echo 'Failed to acquire lock for $squash' >&2; exit 1; }
        if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import: $squash'
        else
            rm -f \"$squash\"
            enroot import -o \"$squash\" docker://$image
        fi
    "
}

# Print the first writable directory among the candidates (creating it if
# needed), or return 1. Used on clusters where the compute-visible shared
# filesystem mounts vary between runner hosts.
#   SHARED_BASE=$(infx_probe_writable_dir /a /b /c) || exit 1
infx_probe_writable_dir() {
    local cand
    for cand in "$@"; do
        if mkdir -p "$cand" 2>/dev/null && touch "$cand/.write-probe.$$" 2>/dev/null; then
            rm -f "$cand/.write-probe.$$" 2>/dev/null
            echo "$cand"
            return 0
        fi
        echo "  not writable: $cand" >&2
    done
    echo "ERROR: no writable directory among: $*" >&2
    return 1
}
