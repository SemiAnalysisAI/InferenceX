#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing disagg dependencies at container start.
#
# Dispatched by $ENGINE (set by server.sh dispatcher):
#   vllm-disagg   -> recipe deps + amd-quark + UCX/RIXL path exports
#                    (base image: vllm/vllm-openai-rocm:nightly)
#   sglang-disagg -> SGLang aiter gluon patch + per-model installs
#                    (base image: lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-*)
#
# Sourced by server_vllm.sh and server_sglang.sh so PATH / LD_LIBRARY_PATH
# exports persist. Each patch is idempotent: skipped if already applied.
#
# Build steps run in subshells to avoid CWD pollution between installers.
# =============================================================================

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
UCX_HOME="${UCX_HOME:-/usr/local/ucx}"
RIXL_HOME="${RIXL_HOME:-/usr/local/rixl}"

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

git_clone_retry() {
    local url="$1" dest="$2" max_tries=3 try=1
    while (( try <= max_tries )); do
        if git clone --quiet "$url" "$dest" 2>/dev/null; then return 0; fi
        echo "[SETUP] git clone attempt $try/$max_tries failed for $url, retrying in 10s..."
        rm -rf "$dest"
        sleep 10
        (( try++ ))
    done
    echo "[SETUP] git clone failed after $max_tries attempts: $url"
    return 1
}

# ---------------------------------------------------------------------------
# 5. Container RDMA/net tools
#    - ibv_devinfo comes from ibverbs-utils
#    - iproute2 provides the `ip` command
#    Used for in-container NIC/RDMA validation and routing checks.
# ---------------------------------------------------------------------------
install_recipe_deps() {
    if command -v ibv_devinfo >/dev/null 2>&1 && command -v ip >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi

    echo "[SETUP] Installing ibv_devinfo + iproute2 in container..."
    # Bound the fetch: the ROCm/amdgpu/pensando apt sources can stall for tens of
    # minutes, and an unbounded apt-get here takes the whole container down with
    # it (no output, no error, srun eventually reaps it). Fail fast instead.
    local apt_opts=(
        -o Acquire::Retries=2
        -o Acquire::http::Timeout=20
        -o Acquire::https::Timeout=20
    )
    # `|| true` is load-bearing: this file is sourced into a `set -e` shell, so a
    # failing apt chain used to abort the whole server script right here -- the
    # container vanished with no error line, the launcher logged nothing past
    # "[SETUP] Installing ...", and the run looked like a mystery hang. Let the
    # command -v check below be the thing that decides.
    { apt-get update -q -y "${apt_opts[@]}" \
        && apt-get install -q -y "${apt_opts[@]}" ibverbs-utils iproute2 \
        && rm -rf /var/lib/apt/lists/* ; } || \
        echo "[SETUP] WARN: apt failed for ibverbs-utils/iproute2 (rc=$?)"

    if ! command -v ibv_devinfo >/dev/null 2>&1 || ! command -v ip >/dev/null 2>&1; then
        # These are diagnostics-only (in-container NIC/routing checks), so a
        # local debug run on a node with a flaky apt mirror can proceed without
        # them. CI leaves ALLOW_MISSING_RDMA_TOOLS unset and still hard-fails.
        if [[ "${ALLOW_MISSING_RDMA_TOOLS:-0}" == "1" ]]; then
            echo "[SETUP] WARN: ibv_devinfo/iproute2 unavailable; continuing (ALLOW_MISSING_RDMA_TOOLS=1)"
            return 0
        fi
        echo "[SETUP] ERROR: Failed to install ibv_devinfo/iproute2"; exit 1
    fi
    _SETUP_INSTALLED+=("ibverbs-utils+iproute2")
}

# ---------------------------------------------------------------------------
# 6b. amd-quark (MXFP4 quantization support for Kimi-K2.5-MXFP4 and similar)
#     Required due to ROCm vLLM missing the quark dependency:
#     https://github.com/vllm-project/vllm/issues/35633
# ---------------------------------------------------------------------------
install_amd_quark() {
    if python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] amd-quark already present"
        return 0
    fi

    echo "[SETUP] Installing amd-quark for MXFP4 quantization support..."
    pip install --quiet amd-quark

    if ! python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] WARN: amd-quark install failed (non-fatal for non-MXFP4 models)"
        return 0
    fi
    _SETUP_INSTALLED+=("amd-quark")
}

# ---------------------------------------------------------------------------
# SGLang: Install latest transformers for GLM model type support.
#
# GLM-5 (zai-org/GLM-5-FP8) requires a transformers build that includes
# the glm_moe_dsa model type. The mori images do not ship it. Gated on any
# GLM model name (not just GLM-5-FP8) so other GLM variants pick up the same
# fix; only installs when a GLM model is active (avoid overhead otherwise).
# ---------------------------------------------------------------------------
install_transformers_glm5() {
    if [[ "$MODEL_NAME" != *GLM* ]]; then
        return 0
    fi

    if python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('zai-org/GLM-5-FP8', trust_remote_code=True)" 2>/dev/null; then
        echo "[SETUP] transformers already supports GLM-5 model type"
        return 0
    fi

    echo "[SETUP] Installing transformers with GLM-5 (glm_moe_dsa) support..."
    pip install --quiet -U --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"
    _SETUP_INSTALLED+=("transformers-glm5")
}

# ---------------------------------------------------------------------------
# Kimi-K3 DSpark: online-update vLLM from the fork branch that carries the K3
# GDN/DSpark + MoRIIO KDA PD-transfer fixes, on top of the prebuilt kimi-k3
# image. Opt-in: only runs when VLLM_K3_FORK_REF is set (recipe additional-
# setting), so other vllm-disagg models keep the image's vLLM untouched.
#   VLLM_K3_FORK_REPO (default https://github.com/YukioZzz/vllm)
#   VLLM_K3_FORK_REF  (e.g. yichaozhu/moriio-k3-dspark)
#
# METHOD: python-source OVERLAY onto the image's installed vLLM, NOT `pip install`.
# The fork's changes (rebased MoRIIO connector + K3 model + DSpark) are pure
# python on top of the same base vLLM the image ships; a `pip install -e` triggers
# the ROCm precompiled-wheel fetch (VLLM_USE_PRECOMPILED) which has no x86_64 wheel
# on pypi.amd.com, and a from-source build is not viable at job start. Overlaying
# only *.py (keeping the image's compiled .so kernels) matches the VALIDATED local
# bring-up (the "/work/k3_moriio_rebased" overlay: IMPORT_OK + MULTI_OK).
# Idempotent via an install marker keyed on the ref.
# ---------------------------------------------------------------------------
install_kimi_k3_vllm_fork() {
    local ref="${VLLM_K3_FORK_REF:-}"
    [[ -z "$ref" ]] && return 0

    local repo="${VLLM_K3_FORK_REPO:-https://github.com/YukioZzz/vllm}"
    local src="${VLLM_K3_FORK_SRC:-/opt/vllm-k3-fork}"
    local marker="${src}/.inferencex_installed_ref"

    # The marker records the resolved commit, not just repo@ref. Keying it on the
    # branch name alone meant that once a node had overlaid the branch, later
    # commits pushed to that same branch were silently skipped -- the container
    # kept running whatever it happened to clone first. Cloud runs start from a
    # pristine image so they never saw it; any reused/committed image would.
    local remote_sha
    remote_sha=$(git ls-remote "$repo" "$ref" 2>/dev/null | awk 'NR==1{print $1}')
    if [[ -f "$marker" ]] && [[ -n "$remote_sha" ]] \
       && [[ "$(cat "$marker" 2>/dev/null)" == "${repo}@${remote_sha}" ]]; then
        echo "[SETUP] Kimi-K3 vLLM fork already overlaid (${repo}@${ref} = ${remote_sha})"
        return 0
    fi

    # Private-fork auth: if VLLM_K3_FORK_TOKEN is set, inject it into an https URL.
    local clone_url="$repo"
    if [[ -n "${VLLM_K3_FORK_TOKEN:-}" && "$repo" == https://github.com/* ]]; then
        clone_url="https://x-access-token:${VLLM_K3_FORK_TOKEN}@github.com/${repo#https://github.com/}"
    fi

    echo "[SETUP] Overlaying Kimi-K3 vLLM fork ${repo}@${ref} onto the installed vLLM..."
    if [[ ! -d "$src/.git" ]]; then
        rm -rf "$src"
        git_clone_retry "$clone_url" "$src" || { echo "[SETUP] ERROR: clone $repo failed"; exit 1; }
    fi
    ( cd "$src" && git fetch --depth 1 origin "$ref" && git checkout -f FETCH_HEAD ) \
        || { echo "[SETUP] ERROR: fetch/checkout $ref failed"; exit 1; }

    local vllm_pkg
    vllm_pkg=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))") \
        || { echo "[SETUP] ERROR: base vLLM not importable in image"; exit 1; }
    if [[ ! -d "$src/vllm" ]]; then
        echo "[SETUP] ERROR: fork has no vllm/ package at $src/vllm"; exit 1
    fi

    # Overlay *.py only (preserve image's compiled kernels).
    python3 - "$src/vllm" "$vllm_pkg" <<'PY'
import os, sys, shutil
srcroot, dst = sys.argv[1], sys.argv[2]
n = 0
for root, _dirs, files in os.walk(srcroot):
    for f in files:
        if not f.endswith(".py"):
            continue
        s = os.path.join(root, f)
        rel = os.path.relpath(s, srcroot)
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        n += 1
print(f"[SETUP] overlaid {n} .py files -> {dst}")
PY

    # Sanity: MultiConnector (MoRIIO + Mooncake) must import after the overlay.
    python3 -c "import vllm; from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector; print('[SETUP] vLLM fork overlay import OK', vllm.__file__)" \
        || { echo "[SETUP] ERROR: vLLM import failed after fork overlay"; exit 1; }

    echo "${repo}@$(cd "$src" && git rev-parse HEAD)" > "$marker"
    _SETUP_INSTALLED+=("vllm-k3-fork-overlay@${ref}")
}

# =============================================================================
# Run installers (engine-gated)
# =============================================================================

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    install_recipe_deps
    install_amd_quark
    install_kimi_k3_vllm_fork

    # =========================================================================
    # vLLM: Export UCX/RIXL paths (persists since this file is sourced)
    # =========================================================================
    export ROCM_PATH="${ROCM_PATH}"
    export UCX_HOME="${UCX_HOME}"
    export RIXL_HOME="${RIXL_HOME}"
    export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
else
    install_transformers_glm5
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
