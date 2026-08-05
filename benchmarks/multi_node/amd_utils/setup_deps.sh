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
    apt-get update -q -y && apt-get install -q -y \
        ibverbs-utils iproute2 \
        && rm -rf /var/lib/apt/lists/*

    if ! command -v ibv_devinfo >/dev/null 2>&1 || ! command -v ip >/dev/null 2>&1; then
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
# Idempotent via an install marker keyed on the ref.
# ---------------------------------------------------------------------------
install_kimi_k3_vllm_fork() {
    local ref="${VLLM_K3_FORK_REF:-}"
    [[ -z "$ref" ]] && return 0

    local repo="${VLLM_K3_FORK_REPO:-https://github.com/YukioZzz/vllm}"
    local src="${VLLM_K3_FORK_SRC:-/opt/vllm-k3-fork}"
    local marker="${src}/.inferencex_installed_ref"

    if [[ -f "$marker" ]] && [[ "$(cat "$marker" 2>/dev/null)" == "${repo}@${ref}" ]]; then
        echo "[SETUP] Kimi-K3 vLLM fork already installed (${repo}@${ref})"
        return 0
    fi

    # Private-fork auth: if VLLM_K3_FORK_TOKEN is set, inject it into an https URL.
    local clone_url="$repo"
    if [[ -n "${VLLM_K3_FORK_TOKEN:-}" && "$repo" == https://github.com/* ]]; then
        clone_url="https://x-access-token:${VLLM_K3_FORK_TOKEN}@github.com/${repo#https://github.com/}"
    fi

    echo "[SETUP] Installing Kimi-K3 vLLM fork ${repo}@${ref} (VLLM_USE_PRECOMPILED=1)..."
    if [[ ! -d "$src/.git" ]]; then
        rm -rf "$src"
        git_clone_retry "$clone_url" "$src" || { echo "[SETUP] ERROR: clone $repo failed"; exit 1; }
    fi
    (
        cd "$src" || exit 1
        git fetch --depth 1 origin "$ref" && git checkout -f FETCH_HEAD
        VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .
    ) || { echo "[SETUP] ERROR: Kimi-K3 vLLM fork install failed"; exit 1; }
    echo "${repo}@${ref}" > "$marker"
    _SETUP_INSTALLED+=("vllm-k3-fork@${ref}")
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
