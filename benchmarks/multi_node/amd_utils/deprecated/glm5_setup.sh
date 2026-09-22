#!/bin/bash

# Historical GLM-5 setup, retained for explicit replay only.
# Source after the shared setup helper when restoring the archived GLM-5 registry.
# GLM-5 needs a transformers build with the glm_moe_dsa model type, which the mori
# images do not ship. Gated on any GLM model name.
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

install_transformers_glm5

# Historical GLM-5 environment overrides.
if [[ "$MODEL_NAME" == "GLM-5-FP8" ]]; then
    export SGLANG_ROCM_FUSED_DECODE_MLA=0
    export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    export SAFETENSORS_FAST_GPU=1
fi
