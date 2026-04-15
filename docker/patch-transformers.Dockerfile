ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Upgrade transformers to get glm_moe_dsa and qwen3_5_moe model type support.
# The SGLang v0.5.10 base image pins an older transformers that doesn't have these.
RUN pip install --no-cache-dir -U \
    "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2" \
    && python3 -c "import transformers; print(f'transformers {transformers.__version__}')" \
    && python3 -c "from transformers import AutoConfig; AutoConfig.for_model('glm_moe_dsa')" \
    && echo "glm_moe_dsa model type verified"
