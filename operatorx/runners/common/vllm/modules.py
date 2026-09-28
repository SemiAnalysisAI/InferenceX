"""A checkpoint's own vLLM MoE module, built as its decoder layer builds it, with dummy
weights and post-load processing (opt-in: OPERATORX_MODEL_MODULES=1)."""
from __future__ import annotations

import importlib

import torch


def _layer(pkg: str):
    from vllm.platforms import current_platform
    return importlib.import_module(f"vllm.models.{pkg}.{'amd' if current_platform.is_rocm() else 'nvidia'}.model")


def _deepseek_v4(pkg):
    def build(cfg, hf, op_args):
        m = _layer(pkg)
        hash_layers = getattr(hf, "num_hash_layers", None) or 0
        i = 0 if op_args["router"]["select"]["kind"] == "hash" else hash_layers
        kw = {"num_hash_layers": hash_layers} if pkg == "deepseek_v4" else {}
        return m.DeepseekV4MoE(cfg, prefix=f"model.layers.{i}.ffn",
                               use_sequence_parallel=m._use_sequence_parallel(cfg), **kw)
    return build


def _minimax_m3(cfg, hf, op_args):
    m = _layer("minimax_m3")
    i = next(i for i in range(hf.num_hidden_layers) if m._is_moe_layer(hf, i))
    # reduce here: the op includes the reduction the model fuses into the next RMSNorm
    return m.MiniMaxM3MoE(config=hf, layer_id=i, quant_config=cfg.quant_config, reduce_results=True,
                          prefix=f"model.layers.{i}.block_sparse_moe")


def _kimi_k3(cfg, hf, op_args):
    m = _layer("kimi_k3")
    i = hf.first_k_dense_replace
    p = cfg.parallel_config
    use_sp = (p.pipeline_parallel_size == 1 and p.enable_expert_parallel and p.tensor_parallel_size > 1
              and (cfg.kernel_config.moe_backend == "deep_gemm_mega_moe" or p.data_parallel_size > 1))
    return m.KimiMoE(config=hf, vllm_config=cfg, quant_config=cfg.quant_config,
                     prefix=f"model.layers.{i}.block_sparse_moe", layer_idx=i, use_sequence_parallel=use_sp)


BUILDERS = {
    "DeepseekV4ForCausalLM": _deepseek_v4("deepseek_v4"),
    "DeepseekV41ForCausalLM": _deepseek_v4("deepseek_v41"),
    "MiniMaxM3SparseForConditionalGeneration": _minimax_m3,
    "KimiK3ForConditionalGeneration": _kimi_k3,
}


def build(op_args: dict) -> torch.nn.Module | None:
    """None when the engine's model is a stand-in or has no entry here."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    cfg = get_current_vllm_config()
    arch = next((a for a in (cfg.model_config.architectures or ()) if a in BUILDERS), None)
    if arch is None:
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    prev = torch.get_default_dtype()
    torch.set_default_dtype(cfg.model_config.dtype)
    try:
        with device:
            module = BUILDERS[arch](cfg, cfg.model_config.hf_text_config, op_args)
    finally:
        torch.set_default_dtype(prev)
    get_model_loader(cfg.load_config).load_weights(module, cfg.model_config)
    process_weights_after_loading(module, cfg.model_config, device)
    return module
