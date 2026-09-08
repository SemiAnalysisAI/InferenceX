"""K3 experiment: retain MXFP4 experts and quantize eligible dense weights."""

import re

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.platforms import current_platform

logger = init_logger(__name__)
_LAYER = re.compile(r"(?:language_model\.)?model\.layers\.\d+\.(.+)")
_ATTENTION = {
    "in_proj_qkvgfab",
    "f_b_proj",
    "o_proj",
    "fused_qkv_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "q_proj",
    "kv_b_proj",
    "g_proj",
}
_MLP = {"gate_up_proj", "down_proj"}


def eligible_prefix(prefix: str) -> bool:
    match = _LAYER.fullmatch(prefix)
    if match is None:
        return False
    suffix = match.group(1)
    if suffix.startswith("self_attn."):
        return suffix.removeprefix("self_attn.") in _ATTENTION
    if suffix.startswith("mlp."):
        return suffix.removeprefix("mlp.") in _MLP
    if suffix.startswith("block_sparse_moe.shared_experts."):
        return suffix.removeprefix("block_sparse_moe.shared_experts.") in _MLP
    return False


class K3DenseFp8LinearMethod(Fp8PtpcOnlineLinearMethod):
    """Load BF16 normally, then reuse vLLM's PTPC FP8 processing and kernels.

    KDA's fused projection initializes padding before checkpoint loading. Keep
    real storage and the original sharded loader so that padding is preserved.
    """

    uses_meta_device = False

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        UnquantizedLinearMethod.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
            raise ValueError("K3 dense PTPC FP8 requires FP8 activations")

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        super().process_weights_after_loading(layer)
        assert layer.weight.dtype == current_platform.fp8_dtype()
        logger.info(
            "K3_DENSE_FP8_LOADED prefix=%s dtype=%s kernel=%s",
            layer.prefix,
            layer.weight.dtype,
            type(self.fp8_linear).__name__,
        )
