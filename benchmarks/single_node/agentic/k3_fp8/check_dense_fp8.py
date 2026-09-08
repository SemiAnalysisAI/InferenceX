"""Check TP8 loading and FP8 numerical error before the K3 serving experiment."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.k3_dense_fp8 import K3DenseFp8LinearMethod

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KimiGDNMergedColumnParallelLinear,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config


def check_outputs(layer, label, rank):
    reference_weight = layer.weight.detach().clone()
    assert isinstance(layer.quant_method, K3DenseFp8LinearMethod)
    layer.quant_method.process_weights_after_loading(layer)
    rows = []
    for tokens in (1, 7, 14, 128):
        x = torch.randn(tokens, reference_weight.shape[1], device="cuda")
        expected = F.linear(x, reference_weight).float()
        actual = layer.quant_method.apply(layer, x).float()
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert torch.isfinite(actual).all(), label
        relative_rmse = (
            (
                (actual - expected).square().mean()
                / expected.square().mean().clamp_min(1e-12)
            )
            .sqrt()
            .item()
        )
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
        assert relative_rmse < 0.06 and cosine > 0.995, (
            label,
            rank,
            tokens,
            relative_rmse,
            cosine,
        )
        rows.append(
            {"tokens": tokens, "relative_rmse": relative_rmse, "cosine": cosine}
        )
    return {
        "layer": label,
        "kernel": type(layer.quant_method.fp8_linear).__name__,
        "checks": rows,
    }


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 8
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    config = VllmConfig(
        model_config=ModelConfig(
            model=os.environ["MODEL_PATH"],
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=16384,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
    )
    with set_current_vllm_config(config), torch.device(f"cuda:{rank}"), torch.no_grad():
        init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=world)
        torch.manual_seed(42 + rank)
        quant = Mxfp4Config()
        base = "language_model.model.layers.0"
        results = []
        for suffix in (
            "self_attn.conv1d",
            "block_sparse_moe.routed_expert_up_proj",
            "block_sparse_moe.gate",
            "self_attention_res_proj",
        ):
            layer = ReplicatedLinear(
                128, 128, bias=False, quant_config=quant, prefix=f"{base}.{suffix}"
            )
            assert isinstance(layer.quant_method, UnquantizedLinearMethod), suffix
        for prefix in (
            "language_model.lm_head",
            "vision_tower.model.layers.0.mlp.down_proj",
        ):
            layer = ReplicatedLinear(
                128, 128, bias=False, quant_config=quant, prefix=prefix
            )
            assert isinstance(layer.quant_method, UnquantizedLinearMethod), prefix

        for label, layer in (
            (
                "shared_column",
                ColumnParallelLinear(
                    7168,
                    12288,
                    bias=False,
                    quant_config=quant,
                    prefix=f"{base}.block_sparse_moe.shared_experts.gate_up_proj",
                ),
            ),
            (
                "shared_row",
                RowParallelLinear(
                    12288,
                    7168,
                    bias=False,
                    quant_config=quant,
                    reduce_results=False,
                    prefix=f"{base}.block_sparse_moe.shared_experts.down_proj",
                ),
            ),
            (
                "mla_kv_b",
                ColumnParallelLinear(
                    512,
                    24576,
                    bias=False,
                    quant_config=quant,
                    prefix=f"{base}.self_attn.kv_b_proj",
                ),
            ),
        ):
            layer.weight.normal_(std=0.02)
            results.append(check_outputs(layer, label, rank))

        sizes = [1024, 1024, 1024, 1024, 128, 24, 104]
        layer = _KimiGDNMergedColumnParallelLinear(
            512,
            sizes,
            replicated_shard_id=4,
            tp_size=world,
            bias=False,
            quant_config=quant,
            prefix=f"{base}.self_attn.in_proj_qkvgfab",
        )
        assert not layer.weight.is_meta
        layer.weight.zero_()
        expected_parts = []
        torch.manual_seed(123)
        for shard_id, width in enumerate(sizes[:-1]):
            full_weight = torch.randn(width, 512, device="cuda") * 0.02
            layer.weight.weight_loader(layer.weight, full_weight, shard_id)
            expected_parts.append(
                full_weight if shard_id == 4 else full_weight.chunk(world, dim=0)[rank]
            )
        expected_parts.append(torch.zeros(sizes[-1] // world, 512, device="cuda"))
        torch.testing.assert_close(
            layer.weight, torch.cat(expected_parts), rtol=0, atol=0
        )
        results.append(check_outputs(layer, "kda_replicated_and_padded", rank))
        torch.distributed.barrier()
        path = Path(os.environ["RESULT_DIR"]) / f"dense-fp8-check-rank{rank}.json"
        path.write_text(json.dumps(results, indent=2) + "\n")
        if rank == 0:
            print("K3_DENSE_FP8_TP8_CHECK_OK", flush=True)
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
