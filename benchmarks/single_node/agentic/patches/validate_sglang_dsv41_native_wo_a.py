"""GPU regression check for the patched native DSpark FP8 WO_A path.

Run inside the pinned, patched SGLang container. Exercises native checkpoint
loading and contiguous/strided projections with eager and CUDA graph execution.
"""

import torch
from types import SimpleNamespace
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dspark_projection import apply_grouped_wo_a

from sglang.srt.runtime_context import get_parallel

parallel_override = get_parallel().override(
    tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
)
parallel_override.__enter__()
print("GPU:", torch.cuda.get_device_name(), flush=True)
torch.manual_seed(42)
for groups, rank, width in [(1, 1024, 4096), (2, 64, 128)]:
    quant = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    with torch.device("cuda"):
        linear = ColumnParallelLinear(
            width,
            groups * rank,
            bias=False,
            quant_config=quant,
            tp_rank=0,
            tp_size=1,
            params_dtype=torch.bfloat16,
        )
        weight = torch.randint(-2, 3, (groups * rank, width)).to(torch.float8_e4m3fn)
        scale = (
            2.0 ** torch.randint(-1, 2, (groups * rank // 32, width // 32)).float()
        ).to(torch.float8_e8m0fnu)
    model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_type="deepseek_v41", n_routed_experts=0)
    model.num_fused_shared_experts = 0
    model.confidence_head = None
    stage = torch.nn.Module()
    stage.self_attn = torch.nn.Module()
    stage.self_attn.wo_a = linear
    model.stages = torch.nn.ModuleList([stage])
    model.load_weights(
        [("mtp.0.attn.wo_a.weight", weight), ("mtp.0.attn.wo_a.scale", scale)]
    )
    assert linear.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        linear.weight.view(torch.uint8), weight.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        linear.weight_scale_inv.float(), scale.float(), rtol=0, atol=0
    )
    linear.quant_method.process_weights_after_loading(linear)
    assert linear.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        linear.weight.view(torch.uint8), weight.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        linear.weight_scale_inv.float(), scale.float(), rtol=0, atol=0
    )
    restored = weight.float() * scale.float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    for layout in ["contiguous", "feature_stride", "token_stride"]:
        for tokens in [1, 8, 64, 0]:
            if layout == "feature_stride":
                x = torch.randint(-2, 3, (tokens, groups, width * 2), device="cuda").to(
                    torch.bfloat16
                )[..., ::2]
            elif layout == "token_stride":
                x = torch.randint(-2, 3, (tokens * 2, groups, width), device="cuda").to(
                    torch.bfloat16
                )[::2]
            else:
                x = torch.randint(-2, 3, (tokens, groups, width), device="cuda").to(
                    torch.bfloat16
                )
            if layout == "feature_stride" and tokens:
                assert not x.is_contiguous(), x.stride()
            if layout == "token_stride" and tokens > 1:
                assert not x.is_contiguous(), x.stride()
            out = apply_grouped_wo_a(linear, x, rank)
            expected = torch.einsum(
                "tgd,grd->tgr", x.float(), restored.view(groups, rank, width)
            ).to(torch.bfloat16)
            torch.testing.assert_close(out, expected, rtol=0.02, atol=1.0)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    apply_grouped_wo_a(linear, x, rank)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = apply_grouped_wo_a(linear, x, rank)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0.02, atol=1.0)
            print(
                f"PASS groups={groups} T={tokens} layout={layout} stride={x.stride()}: native FP8 numerical and CUDA graph replay",
                flush=True,
            )
print("ALL NATIVE WO_A CHECKS PASSED", flush=True)
