#!/usr/bin/env bash
set -eo pipefail
root=${SGLANG_PYTHON_ROOT:-/sgl-workspace/sglang/python}
python3 - "$root/sglang" <<'PY'
import sys
from pathlib import Path

package = Path(sys.argv[1])
edits = [
    (
        "kernels/ops/muse_spark_v12/gemm_ag.py",
        """        and comm.config.num_push_blocks >= 2
        and weight.is_contiguous()
    )
""",
        """        and comm.config.num_push_blocks >= 2
        and (weight.is_contiguous() or getattr(weight, "muse_fp8", None) is not None)
    )
""",
    ),
    (
        "kernels/ops/muse_spark_v12/gemm_ag.py",
        """    n_local = weight.shape[0] // comm.world_size
    w_slice = weight[comm.rank * n_local : (comm.rank + 1) * n_local]
    return bf16_gemm_dispatch(x.contiguous(), w_slice, None)
""",
        """    n_local = weight.shape[0] // comm.world_size
    fp8 = getattr(weight, "muse_fp8", None)
    if fp8 is not None:
        rows = slice(comm.rank * n_local, (comm.rank + 1) * n_local)
        return latent_fp8_gemm(x, fp8[0][rows], fp8[1][rows])
    w_slice = weight[comm.rank * n_local : (comm.rank + 1) * n_local]
    return bf16_gemm_dispatch(x.contiguous(), w_slice, None)


@torch.no_grad()
def quantize_latent_weight(weight: torch.nn.Parameter) -> None:
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    if getattr(weight, "muse_fp8", None) is not None:
        return
    qweight, scale = per_token_group_quant_fp8(weight.data.contiguous(), weight.shape[-1])
    weight.muse_fp8 = (qweight, scale)
    weight.data = weight.data.new_zeros(()).expand(weight.shape)


def latent_fp8_gemm(
    x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    if x.shape[0] == 0:
        return x.new_empty((0, qweight.shape[0]))
    return apply_fp8_linear(
        input=x.contiguous(),
        weight=qweight.t(),
        weight_scale=scale.view(1, -1),
        input_scale=None,
        use_per_token_if_dynamic=True,
    )


def latent_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    fp8 = getattr(weight, "muse_fp8", None)
    if fp8 is not None:
        return latent_fp8_gemm(x, *fp8)
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    return bf16_gemm_dispatch(x, weight, None)
""",
    ),
    (
        "srt/models/muse_spark.py",
        """    gemm_local,
    reserve_projection_workspace,
)
""",
        """    gemm_local,
    latent_gemm,
    quantize_latent_weight,
    reserve_projection_workspace,
)
""",
    ),
    (
        "srt/models/muse_spark.py",
        """                return gemm_ag(comm, output, self.post_expert_proj.weight)
            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)
""",
        """                return gemm_ag(comm, output, self.post_expert_proj.weight)
            output = latent_gemm(output, self.post_expert_proj.weight)
""",
    ),
    (
        "srt/models/muse_spark.py",
        """            else:
                output = self.post_expert_proj(output)
        local_output = get_local_dp_buffer(tp_group)
""",
        """            else:
                output = latent_gemm(output, self.post_expert_proj.weight)
        local_output = get_local_dp_buffer(tp_group)
""",
    ),
    (
        "srt/models/muse_spark.py",
        """                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )
""",
        """                expert_input = latent_gemm(hidden_states, self.pre_expert_proj.weight)
""",
    ),
    (
        "srt/models/muse_spark.py",
        """                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)
""",
        """                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)
        for module in self.modules():
            if isinstance(module, MuseSpark1xMoE) and isinstance(
                module.pre_expert_proj, nn.Linear
            ):
                quantize_latent_weight(module.pre_expert_proj.weight)
                quantize_latent_weight(module.post_expert_proj.weight)
        torch.cuda.empty_cache()
""",
    ),
]
for name, old, new in edits:
    path = package / name
    text = path.read_text()
    if new in text:
        print(f"already patched {path}")
    elif text.count(old) == 1:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
    else:
        sys.exit(f"{path}: anchor found {text.count(old)} times, expected 1:\n{old}")
PY
