import copy

import pytest

from operatorx.ops.gemm import quant, scale
from operatorx.ops.moe import MoeLayerArgs

FP8_X = quant("e4m3", scale("fp32", False, (1, 128)))
FP8_W = quant("e4m3", scale("fp32", True, (128, 128)))
MXFP4_W = quant("e2m1", scale("ue8m0", True, (1, 32)))
BF16 = quant("bf16")

# DeepSeek-R1-style: grouped top-k with score-correction bias, one shared expert, FP8 blocks.
DSR1 = dict(
    tokens=64, hidden=7168,
    experts={"num": 256, "top_k": 8, "inter": 2048,
             "quant": {"x": FP8_X, "w13": FP8_W, "w2": FP8_W, "a2": FP8_X}},
    router={"gate": {"dtype": "fp32"}, "scoring": "sigmoid",
            "select": {"kind": "grouped_topk", "groups": 8, "topk_groups": 4},
            "bias": True, "renormalize": True, "scale": 2.5},
    activation={"kind": "silu"},
    shared={"count": 1, "inter": 2048, "quant": {"x": FP8_X, "w13": FP8_W, "w2": FP8_W}},
)


def test_archetypes_validate():
    MoeLayerArgs(**DSR1)
    MoeLayerArgs(**{**DSR1,  # DSv4-style hash layer, clamped silu, MXFP4 routed experts
                    "experts": {"num": 384, "top_k": 6, "inter": 3072,
                                "quant": {"x": BF16, "w13": MXFP4_W, "w2": MXFP4_W, "a2": BF16}},
                    "router": {"gate": {"dtype": "fp32"}, "scoring": "sqrtsoftplus",
                               "select": {"kind": "hash", "vocab": 129280}, "scale": 2.5},
                    "activation": {"kind": "silu", "limit": 10.0}})
    MoeLayerArgs(**{**DSR1,  # gpt-oss-style: expert biases, swigluoai, no shared expert
                    "experts": {"num": 128, "top_k": 4, "inter": 2880, "bias": True,
                                "quant": {"x": BF16, "w13": MXFP4_W, "w2": MXFP4_W, "a2": BF16}},
                    "router": {"gate": {"dtype": "bf16"}, "scoring": "softmax", "select": {"kind": "topk"}},
                    "activation": {"kind": "swigluoai", "alpha": 1.702, "limit": 7.0},
                    "shared": None})
    MoeLayerArgs(**{**DSR1,  # Qwen-style gated shared expert, forced skewed load
                    "shared": {**DSR1["shared"], "gate": "sigmoid"},
                    "routing": {"distribution": {"kind": "zipf", "s": 1.1}, "seed": 3}})
    MoeLayerArgs(**{**DSR1,  # Kimi-K3-style latent experts
                    "experts": {**DSR1["experts"], "latent": 3584}, "activation": {"kind": "situ"}})


def _with(path, value):
    args = copy.deepcopy(DSR1)
    d = args
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return args


@pytest.mark.parametrize("path,value", [
    (("tokens",), 0),
    (("experts", "top_k"), 300),
    (("experts", "quant", "a2"), {"dtype": "fp8"}),
    (("experts", "quant"), {"x": FP8_X, "w13": FP8_W, "w2": FP8_W}),
    (("experts", "parallel"), {"ep": 8}),
    (("router", "scoring"), "tanh"),
    (("router", "select"), {"kind": "grouped_topk", "groups": 7, "topk_groups": 4}),
    (("router", "select"), {"kind": "topk", "k": 8}),
    (("activation", "kind"), "relu"),
    (("shared", "gate"), "softmax"),
    (("routing",), {"distribution": "uniform", "seed": 0}),
])
def test_bad_args_rejected(path, value):
    with pytest.raises(ValueError):
        MoeLayerArgs(**_with(path, value))


def test_parallelism_not_in_schema():
    with pytest.raises(TypeError):
        MoeLayerArgs(**DSR1, parallel={"tp": 1, "ep": 1})
