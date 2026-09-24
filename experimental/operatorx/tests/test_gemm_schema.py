import pytest

from operatorx.ops.gemm import GemmArgs, quant, scale

FP8_A = quant("e4m3", scale("fp32", False, (1, 128)))
FP8_B = quant("e4m3", scale("fp32", True, (128, 128)))


def test_descriptors_validate_and_default_out():
    args = GemmArgs(m=1, n=128, k=128, a=FP8_A, b=FP8_B)
    assert args.out == "bf16"
    assert FP8_A == {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": False, "group": [1, 128]}}
    GemmArgs(m=1, n=1, k=1, a=quant("bf16"), b=quant("bf16"))


def test_nvfp4_and_asymmetric_descriptors():
    tensor = scale("fp32", True, (-1, -1))
    GemmArgs(m=1, n=16, k=16, a=quant("e2m1", scale("e4m3", False, (1, 16)), tensor),
             b=quant("e2m1", scale("e4m3", True, (1, 16)), tensor))
    assert quant("int4", scale("bf16", True, (1, 32)), symmetric=False)["symmetric"] is False


@pytest.mark.parametrize("bad", [
    {"dtype": "fp8"},
    {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": True, "group": [0, 128]}},
    {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": "yes", "group": [1, 128]}},
    {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": True, "group": [1]}},
    {"dtype": "e4m3", "scale2": {"dtype": "fp32", "static": True, "group": [-1, -1]}},
    {"dtype": "e4m3", "granularity": "block"},
])
def test_bad_descriptors_rejected(bad):
    with pytest.raises(ValueError):
        GemmArgs(m=1, n=1, k=1, a=bad, b=FP8_B)


def test_pre_quantized_input():
    a = quant("e4m3", scale("fp32", True, (-1, -1)), input="e4m3")
    assert a["input"] == "e4m3" and "input" not in FP8_A
    GemmArgs(m=1, n=128, k=128, a=a, b=FP8_B)
    with pytest.raises(ValueError):
        GemmArgs(m=1, n=128, k=128, a={**a, "input": "fp8"}, b=FP8_B)
