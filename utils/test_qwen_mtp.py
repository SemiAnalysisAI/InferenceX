import pytest

from infx.bench_serving.qwen_mtp import build_configs


def prepare(*, mode="thinking_on", tokens=3, eval_only=False, run_eval=False, golden=None, draft=None):
    return build_configs(
        {"quantization_config": {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}},
        draft if draft is not None else {"text_config": {"dtype": "bfloat16"}},
        {"mtp.fc.weight": "shard", "mtp.layers.0.mlp.down_proj.weight": "shard"},
        golden if golden is not None else {
            "qwen3.8-27b-fp8": {"thinking_on": {3: 2.5}, "thinking_off": {3: 3.25}},
        },
        draft_model="reference/head", draft_revision="reference-revision", tokens=tokens,
        thinking_mode=mode, eval_only=eval_only, run_eval=run_eval,
    )


@pytest.mark.parametrize("mode,expected", [("thinking_on", 2.5), ("thinking_off", 3.25)])
def test_throughput_selects_measured_mode_and_preserves_native_head(mode, expected):
    spec, overrides = prepare(mode=mode)
    assert spec == {
        "method": "mtp", "model": "reference/head", "revision": "reference-revision",
        "num_speculative_tokens": 3, "kv_cache_dtype": "auto",
        "rejection_sample_method": "synthetic", "synthetic_acceptance_length": expected,
    }
    assert overrides["quantization_config"]["modules_to_not_convert"] == [
        "lm_head", "mtp.fc", "mtp.layers.0.mlp.down_proj",
    ]


@pytest.mark.parametrize("eval_only,run_eval", [(True, False), (False, True)])
def test_accuracy_paths_keep_real_verification_without_a_golden_curve(eval_only, run_eval):
    spec, _ = prepare(eval_only=eval_only, run_eval=run_eval, golden={})
    assert spec["rejection_sample_method"] == "standard"
    assert "synthetic_acceptance_length" not in spec


@pytest.mark.parametrize("tokens", [0, 5])
def test_unmeasured_draft_lengths_rejected(tokens):
    with pytest.raises(ValueError, match="1-4 draft tokens"):
        prepare(tokens=tokens)


@pytest.mark.parametrize("golden", [
    {}, {"qwen3.8-27b-fp8": {"thinking_on": {3: 4.1}}},
    {"qwen3.8-27b-fp8": {"thinking_on": {3: float("nan")}}},
])
def test_missing_or_invalid_acceptance_rejected(golden):
    with pytest.raises(ValueError, match="golden AL"):
        prepare(golden=golden)


def test_converted_draft_precision_rejected():
    with pytest.raises(ValueError, match="original BF16"):
        prepare(draft={"text_config": {"dtype": "float16"}})
