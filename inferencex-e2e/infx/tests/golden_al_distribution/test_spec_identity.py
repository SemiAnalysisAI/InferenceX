"""Curve identities, identity-based selection, run normalization and the audit."""

import pytest
import yaml

from infx.golden_al_distribution import GOLDEN_DIR, curve_name, load_curve
from infx.golden_al_distribution.identity import (
    Fact,
    canonical_name,
    load_identities,
    parse_identity,
)
from infx.srt_slurm.spec_identity import audit, run_collection, spec_arguments

DEFAULT = {"value": None, "source": "default"}
CHAIN = {"value": "chain", "source": "documented"}


def _role(**args):
    return {"args": args}


def _identity(prefix: str) -> dict:
    return {
        "select": {"model_prefixes": [prefix], "method": "mtp"},
        "collection": {
            "engine": "vllm", "draft": "native", "draft_sample": DEFAULT, "verify": DEFAULT,
            "adaptive": DEFAULT, "topology": CHAIN, "draft_precision": DEFAULT,
            "draft_kv_cache_dtype": DEFAULT, "kv_cache_dtype": DEFAULT,
        },
    }  # fmt: skip


def test_every_curve_file_has_an_identity_matching_its_name() -> None:
    identities = load_identities()
    assert set(identities) == {path.stem for path in GOLDEN_DIR.glob("*.yaml")} - {"identities"}
    for name, identity in identities.items():
        assert canonical_name(load_curve(name).model, identity) == name


def test_overlapping_selectors_are_rejected(tmp_path) -> None:
    (tmp_path / "identities.yaml").write_text(yaml.safe_dump({"a": _identity("m"), "b": _identity("m")}))
    with pytest.raises(ValueError, match="several golden curves"):
        curve_name("m", {"method": "mtp"}, tmp_path)


def test_verify_selector_wins_and_unknown_verifiers_keep_the_general_curve(tmp_path) -> None:
    block = _identity("m")
    block["select"]["verify"] = "block"
    identities = {"general": _identity("m"), "block": block}
    (tmp_path / "identities.yaml").write_text(yaml.safe_dump(identities))
    spec = {"method": "mtp"}
    assert curve_name("m", spec, tmp_path, "block") == "block"
    assert curve_name("m", spec, tmp_path, "standard") == "general"
    assert curve_name("m", spec, tmp_path, None) == "general"
    del identities["general"]
    (tmp_path / "identities.yaml").write_text(yaml.safe_dump(identities))
    with pytest.raises(ValueError, match="verify None; curves need verify=block"):
        curve_name("m", spec, tmp_path, None)


@pytest.mark.parametrize(
    ("field", "fact", "message"),
    [
        ("verify", {"value": "block"}, "verify must be"),
        ("verify", {"value": None, "source": "explicit"}, "null value exactly"),
    ],
)
def test_malformed_fact_is_rejected(field, fact, message) -> None:
    data = _identity("m")
    data["collection"][field] = fact
    with pytest.raises(ValueError, match=message):
        parse_identity("bad", data)


def test_vllm_unset_settings_and_synthetic_sampler() -> None:
    role = _role(**{"speculative-config": {"method": "mtp", "num_speculative_tokens": 3}})
    run = run_collection("qwen3.5", role, "vllm")
    assert run.draft_sample == Fact(None, "default")
    assert run.verify == Fact("standard", "documented")
    assert run.adaptive == Fact(False, "documented")
    synthetic = {
        "method": "dspark", "num_speculative_tokens": 6, "draft_sample_method": "probabilistic",
        "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.77,
    }  # fmt: skip
    assert run_collection("dsv4", _role(**{"speculative-config": synthetic}), "vllm").verify == (
        Fact("block", "documented")
    )


def test_same_engine_defaults_agree_but_cross_engine_defaults_are_unverified() -> None:
    spec = {"method": "mtp", "num_speculative_tokens": 3}
    assert audit("qwen3.5", _role(**{"speculative-config": spec, "kv-cache-dtype": "fp8"}), "vllm")[
        "findings"
    ] == []
    args = {
        "speculative-algorithm": "NEXTN", "speculative-num-steps": 3, "speculative-eagle-topk": 1,
        "kv-cache-dtype": "fp8",
    }  # fmt: skip
    findings = audit("qwen3.5", _role(**args), "sglang")["findings"]
    assert [f[:2] for f in findings] == [
        ["unverified", "draft_sample"], ["unverified", "draft_precision"],
        ["unverified", "draft_kv_cache_dtype"], ["engine", "engine"],
    ]  # fmt: skip


def test_verifier_mismatch_is_reported() -> None:
    spec = {"method": "dspark", "num_speculative_tokens": 4, "draft_sample_method": "probabilistic"}
    findings = audit("dsv41flash", _role(**{"speculative-config": spec}), "vllm")
    assert findings["curve"] == "deepseek-v4.1-flash__dspark__native__probabilistic-block"
    assert ["mismatch", "verify", "standard (documented)", "block (explicit)"] in findings["findings"]


def test_verifier_specific_curve_wins() -> None:
    spec = {"method": "dspark", "num_speculative_tokens": 4, "draft_sample_method": "probabilistic"}
    role = _role(**{"speculative-config": spec, "kv-cache-dtype": "fp8"})
    findings = audit("kimik3", role, "vllm")
    assert findings["curve"] == "kimi-k3__dspark__kimi-k3-dspark__probabilistic-default"
    assert [f for f in findings["findings"] if f[1] == "verify"] == []
    block = {**spec, "rejection_sample_method": "block"}
    role = _role(**{"speculative-config": block, "kv-cache-dtype": "fp8"})
    assert audit("kimik3", role, "vllm")["curve"] == (
        "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block"
    )


def test_sglang_block_verification_and_topk() -> None:
    args = {
        "speculative-algorithm": "EAGLE", "speculative-num-steps": 3,
        "speculative-eagle-topk": 4, "speculative-use-block-verification": True,
    }  # fmt: skip
    run = run_collection("glm5.2", _role(**args), "sglang")
    assert run.verify == Fact("block", "explicit")
    assert run.topology == Fact("tree-topk4", "explicit")
    del args["speculative-eagle-topk"]
    assert run_collection("glm5.2", _role(**args), "sglang").topology.source == "default"


def test_trtllm_relaxed_acceptance_is_its_own_verifier() -> None:
    config = {"decoding_type": "MTP", "max_draft_len": 3, "use_relaxed_acceptance_for_thinking": True}
    run = run_collection("dsv4", _role(speculative_config=config), "dynamo-trt")
    assert run.verify == Fact("relaxed", "explicit")


def test_atom_verifier_is_unknown_and_reported_unverified() -> None:
    args = {"method": "dspark", "num-speculative-tokens": 4, "draft-model": "Inferact/Kimi-K3-DSpark"}
    findings = audit("kimik3", _role(**args), "atom")["findings"]
    assert ["unverified", "verify", "unknown", "block (explicit)"] in findings


@pytest.mark.parametrize(
    ("topk", "topology"),
    [
        (1, Fact("chain", "explicit")),
        (4, Fact("tree-topk4", "explicit")),
        (None, Fact(None, "unknown")),
    ],
)
def test_tokenspeed_identity_keeps_unread_semantics_unknown(
    topk: int | None, topology: Fact
) -> None:
    args = {
        "speculative-algorithm": "MTP",
        "speculative-num-steps": 3,
        "speculative-draft-model-quantization": "modelopt_mixed",
        "kv-cache-dtype": "bfloat16",
        "sampling-backend": "greedy",
    }
    if topk is not None:
        args["speculative-eagle-topk"] = topk
    run = run_collection("qwen3.8next", _role(**args), "tokenspeed")
    assert run.engine == "tokenspeed"
    assert run.draft == "native"
    assert run.topology == topology
    assert run.draft_precision == Fact("modelopt_mixed", "explicit")
    assert run.kv_cache_dtype == Fact("bf16", "explicit")
    assert run.draft_kv_cache_dtype == Fact(None, "default")
    assert run.verify == run.draft_sample == run.adaptive == Fact(None, "unknown")


def test_tokenspeed_dflash_selects_the_draft_curve_and_reports_the_draft_precision() -> None:
    args = {
        "speculative-algorithm": "DFLASH",
        "speculative-draft-model-path": "RadixArk/Muse-Spark-Open-DFlash2",
        "speculative-draft-model-quantization": "fp8_weight_only",
        "speculative-num-steps": 7,
        "speculative-num-draft-tokens": 8,
        "speculative-eagle-topk": 1,
        "kv-cache-dtype": "fp8_e4m3",
    }
    run = run_collection("musespark", _role(**args), "tokenspeed")
    assert run.draft == "RadixArk/Muse-Spark-Open-DFlash2"
    assert run.topology == Fact("chain", "explicit")
    assert run.draft_precision == Fact("fp8_weight_only", "explicit")
    assert run.kv_cache_dtype == Fact("fp8", "explicit")
    result = audit("musespark", _role(**args), "tokenspeed")
    assert result["curve"].startswith("muse-spark-open-816b-a42b-nvfp4__dflash__")
    assert result["draft_tokens"] == 7
    assert ["unverified", "draft_precision", "fp8_weight_only (explicit)", "default"] in result[
        "findings"
    ]


def test_adaptive_run_is_reported() -> None:
    spec = {
        "method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block", "enable_adaptive_verification": True,
    }  # fmt: skip
    findings = audit("dsv41flash", _role(**{"speculative-config": spec}), "vllm")["findings"]
    assert findings[0] == ["adaptive", "adaptive", "True (explicit)", "False (explicit)"]


def test_precision_and_kv_cache_dtypes_are_read_per_engine() -> None:
    spec = {
        "method": "dspark", "num_speculative_tokens": 7, "kv_cache_dtype": "fp8",
        "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
    }  # fmt: skip
    role = _role(**{"speculative-config": spec, "kv-cache-dtype": "nvfp4"})
    assert audit("minimaxm31flash", role, "vllm")["findings"] == []
    args = {
        "speculative-algorithm": "DSPARK", "speculative-dspark-block-size": 7,
        "speculative-draft-model-quantization": "nvfp4_online",
        "speculative-draft-kv-cache-dtype": "fp8_e4m3", "kv-cache-dtype": "auto",
    }  # fmt: skip
    run = run_collection("minimaxm31flash", _role(**args), "sglang")
    assert run.draft_precision == Fact("nvfp4_online", "explicit")
    assert run.draft_kv_cache_dtype == Fact("fp8", "explicit")
    assert run.kv_cache_dtype == Fact(None, "default")
    config = {"decoding_type": "MTP", "max_draft_len": 3}
    trt = _role(speculative_config=config, kv_cache_config={"dtype": "fp8"})
    assert run_collection("dsv4", trt, "dynamo-trt").kv_cache_dtype == Fact("fp8", "explicit")


def test_snapshot_arguments_keep_the_tilert_model_profile() -> None:
    role = _role(model="dsv41", **{"with-mtp": True, "num-mtp": 5, "model-weights-dir": "/w"})
    assert spec_arguments(role) == {"model": "dsv41", "with-mtp": True, "num-mtp": 5}
