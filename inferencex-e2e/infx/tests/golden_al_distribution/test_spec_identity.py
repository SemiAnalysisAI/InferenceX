"""Curve identities, identity-based selection, run normalization and the audit snapshot."""

import itertools
import json
from pathlib import Path

import pytest

from infx.golden_al_distribution import GOLDEN_DIR, curve_name, load_curve
from infx.golden_al_distribution.identity import (
    Collection,
    Fact,
    canonical_name,
    compare,
    load_identities,
    parse_identity,
)
from infx.srt_slurm.spec_identity import audit, run_collection

SNAPSHOT = Path(__file__).with_name("current_spec_roles.json")
# Curve files before the canonical names; the parity test maps the old selection through it.
LEGACY_NAMES = {
    "dsv4_mtp": "deepseek-v4-pro__mtp__native__default-default",
    "dsv4-pro-0813-dspark": "deepseek-v4-pro-0813__dspark__native__probabilistic-default",
    "dsv41flash_dspark": "deepseek-v4.1-flash__dspark__native__probabilistic-block",
    "glm5.2_mtp": "glm-5.2-fp8__mtp__native__default-default",
    "glm5.3_mtp": "glm-5.3-fp8__mtp__native__default-default",
    "kimik3_dspark": "kimi-k3__dspark__kimi-k3-dspark__default-default",
    "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method": (
        "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block"
    ),
    "minimaxm3_eagle3": "minimax-m3__eagle3__minimax-m3-eagle3__default-default",
    "minimaxm3_eagle3_gqa": "minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default",
    "minimaxm31flash_dspark": "minimax-m3.1-flash__dspark__native__probabilistic-block",
    "musespark_dflash": (
        "muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default"
    ),
    "qwen3.5_mtp": "qwen3.5-397b-a17b-nvfp4__mtp__native__default-default",
    "qwen3.8next_mtp": "qwen3.8-flash-next-fp8__mtp__native__default-default",
}


def legacy_curve_name(model: str, spec: dict) -> str:
    """The selection curve_name made before identities.yaml, kept to prove they agree."""
    method = str(spec.get("method", "")).lower()
    if method in ("eagle", "nextn"):
        method = "eagle3" if model == "minimaxm3" else "mtp"
    curve = f"{model}_{method}"
    if model in ("dsv4", "dsv4dspark", "dsv4dsparkprob") and method == "dspark":
        curve = "dsv4-pro-0813-dspark"
    elif model == "minimaxm3" and method == "eagle3":
        if "gqa" in str(spec.get("model", "")).lower():
            curve += "_gqa"
    elif model == "kimik3" and method == "dspark":
        sampling = spec.get("draft_sample_method")
        if sampling == "probabilistic":
            curve += "_probabilistic_sample_method_block_rejection_sample_method"
        elif sampling != "greedy":
            raise ValueError(f"No Kimi DSpark golden curve for draft sampling {sampling!r}")
    if curve not in LEGACY_NAMES:
        raise ValueError(f"No committed golden curve {curve}.yaml")
    return LEGACY_NAMES[curve]


def _role(**args):
    return {"args": args}


def test_every_curve_has_one_identity_and_its_canonical_name() -> None:
    identities = load_identities()
    stems = {path.stem for path in GOLDEN_DIR.glob("*.yaml")} - {"identities"}
    assert set(identities) == stems == set(LEGACY_NAMES.values())
    for name, identity in identities.items():
        assert canonical_name(load_curve(name).model, identity) == name


PREFIXES = [
    "dsv4", "dsv4dspark", "dsv4dsparkprob", "dsv41flash", "glm5.2", "glm5.3", "kimik3",
    "minimaxm3", "minimaxm31flash", "musespark", "qwen3.5", "qwen3.8next", "unknown",
]  # fmt: skip
METHODS = ["mtp", "MTP", "eagle", "EAGLE", "nextn", "NEXTN", "eagle3", "dspark", "dflash", ""]
DRAFTS = [None, "Inferact/MiniMax-M3-EAGLE3", "org/MiniMax-M3-EAGLE3-GQA", "/hf/x-gqa-y"]
SAMPLERS = [None, "greedy", "probabilistic", "other"]


def test_identity_selection_matches_the_legacy_selection_everywhere() -> None:
    """Same curve, or an error in both, for every prefix, method, draft and sampler."""
    for prefix, method, draft, sampler in itertools.product(PREFIXES, METHODS, DRAFTS, SAMPLERS):
        spec = {"method": method}
        if draft:
            spec["model"] = draft
        if sampler:
            spec["draft_sample_method"] = sampler
        try:
            expected = legacy_curve_name(prefix, spec)
        except ValueError:
            with pytest.raises(ValueError):
                curve_name(prefix, spec)
            continue
        assert curve_name(prefix, spec) == expected, spec


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"collection": {"verify": {"value": "block"}}}, "verify must be"),
        ({"collection": {"verify": {"value": None, "source": "explicit"}}}, "null value exactly"),
        ({"collection": {"verify": {"value": "block", "source": "guessed"}}}, "not one of"),
        ({"collection": {"engine": "jax"}}, "unknown engine"),
        ({"select": {"method": "medusa"}}, "known method"),
        ({"select": {"draft_sample": "typical"}}, "draft_sample must be"),
    ],
)
def test_malformed_identity_is_rejected(change, message) -> None:
    default = {"value": None, "source": "default"}
    data = {
        "select": {"model_prefixes": ["m"], "method": "mtp"},
        "collection": {
            "engine": "vllm", "draft": "native", "draft_sample": default, "verify": default,
            "adaptive": default, "topology": {"value": "chain", "source": "documented"},
            "draft_precision": default, "draft_kv_cache_dtype": default, "kv_cache_dtype": default,
        },
    }  # fmt: skip
    for block, fields in change.items():
        data[block] = {**data[block], **fields}
    with pytest.raises(ValueError, match=message):
        parse_identity("bad", data)


def test_snapshot_pins_every_current_speculative_role() -> None:
    """Every matrix role keeps its curve and golden AL, and its findings stay as reviewed.

    Regenerate after a deliberate change, then review the diff:
    python -m infx.matrix.generate full-sweep --config-files configs/nvidia-master.yaml
    configs/amd-master.yaml --scenario-type agentic-coding > matrix.json
    python -m infx.srt_slurm.spec_identity snapshot matrix.json --root . > <this fixture>
    """
    for case in json.loads(SNAPSHOT.read_text()):
        role = {"args": case["args"]}
        assert audit(case["model_prefix"], role, case["framework"]) == case["expected"], case


@pytest.mark.parametrize("algorithm", ["EAGLE", "NEXTN"])
def test_sglang_native_heads_select_the_model_mtp_curve(algorithm) -> None:
    role = _role(**{"speculative-algorithm": algorithm, "speculative-num-steps": 3})
    assert audit("qwen3.5", role, "sglang")["curve"].startswith("qwen3.5-397b-a17b-nvfp4__mtp__")


def test_minimax_sglang_eagle_selects_eagle3() -> None:
    args = {
        "speculative-algorithm": "EAGLE", "speculative-num-steps": 3,
        "speculative-draft-model-path": "Inferact/MiniMax-M3-EAGLE3-GQA",
    }  # fmt: skip
    assert audit("minimaxm3", _role(**args), "sglang")["curve"] == (
        "minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default"
    )


def test_vllm_unset_settings_are_defaults_and_synthetic_is_block() -> None:
    role = _role(**{"speculative-config": {"method": "mtp", "num_speculative_tokens": 3}})
    run = run_collection("qwen3.5", role, "vllm")
    assert run.draft_sample == Fact(None, "default")
    assert run.verify == Fact("standard", "documented")
    assert run.adaptive == Fact(False, "documented")
    synthetic = {
        "method": "dspark", "num_speculative_tokens": 6, "draft_sample_method": "probabilistic",
        "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.77,
    }  # fmt: skip
    run = run_collection("dsv4", _role(**{"speculative-config": synthetic}), "vllm")
    assert run.verify == Fact("block", "documented")
    assert run.draft_sample == Fact("probabilistic", "explicit")


def test_same_engine_defaults_agree_but_cross_engine_defaults_are_unverified() -> None:
    spec = {"method": "mtp", "num_speculative_tokens": 3}
    role = _role(**{"speculative-config": spec, "kv-cache-dtype": "fp8"})
    assert audit("qwen3.5", role, "vllm")["findings"] == []
    args = {
        "speculative-algorithm": "NEXTN", "speculative-num-steps": 3, "speculative-eagle-topk": 1,
        "kv-cache-dtype": "fp8",
    }  # fmt: skip
    findings = audit("qwen3.5", _role(**args), "sglang")["findings"]
    assert [f[:2] for f in findings] == [
        ["unverified", "draft_sample"], ["unverified", "draft_precision"],
        ["unverified", "draft_kv_cache_dtype"], ["engine", "engine"],
    ]  # fmt: skip


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
    config = {
        "decoding_type": "MTP",
        "max_draft_len": 3,
        "use_relaxed_acceptance_for_thinking": True,
    }
    run = run_collection("dsv4", _role(speculative_config=config), "dynamo-trt")
    assert run.verify == Fact("relaxed", "explicit")


def test_atom_verifier_is_unknown_and_reported_unverified() -> None:
    args = {
        "method": "dspark",
        "num-speculative-tokens": 4,
        "draft-model": "Inferact/Kimi-K3-DSpark",
    }
    result = audit("kimik3", _role(**args), "atom")
    assert result["curve"] == "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block"
    assert ["unverified", "verify", "unknown", "block (explicit)"] in result["findings"]


def test_adaptive_run_is_reported() -> None:
    spec = {
        "method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block", "enable_adaptive_verification": True,
    }  # fmt: skip
    findings = audit("dsv41flash", _role(**{"speculative-config": spec}), "vllm")["findings"]
    assert findings[0] == ["adaptive", "adaptive", "True (explicit)", "False (explicit)"]
    assert all(finding[0] == "unverified" for finding in findings[1:])


def test_compare_reports_provisional_curves() -> None:
    run = Collection(
        "vllm", "native", Fact(None, "default"), Fact("standard", "documented"),
        Fact(False, "documented"), Fact("chain", "documented"), Fact(None, "default"),
        Fact(None, "default"), Fact("fp8", "explicit"),
    )  # fmt: skip
    curve = load_identities()["glm-5.3-fp8__mtp__native__default-default"]
    assert [finding.kind for finding in compare(run, curve)] == ["provisional"]


def test_precision_and_kv_cache_dtypes_are_read_per_engine() -> None:
    spec = {
        "method": "dspark", "num_speculative_tokens": 7, "kv_cache_dtype": "fp8",
        "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
    }  # fmt: skip
    role = _role(**{"speculative-config": spec, "kv-cache-dtype": "nvfp4"})
    run = run_collection("minimaxm31flash", role, "vllm")
    assert run.draft_kv_cache_dtype == Fact("fp8", "explicit")
    assert run.kv_cache_dtype == Fact("nvfp4", "explicit")
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


def test_non_speculative_role_has_no_collection() -> None:
    assert run_collection("kimik3", _role(), "vllm") is None
