"""Behavioral checks for llm-d recipe acceptance-length selection."""

import importlib.util
import json
import shlex
from pathlib import Path

import pytest
import yaml

E2E_ROOT = Path(__file__).resolve().parents[3]
RECIPE_MODULE = E2E_ROOT / "benchmarks/multi_node/llm-d/recipe.py"
ENV = {
    "IS_AGENTIC": "1",
    "SPEC_DECODING": "mtp",
    "EVAL_ONLY": "false",
    "RUN_EVAL": "false",
    "MODEL_PREFIX": "dsv4",
    "THINKING_MODE": "thinking_on",
    "KV_OFFLOADING": "none",
}


@pytest.fixture
def renderer():
    spec = importlib.util.spec_from_file_location("llm_d_recipe", RECIPE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def golden_dir(tmp_path: Path) -> Path:
    # Controlled data makes the behavior independent of later curve updates.
    (tmp_path / "dsv4-pro-0813-dspark.yaml").write_text(yaml.safe_dump({
        "test-model": {"thinking_on": {1: 1.7, 3: 2.9, 5: 3.4}},
    }))
    (tmp_path / "dsv4_mtp.yaml").write_text(yaml.safe_dump({
        "test-model": {"thinking_on": {3: 2.5}},
    }))
    return tmp_path


def _role(tokens=3, **overrides):
    config = {
        "method": "dspark",
        "num_speculative_tokens": tokens,
        "draft_sample_method": "probabilistic",
        "attention_backend": "FLASHINFER_MLA_SPARSE_DSV4",
        **overrides,
    }
    return {"extra-args": f"--before yes --speculative-config {json.dumps(config)} --after no"}


def _rendered_spec(output: str) -> dict:
    assignment = shlex.split(output.splitlines()[0])[0]
    extra = assignment.split("=", 1)[1]
    raw = extra.split("--speculative-config ", 1)[1]
    return json.JSONDecoder().raw_decode(raw)[0]


@pytest.mark.parametrize("source_options", [
    {"enable_adaptive_verification": True},
    {"rejection_sample_method": "block"},
    {"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 99},
    {},
])
def test_throughput_enforces_measured_al_despite_recipe_options(renderer, golden_dir, source_options):
    output = renderer.role_assignments(
        {"prefill": _role(5, **source_options)}, "prefill", ENV, golden_dir=golden_dir,
    )
    config = _rendered_spec(output)
    assert config["rejection_sample_method"] == "synthetic"
    assert config["synthetic_acceptance_length"] == 3.4
    assert config["enable_adaptive_verification"] is False
    assert config["attention_backend"] == "FLASHINFER_MLA_SPARSE_DSV4"
    extra = shlex.split(output.splitlines()[0])[0].split("=", 1)[1]
    assert extra.startswith("--before yes ")
    assert extra.endswith(" --after no")


def test_prefill_and_decode_use_their_own_draft_depth(renderer, golden_dir):
    recipe = {"prefill": _role(1), "decode": _role(3)}
    prefill = _rendered_spec(renderer.role_assignments(
        recipe, "prefill", ENV, golden_dir=golden_dir,
    ))
    decode = _rendered_spec(renderer.role_assignments(
        recipe, "decode", ENV, golden_dir=golden_dir,
    ))
    assert prefill["synthetic_acceptance_length"] == 1.7
    assert decode["synthetic_acceptance_length"] == 2.9


def test_native_mtp_uses_its_own_curve(renderer, golden_dir):
    config = _rendered_spec(renderer.role_assignments(
        {"prefill": _role(method="mtp")}, "prefill", ENV, golden_dir=golden_dir,
    ))
    assert config["rejection_sample_method"] == "synthetic"
    assert config["synthetic_acceptance_length"] == 2.5
    assert "enable_adaptive_verification" not in config


def test_eval_uses_explicit_real_verification_for_both_roles(renderer, golden_dir):
    recipe = {"prefill": _role(1), "decode": _role(3, rejection_sample_method="standard")}
    env = {**ENV, "EVAL_ONLY": "true", "RUN_EVAL": "true"}
    prefill = _rendered_spec(renderer.role_assignments(
        recipe, "prefill", env, golden_dir=golden_dir,
    ))
    decode = _rendered_spec(renderer.role_assignments(
        recipe, "decode", env, golden_dir=golden_dir,
    ))
    assert prefill["rejection_sample_method"] == "block"
    assert decode["rejection_sample_method"] == "standard"
    assert "synthetic_acceptance_length" not in prefill
    assert "synthetic_acceptance_length" not in decode


@pytest.mark.parametrize("env_override", [
    {"EVAL_ONLY": "true", "RUN_EVAL": "true"},
    {"IS_AGENTIC": "0"},
    {"SPEC_DECODING": "none"},
])
@pytest.mark.parametrize("sampling, expected", [("synthetic", "block"), ("block", "block")])
def test_real_verification_removes_stale_simulation(renderer, golden_dir, env_override, sampling, expected):
    env = {**ENV, **env_override}
    # Real verification does not require a golden curve or its metadata.
    del env["MODEL_PREFIX"]
    del env["THINKING_MODE"]
    config = _rendered_spec(renderer.role_assignments(
        {"prefill": _role(
            rejection_sample_method=sampling,
            synthetic_acceptance_length=99,
            enable_adaptive_verification=True,
        )},
        "prefill", env, golden_dir=golden_dir,
    ))
    assert "synthetic_acceptance_length" not in config
    assert config["rejection_sample_method"] == expected
    assert config["enable_adaptive_verification"] is True


@pytest.mark.parametrize("key", ["MODEL_PREFIX", "THINKING_MODE", "SPEC_DECODING"])
def test_missing_selection_metadata_fails(renderer, golden_dir, key):
    env = {k: v for k, v in ENV.items() if k != key}
    with pytest.raises(ValueError, match=f"Missing {key}"):
        renderer.role_assignments({"prefill": _role()}, "prefill", env, golden_dir=golden_dir)


@pytest.mark.parametrize("role, env_override, error", [
    (_role(method="unknown"), {}, "No committed golden curve"),
    (_role(), {"MODEL_PREFIX": "unknown"}, "No committed golden curve"),
    (_role(), {"THINKING_MODE": "thinking_off"}, "No golden acceptance"),
    (_role(2), {}, "No golden acceptance"),
    (_role(0), {}, "positive integer draft length"),
])
def test_unmeasured_combinations_fail_closed(renderer, golden_dir, role, env_override, error):
    with pytest.raises(ValueError, match=error):
        renderer.role_assignments(
            {"prefill": role}, "prefill", {**ENV, **env_override}, golden_dir=golden_dir,
        )


def test_combined_eval_cannot_reuse_a_synthetic_server(renderer, golden_dir):
    with pytest.raises(ValueError, match="Run accuracy evals separately"):
        renderer.role_assignments(
            {"prefill": _role(enable_adaptive_verification=True)},
            "prefill", {**ENV, "RUN_EVAL": "true"}, golden_dir=golden_dir,
        )


def test_non_speculative_role_keeps_args_and_topology(renderer, golden_dir):
    output = renderer.role_assignments({"prefill": {
        "extra-args": "--max-num-seqs 32",
        "tp": 8,
        "enable-expert-parallel": False,
        "env": {"KEEP": "space separated"},
    }}, "prefill", ENV, golden_dir=golden_dir)
    assert output == (
        "ROLE_EXTRA_ARGS='--max-num-seqs 32'\n"
        "PREFILL_ENABLE_EP=false\nTP_SIZE=8\nROLE_ENABLE_EP=false\n"
        "export KEEP='space separated'"
    )
