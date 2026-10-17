"""Shared Wan spec (specs/wan.yaml) through the shared loader: #3733 values, seeds, runner agreement."""

import sys
from pathlib import Path

import pytest

from evaluator import mvp_runner as runner

SPECS = Path(__file__).resolve().parents[1] / "specs"
sys.path.insert(0, str(SPECS))
import h3_spec

SPEC_PATH = SPECS / "wan.yaml"
# #3733's experimental/video-generation/mvp/wan22-serving.plan.json at 77f2080ce, hashed as canonical
# JSON outside this package.
PLAN_3733_SHA256 = "1f1a093b391bfd94f64d6e492948989850508ac4dad472eeaa98785fd26327d6"
REQUEST_NAMES = {"frame_count": "num_frames"}


@pytest.fixture(scope="module")
def spec():
    return h3_spec.load_spec(SPEC_PATH)


def test_frozen_set_reproduces_the_3733_plan_byte_for_byte(spec):
    plan = h3_spec.plan_for(spec, "frozen-3733")
    assert plan["plan_id"] == "wan22-t2v-a14b-serving-smoke-v1"
    assert h3_spec.plan_sha256(plan) == PLAN_3733_SHA256


def test_shared_set_adds_a_second_seed_100_above_each_3733_seed(spec):
    frozen = h3_spec.plan_for(spec, "frozen-3733")
    shared = h3_spec.plan_for(spec, "shared-v1")
    assert shared["plan_id"] == "wan22-t2v-a14b-shared-v1"
    assert shared["generation"] == frozen["generation"]
    assert (shared["repetitions"], shared["warmup_runs"]) == (
        frozen["repetitions"],
        frozen["warmup_runs"],
    )
    by_prompt = {}
    for case in shared["cases"]:
        by_prompt.setdefault(case["prompt"], []).append(case)
    assert len(shared["cases"]) == 8
    assert list(by_prompt) == [case["prompt"] for case in frozen["cases"]]
    for base in frozen["cases"]:
        first, second = by_prompt[base["prompt"]]
        assert (first["seed"], second["seed"]) == (base["seed"], base["seed"] + 100)
        assert first["case_id"] == f"{base['case_id']}-s{first['seed']}"
        assert second["case_id"] == f"{base['case_id']}-s{second['seed']}"
        for case in (first, second):
            assert case["requires_motion"] is base["requires_motion"]
            assert case["requires_sound"] is base["requires_sound"]


@pytest.mark.parametrize("set_id", ["frozen-3733", "shared-v1"])
def test_plans_pass_the_runner_and_hash_identically_there(spec, set_id):
    plan = h3_spec.plan_for(spec, set_id)
    assert runner.preview_plan(plan)["plan_sha256"] == h3_spec.plan_sha256(plan)


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
def test_sent_flags_are_exactly_the_runner_request_keys(spec, runtime):
    plan = h3_spec.plan_for(spec, "shared-v1")
    request = runner.preview_plan(plan, runtime=runtime)["slots"][0]["request"]
    sent = {
        REQUEST_NAMES.get(name, name)
        for name, field in spec["generation"].items()
        if field["sent"][runtime]
    }
    assert sent | {"model", "prompt", "seed"} == set(request)
