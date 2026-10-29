import copy
import functools
import json
import operator
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

SPECS = Path(__file__).resolve().parents[1] / "specs"
sys.path.insert(0, str(SPECS))
import h3_spec  # noqa: E402

SPEC_PATH = SPECS / "h3.yaml"
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "h3-serving-2gpu.json"
RUN_39002901777_RECORDED_PLAN_SHA256 = (
    "8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68"
)
FROZEN = "frozen-c124"
CITATION = "feat/example@0123456789abcdef:specs/source.json:1"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return h3_spec.load_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def reference_plan() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["plan"]


def _assign(tree: Any, path: tuple[Any, ...], value: Any) -> None:
    *parents, last = path
    functools.reduce(operator.getitem, parents, tree)[last] = value


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _spec_path(spec: dict[str, Any], plan_path: tuple[Any, ...]) -> tuple[Any, ...]:
    match plan_path:
        case ("model_id",):
            return ("model", "id", "value")
        case ("model_revision",):
            return ("model", "revision", "value")
        case ("generation", name):
            return ("generation", name, "value")
        case ("cases", index, "prompt" | "requires_motion" | "requires_sound" as key):
            prompt = spec["sets"][FROZEN]["cases"][index]["prompt"]
            return ("prompts", prompt, "text" if key == "prompt" else key)
        case ("cases", index, key):
            return ("sets", FROZEN, "cases", index, key)
        case (key,):
            return ("sets", FROZEN, key)
    raise AssertionError(plan_path)


def _mutated(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    return f"{value}x"


def _reverse_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(value)}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def test_frozen_set_reproduces_the_2916_plan_and_its_recorded_hash(
    spec: dict[str, Any], reference_plan: dict[str, Any]
) -> None:
    plan = h3_spec.plan_for(spec, FROZEN)
    assert plan == reference_plan
    # Dict equality treats 3 and 3.0 alike; the hashes do not.
    assert h3_spec.plan_sha256(plan) == RUN_39002901777_RECORDED_PLAN_SHA256
    assert h3_spec.plan_sha256(reference_plan) == RUN_39002901777_RECORDED_PLAN_SHA256


def test_hashes_ignore_key_order_formatting_and_comments(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    body = yaml.safe_dump(_reverse_keys(raw), sort_keys=False, indent=4, width=50)
    rewritten = tmp_path / "h3.yaml"
    rewritten.write_text(f"# rewritten copy\n{body}# end\n\n", encoding="utf-8")
    reloaded = h3_spec.load_spec(rewritten)
    for set_id in spec["sets"]:
        assert h3_spec.plan_sha256(h3_spec.plan_for(reloaded, set_id)) == h3_spec.plan_sha256(
            h3_spec.plan_for(spec, set_id)
        )
    assert h3_spec.spec_sha256(reloaded) == h3_spec.spec_sha256(spec)


def test_each_spec_value_changes_only_its_plan_field_and_the_hash(
    spec: dict[str, Any], reference_plan: dict[str, Any]
) -> None:
    base = h3_spec.plan_for(spec, FROZEN)
    base_leaves = dict(_leaves(base))
    owners = {h3_spec.plan_sha256(base): ("unmutated",)}
    for plan_path, value in _leaves(reference_plan):
        mutated = copy.deepcopy(spec)
        _assign(mutated, _spec_path(mutated, plan_path), _mutated(value))
        plan = h3_spec.plan_for(mutated, FROZEN)
        assert [path for path, item in _leaves(plan) if base_leaves[path] != item] == [plan_path]
        digest = h3_spec.plan_sha256(plan)
        assert digest not in owners, f"{plan_path} hashes like {owners[digest]}"
        owners[digest] = plan_path


def test_count_overrides_rename_the_plan_and_change_its_hash(spec: dict[str, Any]) -> None:
    full = h3_spec.plan_for(spec, FROZEN)
    single = h3_spec.plan_for(spec, FROZEN, repetitions=1, warmup_runs=0)
    assert single == {
        **full,
        "plan_id": f"{full['plan_id']}-r1-w0",
        "repetitions": 1,
        "warmup_runs": 0,
    }
    assert h3_spec.plan_sha256(single) != h3_spec.plan_sha256(full)
    same_counts = h3_spec.plan_for(
        spec, FROZEN, repetitions=full["repetitions"], warmup_runs=full["warmup_runs"]
    )
    assert same_counts == full
    with pytest.raises(ValueError, match=re.escape("repetitions must be an integer >= 1")):
        h3_spec.plan_for(spec, FROZEN, repetitions=0)


def _minimal_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "id": {"value": "org/model", "source": CITATION},
            "revision": {"value": "a" * 40, "source": CITATION},
        },
        "generation": {
            "fps": {
                "value": 24,
                "source": CITATION,
                "sent": {"sglang": False, "vllm-omni": True},
                "checked_on_clip": True,
            },
        },
        "prompts": {
            "p": {
                "text": "a prompt",
                "requires_motion": True,
                "requires_sound": False,
                "source": CITATION,
            },
        },
        "sets": {
            "s": {
                "plan_id": "plan",
                "repetitions": 1,
                "warmup_runs": 0,
                "seed_sources": {7: CITATION},
                "cases": [{"case_id": "c", "prompt": "p", "seed": 7}],
            },
        },
        "reference": {},
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("generation", "fps", "sent"), {"sglang": False}, "generation.fps.sent must be a mapping"),
        (("generation", "fps", "source"), "mvp_runner.py:186", "generation.fps.source must cite"),
        (("generation", "fps", "value"), True, "generation.fps.value must be a number or string"),
        (("sets", "s", "cases", 0, "seed"), 2**63, "sets.s.cases[0].seed must be an integer"),
        (("sets", "s", "cases", 0, "seed"), True, "sets.s.cases[0].seed must be an integer"),
        (("sets", "s", "cases", 0, "prompt"), "q", "sets.s.cases[0].prompt 'q' is not a defined"),
        (
            ("sets", "s", "cases"),
            [{"case_id": "c", "prompt": "p", "seed": n} for n in (7, 7)],
            "is not unique",
        ),
        (("sets", "s", "seed_sources"), {}, "sets.s.seed_sources must cite exactly the seeds [7]"),
        (("sets",), {}, "sets must be a nonempty mapping"),
        (("schema_version",), True, "schema_version must be 1"),
    ],
)
def test_load_spec_rejects_malformed_specs(
    tmp_path: Path, path: tuple[Any, ...], value: Any, message: str
) -> None:
    spec = _minimal_spec()
    _assign(spec, path, value)
    target = tmp_path / "spec.yaml"
    target.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        h3_spec.load_spec(target)
