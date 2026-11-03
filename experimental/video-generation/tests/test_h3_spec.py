import functools
import operator
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

SPECS = Path(__file__).resolve().parents[1] / "specs"
sys.path.insert(0, str(SPECS))
import h3_spec  # noqa: E402

CITATION = "feat/example@0123456789abcdef:specs/source.json:1"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    document = _minimal_spec()
    document["sets"]["s"].update(repetitions=2, warmup_runs=1)
    return document


def _assign(tree: Any, path: tuple[Any, ...], value: Any) -> None:
    *parents, last = path
    functools.reduce(operator.getitem, parents, tree)[last] = value


def _reverse_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(value)}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def test_hashes_ignore_formatting_but_bind_workload_semantics(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    body = yaml.safe_dump(_reverse_keys(spec), sort_keys=False, indent=4, width=50)
    rewritten = tmp_path / "h3.yaml"
    rewritten.write_text(f"# rewritten copy\n{body}# end\n\n", encoding="utf-8")
    reloaded = h3_spec.load_spec(rewritten)
    for set_id in spec["sets"]:
        assert h3_spec.plan_sha256(h3_spec.plan_for(reloaded, set_id)) == h3_spec.plan_sha256(
            h3_spec.plan_for(spec, set_id)
        )
    assert h3_spec.spec_sha256(reloaded) == h3_spec.spec_sha256(spec)
    plan = {
        **h3_spec.plan_for(spec, "s"),
        "fairness": {"encoding": {"crf": 18}, "reference_identity": {"model_revision": "b" * 40}},
        "warmup_case": {"case_id": "warmup", "prompt": "a separate warmup", "seed": 13},
    }
    digest = h3_spec.plan_sha256(plan)
    for field, changed in (
        ("fairness", {**plan["fairness"], "encoding": {"crf": 23}}),
        ("fairness", {**plan["fairness"], "reference_identity": {"model_revision": "c" * 40}}),
        ("warmup_case", {**plan["warmup_case"], "seed": 17}),
    ):
        assert h3_spec.plan_sha256({**plan, field: changed}) != digest


def test_count_overrides_rename_the_plan_and_change_its_hash(
    spec: dict[str, Any],
) -> None:
    full = h3_spec.plan_for(spec, "s")
    assert full == {
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "generation": {"fps": 24},
        "plan_id": "plan",
        "repetitions": 2,
        "warmup_runs": 1,
        "cases": [
            {
                "case_id": "c",
                "prompt": "a prompt",
                "seed": 7,
                "requires_motion": True,
                "requires_sound": False,
            }
        ],
    }
    single = h3_spec.plan_for(spec, "s", repetitions=1, warmup_runs=0)
    assert single == {
        **full,
        "plan_id": f"{full['plan_id']}-r1-w0",
        "repetitions": 1,
        "warmup_runs": 0,
    }
    assert h3_spec.plan_sha256(single) != h3_spec.plan_sha256(full)
    same_counts = h3_spec.plan_for(
        spec, "s", repetitions=full["repetitions"], warmup_runs=full["warmup_runs"]
    )
    assert same_counts == full
    with pytest.raises(ValueError, match=re.escape("repetitions must be an integer >= 1")):
        h3_spec.plan_for(spec, "s", repetitions=0)


def test_fairness_spec_rejects_reused_warmup_and_mismatched_schedule(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load((SPECS / "h3-v0.1.yaml").read_text())
    document["sets"]["release-v1"]["warmup_case"]["prompt"] = document["sets"]["release-v1"][
        "cases"
    ][0]["prompt"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="disjoint"):
        h3_spec.load_spec(path)
    document["sets"]["release-v1"]["warmup_case"]["prompt"] = "warmup-only"
    document["fairness"]["schedule"]["denoising_updates"] = 50
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="schedule"):
        h3_spec.load_spec(path)


def test_fairness_spec_rejects_distinct_ids_with_duplicate_prompt_text(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load((SPECS / "h3-v0.1.yaml").read_text())
    path = tmp_path / "bad.yaml"
    document["prompts"]["paper-fold"]["text"] = document["prompts"]["ceramic-wheel"]["text"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="distinct prompt text"):
        h3_spec.load_spec(path)
    document["prompts"]["paper-fold"]["text"] = "A distinct paper folding prompt."
    document["prompts"]["warmup-only"]["text"] = document["prompts"]["ceramic-wheel"]["text"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="warmup_case text must be disjoint"):
        h3_spec.load_spec(path)


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
        (("sets", "s", "seed_sources"), {}, "sets.s.seed_sources must cite exactly the seeds [7]"),
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
