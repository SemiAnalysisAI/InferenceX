import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from specs import h3_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"
CITATION = "feat/example@0123456789abcdef:specs/source.json:1"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
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
                "repetitions": 2,
                "warmup_runs": 1,
                "seed_sources": {7: CITATION},
                "cases": [{"case_id": "c", "prompt": "p", "seed": 7}],
            },
        },
        "reference": {},
    }


def test_hashes_ignore_formatting_but_bind_workload_semantics(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    body = yaml.safe_dump(spec, sort_keys=False, indent=4, width=50)
    rewritten = tmp_path / "h3.yaml"
    rewritten.write_text(f"# rewritten copy\n{body}# end\n\n", encoding="utf-8")
    reloaded = h3_spec.load_spec(rewritten)
    assert h3_spec.plan_sha256(h3_spec.plan_for(reloaded, "s")) == h3_spec.plan_sha256(
        h3_spec.plan_for(spec, "s")
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


def test_count_overrides_rename_the_plan_and_change_its_hash(spec: dict[str, Any]) -> None:
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
    assert single == {**full, "plan_id": "plan-r1-w0", "repetitions": 1, "warmup_runs": 0}
    assert h3_spec.plan_sha256(single) != h3_spec.plan_sha256(full)
    assert h3_spec.plan_for(spec, "s", repetitions=2, warmup_runs=1) == full
    with pytest.raises(ValueError, match=re.escape("repetitions must be an integer >= 1")):
        h3_spec.plan_for(spec, "s", repetitions=0)


def _reuse_text(source: str, target: str) -> Callable[[dict[str, Any]], None]:
    return lambda doc: doc["prompts"][target].update(text=doc["prompts"][source]["text"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda doc: doc["sets"]["release-v1"]["warmup_case"].update(prompt="ceramic-wheel"),
            "warmup_case prompt must be disjoint",
        ),
        (
            lambda doc: doc["fairness"]["schedule"].update(denoising_updates=50),
            "fairness.schedule must match",
        ),
        (_reuse_text("ceramic-wheel", "paper-fold"), "cases must resolve to distinct prompt text"),
        (_reuse_text("ceramic-wheel", "warmup-only"), "warmup_case text must be disjoint"),
        (
            lambda doc: doc["sets"]["release-v1"]["seed_sources"].pop(11),
            "seed_sources must cite exactly the seeds",
        ),
    ],
)
def test_fairness_spec_rejects_invalid_workloads(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = yaml.safe_load((SPECS / "h3-v0.1.yaml").read_text())
    mutate(document)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=re.escape(message)):
        h3_spec.load_spec(path)
