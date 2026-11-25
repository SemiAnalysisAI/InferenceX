import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from specs import video_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {"id": "org/model", "revision": "a" * 40},
        "generation": {"fps": 24},
        "prompts": {"p": {"text": "a prompt", "requires_motion": True, "requires_sound": False}},
        "sets": {
            "s": {
                "plan_id": "plan",
                "repetitions": 2,
                "warmup_runs": 1,
                "cases": [{"case_id": "c", "prompt": "p", "seed": 7}],
            },
        },
        "reference": {},
    }


def test_hashes_ignore_yaml_formatting(spec: dict[str, Any], tmp_path: Path) -> None:
    body = yaml.safe_dump(spec, sort_keys=False, indent=4, width=50)
    rewritten = tmp_path / "h3.yaml"
    rewritten.write_text(f"# rewritten copy\n{body}# end\n\n", encoding="utf-8")
    reloaded = video_spec.load_spec(rewritten)
    assert video_spec.plan_sha256(video_spec.plan_for(reloaded, "s")) == video_spec.plan_sha256(
        video_spec.plan_for(spec, "s")
    )
    assert video_spec.spec_sha256(reloaded) == video_spec.spec_sha256(spec)


def test_count_overrides_rename_the_plan_and_change_its_hash(spec: dict[str, Any]) -> None:
    full = video_spec.plan_for(spec, "s")
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
    single = video_spec.plan_for(spec, "s", repetitions=1, warmup_runs=0)
    assert single == {**full, "plan_id": "plan-r1-w0", "repetitions": 1, "warmup_runs": 0}
    assert video_spec.plan_sha256(single) != video_spec.plan_sha256(full)
    assert video_spec.plan_for(spec, "s", repetitions=2, warmup_runs=1) == full
    with pytest.raises(ValueError, match=re.escape("repetitions must be an integer >= 1")):
        video_spec.plan_for(spec, "s", repetitions=0)


RECORDED_FROZEN_PLAN_SHA256 = "8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68"


def test_frozen_plan_reproduces_the_hash_its_reference_run_recorded() -> None:
    spec = video_spec.load_spec(SPECS / "h3.yaml")
    recorded = spec["reference"]["sglang"]
    plan = video_spec.plan_for(spec, "frozen-c124")
    assert plan["plan_id"] == recorded["plan_id"]
    assert video_spec.schema_version(plan) == 1
    assert (
        video_spec.plan_sha256(plan)
        == recorded["recorded_plan_sha256"]
        == RECORDED_FROZEN_PLAN_SHA256
    )


def test_v01_plan_declares_schema_2_with_its_fairness_and_warmup() -> None:
    spec = video_spec.load_spec(SPECS / "h3-v2.yaml")
    plan = video_spec.plan_for(spec, "release-v1")
    assert video_spec.schema_version(plan) == 2
    assert plan["fairness"] == spec["fairness"]
    assert plan["warmup_case"]["case_id"] == "warmup-only"


@pytest.fixture
def sized(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **spec,
        "generation": {"width": 64, "height": 32, "num_inference_steps": 5, "guidance_scale": 4.0},
        "workload_size": {
            "latent_frames": 3,
            "latent_tokens_audio": 4,
            "guidance_branches": 2,
            "model": {
                "latent_patch_pixels": 16,
                "dit_weights_per_token": 10,
                "attention_width": 7,
                "layers": 2,
            },
        },
    }


def _load(document: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(document))
    return video_spec.load_spec(path)


def test_workload_size_is_computed_from_generation_and_the_declared_inputs(
    sized: dict[str, Any], tmp_path: Path
) -> None:
    # 3 frames x (64 / 16) x (32 / 16) video tokens + 4 audio; 5 steps x 2 branches;
    # each pass 2 x 28 x 10 + 4 x 28^2 x 7 x 2 = 44464.
    assert video_spec.workload_size(_load(sized, tmp_path)) == {
        **sized["workload_size"],
        "latent_tokens_video": 24,
        "latent_tokens": 28,
        "dit_forward_passes": 10,
        "dit_flops_per_clip": 444640,
    }


def test_workload_size_refuses_a_spec_without_the_inputs(
    spec: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="the spec declares no workload_size"):
        video_spec.workload_size(_load(spec, tmp_path))


def test_workload_size_stays_outside_the_plan_and_both_hashes(sized: dict[str, Any]) -> None:
    unsized = {key: value for key, value in sized.items() if key != "workload_size"}
    assert video_spec.plan_for(sized, "s") == video_spec.plan_for(unsized, "s")
    assert video_spec.spec_sha256(sized) == video_spec.spec_sha256(unsized)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda doc: doc["workload_size"].update(guidance_branches=0),
            "workload_size.guidance_branches must be an integer >= 1",
        ),
        (
            lambda doc: doc["workload_size"]["model"].update(latent_patch_pixels=24),
            "generation.width must divide by workload_size.model.latent_patch_pixels",
        ),
        (
            lambda doc: doc["generation"].update(guidance_scale=1.0),
            "workload_size.guidance_branches must be 2 when generation.guidance_scale exceeds 1",
        ),
        (
            lambda doc: doc.update(
                generation={**doc["generation"], "guidance_scale": 1.0},
                workload_size={**doc["workload_size"], "guidance_branches": 3},
            ),
            "must be 2 when generation.guidance_scale exceeds 1, else 1",
        ),
    ],
)
def test_loader_rejects_an_invalid_workload_size(
    sized: dict[str, Any],
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    mutate(sized)
    with pytest.raises(ValueError, match=re.escape(message)):
        _load(sized, tmp_path)


def _reuse_text(source: str, target: str) -> Callable[[dict[str, Any]], None]:
    return lambda doc: doc["prompts"][target].update(text=doc["prompts"][source]["text"])


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "h3-v2.yaml",
            lambda doc: doc.update(schema_version=1),
            "spec must be a mapping with exactly the keys",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["sets"]["release-v1"]["cases"].pop(),
            "needs 20 distinct measured prompts",
        ),
        (
            "h3-v2.yaml",
            _reuse_text("ceramic-wheel", "paper-fold"),
            "needs 20 distinct measured prompts",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["sets"]["release-v1"]["warmup_case"].update(prompt="ceramic-wheel"),
            "warmup_case must be disjoint",
        ),
        (
            "h3-v2.yaml",
            _reuse_text("ceramic-wheel", "warmup-only"),
            "warmup_case must be disjoint",
        ),
        (
            "h3-v2.yaml",
            lambda doc: doc["fairness"]["schedule"].update(denoising_updates=50),
            "fairness.schedule must match",
        ),
        (
            "h3.yaml",
            lambda doc: doc["sets"]["engine-compare-v1"]["cases"][1].update(seed=11),
            "cases must not repeat a prompt text with the same seed",
        ),
    ],
)
def test_loader_rejects_invalid_workloads(
    tmp_path: Path, name: str, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = yaml.safe_load((SPECS / name).read_text())
    mutate(document)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=re.escape(message)):
        video_spec.load_spec(path)


def test_canonical_json_bytes_is_sorted_compact_utf8_and_finite() -> None:
    assert (
        video_spec.canonical_json_bytes({"b": "é", "a": [1, 2.5]})
        == '{"a":[1,2.5],"b":"é"}'.encode()
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        video_spec.canonical_json_bytes({"a": float("nan")})
