"""Load a controlled Wan spec into the runner's observable plan contract."""

import copy

import h3_spec
import pytest
import yaml

from evaluator import mvp_runner


def controlled_spec():
    citation = "example@abcdef1:spec.yaml:1"
    values = {
        "width": 64,
        "height": 48,
        "frame_count": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 12.0,
        "negative_prompt": "",
    }
    return {
        "schema_version": 1,
        "model": {
            "id": {"value": mvp_runner.WAN_MODEL_ID, "source": citation},
            "revision": {"value": "b" * 40, "source": citation},
        },
        "generation": {
            name: {
                "value": value,
                "source": citation,
                "sent": {"sglang": True, "vllm-omni": True},
                "checked_on_clip": name in {"width", "height", "frame_count", "fps"},
            }
            for name, value in values.items()
        },
        "prompts": {
            "walk": {
                "text": "A person walks across a room.",
                "requires_motion": True,
                "requires_sound": False,
                "source": citation,
            },
        },
        "sets": {
            "sample": {
                "plan_id": "controlled-wan",
                "repetitions": 2,
                "warmup_runs": 1,
                "seed_sources": {7: citation},
                "cases": [{"case_id": "walk-s7", "prompt": "walk", "seed": 7}],
            },
        },
        "reference": {"note": "ignored by the plan hash"},
    }


def write_spec(path, spec):
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return h3_spec.load_spec(path)


def test_loader_builds_runner_plan_and_keeps_reference_out_of_hash(tmp_path):
    source = controlled_spec()
    loaded = write_spec(tmp_path / "spec.yaml", source)
    plan = h3_spec.plan_for(loaded, "sample")
    assert plan == {
        "model_id": mvp_runner.WAN_MODEL_ID,
        "model_revision": "b" * 40,
        "generation": {
            "width": 64,
            "height": 48,
            "frame_count": 81,
            "fps": 16,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "flow_shift": 12.0,
            "negative_prompt": "",
        },
        "plan_id": "controlled-wan",
        "cases": [
            {
                "case_id": "walk-s7",
                "prompt": "A person walks across a room.",
                "seed": 7,
                "requires_motion": True,
                "requires_sound": False,
            }
        ],
        "repetitions": 2,
        "warmup_runs": 1,
    }
    preview = mvp_runner.preview_plan(plan, runtime="vllm-omni")
    assert preview["measurement_count"] == 2
    assert preview["warmup_count"] == 1
    assert preview["slots"][0]["request"]["num_frames"] == 81
    assert preview["plan_sha256"] == h3_spec.plan_sha256(plan)
    changed = copy.deepcopy(source)
    changed["reference"]["note"] = "another note"
    changed["generation"]["fps"]["source"] = "another@abcdef1:spec.yaml:2"
    assert h3_spec.spec_sha256(
        write_spec(tmp_path / "other.yaml", changed)
    ) == h3_spec.spec_sha256(loaded)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda spec: spec["sets"]["sample"]["cases"][0].update(seed=8),
            "seed_sources",
        ),
        (
            lambda spec: spec["generation"]["fps"].update(source="uncited"),
            "generation.fps.source",
        ),
    ],
)
def test_loader_rejects_uncited_or_unmapped_inputs(tmp_path, mutation, match):
    source = controlled_spec()
    mutation(source)
    with pytest.raises(ValueError, match=match):
        write_spec(tmp_path / "invalid.yaml", source)
