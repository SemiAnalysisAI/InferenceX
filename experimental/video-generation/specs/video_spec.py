import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SPEC = Path(__file__).with_name("h3.yaml")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MAX_SEED = 2**63
_ROOT_KEYS = {
    1: {"schema_version", "model", "generation", "prompts", "sets", "reference"},
    2: {"schema_version", "model", "generation", "prompts", "sets", "fairness"},
}


def canonical_json_bytes(value: Any) -> bytes:
    """The bytes every workload hash covers: sorted keys, compact separators, UTF-8, no NaN."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def schema_version(document: dict[str, Any]) -> int:
    """Return a spec's or plan's schema; schema 1 plans omit the field to keep their hashes."""
    return document.get("schema_version", 1)


def _fields(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} must be a mapping with exactly the keys {sorted(keys)}")
    return value


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")


def _flag(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be true or false")


def _integer(value: Any, where: str, *, minimum: int, below: int | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (below is not None and value >= below)
    ):
        bound = f"in [{minimum}, {below})" if below is not None else f">= {minimum}"
        raise ValueError(f"{where} must be an integer {bound}")


def _validate_case(case: Any, where: str, prompts: dict[str, Any]) -> None:
    _fields(case, where, {"case_id", "prompt", "seed"})
    _text(case["case_id"], f"{where}.case_id")
    if case["prompt"] not in prompts:
        raise ValueError(f"{where}.prompt {case['prompt']!r} is not a defined prompt")
    _integer(case["seed"], f"{where}.seed", minimum=0, below=_MAX_SEED)


def _validate_set(set_id: str, prompt_set: Any, prompts: dict[str, Any], version: int) -> None:
    where = f"sets.{set_id}"
    keys = {"plan_id", "repetitions", "warmup_runs", "cases"}
    _fields(prompt_set, where, keys | {"warmup_case"} if version == 2 else keys)
    _text(prompt_set["plan_id"], f"{where}.plan_id")
    _integer(prompt_set["repetitions"], f"{where}.repetitions", minimum=1)
    _integer(prompt_set["warmup_runs"], f"{where}.warmup_runs", minimum=0)
    cases = prompt_set["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{where}.cases must be a nonempty list")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        _validate_case(case, f"{where}.cases[{index}]", prompts)
        if case["case_id"] in case_ids:
            raise ValueError(f"{where}.cases[{index}].case_id {case['case_id']!r} is not unique")
        case_ids.add(case["case_id"])
    texts = [prompts[case["prompt"]]["text"] for case in cases]
    if len(set(zip(texts, [case["seed"] for case in cases], strict=True))) != len(cases):
        raise ValueError(f"{where}.cases must not repeat a prompt text with the same seed")
    if version == 2:
        if len(cases) != 20 or len(set(texts)) != len(cases) or prompt_set["warmup_runs"] != 1:
            raise ValueError(f"{where} needs 20 distinct measured prompts and one warmup run")
        warmup = prompt_set["warmup_case"]
        _validate_case(warmup, f"{where}.warmup_case", prompts)
        if warmup["case_id"] in case_ids or prompts[warmup["prompt"]]["text"] in texts:
            raise ValueError(f"{where}.warmup_case must be disjoint from measurements")


def load_spec(path: str | Path) -> dict[str, Any]:
    """Parse and validate a spec file; raise ValueError naming the first bad field."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    version = spec.get("schema_version") if isinstance(spec, dict) else None
    if type(version) is not int or version not in _ROOT_KEYS:
        raise ValueError(f"schema_version must be one of {sorted(_ROOT_KEYS)}")
    _fields(spec, "spec", _ROOT_KEYS[version])
    model = _fields(spec["model"], "model", {"id", "revision"})
    _text(model["id"], "model.id")
    if not isinstance(model["revision"], str) or not _REVISION.fullmatch(model["revision"]):
        raise ValueError("model.revision must be a 40-character commit")

    generation = spec["generation"]
    if not isinstance(generation, dict) or not generation:
        raise ValueError("generation must be a nonempty mapping")
    for name, value in generation.items():
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"generation.{name} must be a number or string")

    prompts = spec["prompts"]
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("prompts must be a nonempty mapping")
    for name, prompt in prompts.items():
        where = f"prompts.{name}"
        _fields(prompt, where, {"text", "requires_motion", "requires_sound"})
        _text(prompt["text"], f"{where}.text")
        _flag(prompt["requires_motion"], f"{where}.requires_motion")
        _flag(prompt["requires_sound"], f"{where}.requires_sound")

    sets = spec["sets"]
    if not isinstance(sets, dict) or not sets:
        raise ValueError("sets must be a nonempty mapping")
    for set_id, prompt_set in sets.items():
        _validate_set(set_id, prompt_set, prompts, version)

    if version == 2:
        fairness = _fields(
            spec["fairness"],
            "fairness",
            {"schedule", "encoding", "conditioning_cache", "reference_identity"},
        )
        schedule = _fields(
            fairness["schedule"],
            "fairness.schedule",
            {
                "sigma_points",
                "denoising_updates",
                "video_flow_shift",
                "audio_flow_shift",
            },
        )
        _integer(schedule["sigma_points"], "fairness.schedule.sigma_points", minimum=2)
        _integer(
            schedule["denoising_updates"],
            "fairness.schedule.denoising_updates",
            minimum=1,
        )
        if schedule["sigma_points"] != schedule["denoising_updates"] + 1 or any(
            schedule[key] != generation[name]
            for key, name in (
                ("denoising_updates", "num_inference_steps"),
                ("video_flow_shift", "flow_shift"),
                ("audio_flow_shift", "audio_flow_shift"),
            )
        ):
            raise ValueError(
                "fairness.schedule must match the generation request and terminal sigma point"
            )
        encoding = _fields(
            fairness["encoding"],
            "fairness.encoding",
            {
                "video_codec",
                "preset",
                "crf",
                "pixel_format",
                "audio_codec",
                "audio_sample_rate_hz",
                "audio_channels",
            },
        )
        for key in ("video_codec", "preset", "pixel_format", "audio_codec"):
            _text(encoding[key], f"fairness.encoding.{key}")
        for key in ("crf", "audio_sample_rate_hz", "audio_channels"):
            _integer(encoding[key], f"fairness.encoding.{key}", minimum=1)
        if any(
            encoding[key] != generation[key] for key in ("audio_sample_rate_hz", "audio_channels")
        ):
            raise ValueError("fairness.encoding audio properties must match generation")
        if fairness["conditioning_cache"] != "disabled_across_requests":
            raise ValueError("fairness.conditioning_cache must declare disabled_across_requests")
        identity = _fields(
            fairness["reference_identity"],
            "fairness.reference_identity",
            {
                "implementation",
                "minimax_code_revision",
                "diffusers_revision",
                "model_revision",
                "rng_policy",
            },
        )
        for key in ("minimax_code_revision", "diffusers_revision", "model_revision"):
            if not isinstance(identity[key], str) or not _REVISION.fullmatch(identity[key]):
                raise ValueError(f"fairness.reference_identity.{key} must be a 40-character commit")
        if identity["model_revision"] != model["revision"]:
            raise ValueError("fairness.reference_identity.model_revision must match model")
        for key in ("implementation", "rng_policy"):
            _text(identity[key], f"fairness.reference_identity.{key}")

    elif not isinstance(spec["reference"], dict):
        raise ValueError("reference must be a mapping")
    return spec


def _resolve_case(spec: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    prompt = spec["prompts"][case["prompt"]]
    return {
        "case_id": case["case_id"],
        "prompt": prompt["text"],
        "seed": case["seed"],
        "requires_motion": prompt["requires_motion"],
        "requires_sound": prompt["requires_sound"],
    }


def _set_plan_fields(spec: dict[str, Any], set_id: str) -> dict[str, Any]:
    prompt_set = spec["sets"][set_id]
    result = {
        "plan_id": prompt_set["plan_id"],
        "cases": [_resolve_case(spec, case) for case in prompt_set["cases"]],
        "repetitions": prompt_set["repetitions"],
        "warmup_runs": prompt_set["warmup_runs"],
    }
    if schema_version(spec) == 2:
        result["warmup_case"] = _resolve_case(spec, prompt_set["warmup_case"])
    return result


def plan_for(
    spec: dict[str, Any],
    set_id: str,
    *,
    repetitions: int | None = None,
    warmup_runs: int | None = None,
) -> dict[str, Any]:
    """Build the plan object that evaluator/runner.py executes for one prompt set.

    Count overrides that differ from the set's own counts rename the plan
    (``<plan_id>-r<repetitions>-w<warmup_runs>``), so a shortened run never
    shares an identity with the full one.
    """
    plan = {
        "model_id": spec["model"]["id"],
        "model_revision": spec["model"]["revision"],
        "generation": dict(spec["generation"]),
        **_set_plan_fields(spec, set_id),
    }
    if schema_version(spec) == 2:
        plan |= {"schema_version": 2, "fairness": spec["fairness"]}
    counts = (plan["repetitions"], plan["warmup_runs"])
    if repetitions is not None:
        _integer(repetitions, "repetitions", minimum=1)
        plan["repetitions"] = repetitions
    if warmup_runs is not None:
        _integer(warmup_runs, "warmup_runs", minimum=0)
        plan["warmup_runs"] = warmup_runs
    if (plan["repetitions"], plan["warmup_runs"]) != counts:
        plan["plan_id"] += f"-r{plan['repetitions']}-w{plan['warmup_runs']}"
    return plan


def plan_sha256(plan: dict[str, Any]) -> str:
    """The ``plan_sha256`` the runner records for this plan."""
    return _sha256(plan)


def spec_sha256(spec: dict[str, Any]) -> str:
    """Hash the model, generation, set plans, and optional fairness semantics."""
    identity = {
        "model": spec["model"],
        "generation": spec["generation"],
        "sets": {set_id: _set_plan_fields(spec, set_id) for set_id in spec["sets"]},
    }
    if schema_version(spec) == 2:
        identity["fairness"] = spec["fairness"]
    return _sha256(identity)


def main(argv: list[str]) -> None:
    spec = load_spec(argv[1] if len(argv) > 1 else DEFAULT_SPEC)
    plans = {set_id: plan_for(spec, set_id) for set_id in spec["sets"]}
    report = {
        "spec_sha256": spec_sha256(spec),
        "sets": {
            set_id: {"plan_id": plan["plan_id"], "plan_sha256": plan_sha256(plan)}
            for set_id, plan in plans.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv)
