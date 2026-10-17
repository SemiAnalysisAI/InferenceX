import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ENGINES = ("sglang", "vllm-omni")
DEFAULT_SPEC = Path(__file__).with_name("h3.yaml")
_CITATION = re.compile(r"[\w./-]+@[0-9a-f]{7,40}:[\w./-]+:\d+")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MAX_SEED = 2**63


def canonical_json_bytes(value: Any) -> bytes:
    """Stable hash representation, UTF-8, sorted keys, no trailing newline."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def _citation(value: Any, where: str) -> None:
    if not isinstance(value, str) or not _CITATION.fullmatch(value):
        raise ValueError(f"{where} must cite <ref>@<commit>:<path>:<line>")


def _validate_set(set_id: str, prompt_set: Any, prompts: dict[str, Any]) -> None:
    where = f"sets.{set_id}"
    _fields(prompt_set, where, {"plan_id", "repetitions", "warmup_runs", "seed_sources", "cases"})
    _text(prompt_set["plan_id"], f"{where}.plan_id")
    _integer(prompt_set["repetitions"], f"{where}.repetitions", minimum=1)
    _integer(prompt_set["warmup_runs"], f"{where}.warmup_runs", minimum=0)
    cases = prompt_set["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{where}.cases must be a nonempty list")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case_where = f"{where}.cases[{index}]"
        _fields(case, case_where, {"case_id", "prompt", "seed"})
        _text(case["case_id"], f"{case_where}.case_id")
        if case["case_id"] in case_ids:
            raise ValueError(f"{case_where}.case_id {case['case_id']!r} is not unique")
        case_ids.add(case["case_id"])
        if case["prompt"] not in prompts:
            raise ValueError(f"{case_where}.prompt {case['prompt']!r} is not a defined prompt")
        _integer(case["seed"], f"{case_where}.seed", minimum=0, below=_MAX_SEED)
    seed_sources = prompt_set["seed_sources"]
    seeds = {case["seed"] for case in cases}
    if not isinstance(seed_sources, dict) or set(seed_sources) != seeds:
        raise ValueError(f"{where}.seed_sources must cite exactly the seeds {sorted(seeds)}")
    for seed, source in seed_sources.items():
        _citation(source, f"{where}.seed_sources[{seed}]")


def load_spec(path: str | Path) -> dict[str, Any]:
    """Parse and validate a spec file; raise ValueError naming the first bad field."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _fields(spec, "spec", {"schema_version", "model", "generation", "prompts", "sets", "reference"})
    if isinstance(spec["schema_version"], bool) or spec["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    model = _fields(spec["model"], "model", {"id", "revision"})
    for name, entry in model.items():
        _fields(entry, f"model.{name}", {"value", "source"})
        _text(entry["value"], f"model.{name}.value")
        _citation(entry["source"], f"model.{name}.source")
    if not _REVISION.fullmatch(model["revision"]["value"]):
        raise ValueError("model.revision.value must be a 40-character commit")

    generation = spec["generation"]
    if not isinstance(generation, dict) or not generation:
        raise ValueError("generation must be a nonempty mapping")
    for name, field in generation.items():
        where = f"generation.{name}"
        _fields(field, where, {"value", "source", "sent", "checked_on_clip"})
        if isinstance(field["value"], bool) or not isinstance(field["value"], int | float | str):
            raise ValueError(f"{where}.value must be a number or string")
        _citation(field["source"], f"{where}.source")
        sent = _fields(field["sent"], f"{where}.sent", set(ENGINES))
        for engine, carried in sent.items():
            _flag(carried, f"{where}.sent.{engine}")
        _flag(field["checked_on_clip"], f"{where}.checked_on_clip")

    prompts = spec["prompts"]
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("prompts must be a nonempty mapping")
    for name, prompt in prompts.items():
        where = f"prompts.{name}"
        _fields(prompt, where, {"text", "requires_motion", "requires_sound", "source"})
        _text(prompt["text"], f"{where}.text")
        _flag(prompt["requires_motion"], f"{where}.requires_motion")
        _flag(prompt["requires_sound"], f"{where}.requires_sound")
        _citation(prompt["source"], f"{where}.source")

    sets = spec["sets"]
    if not isinstance(sets, dict) or not sets:
        raise ValueError("sets must be a nonempty mapping")
    for set_id, prompt_set in sets.items():
        _validate_set(set_id, prompt_set, prompts)

    if not isinstance(spec["reference"], dict):
        raise ValueError("reference must be a mapping")
    return spec


def generation_values(spec: dict[str, Any]) -> dict[str, Any]:
    return {name: field["value"] for name, field in spec["generation"].items()}


def _set_plan_fields(spec: dict[str, Any], set_id: str) -> dict[str, Any]:
    prompt_set = spec["sets"][set_id]
    cases = []
    for case in prompt_set["cases"]:
        prompt = spec["prompts"][case["prompt"]]
        cases.append(
            {
                "case_id": case["case_id"],
                "prompt": prompt["text"],
                "seed": case["seed"],
                "requires_motion": prompt["requires_motion"],
                "requires_sound": prompt["requires_sound"],
            }
        )
    return {
        "plan_id": prompt_set["plan_id"],
        "cases": cases,
        "repetitions": prompt_set["repetitions"],
        "warmup_runs": prompt_set["warmup_runs"],
    }


def plan_for(
    spec: dict[str, Any],
    set_id: str,
    *,
    repetitions: int | None = None,
    warmup_runs: int | None = None,
) -> dict[str, Any]:
    """Build the #2916 plan object for one prompt set.

    Count overrides that differ from the set's own counts rename the plan
    (``<plan_id>-r<repetitions>-w<warmup_runs>``), so a shortened run never
    shares an identity with the full one.
    """
    plan = {
        "model_id": spec["model"]["id"]["value"],
        "model_revision": spec["model"]["revision"]["value"],
        "generation": generation_values(spec),
        **_set_plan_fields(spec, set_id),
    }
    if repetitions is not None:
        _integer(repetitions, "repetitions", minimum=1)
        plan["repetitions"] = repetitions
    if warmup_runs is not None:
        _integer(warmup_runs, "warmup_runs", minimum=0)
        plan["warmup_runs"] = warmup_runs
    prompt_set = spec["sets"][set_id]
    if (plan["repetitions"], plan["warmup_runs"]) != (
        prompt_set["repetitions"],
        prompt_set["warmup_runs"],
    ):
        plan["plan_id"] += f"-r{plan['repetitions']}-w{plan['warmup_runs']}"
    return plan


def plan_sha256(plan: dict[str, Any]) -> str:
    """Equal to the ``plan_sha256`` that PR #2916's runner records for the same plan."""
    return _sha256(plan)


def spec_sha256(spec: dict[str, Any]) -> str:
    """Hash the model, the generation values, and each set's plan fields; nothing else counts."""
    return _sha256(
        {
            "model": {
                "id": spec["model"]["id"]["value"],
                "revision": spec["model"]["revision"]["value"],
            },
            "generation": generation_values(spec),
            "sets": {set_id: _set_plan_fields(spec, set_id) for set_id in spec["sets"]},
        }
    )


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
