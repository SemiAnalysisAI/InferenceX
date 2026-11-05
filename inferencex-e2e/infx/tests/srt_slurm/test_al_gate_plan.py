"""Plan the report-only acceptance gate of an agentic speculative eval from its resolved recipe."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from infx.golden_al_distribution import GOLDEN_DIR
from infx.srt_slurm import al_gate
from infx.srt_slurm.synthetic_acceptance import plan_commands

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides  # noqa: E402

DSV4 = "deepseek-v4-pro__mtp__native__default-default"
EVAL = {
    "MODEL_PREFIX": "dsv4", "FRAMEWORK": "vllm", "IS_AGENTIC": "1", "SPEC_DECODING": "mtp",
    "EVAL_ONLY": "true", "RUN_EVAL": "true", "THINKING_MODE": "thinking_on",
    "RESULT_FILENAME": "point", "EVAL_FRAMEWORK": "lm-eval", "RUNNER_TYPE": "cluster:test",
    "AL_GATE_DATASET": "al_gate_dataset/speedbench_coding.jsonl", "EVAL_AL_GATE": "true",
}  # fmt: skip
COLLECTION = {
    "renderer": "chat_template",
    "modes": {"thinking_on": {"chat_template_kwargs": None, "sampling": {"temperature": 1.0}}},
}


@pytest.fixture
def golden_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "golden"
    directory.mkdir()
    (directory / "identities.yaml").write_text((GOLDEN_DIR / "identities.yaml").read_text())
    curve = {"deepseek-v4-pro": {"thinking_on": {3: 2.4}}, "collection": COLLECTION}
    (directory / f"{DSV4}.yaml").write_text(yaml.safe_dump(curve))
    glm = {"glm-5.3-nvfp4": {"thinking_on": {3: 2.94}}}
    (directory / "glm-5.3-nvfp4__mtp__native__default-default.yaml").write_text(yaml.safe_dump(glm))
    for name in ("probabilistic-block", "probabilistic-default"):
        kimi = {"kimi-k3": {"thinking_on": {3: 3.0}}, "collection": COLLECTION}
        (directory / f"kimi-k3__dspark__kimi-k3-dspark__{name}.yaml").write_text(yaml.safe_dump(kimi))
    return directory


def vllm_recipe(spec: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
    if spec:
        args["speculative-config"] = json.dumps(spec)
    return {"roles": {"agg": {"args": args}}, "benchmark": {"type": "custom"}, "telemetry": {}}


def apply_native(recipe: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(recipe)
    sets = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--set"]
    unsets = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--unset"]
    apply_overrides_to_recipe(result, parse_overrides(sets, unsets))
    return result


MTP3 = {"method": "mtp", "num_speculative_tokens": 3}


def test_a_drafting_eval_recipe_plans_the_gate_of_its_golden_curve(golden_dir):
    gate = al_gate.plan(vllm_recipe(MTP3), "vllm", EVAL, golden_dir=golden_dir)

    assert gate == {"curve": DSV4, "draft_tokens": 3, "golden_acceptance_length": 2.4}


@pytest.mark.parametrize(
    ("verify", "curve"),
    [
        ({}, "kimi-k3__dspark__kimi-k3-dspark__probabilistic-default"),
        ({"rejection_sample_method": "block"}, "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block"),
    ],
    ids=["standard", "block"],
)
def test_the_gate_measures_the_curve_of_the_recipes_verifier(golden_dir, verify, curve):
    spec = {"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"}
    recipe = vllm_recipe({**spec, **verify}, **{"kv-cache-dtype": "fp8"})

    gate = al_gate.plan(recipe, "vllm", {**EVAL, "MODEL_PREFIX": "kimik3"}, golden_dir=golden_dir)

    assert gate == {"curve": curve, "draft_tokens": 3, "golden_acceptance_length": 3.0}


def test_a_dynamo_trtllm_decode_role_plans_its_gate(golden_dir):
    # srtctl publishes TRT-LLM metrics by default; a recipe without them reports no-counters.
    spec = {"decoding_type": "MTP", "max_draft_len": 3}
    recipe = {"roles": {"prefill": {}, "decode": {"args": {"speculative_config": spec}}}}

    gate = al_gate.plan(recipe, "dynamo-trt", EVAL, golden_dir=golden_dir)

    assert gate == {"curve": DSV4, "draft_tokens": 3, "golden_acceptance_length": 2.4}


def test_a_recipe_that_does_not_draft_has_no_gate(golden_dir):
    assert al_gate.plan(vllm_recipe(), "vllm", EVAL, golden_dir=golden_dir) is None


@pytest.mark.parametrize(
    ("framework", "prefix", "reason"),
    [
        ("tilert", "dsv4", "TileRT"),
        ("vllm", "glm5.3", "records no collection to replay"),
        ("vllm", "minimaxm3", "no golden AL"),
    ],
)
def test_a_gate_that_cannot_run_is_skipped_with_its_reason(golden_dir, framework, prefix, reason):
    environment = {**EVAL, "MODEL_PREFIX": prefix}

    gate = al_gate.plan(vllm_recipe(MTP3), framework, environment, golden_dir=golden_dir)

    assert (gate["status"], gate["result_filename"], gate["eval_framework"]) == (
        "skipped", "point", "lm-eval",
    )  # fmt: skip
    assert reason in gate["error"]


def dp_recipe(dp: int, tp: int, gpus_per_node: int | None, pp: int = 1) -> dict[str, Any]:
    """A vLLM recipe whose generation worker runs ``dp`` replicas of TP x PP GPUs."""
    args = {"data-parallel-size": dp, "tensor-parallel-size": tp, "pipeline-parallel-size": pp}
    recipe = vllm_recipe(MTP3, **args)
    if gpus_per_node is not None:
        recipe["resources"] = {"gpus_per_node": gpus_per_node}
    return recipe


@pytest.mark.parametrize(
    ("framework", "recipe", "cluster_gpus", "skipped"),
    [
        # Eight data-parallel ranks over two 4-GPU nodes run as two processes.
        ("dynamo-vllm", dp_recipe(dp=8, tp=1, gpus_per_node=4), 8, True),
        ("dynamo-vllm", dp_recipe(dp=2, tp=2, pp=2, gpus_per_node=4), 8, True),
        # One 8-GPU node runs a single process for every rank.
        ("dynamo-vllm", dp_recipe(dp=8, tp=1, gpus_per_node=8), 4, False),
        # A replica wider than a node keeps a single API leader for every rank.
        ("dynamo-vllm", dp_recipe(dp=2, tp=8, gpus_per_node=4), 4, False),
        # Without resources.gpus_per_node, srtctl uses the cluster's node size.
        ("dynamo-vllm", dp_recipe(dp=8, tp=1, gpus_per_node=None), 4, True),
        # Only Dynamo hands the benchmark one leader's metrics.
        ("vllm-disagg", dp_recipe(dp=8, tp=1, gpus_per_node=4), 4, False),
    ],
)
def test_dynamo_vllm_data_parallel_workers_across_nodes_are_skipped(
    tmp_path, monkeypatch, golden_dir, framework, recipe, cluster_gpus, skipped
):
    cluster = {"gpus-per-node": cluster_gpus, "arch": "x86_64", "scheduler": "slurm",
               "slurm": {"partition": "batch", "exclusive": True}}  # fmt: skip
    runners = {"labels": {"cluster:test": ["test_0"]}, "clusters": {"test": cluster}}
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump(runners))
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))

    gate = al_gate.plan(recipe, framework, EVAL, golden_dir=golden_dir)

    assert (gate.get("status") == "skipped") is skipped
    if skipped:
        assert "only the leader's metrics" in gate["error"]


def test_the_gate_runs_as_the_benchmark_stage_before_the_eval(tmp_path, golden_dir):
    recipe = vllm_recipe(MTP3)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))

    [planned] = plan_commands(str(path), "vllm", [], EVAL, golden_dir=golden_dir)

    assert planned.acceptance_length is None  # the eval's server verifies for real
    assert planned.al_gate["curve"] == DSV4
    assert dict(planned.environment) == {"EVAL_ONLY": "false", "RUN_EVAL": "true"}
    result = apply_native(recipe, planned.command)
    assert "telemetry" not in result
    assert result["benchmark"] == {
        "type": "custom",
        "command": "bash /infmax-workspace/benchmarks/srt_al_gate.sh",
        "env": {
            "AL_GATE": "true", "AL_GATE_CURVE": DSV4, "AL_GATE_DRAFT_TOKENS": "3",
            "AL_GATE_DATASET": "al_gate_dataset/speedbench_coding.jsonl",
            "AL_GATE_OUTPUT_DIR": "/logs", "THINKING_MODE": "thinking_on",
            "EVAL_FRAMEWORK": "lm-eval",
        },
    }  # fmt: skip


@pytest.mark.parametrize("environment", [{"EVAL_AL_GATE": ""}, {"MODEL_PREFIX": "glm5.3"}])
def test_an_eval_without_a_runnable_gate_is_submitted_unchanged(
    tmp_path, golden_dir, environment
):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(vllm_recipe(MTP3)))

    [planned] = plan_commands(str(path), "vllm", [], {**EVAL, **environment}, golden_dir=golden_dir)

    assert planned.environment == ()
    assert "benchmark" not in " ".join(planned.command)


def test_the_summary_tabulates_every_downloaded_gate(tmp_path, capsys):
    measured = {
        "result_filename": "b", "eval_framework": "lm-eval", "eval_suite": None,
        "curve": DSV4, "draft_tokens": 3, "golden_acceptance_length": 2.4,
        "acceptance_length": 2.2, "ratio": 0.9167, "status": "below", "error": None,
    }  # fmt: skip
    skipped = al_gate.skipped("no SPEED-Bench prompts at ''", {"RESULT_FILENAME": "a"})
    (tmp_path / "eval_x").mkdir()
    (tmp_path / "eval_x" / "al_gate_b.json").write_text(json.dumps(measured))
    (tmp_path / "al_gate_a.json").write_text(json.dumps(skipped))
    (tmp_path / "eval_x" / "al_gate_c.json").write_text("{")

    assert al_gate.main(["summarize", str(tmp_path)]) == 0

    assert capsys.readouterr().out.splitlines()[-2:] == [
        "| `a` | - | - | - | - | - | - | skipped: no SPEED-Bench prompts at '' |",
        f"| `b` | lm-eval | {DSV4} | 3 | 2.40 | 2.20 | 0.917 | below |",
    ]
    assert al_gate.summarize([]) == ""
