import copy
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml
from hypothesis import event, given, strategies as st

from infx.matrix.generate import FullSweepOptions, expand_full_sweep, generate_config_matrix
from infx.matrix.plan import append_only_delta, build_plan
from infx.workflows import benchmark_schema
from infx.workflows.ci_priority import PriorityContext, annotate_jobs, load_policy

from cases import CONCURRENCIES, ROOT, TEXT, recipe, rows_in_plan


@pytest.mark.parametrize("topology", ["single", "aggregate", "disaggregate"])
@pytest.mark.parametrize("mode", ["normal", "trim", "all-evals", "evals-only"])
@given(concs=CONCURRENCIES, offload=st.booleans(), image=TEXT,
       scenarios=st.sets(st.sampled_from(["fixed-seq-len", "agentic-coding"]), min_size=1),
       duplicate_entry=st.booleans())
def test_plan_to_validated_scheduled_matrix(topology, mode, concs, offload, image, scenarios, duplicate_entry):
    master, runners = recipe(topology, concs, offload)
    master["fixture"]["image"] = image
    entry = {"config-keys": ["fixture"], "description": ["Fuzz fixture"],
             "pr-link": "https://github.com/example/project/pull/7", "scenario-type": sorted(scenarios)}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        configs, inventory = root / "master.yaml", root / "runners.yaml"
        configs.write_text(yaml.safe_dump(master))
        inventory.write_text(yaml.safe_dump(runners))
        plan = build_plan([entry] * (2 if duplicate_entry else 1), base_ref="base", head_ref="head",
                          config_files=[str(configs)], runner_config=str(inventory),
                          trim=mode == "trim", all_evals=mode == "all-evals", evals_only=mode == "evals-only")
    payload = json.loads(plan.model_dump_json(by_alias=True, exclude_none=True))
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(output):
        patch.setattr("sys.argv", ["benchmark_schema", "--plan"])
        patch.setattr("sys.stdin", io.StringIO(raw))
        benchmark_schema.main()
    assert output.getvalue() == raw
    family = "single_node" if topology == "single" else "multi_node"
    for scenario in scenarios:
        bucket = "agentic" if scenario == "agentic-coding" else "8k1k"
        actual = [conc for row in payload[family].get(bucket, [])
                  for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])]
        expected = [] if mode == "evals-only" else [min(concs)] if mode == "trim" else concs
        assert sorted(actual) == sorted(expected)
    context = PriorityContext(event_name="pull_request", queue_namespace="fuzz:1",
                              labels=frozenset({"skip_queue"}), pr_number=7)
    policy = load_policy(ROOT / "configs/ci-priority.yaml")
    scheduled = annotate_jobs(payload, policy, context)
    scheduled_rows = list(rows_in_plan(scheduled))
    tokens = [row["queue-token"] for row in scheduled_rows]
    assert len(tokens) == len(set(tokens))
    assert all(row["image"] == image for row in scheduled_rows)
    assert all(row["node-count"] == (2 if topology == "disaggregate" else 1)
               for row in scheduled_rows if topology != "single")
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    assert annotate_jobs(reordered, policy, context) == scheduled
    next_run = annotate_jobs(payload, policy, PriorityContext(queue_namespace="fuzz:2"))
    assert set(tokens).isdisjoint(row["queue-token"] for row in rows_in_plan(next_run))
    event(f"scenarios={','.join(sorted(scenarios))};offload={offload};duplicate={duplicate_entry}")


@pytest.mark.parametrize("topology", ["single", "aggregate", "disaggregate"])
@given(low=st.integers(1, 63), kept=st.integers(64, 128), high=st.integers(129, 256),
       scenario=st.sampled_from(["fixed-seq-len", "agentic-coding"]),
       offload=st.booleans())
def test_full_sweep_filters_concurrency_without_mutating_recipes(topology, low, kept, high, scenario, offload):
    master, runners = recipe(topology, [low, kept, high], offload)
    source = copy.deepcopy((master, runners))
    rows = expand_full_sweep(master, runners, options=FullSweepOptions(
        scenario_types=[scenario], min_conc=64, max_conc=128))
    assert [conc for row in rows for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])] == [kept]
    assert (master, runners) == source


@pytest.mark.parametrize("topology", ["single", "aggregate", "disaggregate"])
@given(concs=CONCURRENCIES, additions=st.lists(st.integers(257, 512), min_size=1, max_size=4, unique=True),
       scenario=st.sampled_from(["fixed-seq-len", "agentic-coding"]))
def test_append_only_preserves_existing_points_and_rejects_replacements(topology, concs, additions, scenario):
    before, runners = recipe(topology, concs)
    after, _ = recipe(topology, concs + additions)
    generate = lambda master: generate_config_matrix(["fixture"], master, runners,
                                                     scenario_types=[scenario], eval_mode="none")
    old, new = generate(before), generate(after)
    delta = append_only_delta(old, new)
    points = [conc for row in delta for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])]
    assert sorted(points) == sorted(additions)
    after["fixture"]["image"] = "different-image"
    with pytest.raises(ValueError):
        append_only_delta(old, generate(after))


@pytest.mark.parametrize("topology", ["single", "aggregate", "disaggregate"])
@given(concs=CONCURRENCIES, scenario=st.sampled_from(["fixed-seq-len", "agentic-coding"]),
       damage=st.sampled_from(["image", "conc", "parallelism", "unknown"]))
def test_schema_rejects_invalid_rows_before_emitting_any_output(topology, concs, scenario, damage):
    master, runners = recipe(topology, concs)
    rows = generate_config_matrix(["fixture"], master, runners, scenario_types=[scenario], eval_mode="none")
    invalid = copy.deepcopy(rows[-1])
    if damage == "image":
        invalid["image"] = 123
    elif damage == "conc":
        invalid["conc"] = True if topology == "single" else [True]
    elif damage == "parallelism":
        worker = invalid if topology == "single" else invalid["prefill"]
        worker["pp"] = 0
    else:
        invalid["typo-field"] = "value"
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(output):
        patch.setattr("sys.argv", ["benchmark_schema"])
        patch.setattr("sys.stdin", io.StringIO(json.dumps([*rows, invalid])))
        with pytest.raises(SystemExit) as error:
            benchmark_schema.main()
    assert error.value.code == 2
    assert output.getvalue() == ""
