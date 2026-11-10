"""A published curve's rows are pooled per dashboard model and bound to their matrix points."""

import argparse
import contextlib
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from infx.results import published_curve as curve

MODELS = [
    {"name": "GLM-5.2", "dbKeys": ["glm5.2", "glm5.3"], "category": "default"},
    {"name": "GLM-5.3-Flash", "dbKeys": ["glm5.3flash"], "category": "experimental"},
]
RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39187209967"


@pytest.mark.parametrize("name,display,key", [
    ("GLM-5.3", "GLM-5.2", "glm5.3"),
    ("glm5.3", "GLM-5.2", "glm5.3"),
    ("GLM-5.2", "GLM-5.2", "glm5.2"),
    ("GLM-5.3-Flash", "GLM-5.3-Flash", "glm5.3flash"),
])
def test_family_names_the_dashboard_model_that_pools_a_model(name, display, key):
    found = curve.family(MODELS, name)

    assert (found["display_model"], found["model_key"]) == (display, key)
    entry = next(m for m in MODELS if m["name"] == display)
    assert (found["pooled_models"], found["category"]) == (entry["dbKeys"], entry["category"])


def test_family_refuses_a_model_the_dashboard_does_not_show():
    with pytest.raises(ValueError, match="GLM-9"):
        curve.family(MODELS, "GLM-9")


def test_family_rows_keep_every_curve_of_the_pooled_models_in_curve_order():
    def row(id, conc, **fields):
        return {"id": id, "conc": conc, "model": "glm5.2", "hardware": "gb300",
                "framework": "dynamo-sglang", "precision": "fp4",
                "benchmark_type": "agentic_traces", **fields}

    rows = [
        row("a", 64), row("b", 1, model="glm5.3"), row("c", 4, model="glm5.3flash"),
        row("d", 4, hardware="gb200"), row("e", 4, framework="dynamo-trt"),
        row("f", 4, precision="fp8"), row("g", 8, benchmark_type="single_turn", isl=8192),
    ]

    agentic = curve.family_rows(rows, ["glm5.2", "glm5.3"], True)
    every = curve.family_rows(rows, ["glm5.2", "glm5.3"], False)

    assert [r["id"] for r in agentic] == ["d", "b", "a", "f", "e"]
    assert [r["id"] for r in every] == ["d", "b", "a", "g", "f", "e"]
    assert curve.curve_of(rows[6]) == {
        "hardware": "gb300", "framework": "dynamo-sglang", "precision": "fp4",
        "benchmark_type": "single_turn", "isl": 8192, "osl": None,
    }


def point(id, itl, tput, full=True):
    key = "p90_full_response_itl" if full else "p90_itl"
    return {"id": id, "benchmark_type": "agentic_traces", "metrics": {key: itl, "tput_per_gpu": tput}}


def test_interactivity_is_one_over_p90_itl_full_response_first():
    assert curve.interactivity(point("a", 0.004, 1)) == 250
    assert curve.interactivity(point("a", 0.005, 1, full=False)) == 200
    partial = {"benchmark_type": "agentic_traces",
               "metrics": {"p90_itl": 0.005, "p75_full_response_itl": 0.004}}
    assert curve.interactivity(partial) is None
    assert curve.interactivity({"benchmark_type": "agentic_traces", "metrics": {"p90_itl": 0}}) is None


def test_a_fixed_sequence_row_plots_its_published_p90_interactivity():
    row = {"benchmark_type": "single_turn", "metrics": {"p90_itl": 0.004, "p90_intvty": 310.5}}

    assert curve.interactivity(row) == 310.5
    assert curve.interactivity({**row, "metrics": {"p90_itl": 0.004}}) is None


def test_frontier_keeps_rows_no_other_beats_on_interactivity_and_throughput():
    rows = [
        point("slow-big", 0.010, 900), point("mid", 0.005, 500), point("dominated", 0.005, 400),
        point("tie-low", 0.004, 100), point("tie-high", 0.004, 300), point("fast", 0.002, 50),
        point("beaten", 0.008, 450), {"id": "no-metrics", "metrics": {}},
    ]

    assert [row["id"] for row in curve.frontier(rows)] == ["slow-big", "mid", "tie-high", "fast"]


def test_frontier_takes_a_lower_better_latency_axis():
    rows = [
        {"id": id, "metrics": {"p90_e2el": e2el, "tput_per_gpu": tput}}
        for id, e2el, tput in [("fast", 10, 100), ("dominated", 20, 90), ("big", 30, 400),
                               ("no-e2el", None, 999)]
    ]

    assert [row["id"] for row in curve.frontier(rows, ("p90_e2el", "tput_per_gpu"))] == [
        "big", "fast"]
    assert not curve.plotted(rows[3], ("p90_e2el", "tput_per_gpu"))


def test_frontier_axes_name_known_metrics_or_fail_listing_them():
    assert curve.frontier_axes("p90_ttft, tok_per_usd") == ("p90_ttft", "tok_per_usd")
    with pytest.raises(argparse.ArgumentTypeError, match="median_intvty.*p90_e2el.*tok_per_usd"):
        curve.frontier_axes("p90_ttft,dollars")


def test_lines_split_a_curve_by_date_and_a_fixed_sequence_by_spec_method():
    agentic = {"hardware": "b300", "framework": "sglang", "precision": "fp4",
               "benchmark_type": "agentic_traces", "spec_method": "mtp", "curve_date": "2026-10-12"}
    fixed = {**agentic, "benchmark_type": "single_turn", "isl": 8192, "osl": 1024}

    assert curve.line_of(agentic) == curve.line_of({**agentic, "spec_method": "none"})
    assert curve.line_of(agentic) != curve.line_of({**agentic, "curve_date": "2026-10-29"})
    assert curve.line_of(fixed) != curve.line_of({**fixed, "spec_method": "none"})


@pytest.mark.parametrize("fields,gpus", [
    ({"disagg": False, "prefill_tp": 8, "prefill_ep": 8, "num_prefill_gpu": 64}, 8),
    ({"disagg": False, "prefill_tp": 4, "prefill_ep": 1, "num_prefill_gpu": 4}, 4),
    ({"disagg": False, "prefill_tp": 4, "prefill_ep": 4, "num_prefill_gpu": 4}, 4),
    ({"disagg": True, "num_prefill_gpu": 8, "num_decode_gpu": 16}, 24),
    ({"disagg": True, "num_prefill_gpu": 8}, None),
])
def test_physical_gpus_counts_shared_tp_and_ep_gpus_once(fields, gpus):
    assert curve.physical_gpus(fields) == gpus


def test_published_adds_curve_gpus_price_and_engine():
    row = {
        "id": "1", "hardware": "h100", "framework": "dynamo-sglang", "precision": "fp4",
        "benchmark_type": "agentic_traces", "disagg": False, "prefill_tp": 8, "prefill_ep": 8,
        "num_prefill_gpu": 64, "image": "lmsysorg/sglang:nightly-dev-20261016-a93f16f0@sha256:00",
        "metrics": {"tput_per_gpu": 100.0, "total_tput_tps": 800.0, "p90_itl": 0.01},
    }

    entry = curve.published(row, {"h100": {"h": 2.0, "r": 4.0}})

    assert entry["num_prefill_gpu"] == 64 and entry["physical_gpus"] == 8
    assert entry["tput_per_physical_gpu"] == 100.0
    assert (entry["tok_per_usd"], entry["price_key"]) == (180000.0, "h")
    assert entry["curve"]["hardware"] == "h100"
    assert entry["engine"] == {"repository": "sgl-project/sglang", "commit": "a93f16f0"}
    assert curve.published(row, {})["tok_per_usd"] is None


@pytest.mark.parametrize("framework,image,source", [
    ("dynamo-trt", "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc28", ("NVIDIA/TensorRT-LLM", None)),
    ("vllm", "vllm/vllm-openai:nightly-20261019", ("vllm-project/vllm", None)),
    ("atom", "rocm/atom:v1-0123abc", (None, "0123abc")),
])
def test_engine_source_reads_a_commit_but_not_a_date_from_the_image_tag(framework, image, source):
    found = curve.engine_source({"framework": framework, "image": image})
    assert (found["repository"], found["commit"]) == source


def test_model_key_is_the_one_prefix_the_matrix_runs_a_checkpoint_under(tmp_path):
    configs = [tmp_path / "nvidia.yaml", tmp_path / "amd.yaml"]
    configs[0].write_text(yaml.safe_dump({
        "a": {"model": "nvidia/GLM-5.3-NVFP4", "model-prefix": "glm5.3"},
        "b": {"model": "nvidia/GLM-5.2-NVFP4", "model-prefix": "glm5.2"},
        "c": {"model": "org/Shared", "model-prefix": "one"},
    }))
    configs[1].write_text(yaml.safe_dump({"d": {"model": "org/Shared", "model-prefix": "two"}}))

    assert curve.model_key(configs, "NVIDIA/glm-5.3-nvfp4") == "glm5.3"
    assert curve.model_key(configs, "org/Shared") is None
    assert curve.model_key(configs, "org/Missing") is None


def test_hints_name_the_commits_points_and_scripts_near_an_unbound_row(tmp_path):
    for name in ("glm5.2_fp4_gb300_mtp.sh", "glm5.2_fp4_b300.sh", "glm5.3_fp4_gb300.sh"):
        script = tmp_path / "benchmarks/single_node/agentic" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "scripts"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    matrix = [
        {"framework": "dynamo-trt", "runner": "cluster:gb300-nv", "conc": 4, "exp-name": "a",
         "srt-recipe": "r/a.yaml"},
        {"framework": "trt", "runner": "gb300", "conc": 8, "exp-name": "b", "srt-recipe": None},
        {"framework": "dynamo-trt", "runner": "cluster:gb200-nv", "conc": 4, "exp-name": "c"},
        {"framework": "dynamo-sglang", "runner": "cluster:gb300-nv", "conc": 4, "exp-name": "d"},
    ]
    row = {"model": "glm5.2", "precision": "fp4", "hardware": "gb300", "framework": "dynamo-trt"}

    at_four = curve.hints(tmp_path, commit, matrix, {**row, "conc": 4})
    elsewhere = curve.hints(tmp_path, commit, matrix, {**row, "conc": 16})

    assert at_four == {
        "points": [{"exp_name": "a", "recipe": "r/a.yaml"}],
        "scripts": ["benchmarks/single_node/agentic/glm5.2_fp4_gb300_mtp.sh"],
    }
    assert [hint["exp_name"] for hint in elsewhere["points"]] == ["a", "b"]


def test_producer_reads_the_run_head_and_pull_from_workflow_info():
    row = {"run_url": RUN + "/attempts/2", "run_started_at": "2026-10-19T23:50:00Z",
           "date": "2026-10-20"}
    infos = {
        "2026-10-19": {"runConfigs": [{"github_run_id": "1", "head_sha": "f" * 40}]},
        "2026-10-20": {
            "runConfigs": [{"github_run_id": "39187209967", "head_sha": "8f965ad4",
                            "html_url": RUN}] * 2,
            "changelogs": [{"workflow_run_id": "39187209967",
                            "pr_link": "https://github.com/SemiAnalysisAI/InferenceX/pull/3701"}],
        },
    }

    assert curve.run_dates(row) == ["2026-10-19", "2026-10-20"]
    assert curve.producer(row, infos) == {
        "repository": "SemiAnalysisAI/InferenceX", "run_id": "39187209967",
        "commit": "8f965ad4", "pull": "3701",
    }
    assert curve.producer(row, {})["unbound"].startswith("workflow-info lists no")
    assert curve.producer({"run_url": None}, infos) == {"unbound": "no GitHub Actions run URL"}


def test_producer_binds_only_runs_of_the_trusted_repository():
    elsewhere = "https://github.com/example/fork/actions/runs/39187209967"
    infos = {"2026-10-20": {"runConfigs": [
        {"github_run_id": "39187209967", "head_sha": "beef", "html_url": elsewhere},
    ]}}

    assert curve.producer({"run_url": elsewhere, "date": "2026-10-20"}, infos) == {
        "repository": "example/fork", "run_id": "39187209967",
        "unbound": "its run is in example/fork, not SemiAnalysisAI/InferenceX",
    }
    assert "commit" not in curve.producer({"run_url": RUN, "date": "2026-10-20"}, infos)
    assert curve.producer({"run_url": elsewhere, "date": "2026-10-20"}, infos,
                          "example/fork")["commit"] == "beef"


def test_points_generate_only_the_requested_gpus_configs_when_it_has_any(tmp_path):
    configs = tmp_path / "master.yaml"
    configs.write_text(yaml.safe_dump({
        "glm-h100": {"model-prefix": "glm5.2", "precision": "fp8", "runner": "cluster:h100-dgxc"},
        "glm-b300": {"model-prefix": "glm5.2", "precision": "fp8", "runner": "cluster:b300-nv"},
        "glm-fp4": {"model-prefix": "glm5.2", "precision": "fp4", "runner": "cluster:h100-dgxc"},
    }))
    generated = []

    def generate(keys, _):
        generated.append(keys)
        return [{"conc": [1, 2]}]

    revision = SimpleNamespace(master_configs=[str(configs)], generate=generate)

    assert curve.points(revision, ["glm5.2"], "fp8", "h100") == [{"conc": 1}, {"conc": 2}]
    curve.points(revision, ["glm5.2"], "fp8", "gb200")
    assert generated == [["glm-h100"], ["glm-b300", "glm-h100"]]


def test_a_commit_whose_generator_fails_unbinds_its_rows_with_the_generators_error(
        monkeypatch, tmp_path):
    refused = subprocess.CalledProcessError(
        1, ["python", "-m", "infx.matrix.generate"],
        stderr="Traceback ...\nValueError: Runner 'cluster:b300-nv' requires metadata\n")

    @contextlib.contextmanager
    def snapshot(_):
        yield SimpleNamespace()

    def points(*_):
        raise refused

    monkeypatch.setattr(curve, "fetch_commit", lambda *_: None)
    monkeypatch.setattr(curve, "snapshot", snapshot)
    monkeypatch.setattr(curve, "points", points)
    producers = {"1": {"repository": curve.INFERENCEX, "pull": None}}

    _, unbound, _ = curve.bind_commits({"0" * 40: [{"id": 1}]}, producers, ["glm5.2"], "fp4",
                                       "b300", tmp_path)

    assert unbound == {"1": "the commit's matrix cannot be generated: "
                            "ValueError: Runner 'cluster:b300-nv' requires metadata"}


def test_bind_matches_rows_to_points_by_fingerprint_and_concurrency():
    matrix = [
        {"recipe-fingerprint": "agg", "conc": 4, "exp-name": "tp4"},
        {"recipe-fingerprint": "agg", "conc": 8, "exp-name": "tp4"},
        {"recipe-fingerprint": "disagg", "conc": 4, "exp-name": "1p1d"},
    ]
    rows = [
        {"id": 1, "conc": 8, "recipe_fingerprint": "agg"},
        {"id": 2, "conc": "4", "recipe_fingerprint": "disagg"},
        {"id": 3, "conc": 4, "recipe_fingerprint": "agg", "offload_mode": "off"},
        {"id": 4, "conc": 16, "recipe_fingerprint": "agg"},
        {"id": 5, "conc": 4, "recipe_fingerprint": None},
    ]

    matched, unbound = curve.bind(rows, matrix)

    assert [(point["exp-name"], point["conc"], ids) for point, ids in matched] == [
        ("tp4", 8, ["1"]), ("1p1d", 4, ["2"]), ("tp4", 4, ["3"]),
    ]
    assert unbound == {
        "4": "no matrix point has its fingerprint and concurrency",
        "5": "the row has no recipe fingerprint",
    }
    assert curve.bind(rows[:1], [{"conc": 8}])[1] == {
        "1": "its commit's matrix points carry no recipe fingerprints",
    }


def test_write_variant_keeps_the_selected_variant_and_the_committed_file(tmp_path):
    point = {"recipe-fingerprint": "agg", "conc": 4, "image": "img", "model": "nvidia/GLM-5.2-NVFP4",
             "model-prefix": "glm5.2"}
    found = {"path": "benchmarks/recipes/agg-variants.yaml", "selector": "override_tp4_c4",
             "text": "base: {}\noverride_tp4_c4: {}\n", "recipe": {"name": "tp4", "tp": 4}}

    entry = curve.write_variant(tmp_path, "c0ffee", point, found, ["7"])

    assert entry["file"] == "recipes/override_tp4_c4-4d9f8676.yaml"
    assert entry["source_file"] == "sources/c0ffee/benchmarks/recipes/agg-variants.yaml"
    assert (entry["model"], entry["model_prefix"], entry["row_ids"]) == (
        "nvidia/GLM-5.2-NVFP4", "glm5.2", ["7"],
    )
