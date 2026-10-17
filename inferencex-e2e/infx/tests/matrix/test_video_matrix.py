"""Video-generation rows of the vendor master configs: one job per deployment."""

import copy
import re

import pytest
from pydantic import ValidationError

from infx.config import MASTER_CONFIGS, RUNNER_CONFIG, repository_root
from infx.matrix.generate import generate_config_matrix, trim_conc
from infx.srt_slurm.video import load_spec
from infx.workflows.benchmark_schema import validate_matrix
from infx.matrix.validation import (
    SingleNodeMasterConfigEntry,
    load_config_files,
    load_runner_file,
    validate_video_matrix_entry,
)

ROOT = repository_root()
H200 = "h3-bf16-h200-sglang-video"


@pytest.fixture(scope="module")
def masters():
    return load_config_files([str(ROOT / path) for path in MASTER_CONFIGS])


@pytest.fixture(scope="module")
def runners():
    return load_runner_file(str(ROOT / RUNNER_CONFIG))


def test_the_h200_video_config_generates_one_job_for_every_concurrency(masters, runners):
    [row] = generate_config_matrix([H200], masters, runners, root=ROOT)
    assert row["scenario-type"] == "video-generation"
    assert row["runner"] == "cluster:h200-dgxc" and row["gpus"] == 2
    assert row["conc-list"] == [1, 2, 4] and row["spec"] == "h3-serving-2gpu.json"
    assert row["srt-recipe"].endswith("srt-slurm-recipes/h3/sglang/h200-bf16/serving-client.yaml")
    assert row["exp-name"] == "h3_video_h3-serving-2gpu_g2"
    assert "tp" not in row and "isl" not in row
    assert validate_video_matrix_entry(row) == row


def test_every_video_config_targets_one_cluster_and_the_same_frozen_workload(masters, runners):
    keys = sorted(key for key in masters if masters[key]["scenarios"].get("video-generation"))
    assert keys == [
        "h3-bf16-b200-sglang-video", "h3-bf16-h100-sglang-video",
        "h3-bf16-h200-sglang-video", "h3-bf16-mi355x-sglang-video",
    ]  # fmt: skip
    rows = generate_config_matrix(keys, masters, runners, root=ROOT)
    assert len(rows) == len(keys)
    assert all(row["runner"].startswith("cluster:") for row in rows)
    # A SKU may pin a different interpreter for its runtime image, but every point must
    # serve the identical frozen workload: same plan, model, server settings and limits.
    specs = {}
    for row in rows:
        spec = load_spec(row["spec"], root=ROOT)
        for role in ("baseline", "candidate"):
            spec[role].pop("python")
        specs[row["spec"]] = spec
    assert {"h3-serving-2gpu.json", "h3-serving-2gpu-b200.json"} <= set(specs)
    reference = specs["h3-serving-2gpu.json"]
    assert all(spec == reference for spec in specs.values())


def test_scenario_type_filters_video_rows(masters, runners):
    assert generate_config_matrix([H200], masters, runners, scenario_types=["fixed-seq-len"], root=ROOT) == []
    [row] = generate_config_matrix([H200], masters, runners, scenario_types=["video-generation"], root=ROOT)
    assert row["conc-list"] == [1, 2, 4]


def test_video_rows_keep_scalars_where_the_benchmark_template_reads_config_fields(masters, runners):
    """benchmark-tmpl.yml copies ``fromJSON(inputs.config).<field>`` into env values and its job
    name; a list there fails the job at "Set up job" (run 39001040568)."""
    template = (ROOT.parent / ".github/workflows/benchmark-tmpl.yml").read_text()
    scalar_reads = set(re.findall(r"(?:\$\{\{|\|\|) fromJSON\(inputs\.config\)\.([a-z-]+) \}\}", template))
    assert {"conc", "isl", "model-prefix"} <= scalar_reads
    [row] = generate_config_matrix([H200], masters, runners, root=ROOT)
    nested = {field: row[field] for field in scalar_reads if isinstance(row.get(field), (list, dict))}
    assert nested == {}
    assert trim_conc([row])[0]["conc-list"] == [1]


def test_the_generated_video_row_passes_the_get_jobs_schema_gate(masters, runners):
    """get-jobs pipes the generated matrix through infx.workflows.benchmark_schema (run 39001504768)."""
    [row] = generate_config_matrix([H200], masters, runners, root=ROOT)
    validate_matrix([row])
    with pytest.raises(ValueError, match="conc-list"):
        validate_matrix([{**row, "conc-list": [1, 1]}])


def test_video_configs_require_a_cluster_runner_and_a_spec_file(masters):
    config = copy.deepcopy(masters[H200])
    config["runner"] = "h200"
    with pytest.raises(ValidationError, match="cluster"):
        SingleNodeMasterConfigEntry(**config)
    config = copy.deepcopy(masters[H200])
    config["scenarios"]["video-generation"][0]["spec"] = "../specs/h3-serving-2gpu.json"
    with pytest.raises(ValidationError, match="specs"):
        SingleNodeMasterConfigEntry(**config)


def test_video_rows_cannot_be_multinode(masters, runners):
    config = copy.deepcopy(masters)
    config[H200]["multinode"] = True
    with pytest.raises((ValidationError, ValueError), match="multinode|one node"):
        generate_config_matrix([H200], config, runners, root=ROOT)
