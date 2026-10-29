"""Video matrix generation uses one job for each server deployment."""

import copy

import pytest
from pydantic import ValidationError

from infx.matrix.generate import generate_config_matrix, trim_conc, with_fingerprints
from infx.matrix.validation import SingleNodeMasterConfigEntry
from infx.workflows.benchmark_schema import validate_matrix


@pytest.fixture
def master():
    return {"video": {"image": "example/runtime:1", "model": "org/video", "model-prefix": "video",
        "precision": "bf16", "framework": "sglang", "runner": "cluster:fixture", "multinode": False,
        "srt-recipe-dir": "video", "scenarios": {"video-generation": [{"spec": "clip.yaml",
        "set": "shared-v1",
        "search-space": [{"gpus": 2, "conc-list": [1, 2, 4], "srt-recipe": "serve.yaml"}]}]}}}


RUNNERS = {"labels": {"cluster:fixture": ["node"]}, "clusters": {"fixture": {
    "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "slurm": {"partition": "batch"}}}}


def test_video_generation_and_smoke_keep_one_server_job(master):
    [row] = generate_config_matrix(["video"], master, RUNNERS)
    assert (row["gpus"], row["conc-list"], row["spec"], row["set"]) == (2, [1, 2, 4], "clip.yaml", "shared-v1")
    assert row["exp-name"] == "video_video_clip_shared-v1_g2"
    validate_matrix([row])
    assert trim_conc([row])[0]["conc-list"] == [1]
    assert generate_config_matrix(["video"], master, RUNNERS, scenario_types=["fixed-seq-len"]) == []
    with pytest.raises(ValueError, match="conc-list"):
        validate_matrix([{**row, "conc-list": [1, 1]}])


@pytest.mark.parametrize("edit,match", [
    ({"runner": "fixture"}, "cluster"), ({"multinode": True}, "multinode|one node")])
def test_video_configs_require_one_cluster(master, edit, match):
    config = {**master["video"], **edit}
    with pytest.raises((ValidationError, ValueError), match=match):
        SingleNodeMasterConfigEntry(**config)
        generate_config_matrix(["video"], {"video": config}, RUNNERS)


@pytest.mark.parametrize("edit,match", [
    ({"spec": "../clip.yaml"}, "specs"), ({"spec": "clip.json"}, "YAML"), ({"set": None}, "set")])
def test_video_scenarios_name_a_yaml_workload_and_its_set(master, edit, match):
    config = copy.deepcopy(master["video"])
    scenario = config["scenarios"]["video-generation"][0]
    for key, value in edit.items():
        if value is None:
            del scenario[key]
        else:
            scenario[key] = value
    with pytest.raises(ValidationError, match=match):
        SingleNodeMasterConfigEntry(**config)


def test_video_cannot_upload_without_a_renderable_recipe_fingerprint(master, tmp_path):
    [row] = generate_config_matrix(["video"], master, RUNNERS)
    with pytest.raises(ValueError, match="serve.yaml"):
        with_fingerprints([row], tmp_path)
