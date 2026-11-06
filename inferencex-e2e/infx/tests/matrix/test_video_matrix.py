"""Video matrix generation uses one job for each server deployment."""

import pytest

from infx.matrix.generate import generate_config_matrix, trim_conc, with_fingerprints
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


VIDEO = ["video-generation"]


def test_video_generation_and_smoke_keep_one_server_job(master):
    [row] = generate_config_matrix(["video"], master, RUNNERS, scenario_types=VIDEO)
    assert (row["gpus"], row["conc-list"], row["spec"], row["set"]) == (2, [1, 2, 4], "clip.yaml", "shared-v1")
    assert row["exp-name"] == "video_video_clip_shared-v1_g2"
    validate_matrix([row])
    assert trim_conc([row])[0]["conc-list"] == [1]
    assert generate_config_matrix(["video"], master, RUNNERS, scenario_types=["fixed-seq-len"]) == []
    assert generate_config_matrix(["video"], master, RUNNERS) == []
    with pytest.raises(ValueError, match="conc-list"):
        validate_matrix([{**row, "conc-list": [1, 1]}])


def test_changelog_entry_for_a_video_config_must_name_video_generation(master, monkeypatch):
    from infx.matrix import plan

    monkeypatch.setattr(plan, "load_config_files", lambda *_, **__: master)
    monkeypatch.setattr(plan, "load_runner_file", lambda *_: RUNNERS)
    entry = {"config-keys": ["video"], "description": ["Add a video deployment"],
             "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1"}

    def build(**fields):
        return plan.build_plan([{**entry, **fields}], base_ref="base", head_ref="head",
                               config_files=["configs/master.yaml"])

    for fields in ({}, {"scenario-type": ["fixed-seq-len"]}):
        with pytest.raises(ValueError, match=r"add video-generation.*plan no jobs: \['video'\]"):
            build(**fields)
    build(**{"config-keys": ["*"]})


def test_video_cannot_upload_without_a_renderable_recipe_fingerprint(master, tmp_path):
    [row] = generate_config_matrix(["video"], master, RUNNERS, scenario_types=VIDEO)
    with pytest.raises(ValueError, match="serve.yaml"):
        with_fingerprints([row], tmp_path)
