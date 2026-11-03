"""AgentX server-reuse identities produced from bound single-node recipes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from infx.bench.agentic import reuse
from infx.bench.env import InputError
from infx.srt_slurm.reuse_identity import identities, main
from infx.srt_slurm.workload import bind_workload

IMAGE = "vllm/vllm-openai@sha256:" + "2b" * 32
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
RECIPE = {
    "name": "dsv41-flash-tp1-dp2-ep2",
    "model": {},
    "engine": {"type": "vllm", "connector": None},
    "frontend": {"type": "vllm", "enable_multiple_frontends": False},
    "roles": {
        "agg": {
            "gpus": 2, "nodes": 1, "workers": 1,
            "args": {
                "tensor-parallel-size": 1, "data-parallel-size": 2,
                "enable-expert-parallel": True, "max-num-seqs": 16,
                "compilation-config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
            },
            "env": {"VLLM_SERVER_DEV_MODE": "1"},
        },
    },
    "benchmark": {"type": "custom", "command": "bash srt_agentic.sh", "env": {}},
}  # fmt: skip


def _bound(conc: int, image: str = IMAGE) -> dict:
    environment = {"IMAGE": image, "MODEL": MODEL, "PRECISION": "fp4", "CONC": str(conc)}
    return bind_workload(RECIPE, environment, agentic=True, multinode=False)


def test_points_bound_from_one_recipe_share_one_server_identity(tmp_path):
    paths = {}
    for conc in (8, 16):
        paths[conc] = tmp_path / f"c{conc}" / "recipe.yaml"
        paths[conc].parent.mkdir()
        paths[conc].write_text(yaml.safe_dump(_bound(conc), sort_keys=False))
    output = tmp_path / "identities.json"

    main(["--point", f"8={paths[8]}", "--point", f"16={paths[16]}", "--output", str(output)])

    document = json.loads(output.read_text())
    assert document["schema"] == "infx-agentx-server-reuse/v1"
    assert set(document["points"]) == {"8", "16"}
    point = document["points"]["8"]
    assert (point["image"], point["model"]) == (IMAGE, MODEL)
    assert point["server"]["roles"]["agg"]["args"]["max-num-seqs"] == 16
    assert {"benchmark", "name"}.isdisjoint(point["server"])
    assert document["points"]["16"] == point
    assert reuse.check_identities(document, (8, 16), MODEL, IMAGE) == reuse.check_identities(
        document, (16, 8), MODEL, None
    )


def test_different_images_are_refused_in_both_recipe_and_identity():
    with pytest.raises(
        InputError,
        match=re.escape("declare different server launches (image, server.model.container)"),
    ):
        identities({8: _bound(8), 16: _bound(16, image="vllm/vllm-openai:other")})


def test_a_recipe_bound_to_another_concurrency_is_refused():
    with pytest.raises(ValueError, match="the recipe given for point 16 was bound to CONC=8"):
        identities({8: _bound(8), 16: _bound(8)})


def test_the_command_reports_an_unusable_point_without_writing(tmp_path, capsys):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(_bound(8)))
    output = tmp_path / "identities.json"

    with pytest.raises(SystemExit) as exit_info:
        main(["--point", f"8={recipe}", "--point", f"16={recipe}", "--output", str(output)])

    assert exit_info.value.code == 2
    assert "bound to CONC=8" in capsys.readouterr().err
    assert not output.exists()
