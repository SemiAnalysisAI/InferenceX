import json

import pytest
import yaml

from infx.serving import scenarios
from infx.serving.to_matrix import build, main, spec_decoding

ENGINE = """\
name: qwen-tp1
runner: cluster:hcm-hgx02
image: vllm/vllm-openai:v0.29.0
model: Qwen/Qwen3.8-27B
gpus: 1
serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144 --port 9000
"""
ROW_KEYS = {
    "image", "model", "model-prefix", "precision", "framework", "runner", "tp", "pp", "dcp-size",
    "pcp-size", "ep", "dp-attn", "spec-decoding", "conc", "kv-offloading", "total-cpu-dram-gb",
    "duration", "exp-name", "scenario-type",
}


def test_one_point_per_conc_with_the_rendered_scenario():
    result = build(ENGINE + "scenario: agentx\noverrides: {duration: 180}\nconc: [8, 16]\n")
    points = result["points"]
    assert [p["config"]["exp-name"] for p in points] == ["qwen-tp1_c8", "qwen-tp1_c16"]
    assert set(points[0]["config"]) == ROW_KEYS
    assert points[0]["config"]["duration"] == 180 and points[0]["config"]["tp"] == 1
    serving = json.loads(points[0]["serving"])
    assert serving["scenario_label"] == "agentx-cc-062126-smoke"
    assert serving["search"] is False and serving["dataset_kind"] == "public"
    assert yaml.safe_load(serving["scenario_yaml"]) == scenarios.render("agentx", {"duration": 180}).doc
    assert "--port" not in serving["serve_cmd"]
    assert result["title"] == "gn-benchmark qwen-tp1 agentx-cc-062126-smoke (180s) c8,16 cluster:hcm-hgx02"


def test_grid_is_one_job_per_point():
    points = build(ENGINE + "scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: 50, search: true}\nconc: 1\n")["points"]
    assert [p["config"]["exp-name"] for p in points] == [
        "qwen-tp1-sweep-isl8k-osl1k-cache50", "qwen-tp1-sweep-isl32k-osl1k-cache50"]
    assert all(json.loads(p["serving"])["search"] for p in points)


def test_single_search_point_keeps_the_search_name():
    [point] = build(ENGINE + "scenario: synthetic\nworkload: {isl: 8k, osl: 1k, cache: 0, search: true}\nconc: 1\n")["points"]
    assert point["config"]["exp-name"] == "qwen-tp1_search"


def test_cli_check_and_errors(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text(ENGINE + "scenario: tau2\nconc: 4\n")
    assert main(["--check", str(good)]) == 0
    assert capsys.readouterr().out.startswith("gn-benchmark qwen-tp1 custom-agentic-tau2")
    bad = tmp_path / "bad.yaml"
    bad.write_text(ENGINE + "scenario: tau2\nconc: 4\noverrides: {nope: 1}\n")
    assert main([str(bad)]) == 2
    assert "ERROR overrides: unknown key(s) nope" in capsys.readouterr().err


REMOTE = """\
name: gpt5-remote
runner: cluster:hcm-rtx01
endpoint: {url: "https://api.openai.com/v1", model: gpt-5, api-key-secret: OPENAI_API_KEY}
scenario: agentx
overrides: {duration: 180}
conc: [4, 8]
"""


def test_remote_point_runs_the_client_image_against_the_endpoint():
    points = build(REMOTE)["points"]
    assert [p["config"]["exp-name"] for p in points] == ["gpt5-remote_c4", "gpt5-remote_c8"]
    row = points[0]["config"]
    assert set(row) == ROW_KEYS
    assert row["image"] == "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
    assert (row["model"], row["framework"], row["precision"], row["tp"]) == ("gpt-5", "remote", "remote", 1)
    assert points[0]["api_key_secret"] == "OPENAI_API_KEY"
    serving = json.loads(points[0]["serving"])
    assert serving["remote_url"] == "https://api.openai.com"
    assert serving["tokenizer"] == "builtin"
    assert serving["api_key_secret"] == "OPENAI_API_KEY"
    assert serving["serve_cmd"] == "" and serving["router_cmd"] == ""


def test_engine_point_carries_no_remote_fields():
    point = build(ENGINE + "scenario: tau2\nconc: 4\n")["points"][0]
    serving = json.loads(point["serving"])
    assert (serving["remote_url"], serving["tokenizer"], point["api_key_secret"]) == ("", "", "")


@pytest.mark.parametrize("serve, expected", [
    ("", "none"),
    ("vllm serve m --tensor-parallel-size 1", "none"),
    ("vllm serve m --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":6}'", "mtp"),
    ("vllm serve m --speculative-config '{\"method\":\"eagle3\",\"model\":\"d\"}'", "draft_model"),
    ("python3 -m sglang.launch_server --speculative-algorithm NEXTN", "mtp"),
    ("python3 -m sglang.launch_server --speculative-algorithm EAGLE3", "draft_model"),
])
def test_spec_decoding_follows_the_serve_command(serve, expected):
    assert spec_decoding(serve) == expected


def test_arms_are_one_job_each_with_their_own_engine_in_one_run():
    text = (ENGINE + "arms:\n  base: {}\n  mtp:\n    append: |\n      --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 4}'\n"
            "scenario: agentx\nconc: [8]\n")
    result = build(text)
    names = [p["config"]["exp-name"] for p in result["points"]]
    assert names == ["qwen-tp1-base_c8", "qwen-tp1-mtp_c8"]
    assert [p["config"]["spec-decoding"] for p in result["points"]] == ["none", "mtp"]
    commands = [json.loads(p["serving"])["serve_cmd"] for p in result["points"]]
    assert "--speculative-config" not in commands[0] and "--speculative-config" in commands[1]
    assert "2 arms (base,mtp)" in result["title"]
