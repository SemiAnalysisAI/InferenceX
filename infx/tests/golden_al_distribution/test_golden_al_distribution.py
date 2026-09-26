import json

import pytest
import yaml

from infx.golden_al_distribution import (
    curve_name,
    golden_length,
    load_curve,
)
from infx.golden_al_distribution.__main__ import main


@pytest.fixture
def golden_dir(tmp_path):
    (tmp_path / "qwen3.5_mtp.yaml").write_text(
        "fixture-model:\n"
        "  thinking_on: {3: '2.5', 1: 1.4}\n"
        "  thinking_off: {1: 1.5}\n"
    )
    return tmp_path


@pytest.mark.parametrize(
    ("model", "spec", "expected"),
    [
        ("qwen3.5", {"method": "mtp"}, "qwen3.5_mtp"),
        ("qwen3.5", {"method": "NEXTN"}, "qwen3.5_mtp"),
        ("glm5.2", {"method": "eagle"}, "glm5.2_mtp"),
        ("minimaxm3", {"method": "eagle"}, "minimaxm3_eagle3"),
        ("minimaxm3", {"method": "eagle3", "model": "org/MiniMax-M3-EAGLE3-GQA"}, "minimaxm3_eagle3_gqa"),
        ("dsv4dsparkprob", {"method": "dspark"}, "dsv4-pro-0813-dspark"),
        ("dsv41flash", {"method": "dspark"}, "dsv41flash_dspark"),
        ("kimik3", {"method": "dspark", "draft_sample_method": "greedy"}, "kimik3_dspark"),
        (
            "kimik3",
            {"method": "dspark", "draft_sample_method": "probabilistic"},
            "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method",
        ),
    ],
)
def test_curve_name_resolves_model_and_speculation_method(model, spec, expected) -> None:
    assert curve_name(model, spec) == expected


@pytest.mark.parametrize("mode,tokens,expected", [("thinking_on", 3, 2.5), ("thinking_off", 1, 1.5)])
def test_golden_length_selects_mode_and_converts_values(golden_dir, mode, tokens, expected):
    spec = {"method": "mtp", "num_speculative_tokens": tokens}
    assert golden_length("qwen3.5", spec, mode, golden_dir) == expected


def test_tokens_are_sorted_and_missing_modes_are_empty(golden_dir):
    curve = load_curve("qwen3.5_mtp", golden_dir)
    assert curve.tokens("thinking_on") == [1, 3]
    assert curve.tokens("missing") == []


@pytest.mark.parametrize(
    ("tokens", "mode", "message"),
    [
        (0, "thinking_on", "positive integer"),
        (True, "thinking_on", "positive integer"),
        ("3", "thinking_on", "positive integer"),
        (99, "thinking_on", "No golden acceptance"),
        (3, "thinking_off", "No golden acceptance"),
        (3, "missing", "No golden acceptance"),
    ],
)
def test_golden_length_rejects_invalid_or_unmeasured_cells(golden_dir, tokens, mode, message):
    spec = {"method": "mtp", "num_speculative_tokens": tokens}
    with pytest.raises(ValueError, match=message):
        golden_length("qwen3.5", spec, mode, golden_dir)


@pytest.mark.parametrize("value", [0.9, 4.1, float("inf"), float("nan"), "invalid"])
def test_golden_length_rejects_invalid_measurements(tmp_path, value):
    (tmp_path / "fixture_mtp.yaml").write_text(
        yaml.safe_dump({"fixture": {"thinking_on": {3: value}}})
    )
    with pytest.raises(ValueError, match="golden acceptance"):
        golden_length("fixture", {"method": "mtp", "num_speculative_tokens": 3}, "thinking_on", tmp_path)


@pytest.mark.parametrize(
    "data,message",
    [(None, "one model"), ({}, "one model"), ({"a": {}, "b": {}}, "one model"), ({"a": []}, "thinking modes")],
)
def test_load_curve_rejects_malformed_data(tmp_path, data, message):
    (tmp_path / "fixture.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_curve("fixture", tmp_path)


def test_cli_lookup_formats_text_and_json(golden_dir, capsys):
    args = ["--golden-dir", str(golden_dir), "lookup", "qwen3.5", "eagle", "3"]
    assert main(args) == 0
    assert capsys.readouterr().out == "2.5\n"

    assert main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "curve": "qwen3.5_mtp",
        "thinking": "thinking_on",
        "tokens": 3,
        "acceptance_length": 2.5,
    }


def test_cli_lookup_uses_the_requested_draft_sampler(golden_dir, capsys):
    curve = "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method"
    (golden_dir / f"{curve}.yaml").write_text("fixture-model:\n  thinking_on: {7: 3.5}\n")
    assert main([
        "--golden-dir", str(golden_dir), "lookup", "kimik3", "dspark", "7",
        "--draft-sample-method", "probabilistic", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "curve": curve,
        "thinking": "thinking_on",
        "tokens": 7,
        "acceptance_length": 3.5,
    }


def test_cli_list_serializes_curves(golden_dir, capsys):
    assert main(["--golden-dir", str(golden_dir), "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [{
        "curve": "qwen3.5_mtp",
        "model": "fixture-model",
        "modes": {"thinking_on": {"1": 1.4, "3": 2.5}, "thinking_off": {"1": 1.5}},
    }]


def test_cli_show_marks_unmeasured_cells(golden_dir, capsys):
    assert main(["--golden-dir", str(golden_dir), "show", "qwen3.5_mtp"]) == 0
    assert [line.split() for line in capsys.readouterr().out.splitlines()] == [
        ["qwen3.5_mtp", "(fixture-model)"],
        ["tokens", "thinking_on", "thinking_off"],
        ["1", "1.40", "1.50"],
        ["3", "2.50", "-"],
    ]


def test_cli_reports_errors(golden_dir, capsys):
    assert main(["--golden-dir", str(golden_dir), "show", "missing_curve"]) == 1
    assert "No committed golden curve" in capsys.readouterr().err
