"""Exercise the golden AL package: committed curves, curve resolution and the query CLI."""

import json
import math

import pytest

from infx.golden_al_distribution import (
    GOLDEN_DIR,
    THINKING_MODES,
    curve_name,
    golden_length,
    list_curves,
)
from infx.golden_al_distribution.__main__ import main


def test_every_committed_curve_is_valid() -> None:
    curves = list_curves()
    assert {curve.name for curve in curves} == {path.stem for path in GOLDEN_DIR.glob("*.yaml")}
    for curve in curves:
        assert set(curve.modes) <= set(THINKING_MODES), curve.name
        for mode in curve.modes:
            assert curve.tokens(mode), (curve.name, mode)
            for tokens in curve.tokens(mode):
                value = curve.acceptance(mode, tokens)
                assert math.isfinite(value)
                assert 1 <= value <= tokens + 1


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
        ("musespark", {"method": "dflash"}, "musespark_dflash"),
        ("minimaxm31flash", {"method": "dspark"}, "minimaxm31flash_dspark"),
        ("kimik3", {"method": "dspark", "draft_sample_method": "greedy"}, "kimik3_dspark"),
        (
            "kimik3",
            {"method": "dspark", "draft_sample_method": "probabilistic"},
            "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method",
        ),
    ],
)
def test_curve_name_resolves_committed_curves(model, spec, expected) -> None:
    assert curve_name(model, spec) == expected
    assert (GOLDEN_DIR / f"{expected}.yaml").is_file()


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"method": "dspark", "num_speculative_tokens": 7}, "Kimi DSpark golden curve"),
        ({"method": "dspark", "draft_sample_method": "greedy", "num_speculative_tokens": 0}, "positive integer"),
        ({"method": "dspark", "draft_sample_method": "greedy", "num_speculative_tokens": 99}, "No golden acceptance"),
        ({"method": "eagle3", "num_speculative_tokens": 3}, "No committed golden curve"),
    ],
)
def test_golden_length_rejects_unknown_cells(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        golden_length("kimik3", spec, "thinking_on")


def test_cli_lookup_matches_library(capsys) -> None:
    assert main(["lookup", "qwen3.5", "eagle", "3", "--thinking", "thinking_off"]) == 0
    expected = golden_length("qwen3.5", {"method": "mtp", "num_speculative_tokens": 3}, "thinking_off")
    assert float(capsys.readouterr().out) == expected

    assert main(["lookup", "kimik3", "dspark", "7", "--draft-sample-method", "probabilistic", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["curve"] == "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method"
    assert result["tokens"] == 7


def test_cli_list_and_show(capsys) -> None:
    assert main(["list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["curve"] for row in listed] == [curve.name for curve in list_curves()]

    assert main(["show", "glm5.3_mtp"]) == 0
    assert "glm-5.3-fp8" in capsys.readouterr().out


def test_cli_reports_errors(capsys) -> None:
    assert main(["show", "missing_curve"]) == 1
    assert "No committed golden curve" in capsys.readouterr().err


def test_cli_show_rejects_a_curve_outside_the_golden_dir(tmp_path, capsys) -> None:
    golden = tmp_path / "golden"
    golden.mkdir()
    (tmp_path / "outside.yaml").write_text("some-model:\n  thinking_on:\n    3: 2.5\n")

    assert main(["--golden-dir", str(golden), "show", "../outside"]) == 1
    assert "Invalid golden curve identity" in capsys.readouterr().err


def test_a_thinking_mode_written_as_a_list_is_rejected(tmp_path) -> None:
    (tmp_path / "m_mtp.yaml").write_text("m:\n  thinking_on: [1.5, 2.3, 2.9]\n")
    spec = {"method": "mtp", "num_speculative_tokens": 2}

    with pytest.raises(ValueError, match="thinking_on of golden curve m_mtp.yaml"):
        golden_length("m", spec, "thinking_on", tmp_path)


def test_cli_show_marks_every_cell_of_an_empty_thinking_mode(tmp_path, capsys) -> None:
    (tmp_path / "m_mtp.yaml").write_text("m:\n  thinking_on:\n    1: 1.5\n  thinking_off:\n")

    assert main(["--golden-dir", str(tmp_path), "show", "m_mtp"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "m_mtp (m)",
        "tokens  thinking_on  thinking_off",
        "1       1.50         -",
    ]
