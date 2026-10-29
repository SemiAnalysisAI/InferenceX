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
    stems = {path.stem for path in GOLDEN_DIR.glob("*.yaml")} - {"identities"}
    assert {curve.name for curve in curves} == stems
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
        ("qwen3.5", {"method": "mtp"}, "qwen3.5-397b-a17b-nvfp4__mtp__native__default-default"),
        ("qwen3.5", {"method": "NEXTN"}, "qwen3.5-397b-a17b-nvfp4__mtp__native__default-default"),
        ("glm5.2", {"method": "eagle"}, "glm-5.2-fp8__mtp__native__default-default"),
        (
            "minimaxm3",
            {"method": "eagle"},
            "minimax-m3__eagle3__minimax-m3-eagle3__default-default",
        ),
        (
            "minimaxm3",
            {"method": "eagle3", "model": "org/MiniMax-M3-EAGLE3-GQA"},
            "minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default",
        ),
        (
            "dsv4dsparkprob",
            {"method": "dspark"},
            "deepseek-v4-pro-0813__dspark__native__probabilistic-default",
        ),
        (
            "dsv41flash",
            {"method": "dspark"},
            "deepseek-v4.1-flash__dspark__native__probabilistic-block",
        ),
        (
            "musespark",
            {"method": "dflash"},
            "muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default",
        ),
        (
            "minimaxm31flash",
            {"method": "dspark"},
            "minimax-m3.1-flash__dspark__native__probabilistic-block",
        ),
        (
            "kimik3",
            {"method": "dspark", "draft_sample_method": "greedy"},
            "kimi-k3__dspark__kimi-k3-dspark__default-default",
        ),
        (
            "kimik3",
            {"method": "dspark", "draft_sample_method": "probabilistic"},
            "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block",
        ),
    ],
)
def test_curve_name_resolves_committed_curves(model, spec, expected) -> None:
    assert curve_name(model, spec) == expected
    assert (GOLDEN_DIR / f"{expected}.yaml").is_file()


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"method": "dspark", "num_speculative_tokens": 7}, "No kimik3 dspark golden curve"),
        (
            {"method": "dspark", "draft_sample_method": "greedy", "num_speculative_tokens": 0},
            "positive integer",
        ),
        (
            {"method": "dspark", "draft_sample_method": "greedy", "num_speculative_tokens": 99},
            "No golden acceptance",
        ),
        ({"method": "eagle3", "num_speculative_tokens": 3}, "No golden curve for model prefix"),
    ],
)
def test_golden_length_rejects_unknown_cells(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        golden_length("kimik3", spec, "thinking_on")


def test_cli_lookup_matches_library(capsys) -> None:
    assert main(["lookup", "qwen3.5", "eagle", "3", "--thinking", "thinking_off"]) == 0
    expected = golden_length(
        "qwen3.5", {"method": "mtp", "num_speculative_tokens": 3}, "thinking_off"
    )
    assert float(capsys.readouterr().out) == expected

    assert (
        main(
            ["lookup", "kimik3", "dspark", "7", "--draft-sample-method", "probabilistic", "--json"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["curve"] == "kimi-k3__dspark__kimi-k3-dspark__probabilistic-block"
    assert result["tokens"] == 7


def test_cli_list_and_show(capsys) -> None:
    assert main(["list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["curve"] for row in listed] == [curve.name for curve in list_curves()]

    assert main(["show", "glm-5.3-fp8__mtp__native__default-default"]) == 0
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
    fact = "{value: null, source: default}"
    (tmp_path / "identities.yaml").write_text(
        "m_mtp:\n  select: {model_prefixes: [m], method: mtp}\n  collection:\n"
        "    engine: vllm\n    draft: native\n"
        + "".join(f"    {field}: {fact}\n" for field in (
            "draft_sample", "verify", "adaptive", "draft_precision", "draft_kv_cache_dtype",
            "kv_cache_dtype",
        ))
        + "    topology: {value: chain, source: documented}\n"
    )
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
