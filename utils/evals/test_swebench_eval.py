"""Tests for the SWE-bench Lite eval MVP (generation -> scoring -> lm-eval shape).

Pure-stdlib paths (extract_patch, predictions, report parsing, results shape)
run on any interpreter. The dataset filter and the collect/validate integration
guard on optional deps / interpreter version so the file imports cleanly even on
the macOS system python 3.9 used for local spot-checks.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))          # utils/evals
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # utils

import swebench_score as sbs


# --- diff extraction -------------------------------------------------------

def test_extract_patch_from_diff_fence():
    text = (
        "Here is the fix:\n\n```diff\n"
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1 +1 @@\n-old\n+new\n```\nDone."
    )
    patch = sbs.extract_patch(text)
    assert patch.startswith("diff --git a/f.py b/f.py")
    assert patch.endswith("\n")
    assert "Here is the fix" not in patch
    assert "Done." not in patch


def test_extract_patch_bare_diff_git():
    text = "no fence\ndiff --git a/x b/x\n@@ @@\n-a\n+b\n"
    patch = sbs.extract_patch(text)
    assert patch.startswith("diff --git a/x b/x")
    assert "no fence" not in patch


def test_extract_patch_bare_diff_strips_trailing_prose():
    # A bare diff followed by an explanation must not glue the prose onto the
    # patch (git apply would reject it -> instance scored unresolved).
    text = (
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
        "\nNotes:\nThis fixes #123.\n"
    )
    patch = sbs.extract_patch(text)
    assert patch.rstrip().endswith("+new")
    assert "Notes:" not in patch
    assert "This fixes" not in patch


def test_extract_patch_keeps_multi_file_and_interior_context():
    # Multiple files + a blank context line (represented as " ") stay intact.
    text = (
        "```diff\n"
        "diff --git a/a b/a\n@@ -1,2 +1,2 @@\n context\n-x\n+y\n"
        "diff --git a/b b/b\n@@ -1 +1 @@\n-p\n+q\n"
        "```\nthanks!"
    )
    patch = sbs.extract_patch(text)
    assert "diff --git a/a b/a" in patch
    assert "diff --git a/b b/b" in patch
    assert "thanks" not in patch


def test_extract_patch_empty_when_no_diff():
    assert sbs.extract_patch("") == ""
    # Prose with no diff markers falls back to the raw text (harness will reject).
    assert sbs.extract_patch("just words").strip() == "just words"


# --- samples -> predictions ------------------------------------------------

def _write_samples(dirpath: Path, records: list[dict]) -> None:
    with (dirpath / "samples_swebench_lite_2026.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_build_predictions_extracts_instance_and_patch(tmp_path):
    _write_samples(tmp_path, [
        {
            "doc": {"instance_id": "repo__proj-1"},
            "filtered_resps": ["```diff\ndiff --git a/a b/a\n+x\n```"],
        },
        {
            "doc": {"instance_id": "repo__proj-2"},
            "resps": [["diff --git a/b b/b\n+y\n"]],
        },
    ])
    preds = sbs.build_predictions(tmp_path, "my-model")
    by_id = {p["instance_id"]: p for p in preds}
    assert set(by_id) == {"repo__proj-1", "repo__proj-2"}
    assert by_id["repo__proj-1"]["model_name_or_path"] == "my-model"
    assert by_id["repo__proj-1"]["model_patch"].startswith("diff --git a/a b/a")
    assert by_id["repo__proj-2"]["model_patch"].startswith("diff --git a/b b/b")


def test_build_predictions_raises_without_samples(tmp_path):
    with pytest.raises(FileNotFoundError):
        sbs.build_predictions(tmp_path, "m")


# --- report parsing --------------------------------------------------------

def test_parse_resolved_classic_counts():
    assert sbs.parse_resolved(
        {"resolved_instances": 80, "total_instances": 196}
    ) == (80, 196)


def test_parse_resolved_from_id_lists():
    report = {"resolved_ids": ["a", "b", "c"], "completed_ids": ["a", "b", "c", "d"]}
    # no total_instances -> falls back to completed_ids length
    assert sbs.parse_resolved(report) == (3, 4)


def test_parse_resolved_raises_on_garbage():
    with pytest.raises(ValueError):
        sbs.parse_resolved({"nope": 1})


# --- lm-eval-shaped results ------------------------------------------------

def test_build_results_json_is_lm_eval_shaped():
    res = sbs.build_results_json(
        "swebench_lite", 49, 196, "m", "0.4.12", {"resolved_instances": 49}
    )
    assert "lm_eval_version" in res  # detection key for collect_eval_results
    task = res["results"]["swebench_lite"]
    assert task["exact_match,resolved"] == pytest.approx(0.25)
    cfg = res["configs"]["swebench_lite"]
    assert cfg["filter_list"] == [{"name": "resolved"}]
    assert res["n-samples"]["swebench_lite"]["effective"] == 196


def test_score_offline_end_to_end(tmp_path):
    """--report path: samples -> predictions + results JSON, no Docker."""
    samples = tmp_path / "gen"
    samples.mkdir()
    _write_samples(samples, [
        {"doc": {"instance_id": "r__p-1"}, "filtered_resps": ["```diff\ndiff --git a/a b/a\n+x\n```"]},
    ])
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"resolved_instances": 1, "total_instances": 1}))
    out = tmp_path / "out"
    rc = sbs.main([
        "--samples-dir", str(samples), "--out-dir", str(out),
        "--model-name", "m", "--report", str(report),
    ])
    assert rc == 0
    assert (out / "predictions.jsonl").exists()
    results = json.loads((out / "results_swebench_lite.json").read_text())
    assert results["results"]["swebench_lite"]["exact_match,resolved"] == 1.0


def test_predictions_only_writes_predictions_no_results(tmp_path):
    """SWEBENCH_SKIP_SCORE path: predictions only, no Docker, no results JSON."""
    samples = tmp_path / "gen"
    samples.mkdir()
    _write_samples(samples, [
        {"doc": {"instance_id": "r__p-1"}, "filtered_resps": ["```diff\ndiff --git a/a b/a\n+x\n```"]},
    ])
    out = tmp_path / "out"
    rc = sbs.main([
        "--samples-dir", str(samples), "--out-dir", str(out),
        "--model-name", "m", "--predictions-only",
    ])
    assert rc == 0
    assert (out / "predictions.jsonl").exists()
    assert not (out / "results_swebench_lite.json").exists()


# --- integration with the existing pipeline (needs tabulate + py3.10+) -----

@pytest.mark.skipif(sys.version_info < (3, 10), reason="repo modules use py3.10 syntax")
def test_results_json_flows_through_collect_and_validate(tmp_path, monkeypatch):
    pytest.importorskip("tabulate")
    import collect_eval_results as cer
    import validate_scores as vs

    art = tmp_path / "eval"
    art.mkdir()
    (art / "meta_env.json").write_text(json.dumps({
        "infmax_model_prefix": "dsr1", "hw": "b200", "framework": "sglang",
        "precision": "fp8", "isl": 8192, "osl": 1024,
    }))
    res = sbs.build_results_json(
        "swebench_lite", 150, 300, "dsr1", "0.4.12", None
    )
    (art / "results_swebench_lite.json").write_text(json.dumps(res))

    # collect surfaces the resolved-rate as the unified `score`.
    rows = cer.collect_eval_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "swebench_lite"
    assert rows[0]["score"] == pytest.approx(0.5)

    # validate_scores gates exact_match,resolved against thresholds.json (0.10).
    monkeypatch.chdir(art)
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--results-glob", "results_swebench_lite.json",
    ])
    assert vs.main() == 0  # 0.5 >= 0.10 default threshold
