"""Exercise the verifier's deterministic facts against controlled inputs."""

import json

import pytest

from infx.klaud.models import Feed
from infx.workflows import signoff_facts

SERIES = ["--model", "m", "--precision", "fp4", "--framework", "sglang", "--hardware", "b200"]
SERIES += ["--spec-method", "mtp"]


def row(run, date, metrics, **overrides):
    return {
        "model": "m",
        "precision": "fp4",
        "framework": "sglang",
        "hardware": "b200",
        "spec_method": "mtp",
        "task": "gsm8k",
        "date": date,
        "conc": 64,
        "decode_tp": 8,
        "metrics": metrics,
        "run_url": f"https://github.com/o/r/actions/runs/{run}",
        **overrides,
    }


def run_cli(monkeypatch, capsys, rows, argv):
    feed = Feed(url="https://x/api", retrieved_at="2026-10-24T00:00:00Z", sha256="0" * 64, payload=rows)
    calls = []

    def fetch(resource, **kwargs):
        calls.append((resource, kwargs))
        return feed

    monkeypatch.setattr(signoff_facts.api, "fetch", fetch)
    signoff_facts.main(argv)
    return json.loads(capsys.readouterr().out), calls


def test_evals_split_exact_and_wider_peers_newest_first_and_limited(monkeypatch, capsys):
    score = {"em_strict": 0.95, "em_strict_se": 0.006, "n_eff": 1319}
    rows = [
        row(1, "2026-10-01", score),
        row(2, "2026-10-20", score),
        row(3, "2026-10-10", score),
        row(4, "2026-10-21", score, hardware="mi355x"),
        row(5, "2026-10-22", score, spec_method="none"),
        row(6, "2026-10-23", score, precision="fp8"),
        row(7, "2026-10-23", score, task="minimax_m3_full"),
        "not-a-row",
    ]
    report, calls = run_cli(monkeypatch, capsys, rows, ["evals", *SERIES, "--task", "gsm8k", "--limit", "2"])
    assert calls == [("evaluations", {})]
    assert [r["run_url"][-1] for r in report["exact"]] == ["2", "3"]
    assert report["exact_total"] == 3
    assert [r["run_url"][-1] for r in report["same_model_precision_task"]] == ["5", "4"]
    assert report["exact"][0]["em_strict_se"] == 0.006


def test_points_keep_only_the_requested_series(monkeypatch, capsys):
    metrics = {"tput_per_gpu": 812.5, "mean_intvty": 41.0, "ttft": 1}
    rows = [
        row(1, "2026-10-01", metrics, image="lmsysorg/sglang:v1"),
        row(2, "2026-10-02", metrics, image="lmsysorg/sglang:v2"),
        row(3, "2026-10-03", metrics, framework="vllm"),
    ]
    report, calls = run_cli(monkeypatch, capsys, rows, ["points", "--model-name", "M", *SERIES])
    assert calls == [("benchmarks", {"model": "M"})]
    assert [p["image"] for p in report["points"]] == ["lmsysorg/sglang:v2", "lmsysorg/sglang:v1"]
    assert "ttft" not in report["points"][0]


def check(name, conclusion, run_id=1, check_id=1):
    url = f"https://github.com/o/r/actions/runs/{run_id}/job/9"
    return {"id": check_id, "name": name, "conclusion": conclusion, "details_url": url}


@pytest.mark.parametrize(
    ("runs", "satisfied"),
    [
        ([check("agentic / (img)", "success"), check("agentic eval / (img)", "success", 7)], True),
        ([check("single-node 1k1k / x", "success"), check("eval / x", "skipped")], False),
        ([check("canary / x", "success"), check("eval / x", "success")], False),
        (
            [
                check("agentic / a", "success"),
                check("agentic / b", "failure"),
                check("agentic eval / a", "success"),
            ],
            False,
        ),
        (
            [
                check("agentic / a", "failure", check_id=1),
                check("agentic / a", "success", check_id=2),
                check("agentic eval / a", "success"),
            ],
            True,
        ),
        ([check("agentic / a", "success"), check("collect-evals / collect-evals", "success")], False),
    ],
)
def test_commit_validation_needs_executed_benchmark_and_eval_without_failures(runs, satisfied):
    assert signoff_facts.commit_validation(runs)["satisfied"] is satisfied


def test_commit_validation_reports_run_ids_of_successful_jobs():
    runs = [check("agentic / a", "success", 11), check("agentic eval / a", "success", 12)]
    assert signoff_facts.commit_validation(runs)["run_ids"] == [11, 12]


def candidate(em, se, tp=8, **overrides):
    return {
        "model_prefix": "m",
        "precision": "fp4",
        "framework": "sglang",
        "hw": "CLUSTER:B200-NSCALE",
        "spec_decoding": "mtp",
        "task": "gsm8k",
        "decode_tp": tp,
        "decode_ep": 1,
        "decode_dp_attention": "false",
        "em_strict": em,
        "em_strict_se": se,
        **overrides,
    }


def test_eval_gap_fails_only_beyond_two_combined_standard_errors():
    peer = row(5, "2026-10-20", {"em_strict": 0.97, "em_strict_se": 0.003}, decode_ep=1)
    peer["decode_dp_attention"] = False
    # threshold = 2 * sqrt(0.003^2 + 0.004^2) = 0.01
    weak, ok = candidate(0.9590, 0.004), candidate(0.9610, 0.004, tp=4)
    report = signoff_facts.eval_gaps([weak, ok], [peer], run_id=None, hardware=None)
    assert report[0]["candidate"]["hardware"] == "b200"
    assert report[0]["cohort"]["threshold"] == pytest.approx(0.01)
    assert report[0]["cohort"]["verdict"] == "materially-weaker"
    assert report[0]["cohort_same_topology"] is True
    assert report[1]["cohort"]["verdict"] == "ok"
    assert report[1]["cohort_same_topology"] is False
    assert report[0]["sibling"]["verdict"] == "ok"


def test_eval_gap_excludes_own_run_and_reports_missing_cohort():
    own = row(42, "2026-10-24", {"em_strict": 0.99, "em_strict_se": 0.001})
    report = signoff_facts.eval_gaps([candidate(0.9, 0.01)], [own], run_id=42, hardware=None)
    assert report[0]["cohort_scope"] == "none"
    assert report[0]["cohort"] is None


def point(label, tput, intvty):
    return {"label": label, "tput_per_gpu": tput, "mean_intvty": intvty}


def test_dominance_requires_removed_points_to_be_dominated():
    before = [point("a", 100, 50), point("b", 200, 30), point("keep", 150, 40)]
    after = [point("keep", 165, 40), point("new", 210, 30)]
    report = signoff_facts.dominance(before, after)
    assert report["removed"] == [
        {"label": "a", "dominated_by": None},
        {"label": "b", "dominated_by": "new"},
    ]
    assert report["undominated_removed"] == ["a"]
    assert report["deltas"] == [{"label": "keep", "tput_per_gpu_pct": 10.0, "mean_intvty_pct": 0.0}]


def test_dominance_rejects_missing_metrics():
    with pytest.raises(ValueError, match="positive"):
        signoff_facts.dominance([point("a", 0, 50)], [])


def matrix(points):
    return {"key": ({(p["recipe-fingerprint"], p["conc"]): p for p in points}, [])}


def gen(name, tp, conc, fingerprint):
    return {
        "exp-name": name,
        "tp": tp,
        "ep": 1,
        "dp-attn": False,
        "conc": conc,
        "image": "img",
        "recipe-fingerprint": fingerprint,
    }


def measured(fingerprint, conc, tput, intvty, date="2026-10-20"):
    return {
        "recipe_fingerprint": fingerprint,
        "conc": conc,
        "date": date,
        "metrics": {"tput_per_gpu": tput, "mean_intvty": intvty},
    }


def test_points_diff_flags_undominated_removals_and_reports_retuned_deltas():
    base = matrix([gen("x", 4, 8, "f4"), gen("x", 4, 64, "f4"), gen("y", 2, 32, "old")])
    head = matrix([gen("y", 2, 32, "new"), gen("x", 4, 1, "f4")])
    ingested = [
        measured("f4", 8, 100, 500),
        measured("f4", 64, 900, 100),
        measured("old", 32, 1000, 150),
    ]
    new_run = [measured("new", 32, 1100, 150, "2026-10-22")]
    [change] = signoff_facts.points_diff(base, head, ingested, new_run)
    assert change["undominated_removed"] == ["x tp4 ep1 dpaFalse conc 8"]
    assert [item["dominated_by"] for item in change["removed"] if item["conc"] == 64] == [
        "y tp2 ep1 dpaFalse conc 32"
    ]
    assert change["unmeasured_removed"] == []
    assert change["retuned_deltas"] == [
        {"label": "y tp2 ep1 dpaFalse conc 32", "tput_per_gpu_pct": 10.0, "mean_intvty_pct": 0.0}
    ]


def test_image_refs_normalize_registry_separator():
    text = 'image: "nvcr.io#nvidia/trt:1.0"\n  container: lmsysorg/sglang:v1@sha256:ab\nfoo: bar'
    assert signoff_facts.image_refs(text) == {"nvcr.io/nvidia/trt:1.0", "lmsysorg/sglang:v1@sha256:ab"}


@pytest.mark.parametrize(("framework", "setting", "pinned", "matches"), [("dynamo-trt", 2.5, 2.5, True), ("vllm", 3.5, 2.5, False)])
def test_golden_al_subtracts_bonus_token_only_for_trt(monkeypatch, framework, setting, pinned, matches):
    monkeypatch.setattr(signoff_facts.curves, "curve_name", lambda model, spec: "m_mtp")
    monkeypatch.setattr(signoff_facts.curves, "golden_length", lambda model, spec, thinking: 3.5)
    report = signoff_facts.golden_al("m", {}, "thinking_on", framework, pinned)
    assert report["expected_setting"] == setting
    assert report["matches"] is matches


def test_eval_gap_estimates_zero_standard_errors_from_sample_size():
    # 0.98 vs 0.97 at n=102: binomial SEs 0.01386 and 0.01689, threshold 0.0437 > gap 0.01
    peer = row(5, "2026-10-20", {"em_strict": 0.98, "em_strict_se": 0, "n_eff": 102})
    weak = candidate(0.97, 0, n_eff=102)
    [report] = signoff_facts.eval_gaps([weak], [peer], run_id=None, hardware=None)
    assert report["cohort"]["verdict"] == "ok"
    assert report["cohort"]["threshold"] == pytest.approx(0.0437, abs=1e-4)
    assert report["cohort"]["standard_error_estimated_from_n_eff"] is True


@pytest.mark.parametrize(
    ("image", "inspected"),
    [
        ("vllm/vllm-openai:nightly-1@sha256:ab", "vllm/vllm-openai@sha256:ab"),
        ("localhost:5000/repo@sha256:ab", "localhost:5000/repo@sha256:ab"),
        ("lmsysorg/sglang:v1", "lmsysorg/sglang:v1"),
    ],
)
def test_digest_pins_are_inspected_by_digest(monkeypatch, image, inspected):
    monkeypatch.setattr(signoff_facts, "_inspect", lambda reference: {"inspected": reference})
    assert signoff_facts.inspect_image(image)["inspected"] == inspected
