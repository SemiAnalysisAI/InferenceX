import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx import github as gh
from infx.bench.eval import meta as eval_meta
from infx.config import project_root
from infx.klaud import github, lifecycle, reporting, validation
from infx.klaud.models import OwnedCandidate, identity
from infx.results.eval_artifacts import eval_key
from infx.results.evals import build_row
from infx.tests.klaud.test_klaud_github import CANDIDATE_ID, FAMILY
from infx.tests.workflows.test_validate_reusable_sweep_artifacts import write_eval_aggregate

ROOT = project_root(Path(__file__).resolve().parents[4])
WORKFLOWS = ROOT.parent / ".github/workflows"


@pytest.fixture
def sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr(github, "sleep", delays.append)
    return delays


def failing_then(monkeypatch, errors, result):
    """Make gh.api raise each of ``errors`` in turn and then return ``result``; count calls."""
    calls = []

    def api(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result

    monkeypatch.setattr(gh, "api", api)
    return calls


def http_error(status):
    return subprocess.CalledProcessError(
        1, ["gh", "api"], stderr=f"gh: Something went wrong (HTTP {status})\n"
    )


TIMEOUT = subprocess.TimeoutExpired(["gh", "api"], 60)


def gh_error(stderr):
    return subprocess.CalledProcessError(1, ["gh", "api"], stderr=stderr)


@pytest.mark.parametrize("errors", [
    [TIMEOUT],
    [http_error(502), http_error(503)],
    [TIMEOUT, http_error(500)],
    pytest.param([gh_error("gh: HTTP 502\n")], id="status-without-parentheses"),
    pytest.param([gh_error("read tcp 10.0.0.1:443: read: connection reset by peer\n")],
                 id="connection-reset"),
    pytest.param([gh_error('Get "https://api.github.com/repos": unexpected EOF\n')], id="eof"),
    pytest.param([gh_error("net/http: TLS handshake timeout\n")], id="tls-timeout"),
    pytest.param([gh_error("dial tcp 140.82.112.6:443: i/o timeout\n")], id="io-timeout"),
])  # fmt: skip
def test_reads_retry_transient_failures_then_succeed(monkeypatch, sleeps, errors):
    calls = failing_then(monkeypatch, errors, {"ok": True})

    assert github.read("example/project", "pulls/7") == {"ok": True}
    assert len(calls) == len(errors) + 1
    assert sleeps == [1, 4][: len(errors)]


def test_reads_give_up_after_three_attempts(monkeypatch, sleeps):
    calls = failing_then(monkeypatch, [TIMEOUT] * 3, {"ok": True})

    with pytest.raises(subprocess.TimeoutExpired):
        github.read("example/project", "pulls/7")
    assert len(calls) == 3
    assert sleeps == [1, 4]


@pytest.mark.parametrize("error", [
    http_error(404),
    http_error(422),
    http_error(429),
    subprocess.CalledProcessError(
        1, ["gh", "api"], stderr="gh: API rate limit exceeded for user (HTTP 403)\n"
    ),
    subprocess.CalledProcessError(1, ["gh", "api"], stderr=None),
    gh_error("gh: HTTP 429\n"),
    gh_error("gh: HTTP 403: API rate limit exceeded\n"),
    gh_error("gh: Not Found (HTTP 404)\nunexpected EOF\n"),
])  # fmt: skip
def test_reads_never_retry_client_errors_or_rate_limits(monkeypatch, sleeps, error):
    calls = failing_then(monkeypatch, [error], {"ok": True})

    with pytest.raises(subprocess.CalledProcessError):
        github.read("example/project", "pulls/7")
    assert len(calls) == 1
    assert sleeps == []


def test_listings_retry_an_incomplete_listing_but_not_a_malformed_one(monkeypatch, sleeps):
    pages = iter([
        [{"total_count": 2, "runs": [{"id": 1}]}],
        [{"total_count": 2, "runs": [{"id": 1}, {"id": 2}]}],
    ])  # fmt: skip
    monkeypatch.setattr(gh, "api", lambda *_args, **_kwargs: next(pages))

    assert github.items("example/project", "actions/runs", "runs") == [{"id": 1}, {"id": 2}]
    assert sleeps == [1]

    calls = failing_then(monkeypatch, [], [{"runs": "not-a-list", "total_count": 1}])
    with pytest.raises(github.ListingUnavailable, match="unexpected shape"):
        github.items("example/project", "actions/runs", "runs")
    assert len(calls) == 1
    assert sleeps == [1]


def test_listings_surface_a_persistently_incomplete_listing(monkeypatch, sleeps):
    calls = failing_then(monkeypatch, [], [{"total_count": 3, "runs": [{"id": 1}]}])

    with pytest.raises(github.ListingUnavailable, match="Incomplete GitHub listing"):
        github.items("example/project", "actions/runs", "runs")
    assert len(calls) == 3
    assert sleeps == [1, 4]


@pytest.mark.parametrize("error", [TIMEOUT, http_error(502)])
def test_writes_are_never_retried(monkeypatch, sleeps, error):
    calls = failing_then(monkeypatch, [error], {})

    with pytest.raises(type(error)):
        github.write("example/project", "issues/7/comments", "POST", {"body": "x"})
    assert len(calls) == 1
    assert sleeps == []


def coverage_entry(conc, name="fixture"):
    return {
        "image": "fixture:latest", "model": "fixture/model", "model-prefix": "gptoss",
        "precision": "fp4", "framework": "vllm", "spec-decoding": "none",
        "runner": "h100-dgxc-slurm", "isl": 8192, "osl": 1024,
        "tp": 2, "pp": 1, "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False,
        "conc": conc, "max-model-len": 9216, "exp-name": name, "disagg": False,
        "run-eval": True, "recipe-fingerprint": "a" * 64, "eval-suite": "gsm8k",
    }  # fmt: skip


def test_coverage_failure_names_missing_points_from_the_matrix_only(tmp_path):
    entries = [coverage_entry(conc) for conc in (1, 2, 4, 8, 16, 32, 64)]
    matrix = {
        "single_node": {"all": entries}, "evals": [entries[0]],
        "changelog_metadata": {"head_ref": "head", "entries": [{"config-keys": ["fixture"]}]},
    }  # fmt: skip
    manifest = {"head": "head", "run-id": 42, "run-attempt": 2, "full-sweep": True, "matrix": matrix}
    (tmp_path / "results_bmk").mkdir()
    (tmp_path / "results_bmk/agg.json").write_text(json.dumps([
        {"recipe_fingerprint": "a" * 64, "conc": 1, "image": "fixture:latest"},
        {"recipe_fingerprint": "<script>", "conc": 3, "image": "attacker"},
    ]))  # fmt: skip
    write_eval_aggregate(tmp_path, [])

    with pytest.raises(github.VerificationError) as raised:
        validation.check_coverage(
            tmp_path, manifest, {"head_sha": "head", "id": 42, "run_attempt": 2},
            "configs/nvidia-master.yaml:fixture", matrix,
        )  # fmt: skip

    assert str(raised.value) == (
        "Full-sweep result coverage or consistency failed; 6 benchmark point(s) missing "
        "(fixture c16, fixture c2, fixture c32, fixture c4, fixture c64 and 1 more), "
        "1 unexpected; 1 eval(s) missing (fixture gsm8k c1), 0 unexpected"
    )
    assert "<script>" not in str(raised.value)
    assert "attacker" not in str(raised.value)


def test_coverage_failure_labels_evals_from_the_canonical_matrix_not_the_manifest(tmp_path):
    entry = coverage_entry(1)
    canonical = {
        "single_node": {"all": [entry]}, "evals": [entry],
        "changelog_metadata": {"head_ref": "head", "entries": [{"config-keys": ["fixture"]}]},
    }  # fmt: skip
    matrix = {**canonical, "evals": [coverage_entry(1, name="<script>")]}
    manifest = {
        "head": "head",
        "run-id": 42,
        "run-attempt": 2,
        "full-sweep": True,
        "matrix": matrix,
    }
    (tmp_path / "results_bmk").mkdir()
    (tmp_path / "results_bmk/agg.json").write_text(json.dumps([
        {"recipe_fingerprint": "a" * 64, "conc": 1, "image": "fixture:latest"},
    ]))  # fmt: skip
    write_eval_aggregate(tmp_path, [])

    with pytest.raises(github.VerificationError) as raised:
        validation.check_coverage(
            tmp_path, manifest, {"head_sha": "head", "id": 42, "run_attempt": 2},
            "configs/nvidia-master.yaml:fixture", canonical,
        )  # fmt: skip

    assert "1 eval(s) missing (fixture gsm8k c1)" in str(raised.value)
    assert "<script>" not in str(raised.value)


AGENTX = {
    "image": "fixture:new", "model-prefix": "dsr1", "runner": "b200-dgxc", "framework": "sglang",
    "precision": "fp4", "spec-decoding": "none", "scenario-type": "agentic-coding", "tp": 8,
    "ep": 1, "dp-attn": False, "exp-name": "fixture", "recipe-fingerprint": "f" * 64,
    "eval-suite": "minimax_m3_full",
}  # fmt: skip
CORPUS = "semianalysisai/cc-traces-weka-062126"


def agentx_row(conc, output, errors=0, dataset=CORPUS):
    """An AgentX row at ``conc`` with ``output`` tok/s/GPU and ``errors`` of 1000 requests."""
    return {
        "recipe_fingerprint": "f" * 64, "conc": conc, "image": "fixture:new",
        "metrics": {"request_metrics": {
            "throughput": {"per_gpu": {"total_tput_tps": output * 10, "output_tput_tps": output}},
            "latency": {"ttft": {"mean": 0.1}, "tpot": {"mean": 0.01}},
        }},
        "request_accounting": {"records_error_dropped": errors, "records_total": 1000},
        "dataset": {"source_type": "public_dataset", "hf_dataset_name": dataset},
    }  # fmt: skip


def agentx_eval_row(conc, score, suite="minimax_m3_full"):
    """The aggregate ``suite`` eval row of the AGENTX recipe at ``conc`` scoring ``score``."""
    row = {key.replace("-", "_"): value for key, value in AGENTX.items()}
    row.update(is_multinode=False, hw=AGENTX["runner"], isl=0, osl=0, dp_attention=False)
    return {**row, "eval_suite": suite, "conc": conc, "score": score, "n_eff": 102}


def agentx_baseline(output=100.0, dataset="cc-traces-weka-062126", evals=()):
    """A frozen baseline of the AGENTX c4 and c8 points, each at ``output`` tok/s/GPU."""
    points = [
        reporting.Point(
            key=reporting.point_key({**AGENTX, "conc": conc}),
            label=reporting.point_label({**AGENTX, "conc": conc}), conc=conc,
            scenario="agentic-coding", dataset=dataset,
            values=reporting.Values(output_tps_gpu=output), result="passed",
        )
        for conc in (4, 8)
    ]  # fmt: skip
    return reporting.Baseline(
        family=FAMILY, date="2026-09-01", image="fixture:old",
        goal=reporting.Prose(en="Update the image.", zh="更新镜像。"), sources=[], points=points,
        evals=list(evals),
    )  # fmt: skip


def agentx_matrix(suite="minimax_m3_full"):
    """The AGENTX c4/c8 curve with one default ``suite`` eval at c8."""
    return {
        "single_node": {"all": [{**AGENTX, "conc": [4, 8]}]},
        "evals": [{**AGENTX, "eval-suite": suite, "conc": 8}],
    }  # fmt: skip


AGENTX_MATRIX = agentx_matrix()
FINAL_RUN = {"id": 9, "head_sha": "c" * 40, "run_attempt": 1}
OTHER_CORPUS = "semianalysisai/cc-traces-weka-062126-256k"


@pytest.mark.parametrize(("rows", "baseline", "blocked"), [
    pytest.param([agentx_row(4, 75.0), agentx_row(8, 71.0, errors=50)], agentx_baseline(), None,
                 id="within-thresholds"),
    pytest.param([agentx_row(4, 69.0), agentx_row(8, 100.0)], agentx_baseline(),
                 "output tok/s/GPU fell more than 30% at AgentX c4 TP8 EP1", id="throughput"),
    pytest.param([agentx_row(4, 100.0), agentx_row(8, 100.0, errors=51)], agentx_baseline(),
                 "request errors exceed 5% at AgentX c8 TP8 EP1", id="request-errors"),
    pytest.param([agentx_row(4, 75.0), agentx_row(8, 75.0)], agentx_baseline(dataset=None), None,
                 id="legacy-baseline-within-thresholds"),
    pytest.param([agentx_row(4, 25.0), agentx_row(8, 100.0)], agentx_baseline(dataset=None),
                 "output tok/s/GPU fell more than 30% at AgentX c4 TP8 EP1",
                 id="legacy-baseline-regressed"),
    pytest.param([agentx_row(4, 25.0, dataset=OTHER_CORPUS), agentx_row(8, 25.0,
                  dataset=OTHER_CORPUS)], agentx_baseline(), None, id="other-corpus"),
    pytest.param([agentx_row(4, 100.0), agentx_row(8, 100.0)], agentx_baseline(output=None),
                 "output throughput cannot be compared (missing or zero) at AgentX c4",
                 id="baseline-throughput-missing"),
])  # fmt: skip
def test_final_validation_blocks_regressions_against_the_frozen_baseline(rows, baseline, blocked):
    evidence = (AGENTX_MATRIX, rows, [agentx_eval_row(8, 0.95)])
    if blocked is None:
        reporting.check_regressions(FINAL_RUN, evidence, baseline)
        return
    with pytest.raises(github.VerificationError, match=re.escape(blocked)):
        reporting.check_regressions(FINAL_RUN, evidence, baseline)


@pytest.mark.parametrize(("suite", "score", "frozen", "blocked"), [
    pytest.param("minimax_m3_full", 0.15, None,
                 "eval score below 50% at fixture minimax_m3_full c8 (15%)", id="floored-suite"),
    pytest.param("bfcl_vllm_kimi_multi_turn", 0.3, None, None, id="unfloored-suite"),
    pytest.param("bfcl_vllm_kimi_multi_turn", 0.2, 0.3, None, id="within-baseline-retention"),
    pytest.param("bfcl_vllm_kimi_multi_turn", 0.1, 0.3,
                 "eval score fell below 50% of its baseline at fixture bfcl_vllm_kimi_multi_turn "
                 "c8 (10% vs 30%)", id="below-baseline-retention"),
    pytest.param("gsm8k", 0.48, 0.95, "eval score below 50% at fixture gsm8k c8 (48%)",
                 id="floor-despite-baseline"),
])  # fmt: skip
def test_final_validation_blocks_evals_far_below_baseline_or_floor(suite, score, frozen, blocked):
    row = agentx_eval_row(8, score, suite)
    evals = [] if frozen is None else [reporting.Evaluation(
        key=identity(eval_key(row)), suite=suite, metric="em_strict", score=frozen,
        samples=102, result="passed",
    )]  # fmt: skip
    evidence = (agentx_matrix(suite), [agentx_row(4, 100.0), agentx_row(8, 100.0)], [row])
    if blocked is None:
        reporting.check_regressions(FINAL_RUN, evidence, agentx_baseline(evals=evals))
        return
    with pytest.raises(github.VerificationError, match=re.escape(blocked)):
        reporting.check_regressions(FINAL_RUN, evidence, agentx_baseline(evals=evals))


def test_final_validation_runs_the_regression_guard(monkeypatch):
    session = lifecycle.Session(
        "example/project", {"id": 42},
        OwnedCandidate(id=CANDIDATE_ID, family=FAMILY, base="a" * 40),
    )  # fmt: skip
    monkeypatch.setattr(session, "pulls", lambda: [{"number": 7}])
    monkeypatch.setattr(reporting, "baseline_for", lambda *_: agentx_baseline())
    monkeypatch.setattr(reporting, "check_baseline_coverage", lambda *_: None)
    monkeypatch.setattr(lifecycle, "verify_sweep", lambda *_: (
        AGENTX_MATRIX, [agentx_row(4, 20.0), agentx_row(8, 20.0)], [agentx_eval_row(8, 0.95)],
    ))  # fmt: skip

    with pytest.raises(github.VerificationError, match="fell more than 30%"):
        session.validation(FINAL_RUN)


def attempt_evals(*keys):
    return [
        reporting.Evaluation(key=key * 64, suite="gsm8k", metric="em_strict", result="passed")
        for key in keys
    ]


@pytest.mark.parametrize(("baseline", "attempt", "warned"), [
    pytest.param("a", "b", True, id="no-match"),
    pytest.param("ab", "bc", False, id="one-match"),
    pytest.param("", "b", False, id="baseline-without-evals"),
    pytest.param("a", "", False, id="attempt-without-evals"),
    pytest.param(None, "b", False, id="no-baseline"),
])  # fmt: skip
def test_publishing_an_attempt_warns_when_no_eval_matches_the_baseline(
    monkeypatch, capsys, baseline, attempt, warned
):
    session = SimpleNamespace(
        repository="example/project",
        candidate=SimpleNamespace(family=FAMILY, id=CANDIDATE_ID),
        pulls=lambda: [{"number": 7}],
        refresh=lambda pull: pull,
        runs=lambda: [{"id": 9, "head_sha": "c" * 40, "run_attempt": 1}],
    )
    frozen = None if baseline is None else reporting.Baseline.model_construct(
        evals=attempt_evals(*baseline)
    )
    record = reporting.Attempt.model_construct(
        kind="initial", number=0, head="c" * 40, run_id=9, run_attempt=1, status="running",
        points=[], evals=attempt_evals(*attempt),
    )
    published = []
    monkeypatch.setattr(reporting, "baseline_for", lambda *_: frozen)
    monkeypatch.setattr(reporting, "stored", lambda *_: None)
    monkeypatch.setattr(reporting, "render_attempt", lambda *_: "Rendered.")
    monkeypatch.setattr(reporting, "upsert", lambda *args: published.append(args[2]))

    reporting.publish(session, record)

    assert published == ["run-9-1"]
    warning = "::warning::Klaud: none of the 1 evals of run 9 match a frozen baseline eval"
    assert (warning in capsys.readouterr().out) is warned


def workflow_env(template):
    return yaml.safe_load((WORKFLOWS / template).read_text())["env"]


CONFIG_EXPRESSION = re.compile(
    r"\$\{\{ (?:inputs\.scenario-type == 'agentic-coding' && '(?P<agentic>[^']*)' \|\| )?"
    r"fromJSON\(inputs\.config\)(?:\.(?P<dotted>[\w.-]+)|\['(?P<indexed>[\w-]+)'\])"
    r"(?: \|\| '?(?P<default>[^' }]*)'?)? \}\}"
)


def rendered(value):
    """``value`` as GitHub Actions renders it into an environment variable."""
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def eval_environment(entry, template):
    """The environment ``template`` gives an eval job of matrix ``entry``.

    Config fields resolve through the template's own env expressions, so renaming a field in
    either the workflow or the generator breaks the round trip.
    """
    environ = {"RUNNER_TYPE": entry["runner"], "EVAL_SUITE": entry.get("eval-suite") or ""}
    for name, expression in workflow_env(template).items():
        if expression == "${{ inputs.dp-attn }}":
            environ[name] = rendered(entry.get("dp-attn"))
        elif isinstance(expression, str) and (match := CONFIG_EXPRESSION.fullmatch(expression)):
            if match["agentic"] is not None:
                environ[name] = match["agentic"]
                continue
            value = entry
            for part in (match["dotted"] or match["indexed"]).split("."):
                value = value.get(part) if isinstance(value, dict) else None
            environ[name] = rendered(value) or (match["default"] or "")
        elif not isinstance(expression, str) or "${{" not in expression:
            environ[name] = rendered(expression)
    return environ


def canonical_evals():
    """Every default eval of every live agentic family on this checkout, with its family."""
    from infx.matrix.generate import generate_test_config_sweep, mark_eval_entries
    from infx.matrix.validation import load_config_files, load_runner_file

    runners = load_runner_file(str(ROOT / "configs/runners.yaml"))
    for path in sorted((ROOT / "configs").glob("*-master.yaml")):
        configs = load_config_files([str(path)])
        for key in configs:
            entries = [
                entry
                for entry in generate_test_config_sweep(
                    SimpleNamespace(config_keys=[key]), configs, runners, ROOT
                )
                if validation.in_scope(entry)
            ]
            for entry in mark_eval_entries(deepcopy(entries), ROOT):
                if entry.get("run-eval"):
                    yield f"{path.name}:{key}", entry


def test_every_canonical_eval_identity_round_trips_through_the_collector():
    checked, multinode = 0, set()
    for family, entry in canonical_evals():
        multi = entry.get("prefill") is not None
        template = "benchmark-multinode-tmpl.yml" if multi else "benchmark-tmpl.yml"
        environ = eval_environment(entry, template)
        for key, _, conc in validation.eval_identities({"evals": [entry]}):
            suite = entry.get("eval-suite") or "gsm8k"
            document = json.loads(json.dumps(eval_meta.build(environ, conc=conc, suite=suite)))
            row = build_row(document, {"task": suite, "strict": 0.9, "n_eff": 100})
            assert eval_key(row) == key, family
            checked += 1
            if multi:
                multinode.add(family)
    assert checked
    assert multinode
