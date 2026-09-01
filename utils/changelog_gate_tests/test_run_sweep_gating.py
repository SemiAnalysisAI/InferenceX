"""Exercise the real workflow conditions against concrete authorization and failure cases.

The small evaluator below supports only the GitHub Actions expressions used by
these gates; this is not a substitute for executing the workflow in Actions.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_WF = yaml.load(
    (REPO_ROOT / ".github/workflows/run-sweep.yml").read_text(),
    Loader=yaml.BaseLoader,
)
CHECK_IF = _WF["jobs"]["check-changelog"]["if"]
GATE_IF = _WF["jobs"]["reuse-sweep-gate"]["if"]
CLASSIFIER_STEP = next(
    step for step in _WF["jobs"]["setup"]["steps"] if step.get("id") == "classify"
)
CLASSIFIER_IF = CLASSIFIER_STEP["if"]
SETUP_IF = _WF["jobs"]["setup"]["if"]

# --------------------------------------------------------------------------
# Minimal GitHub Actions expression engine (supports the subset used by the
# gating conditions: && || ! == != contains() always(), parens, paths).
# --------------------------------------------------------------------------
def _tokenize(s: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "'":
            j = i + 1
            while j < n and s[j] != "'":
                j += 1
            toks.append(("str", s[i + 1 : j]))
            i = j + 1
            continue
        if s[i : i + 2] in ("==", "!=", "&&", "||"):
            toks.append(("op", s[i : i + 2]))
            i += 2
            continue
        if c in "!(),":
            kind = {"!": "op", "(": "lp", ")": "rp", ",": "comma"}[c]
            toks.append((kind, c))
            i += 1
            continue
        m = re.match(r"[A-Za-z0-9_.*\-]+", s[i:])
        if not m:
            raise SyntaxError(f"bad char {c!r} in {s!r}")
        toks.append(("word", m.group(0)))
        i += len(m.group(0))
    return toks


def _truthy(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (str, list, dict)):
        return len(v) > 0
    return bool(v)


class _Parser:
    def __init__(self, toks: list[tuple[str, str]], ctx: dict) -> None:
        self.t, self.i, self.ctx = toks, 0, ctx

    def _peek(self) -> tuple[str | None, str | None]:
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def _next(self) -> tuple[str, str]:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self) -> object:
        v = self._or()
        if self.i != len(self.t):
            raise SyntaxError(f"trailing tokens: {self.t[self.i:]}")
        return v

    def _or(self) -> object:
        v = self._and()
        while self._peek() == ("op", "||"):
            self._next()
            # Bind the operand before combining: it must always consume its
            # tokens, even when `or`/`and` would short-circuit on truthiness.
            rhs = self._and()
            v = _truthy(v) or _truthy(rhs)
        return v

    def _and(self) -> object:
        v = self._eq()
        while self._peek() == ("op", "&&"):
            self._next()
            rhs = self._eq()
            v = _truthy(v) and _truthy(rhs)
        return v

    def _eq(self) -> object:
        v = self._unary()
        if self._peek() in (("op", "=="), ("op", "!=")):
            op = self._next()[1]
            eq = v == self._unary()
            return eq if op == "==" else not eq
        return v

    def _unary(self) -> object:
        if self._peek() == ("op", "!"):
            self._next()
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self) -> object:
        kind, val = self._peek()
        if kind == "lp":
            self._next()
            v = self._or()
            assert self._next()[0] == "rp"
            return v
        if kind == "str":
            self._next()
            return val
        if kind == "word":
            self._next()
            if self._peek()[0] == "lp":
                self._next()
                args: list[object] = []
                if self._peek()[0] != "rp":
                    args.append(self._or())
                    while self._peek()[0] == "comma":
                        self._next()
                        args.append(self._or())
                assert self._next()[0] == "rp"
                return _call(val, args)
            if val in ("true", "false"):
                return val == "true"
            return self.ctx.get(val)
        raise SyntaxError(f"unexpected token {self._peek()}")


def _call(name: str, args: list[object]) -> object:
    if name in ("always", "success"):
        return True
    if name == "contains":
        haystack, needle = args[0], args[1]
        return False if haystack is None else needle in haystack
    raise SyntaxError(f"unsupported function {name}()")


@lru_cache(maxsize=None)
def _tokens(expr: str) -> tuple[tuple[str, str], ...]:
    return tuple(_tokenize(expr))


def _eval(expr: str, ctx: dict) -> bool:
    return _truthy(_Parser(_tokens(expr), ctx).parse())


def test_expression_evaluator_handles_truthiness_and_precedence() -> None:
    # The workflow checks depend on this helper; verify it with independent examples.
    for expression, context, expected in [
        ("always()", {}, True),
        ("!false", {}, True),
        ("'a' == 'b'", {}, False),
        ("x != 'true'", {"x": "true"}, False),
        ("x != 'true'", {"x": ""}, True),
        ("contains(labels, 'sweep')", {"labels": ["sweep"]}, True),
        ("contains(labels, 'sweep')", {}, False),
        ("true || false && false", {}, True),
        ("(true || false) && false", {}, False),
    ]:
        assert _eval(expression, context) is expected, expression


# --------------------------------------------------------------------------
# DAG evaluation: check-changelog -> reuse-sweep-gate -> setup
# --------------------------------------------------------------------------
def _ctx(sc: dict) -> dict:
    return {
        "github.event_name": sc["event"],
        "github.repository": "SemiAnalysisAI/InferenceX",
        "github.event.action": sc.get("action"),
        "github.event.pull_request.draft": sc.get("draft", False),
        "github.event.pull_request.head.repo.full_name": sc.get(
            "head_repo", "SemiAnalysisAI/InferenceX"
        ),
        "github.event.pull_request.labels.*.name": sc.get("labels", []),
        "github.event.label.name": sc.get("label_name"),
        "vars.PRIORITY_SCHEDULER_ENABLED": sc.get("scheduler_enabled", "true"),
        "github.event.head_commit.message": sc.get("msg", ""),
    }


def run_dag(sc: dict) -> tuple[str, str, str]:
    """Return (check-changelog result, reuse-sweep-gate result, setup decision)."""
    ctx = _ctx(sc)

    if not _eval(CHECK_IF, ctx):
        check_result = "skipped"
    else:
        check_result = sc.get("check", "success")
    ctx["needs.check-changelog.result"] = check_result
    ctx["needs.check-changelog.outputs.skip-pr-sweep"] = (
        sc.get("check_skip", "false")
    )

    if not _eval(GATE_IF, ctx):
        gate_result, skip = "skipped", ""
    else:
        gate_result = "success"
        skip = "true" if sc.get("reuse_auth") else ""
    ctx["needs.reuse-sweep-gate.result"] = gate_result
    ctx["needs.reuse-sweep-gate.outputs.skip-pr-sweep"] = skip


    setup = "RUN" if _eval(SETUP_IF, ctx) else "SKIP"
    return check_result, gate_result, setup


_PR = {"event": "pull_request", "draft": False}

# (id, scenario, expected (check, reuse, setup))
CASES = [
    ("PR-sync-full-noreuse",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "reuse_auth": False}, ("success", "success", "RUN")),
    ("PR-sync-full-reuse-authorized",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "reuse_auth": True}, ("success", "success", "SKIP")),
    ("PR-sync-full-changelog-failure",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "check": "failure"}, ("failure", "skipped", "SKIP")),
    ("PR-sync-trim-sweep-enabled",
     {**_PR, "action": "synchronize", "labels": ["sweep-enabled"]},
     ("success", "skipped", "RUN")),
    ("PR-sync-all-evals-without-sweep-label",
     {**_PR, "action": "synchronize", "labels": ["all-evals"]},
     ("success", "skipped", "SKIP")),
    ("PR-sync-evals-only-without-sweep-label",
     {**_PR, "action": "synchronize", "labels": ["evals-only"]},
     ("success", "skipped", "SKIP")),
    ("PR-sync-agentx-fast-without-sweep-label",
     {**_PR, "action": "synchronize", "labels": ["agentx-fast"]},
     ("success", "skipped", "SKIP")),
    ("PR-sync-full-with-all-evals-uses-reuse",
     {**_PR, "action": "synchronize",
      "labels": ["full-sweep-enabled", "all-evals"],
      "reuse_auth": True}, ("success", "success", "SKIP")),
    ("PR-sync-full-with-evals-only-ignores-reuse",
     {**_PR, "action": "synchronize",
      "labels": ["full-sweep-enabled", "evals-only"],
      "reuse_auth": True}, ("success", "skipped", "RUN")),
    ("PR-sync-full-with-agentx-fast-ignores-reuse",
     {**_PR, "action": "synchronize",
      "labels": ["full-sweep-enabled", "agentx-fast"],
      "reuse_auth": True}, ("success", "skipped", "RUN")),
    ("PR-sync-full-with-both-modifiers-ignores-reuse",
     {**_PR, "action": "synchronize",
      "labels": ["full-sweep-enabled", "all-evals", "evals-only"],
      "reuse_auth": True}, ("success", "skipped", "RUN")),
    ("PR-sync-no-sweep-label",
     {**_PR, "action": "synchronize", "labels": []},
     ("success", "skipped", "SKIP")),
    ("PR-sync-external-fork-defers-to-trusted-dispatch",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "head_repo": "external/InferenceX"},
     ("success", "success", "SKIP")),
    ("PR-labeled-with-sweep-label",
     {**_PR, "action": "labeled", "label_name": "full-sweep-enabled",
      "labels": ["full-sweep-enabled"]}, ("success", "skipped", "RUN")),
    ("PR-labeled-with-all-evals-without-sweep-label",
     {**_PR, "action": "labeled", "label_name": "all-evals",
      "labels": ["all-evals"]}, ("success", "skipped", "SKIP")),
    ("PR-labeled-with-evals-only-without-sweep-label",
     {**_PR, "action": "labeled", "label_name": "evals-only",
      "labels": ["evals-only"]}, ("success", "skipped", "SKIP")),
    ("PR-labeled-with-agentx-fast-without-sweep-label",
     {**_PR, "action": "labeled", "label_name": "agentx-fast",
      "labels": ["agentx-fast"]}, ("success", "skipped", "SKIP")),
    ("PR-labeled-all-evals-modifies-full-sweep",
     {**_PR, "action": "labeled", "label_name": "all-evals",
      "labels": ["full-sweep-enabled", "all-evals"]},
     ("success", "skipped", "RUN")),
    ("PR-labeled-evals-only-modifies-full-sweep",
     {**_PR, "action": "labeled", "label_name": "evals-only",
      "labels": ["full-sweep-enabled", "evals-only"]},
     ("success", "skipped", "RUN")),
    ("PR-labeled-agentx-fast-modifies-full-sweep",
     {**_PR, "action": "labeled", "label_name": "agentx-fast",
      "labels": ["full-sweep-enabled", "agentx-fast"]},
     ("success", "skipped", "RUN")),
    ("PR-labeled-skip-queue-restarts-full-sweep",
     {**_PR, "action": "labeled", "label_name": "skip_queue",
      "labels": ["full-sweep-enabled", "skip_queue"]},
     ("success", "skipped", "RUN")),
    ("PR-unlabeled-skip-queue-restarts-numeric-sweep",
     {**_PR, "action": "unlabeled", "label_name": "skip_queue",
      "labels": ["full-sweep-enabled"]},
     ("success", "skipped", "RUN")),
    ("PR-labeled-patchwork-restarts-full-sweep",
     {**_PR, "action": "labeled", "label_name": "ci-patchwork",
      "labels": ["full-sweep-enabled", "ci-patchwork"]},
     ("success", "skipped", "RUN")),
    ("PR-unlabeled-patchwork-restarts-full-sweep",
     {**_PR, "action": "unlabeled", "label_name": "ci-patchwork",
      "labels": ["full-sweep-enabled"]},
     ("success", "skipped", "RUN")),
    ("PR-labeled-with-unrelated-label",
     {**_PR, "action": "labeled", "label_name": "documentation",
      "labels": ["full-sweep-enabled"]}, ("skipped", "skipped", "SKIP")),
    ("PR-unlabeled-removed-sweep-label",
     {**_PR, "action": "unlabeled", "label_name": "full-sweep-enabled",
      "labels": []}, ("success", "skipped", "SKIP")),
    ("PR-draft",
     {**_PR, "action": "synchronize", "draft": True,
      "labels": ["full-sweep-enabled"]}, ("skipped", "skipped", "SKIP")),
    ("PR-ready-for-review",
     {**_PR, "action": "ready_for_review", "labels": ["full-sweep-enabled"],
      "reuse_auth": False}, ("success", "skipped", "RUN")),
    ("PR-sync-validation-requests-skip",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "check_skip": "true"},
     ("success", "success", "SKIP")),
    ("push-additions-no-skip",
     {"event": "push", "msg": "feat: add model"},
     ("skipped", "skipped", "RUN")),
    ("push-skip-sweep-tag-ignored",
     {"event": "push", "msg": "fix: x [skip-sweep]"},
     ("skipped", "skipped", "RUN")),
]


@pytest.mark.parametrize("scenario,expected", [(c[1], c[2]) for c in CASES],
                         ids=[c[0] for c in CASES])
def test_gating_decision(
    scenario: dict,
    expected: tuple[str, str, str],
) -> None:
    assert run_dag(scenario) == expected


def test_priority_classifier_runs_only_for_enabled_pull_requests() -> None:
    scenario = {
        **_PR,
        "action": "synchronize",
        "labels": ["full-sweep-enabled"],
    }
    disabled = _ctx({**scenario, "scheduler_enabled": "false"})
    enabled_pr = _ctx({**scenario, "scheduler_enabled": "true"})
    enabled_push = _ctx({"event": "push", "scheduler_enabled": "true"})

    assert not _eval(CLASSIFIER_IF, disabled)
    assert _eval(CLASSIFIER_IF, enabled_pr)
    assert not _eval(CLASSIFIER_IF, enabled_push)


@pytest.mark.parametrize("failed_job", ["check-changelog", "reuse-sweep-gate"])
@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_setup_does_not_run_when_a_prerequisite_fails(failed_job, result) -> None:
    ctx = _ctx({**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"]})
    ctx.update({
        "needs.check-changelog.result": "success",
        "needs.check-changelog.outputs.skip-pr-sweep": "false",
        "needs.reuse-sweep-gate.result": "success",
        "needs.reuse-sweep-gate.outputs.skip-pr-sweep": "false",
        f"needs.{failed_job}.result": result,
    })

    assert not _eval(SETUP_IF, ctx)


@pytest.mark.parametrize("labels,returncode", [
    ([], 0),
    (["full-sweep-enabled", "all-evals"], 0),
    (["full-sweep-enabled", "sweep-enabled"], 1),
])
def test_conflicting_sweep_labels_are_rejected(labels, returncode) -> None:
    step = next(step for step in _WF["jobs"]["check-changelog"]["steps"]
                if step.get("name") == "Reject conflicting sweep labels")
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, "SWEEP_LABELS": json.dumps(labels)},
        capture_output=True, text=True,
    )

    assert result.returncode == returncode, result.stderr


@pytest.mark.parametrize("message,expected", [
    ("fix: normal change", "false"),
    ("fix: docs\n\n[skip-sweep]", "true"),
])
def test_skip_policy_reads_the_commit_message(tmp_path, message, expected) -> None:
    step = next(step for step in _WF["jobs"]["check-changelog"]["steps"]
                if step.get("id") == "sweep_policy")
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-e", "-c", 'git() { printf "%s\\n" "$TEST_COMMIT_MESSAGE"; };\n' + step["run"]],
        env={**os.environ, "HEAD_SHA": "test-head", "TEST_COMMIT_MESSAGE": message,
             "GITHUB_OUTPUT": str(output)},
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == f"skip-pr-sweep={expected}"
