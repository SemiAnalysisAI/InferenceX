"""Exhaustively verify run-sweep.yml's changelog gating for every case.

The simulation jobs in `.github/workflows/test-changelog-gate.yml` hand-copy
two of the gating `if` conditions and exercise two scenarios. This test instead
parses the REAL `check-changelog` -> `reuse-sweep-gate` -> `setup` conditions
out of `run-sweep.yml` and evaluates them with a minimal GitHub Actions
expression engine, so it cannot drift from production and it covers every
distinct skip/run decision (PR and push, draft, label, reuse, metadata-only,
and validation-failure paths).

The engine is grounded against a real GitHub outcome in
`test_engine_matches_real_github_run`.
"""

from __future__ import annotations

import itertools
import re
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
SETUP_IF = _WF["jobs"]["setup"]["if"]
PR_TYPES = set(_WF["on"]["pull_request"]["types"])

# All sweep labels, and the subset that authorizes artifact reuse. Kept here
# (not parsed) so the reference spec is an INDEPENDENT encoding of intent that
# the real run-sweep.yml conditions are cross-checked against.
SWEEP_LABELS = {
    "sweep-enabled",
    "full-sweep-enabled",
    "non-canary-full-sweep-enabled",
    "full-sweep-fail-fast",
    "full-sweep-fail-fast-no-canary",
}
REUSE_ELIGIBLE_LABELS = SWEEP_LABELS - {"sweep-enabled"}


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


# --------------------------------------------------------------------------
# DAG evaluation: check-changelog -> reuse-sweep-gate -> setup
# --------------------------------------------------------------------------
def _ctx(sc: dict) -> dict:
    return {
        "github.event_name": sc["event"],
        "github.event.action": sc.get("action"),
        "github.event.pull_request.draft": sc.get("draft", False),
        "github.event.pull_request.labels.*.name": sc.get("labels", []),
        "github.event.label.name": sc.get("label_name"),
        "github.event.head_commit.message": sc.get("msg", ""),
    }


def _ran_check_outcome(sc: dict) -> tuple[str, str]:
    """Outcome of a check-changelog job that actually runs (not a draft PR).

    Models the PR-only "Reject conflicting sweep labels" step (which fails the
    job when more than one sweep label is present); otherwise the validator
    step's success/failure governs, and has-additions is empty unless success.
    """
    labels = set(sc.get("labels", []))
    if sc["event"] == "pull_request" and len(labels & SWEEP_LABELS) > 1:
        return "failure", ""
    result = sc["check"]
    return result, (sc.get("has", "") if result == "success" else "")


def run_dag(sc: dict) -> tuple[str, str, str]:
    """Return (check-changelog result, reuse-sweep-gate result, setup decision).

    The job `if` conditions are evaluated from the REAL run-sweep.yml strings;
    only the in-job step outcomes (validator / conflicting-labels) are modelled.
    """
    ctx = _ctx(sc)

    if not _eval(CHECK_IF, ctx):
        cc_result, has = "skipped", ""
    else:
        cc_result, has = _ran_check_outcome(sc)
    ctx["needs.check-changelog.result"] = cc_result
    ctx["needs.check-changelog.outputs.has-additions"] = has

    if not _eval(GATE_IF, ctx):
        gate_result, skip = "skipped", ""
    else:
        gate_result = "success"
        skip = "true" if sc.get("reuse_auth") else ""
    ctx["needs.reuse-sweep-gate.result"] = gate_result
    ctx["needs.reuse-sweep-gate.outputs.skip-pr-sweep"] = skip

    setup = "RUN" if _eval(SETUP_IF, ctx) else "SKIP"
    return cc_result, gate_result, setup


_PR = {"event": "pull_request", "draft": False, "check": "success"}

# (id, scenario, expected (check, reuse, setup))
CASES = [
    ("PR-sync-full-noreuse",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "has": "true", "reuse_auth": False}, ("success", "success", "RUN")),
    ("PR-sync-full-reuse-authorized",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "has": "true", "reuse_auth": True}, ("success", "success", "SKIP")),
    ("PR-sync-trim-sweep-enabled",
     {**_PR, "action": "synchronize", "labels": ["sweep-enabled"],
      "has": "true"}, ("success", "skipped", "RUN")),
    ("PR-sync-no-sweep-label",
     {**_PR, "action": "synchronize", "labels": [], "has": "true"},
     ("success", "skipped", "SKIP")),
    ("PR-labeled-with-sweep-label",
     {**_PR, "action": "labeled", "label_name": "full-sweep-enabled",
      "labels": ["full-sweep-enabled"], "has": "true"},
     ("success", "skipped", "RUN")),
    ("PR-labeled-with-unrelated-label",
     {**_PR, "action": "labeled", "label_name": "documentation",
      "labels": ["full-sweep-enabled"], "has": "true"},
     ("success", "skipped", "SKIP")),
    ("PR-unlabeled-removed-sweep-label",
     {**_PR, "action": "unlabeled", "label_name": "full-sweep-enabled",
      "labels": [], "has": "true"}, ("success", "skipped", "SKIP")),
    ("PR-draft",
     {**_PR, "action": "synchronize", "draft": True,
      "labels": ["full-sweep-enabled"], "has": "true"},
     ("skipped", "skipped", "SKIP")),
    ("PR-ready-for-review",
     {**_PR, "action": "ready_for_review", "labels": ["full-sweep-enabled"],
      "has": "true", "reuse_auth": False}, ("success", "skipped", "RUN")),
    ("PR-sync-metadata-only",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "has": "false"}, ("success", "skipped", "SKIP")),
    ("PR-sync-changelog-invalid",
     {**_PR, "action": "synchronize", "labels": ["full-sweep-enabled"],
      "check": "failure"}, ("failure", "skipped", "SKIP")),
    ("PR-sync-conflicting-sweep-labels",
     {**_PR, "action": "synchronize",
      "labels": ["sweep-enabled", "full-sweep-enabled"], "check": "failure"},
     ("failure", "skipped", "SKIP")),
    ("push-additions-no-skip",
     {"event": "push", "check": "success", "has": "true",
      "msg": "feat: add model"}, ("success", "skipped", "RUN")),
    ("push-skip-sweep-tag",
     {"event": "push", "check": "success", "has": "true",
      "msg": "fix: x [skip-sweep]"}, ("success", "skipped", "SKIP")),
    ("push-metadata-only",
     {"event": "push", "check": "success", "has": "false",
      "msg": "fix: link"}, ("success", "skipped", "SKIP")),
    # malformed changelog merged to main -> ingest blocked (recovery needed).
    ("push-changelog-invalid",
     {"event": "push", "check": "failure", "msg": "feat: add model"},
     ("failure", "skipped", "SKIP")),
]


@pytest.mark.parametrize("scenario,expected", [(c[1], c[2]) for c in CASES],
                         ids=[c[0] for c in CASES])
def test_gating_decision(scenario: dict, expected: tuple[str, str, str]) -> None:
    assert run_dag(scenario) == expected


def test_engine_matches_real_github_run() -> None:
    # Ground truth: run-sweep run 27737489942 on PR #1821 (no labels, not a
    # draft, metadata-only) recorded check-changelog=success,
    # reuse-sweep-gate=skipped, setup=skipped.
    real = {**_PR, "action": "synchronize", "labels": [], "has": "false"}
    assert run_dag(real) == ("success", "skipped", "SKIP")


def test_engine_self_consistency() -> None:
    checks = [
        ("always()", {}, True),
        ("!false", {}, True),
        ("'a' == 'a'", {}, True),
        ("'a' != 'b'", {}, True),
        ("x != 'true'", {"x": ""}, True),
        ("x != 'true'", {"x": "true"}, False),
        ("a && b", {"a": "true", "b": ""}, False),
        ("a || b", {"a": "", "b": "true"}, True),
        ("contains(L, 'z')", {"L": ["z"]}, True),
        ("contains(L, 'z')", {"L": ["q"]}, False),
        ("contains(M, '[skip-sweep]')", {"M": "x [skip-sweep]"}, True),
        ("!d", {"d": True}, False),
        ("(a || b) && c", {"a": "", "b": "true", "c": "true"}, True),
    ]
    for expr, ctx, want in checks:
        assert _eval(expr, ctx) is want, expr


def test_trigger_types_enable_gated_events() -> None:
    assert {"synchronize", "labeled", "unlabeled", "ready_for_review"} <= PR_TYPES
    # opened/reopened are intentionally excluded so opening or reopening a PR
    # that already carries a sweep label does not start a sweep.
    assert {"opened", "reopened"}.isdisjoint(PR_TYPES)


# --------------------------------------------------------------------------
# Independent reference spec of the INTENDED gating, plus an exhaustive
# cross-product cross-check: every combination of the input axes is fed to
# both the reference spec and the engine driving the REAL run-sweep.yml `if`
# strings; any disagreement is either a spec error or a gating bug.
# --------------------------------------------------------------------------
def reference_gate(sc: dict) -> tuple[str, str, str]:
    """Hand-written reference for (check, reuse, setup) from documented intent."""
    labels = set(sc.get("labels", []))
    draft = sc.get("draft", False)
    is_pr = sc["event"] == "pull_request"

    if is_pr and draft:
        check, has = "skipped", ""
    elif is_pr and len(labels & SWEEP_LABELS) > 1:
        check, has = "failure", ""
    else:
        check = sc["check"]
        has = sc.get("has", "") if check == "success" else ""

    gate_runs = (
        check == "success"
        and has == "true"
        and is_pr
        and sc.get("action") == "synchronize"
        and not draft
        and bool(labels & REUSE_ELIGIBLE_LABELS)
    )
    reuse = "success" if gate_runs else "skipped"
    authorized = gate_runs and sc.get("reuse_auth", False)
    reuse_clause = (reuse == "skipped") or (reuse == "success" and not authorized)

    if is_pr:
        action = sc.get("action")
        action_ok = action not in ("labeled", "unlabeled") or (
            sc.get("label_name") in SWEEP_LABELS
        )
        event_ok = (not draft) and bool(labels & SWEEP_LABELS) and action_ok
    else:
        event_ok = "[skip-sweep]" not in sc.get("msg", "")

    runs = check == "success" and has == "true" and reuse_clause and event_ok
    return check, reuse, ("RUN" if runs else "SKIP")


def _all_scenarios() -> list[dict]:
    label_cfgs = [
        [],
        ["sweep-enabled"],
        ["full-sweep-enabled"],
        ["non-canary-full-sweep-enabled"],
        ["full-sweep-fail-fast"],
        ["full-sweep-fail-fast-no-canary"],
        ["documentation"],
        ["sweep-enabled", "full-sweep-enabled"],
        ["full-sweep-enabled", "full-sweep-fail-fast"],
    ]
    pr_axes = itertools.product(
        ["ready_for_review", "synchronize", "labeled", "unlabeled"],  # action
        [False, True],                      # draft
        label_cfgs,                         # labels
        ["full-sweep-enabled", "sweep-enabled", "documentation", None],  # label.name
        ["success", "failure"],             # validator outcome
        ["true", "false"],                  # has-additions
        [False, True],                      # reuse authorized
    )
    scenarios = [
        {"event": "pull_request", "action": a, "draft": d, "labels": labs,
         "label_name": ln, "check": chk, "has": h, "reuse_auth": r}
        for a, d, labs, ln, chk, h, r in pr_axes
    ]
    push_axes = itertools.product(
        ["success", "failure"],                       # validator outcome
        ["true", "false"],                            # has-additions
        ["feat: add model", "fix: thing [skip-sweep]"],  # commit message
    )
    scenarios += [
        {"event": "push", "check": chk, "has": h, "msg": m}
        for chk, h, m in push_axes
    ]
    return scenarios


def test_exhaustive_cross_product() -> None:
    scenarios = _all_scenarios()
    mismatches = [
        (sc, run_dag(sc), reference_gate(sc))
        for sc in scenarios
        if run_dag(sc) != reference_gate(sc)
    ]
    assert not mismatches, mismatches[:10]
    # Sanity: confirm the sweep actually covered the whole input space
    # (4 actions x 2 draft x 9 label-configs x 4 label-names x 2 check x
    # 2 has-additions x 2 reuse = 2304 PR cases, plus 8 push cases).
    assert len(scenarios) == 2312


def test_named_cases_match_reference_spec() -> None:
    for case_id, scenario, expected in CASES:
        assert reference_gate(scenario) == expected, case_id
