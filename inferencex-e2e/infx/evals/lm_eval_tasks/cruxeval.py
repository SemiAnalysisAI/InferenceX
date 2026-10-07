"""CRUXEval scoring for the vendored lm-eval task YAMLs.

Upstream lm-eval's cruxeval utils (EleutherAI/lm-evaluation-harness#3699) strip
quotes from predicted outputs, so a correct string answer executes as a bare
name, and its few-shot examples leave string values unquoted. This module
follows the reference CRUXEval check instead: execute the dataset function with
the completed assertion and count it correct when the assertion passes.

Only the model's assertion operand is taken from the generation. It is parsed
with ``ast`` and spliced into an assertion built from the dataset's own code,
input, and output, so the model cannot replace the function under test. Output
predictions must be literals, as the prompt requires. Input predictions may be
expressions, as in the reference (14 dataset inputs are lambdas, ``range`` or
``dict()`` calls), so each check runs in a separate isolated interpreter with a
time and memory limit.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

TIMEOUT_S = 5
MEMORY_BYTES = 2 << 30
_ANSWER = re.compile(r"\[ANSWER\](.*?)(?:\[/ANSWER\]|$)", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")

# Runs in a fresh isolated interpreter; the program arrives on stdin.
_RUNNER = """
import resource, sys
try:
    resource.setrlimit(resource.RLIMIT_AS, ({memory}, {memory}))
except (ValueError, OSError):
    pass
program = sys.stdin.read()
exec(compile(program, "<cruxeval>", "exec"), {{"__name__": "__cruxeval__"}})
"""


def _assertion(generation: str) -> ast.Compare | None:
    """Return ``f(...) == value`` from the last answer block of a generation."""
    blocks = _ANSWER.findall(generation or "")
    text = blocks[-1] if blocks else (generation or "")
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line or _FENCE.match(line):
            continue
        if not line.startswith("assert"):
            line = f"assert {line}"
        try:
            node = ast.parse(line).body[0]
        except (SyntaxError, IndexError):
            continue
        test = node.test if isinstance(node, ast.Assert) else None
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Call)
            and isinstance(test.left.func, ast.Name)
            and test.left.func.id == "f"
        ):
            return test
    return None


def build_program(doc: dict[str, Any], generation: str, mode: str) -> str | None:
    """Complete the dataset assertion with the model's prediction."""
    test = _assertion(generation)
    if test is None:
        return None
    if mode == "output":
        predicted = ast.unparse(test.comparators[0])
        # The prompt asks for a literal; every dataset output is one.
        try:
            ast.literal_eval(predicted)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
        return f"{doc['code']}\nassert f({doc['input']}) == {predicted}\n"
    call = test.left
    arguments = [ast.unparse(arg) for arg in call.args]
    arguments += [ast.unparse(keyword) for keyword in call.keywords]
    return f"{doc['code']}\nassert f({', '.join(arguments)}) == {doc['output']}\n"


def passes(program: str) -> bool:
    """Execute one completed assertion in an isolated, time-limited process."""
    with tempfile.TemporaryDirectory(prefix="cruxeval-") as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER.format(memory=MEMORY_BYTES)],
                input=program,
                capture_output=True,
                text=True,
                cwd=workdir,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                timeout=TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def _score(doc: dict[str, Any], results: list[str], mode: str) -> dict[str, float]:
    program = build_program(doc, results[0] if results else "", mode)
    return {"exact_match": float(program is not None and passes(program))}


def process_results_output(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _score(doc, results, "output")


def process_results_input(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _score(doc, results, "input")
