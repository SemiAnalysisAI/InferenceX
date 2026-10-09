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

Reasoning models draft and revise assertions, answer tags included, while they
think, so only the text after the last ``</think>`` can hold the answer. A
generation without ``</think>`` counts only when it ends with a closed answer
block, as when the server returns the reasoning separately; otherwise it was
cut off mid-reasoning and scores zero rather than whatever draft it last wrote.
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
_THINK_END = "</think>"
_ANSWER_END = "[/ANSWER]"

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


def _final_answer(generation: str) -> str | None:
    """The part of a generation that can hold its answer; ``None`` if it was cut off."""
    generation = generation or ""
    if _THINK_END in generation:
        return generation.rsplit(_THINK_END, 1)[1]
    return generation if generation.rstrip().endswith(_ANSWER_END) else None


def _assertion(generation: str) -> ast.Compare | None:
    """Return ``f(...) == value`` from the last answer block of a generation."""
    answer = _final_answer(generation)
    if answer is None:
        return None
    blocks = _ANSWER.findall(answer)
    text = blocks[-1] if blocks else answer
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line or _FENCE.match(line):
            continue
        if not line.startswith("assert"):
            line = f"assert {line}"
        try:
            node = ast.parse(line).body[0]
        # ValueError: an integer literal past Python's int-to-str digit limit.
        except (SyntaxError, IndexError, ValueError, RecursionError, MemoryError):
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
                # Only the exit status matters; discarding output keeps a model that
                # prints in a loop from growing this unsandboxed process's memory.
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
