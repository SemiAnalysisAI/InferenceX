"""CRUXEval scorer: dataset-built assertions, literal outputs, isolated execution."""

from __future__ import annotations

from pathlib import Path

import yaml

from infx.evals import cruxeval

EVALS = Path(cruxeval.__file__).parent
DOC = {
    "code": 'def f(s):\n    return s + "a"',
    "input": '"x9j"',
    "output": '"x9ja"',
}


def _score(generation: str, mode: str) -> float:
    return cruxeval._score(DOC, [generation], mode)["exact_match"]


def test_quoted_string_output_passes() -> None:
    assert _score('Reasoning.\n[ANSWER]\nassert f("x9j") == "x9ja"\n[/ANSWER]', "output") == 1.0


def test_wrong_output_fails() -> None:
    assert _score('[ANSWER]\nassert f("x9j") == "x9j"\n[/ANSWER]', "output") == 0.0


def test_last_answer_block_and_code_fence_are_used() -> None:
    generation = (
        '[ANSWER]\nassert f("x9j") == "no"\n[/ANSWER]\n'
        '[ANSWER]\n```python\nassert f("x9j") == "x9ja"\n```\n[/ANSWER]'
    )
    assert _score(generation, "output") == 1.0


def test_output_prediction_must_be_literal() -> None:
    generation = '[ANSWER]\nassert f("x9j") == "x9j" + "a"\n[/ANSWER]'
    assert cruxeval.build_program(DOC, generation, "output") is None


def test_model_cannot_replace_the_dataset_input_or_function() -> None:
    program = cruxeval.build_program(DOC, 'assert f("zz") == "x9ja"', "output")
    assert program == f'{DOC["code"]}\nassert f("x9j") == "x9ja"\n'
    assert cruxeval.build_program(DOC, 'assert f("x9j") == "x9ja" or True', "output") is None


def test_input_prediction_accepts_any_passing_input() -> None:
    assert _score('[ANSWER]\nassert f("x9j") == "x9ja"\n[/ANSWER]', "input") == 1.0
    assert _score('[ANSWER]\nassert f("x9" + "j") == "anything"\n[/ANSWER]', "input") == 1.0
    assert _score('[ANSWER]\nassert f("q") == "x9ja"\n[/ANSWER]', "input") == 0.0


def test_unparseable_generation_fails() -> None:
    assert _score("I am not sure.", "output") == 0.0


def test_non_terminating_function_times_out() -> None:
    doc = {"code": "def f(x):\n    while True:\n        pass", "input": "1", "output": "1"}
    assert cruxeval._score(doc, ["assert f(1) == 1"], "output")["exact_match"] == 0.0


def test_task_yamls_use_the_repo_scorer() -> None:
    for task, hook in (("cruxeval_output", "output"), ("cruxeval_input", "input")):
        text = (EVALS / f"{task}.yaml").read_text()
        assert f"process_results: !function cruxeval.process_results_{hook}" in text
        config = yaml.safe_load(text.replace("!function ", ""))
        assert config["task"] == task
        assert config["unsafe_code"] is True
        assert config["filter_list"][0]["name"] == "strict-match"
