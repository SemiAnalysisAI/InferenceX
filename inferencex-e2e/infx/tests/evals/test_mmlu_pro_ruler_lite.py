"""Deterministic MMLU-Pro sampling and RULER-lite generation and scoring."""

from __future__ import annotations

import datasets

from infx.evals.lm_eval_tasks import mmlu_pro_utils, ruler_lite


def test_mmlu_pro_sample_is_stratified_and_deterministic() -> None:
    rows = [
        {"question_id": i, "category": category, "question": "q", "options": ["a", "b"]}
        for category in ("law", "math")
        for i in range(300 if category == "law" else 150)
    ]
    for i, row in enumerate(rows):
        row["question_id"] = i
    dataset = datasets.Dataset.from_list(rows)
    first = mmlu_pro_utils.process_docs(dataset)
    second = mmlu_pro_utils.process_docs(dataset)
    assert first["question_id"] == second["question_id"]
    counts = {c: first["category"].count(c) for c in ("law", "math")}
    assert counts == {"law": mmlu_pro_utils.PER_SUBJECT, "math": 150}


def test_mmlu_pro_prompt_asks_for_the_letter_format() -> None:
    doc = {"category": "math", "question": "1+1?", "options": ["1", "2"]}
    text = mmlu_pro_utils.doc_to_text(doc)
    assert 'the answer is (X)' in text
    assert "A. 1\nB. 2" in text


def test_ruler_lite_docs_are_deterministic_and_gold_scores_one(monkeypatch) -> None:
    monkeypatch.setattr(ruler_lite, "_tokenizer", lambda: None)
    for task in ruler_lite.TASKS:
        first = ruler_lite._doc(task, 8192, 0)
        assert first == ruler_lite._doc(task, 8192, 0)
        assert first["length_source"] == "chars/4"
        assert 7000 < first["input_tokens"] <= 8192
        assert all(value in first["input"] for value in first["outputs"])
        gold = ruler_lite.process_results(first, [" ".join(first["outputs"])])
        assert gold == {"exact_match": 1.0}
        assert ruler_lite.process_results(first, ["nothing"]) == {"exact_match": 0.0}


def test_ruler_lite_partial_credit() -> None:
    doc = {"outputs": ["111", "222", "333", "444"]}
    assert ruler_lite.process_results(doc, ["111 and 333"]) == {"exact_match": 0.5}
