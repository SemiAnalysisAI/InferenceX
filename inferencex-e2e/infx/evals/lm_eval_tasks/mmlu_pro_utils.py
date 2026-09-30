"""MMLU-Pro stratified sample for the mmlu_pro_2800 lm-eval task.

The bundled lm-eval mmlu_pro is a 14-subtask group whose --limit applies per
subject and whose group row carries no task config for collection. This task is
one flat task over a fixed, seeded sample of PER_SUBJECT questions from every
category, prompted zero-shot with the upstream chain-of-thought format.
"""

from __future__ import annotations

import random

import datasets

PER_SUBJECT = 200
SEED = 1234
CHOICES = "ABCDEFGHIJ"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    by_category: dict[str, list[int]] = {}
    for index, category in enumerate(dataset["category"]):
        by_category.setdefault(category, []).append(index)
    selected = []
    for category in sorted(by_category):
        indices = sorted(by_category[category], key=lambda i: dataset[i]["question_id"])
        random.Random(f"{SEED}:{category}").shuffle(indices)
        selected += indices[:PER_SUBJECT]
    return dataset.select(sorted(selected))


def doc_to_text(doc: dict) -> str:
    options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(CHOICES, doc["options"], strict=False)
    )
    return (
        f"The following is a multiple choice question about {doc['category']}. "
        'Think step by step and then finish your answer with "the answer is (X)" '
        "where X is the correct letter choice.\n\n"
        f"Question:\n{doc['question']}\nOptions:\n{options}\n"
        "Answer: Let's think step by step."
    )
