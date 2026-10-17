import random

import datasets

LETTERS = "ABCD"


def process_docs(
    dataset: datasets.Dataset, n_repeats: int = 2, seed: int = 3407
) -> datasets.Dataset:
    rng = random.Random(seed)  # noqa: S311
    docs = list(dataset)

    rows = []
    for r in range(n_repeats):
        for doc in docs:
            base_choices = [
                doc["Correct Answer"],
                doc["Incorrect Answer 1"],
                doc["Incorrect Answer 2"],
                doc["Incorrect Answer 3"],
            ]
            perm = rng.sample(range(4), 4)

            new_choices = [base_choices[j] for j in perm]
            correct_letter = LETTERS[perm.index(0)]  # where correct ended up

            new_doc = dict(doc)
            new_doc["A"], new_doc["B"], new_doc["C"], new_doc["D"] = new_choices
            new_doc["answer"] = correct_letter
            new_doc["repeat_id"] = r
            rows.append(new_doc)

    return datasets.Dataset.from_list(rows)


MMLU_PRO_LETTERS = "ABCDEFGHIJ"


def mmlu_pro_tenth(dataset: datasets.Dataset) -> datasets.Dataset:
    """Every 10th question of each category, in question_id order: 10% of each subject."""
    by_category: dict[str, list[int]] = {}
    for doc in dataset:
        by_category.setdefault(doc["category"], []).append(doc["question_id"])
    keep = {qid for ids in by_category.values() for qid in sorted(ids)[::10]}
    return dataset.filter(lambda doc: doc["question_id"] in keep)


def mmlu_pro_third(dataset: datasets.Dataset) -> datasets.Dataset:
    """Throwaway: a fixed third of MMLU-Pro test (question_id % 3 == 0), all subjects."""
    return dataset.filter(lambda doc: doc["question_id"] % 3 == 0)


def mmlu_pro_doc_to_text(doc: dict) -> str:
    options = "\n".join(f"({MMLU_PRO_LETTERS[i]}) {opt}" for i, opt in enumerate(doc["options"]))
    return (
        f"{doc['question']}\n\n{options}\n\n"
        "Think step by step, then finish with \"The answer is (X)\" where X is the letter of the correct option."
    )
