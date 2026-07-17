import random
import re
import datasets

LETTERS = "ABCD"

_AIME_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(\d{1,3})\s*\}")
_AIME_INTEGER_RE = re.compile(r"(?<![\d.])(\d{1,3})(?!\d)(?!\.\d)")


def process_aime_results(doc: dict, results: list[str]) -> dict[str, int]:
    response = results[0]
    boxed = _AIME_BOXED_RE.findall(response)
    candidates = boxed or _AIME_INTEGER_RE.findall(response)
    if not candidates:
        return {"exact_match": 0}

    prediction = int(candidates[-1])
    target = int(doc["answer"])
    return {"exact_match": int(prediction == target)}


def process_docs(
    dataset: datasets.Dataset, n_repeats: int = 2, seed: int = 3407
) -> datasets.Dataset:
    rng = random.Random(seed)
    docs = list(dataset)

    rows = []
    for r in range(n_repeats):
        for i, doc in enumerate(docs):
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
