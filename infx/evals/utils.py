import random
import datasets

LETTERS = "ABCD"

def process_docs(dataset: datasets.Dataset, n_repeats: int = 2, seed: int = 3407) -> datasets.Dataset:
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