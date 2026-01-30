import random
import datasets

LETTERS = "ABCD"

def process_docs(dataset: datasets.Dataset, n_repeats: int = 8, seed: int = 0) -> datasets.Dataset:
    """
    Expand each GPQA row into n_repeats rows with independently permuted options.
    Converts:
      - Question
      - Correct Answer / Incorrect Answer 1..3
    into:
      - choice1..choice4
      - answer = "(A)" / "(B)" / "(C)" / "(D)" (the correct option after permutation)
    """
    rows = []
    for i, doc in enumerate(dataset):
        # Base choices in a known order: correct first
        choices = [
            doc["Correct Answer"],
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
        ]
        correct_idx = 0

        for r in range(n_repeats):
            rng = random.Random(seed + i * 1000 + r)
            perm = list(range(4))
            rng.shuffle(perm)

            new_choices = [choices[j] for j in perm]
            new_correct_idx = perm.index(correct_idx)
            new_ans = f"({LETTERS[new_correct_idx]})"

            new_doc = dict(doc)
            # Your YAML expects these:
            new_doc["choice1"], new_doc["choice2"], new_doc["choice3"], new_doc["choice4"] = new_choices
            new_doc["answer"] = new_ans
            new_doc["repeat_id"] = r  # optional debug

            rows.append(new_doc)

    return datasets.Dataset.from_list(rows)