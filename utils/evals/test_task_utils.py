from datasets import Dataset

from utils import process_aime_results, process_docs


def test_aime_prefers_boxed_answer_over_trailing_numbers():
    result = process_aime_results(
        {"answer": 42},
        ["After 12 steps, the answer is \\boxed{042}. Confidence: 99%."],
    )

    assert result == {"exact_match": 1}


def test_aime_uses_last_standalone_integer_without_boxed_answer():
    result = process_aime_results(
        {"answer": 7},
        ["The intermediate value is 12, so the final answer is 007."],
    )

    assert result == {"exact_match": 1}


def test_aime_rejects_missing_integer_answer():
    result = process_aime_results(
        {"answer": 3},
        ["I cannot determine the answer."],
    )

    assert result == {"exact_match": 0}


def test_gpqa_shuffles_choices_without_losing_the_correct_answer():
    source = Dataset.from_list(
        [
            {
                "Question": "Which choice is correct?",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong one",
                "Incorrect Answer 2": "wrong two",
                "Incorrect Answer 3": "wrong three",
            }
        ]
    )

    processed = process_docs(source, n_repeats=2, seed=3407)

    assert len(processed) == 2
    assert {row["repeat_id"] for row in processed} == {0, 1}
    for row in processed:
        assert row[row["answer"]] == "correct"
        assert {row[letter] for letter in "ABCD"} == {
            "correct",
            "wrong one",
            "wrong two",
            "wrong three",
        }
