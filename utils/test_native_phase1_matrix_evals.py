"""The native pilot eval boundary is narrower than legacy all-evals selection."""

import pytest

from infx.matrix.generate import select_matrix_evals


def agentic_row(name, concurrency, execution=None):
    row = {
        "exp-name": name,
        "model": "fixture/model",
        "model-prefix": "fixture",
        "runner": "cluster:fixture",
        "framework": "vllm",
        "precision": "fp4",
        "scenario-type": "agentic-coding",
        "conc": concurrency,
        "tp": 8,
        "run-eval": False,
    }
    if execution is not None:
        row["execution"] = execution
    return row


@pytest.mark.parametrize(
    "native_concurrencies, expected_native",
    [([1, 16, 28], [("native-28", 28)]), ([1, 16], [])],
)
def test_all_evals_keeps_native_c28_only_and_all_legacy_points(
    native_concurrencies, expected_native
):
    rows = [
        agentic_row(
            f"native-{concurrency}",
            concurrency,
            {"runtime": "srt-slurm", "contract-version": 1},
        )
        for concurrency in native_concurrencies
    ]
    rows.extend(
        agentic_row(f"legacy-{concurrency}", concurrency) for concurrency in (1, 16, 28)
    )
    result = select_matrix_evals(rows, mode="all")
    assert [(row["exp-name"], row["conc"]) for row in result] == expected_native + [
        ("legacy-1", 1),
        ("legacy-16", 16),
        ("legacy-28", 28),
    ]
    assert [
        (row["run-eval"], row["eval-only"], row["eval-framework"]) for row in result
    ] == [(True, True, "lm-eval")] * len(result)


def test_all_evals_clears_representative_selection_when_native_c28_is_absent():
    result = select_matrix_evals(
        [
            agentic_row("native-1", 1, {"runtime": "srt-slurm", "contract-version": 1}),
            agentic_row(
                "native-16", 16, {"runtime": "srt-slurm", "contract-version": 1}
            ),
        ],
        mode="all",
    )
    # Default selection initially chooses c16. Expansion must remove that
    # selection rather than submitting an eval the native adapter will reject.
    assert result == []
