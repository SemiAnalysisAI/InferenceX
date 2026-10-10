"""``python -m infx.bench.speedbench`` against a stand-in for NeMo-Skills' prepare.py."""

from __future__ import annotations

import hashlib
import json

import pytest

from infx.bench import speedbench

# The names the command uses, over a split of its own. Resolving a non-coding row fails, as
# resolving one from a gated dataset would.
FAKE_PREPARE_PY = """
PLACEHOLDER = "FETCH FROM SOURCE"
SPLITS = {
    ("nvidia/SPEED-Bench", "qualitative", "test", "rev-under-test"): [
        {"question_id": "q1", "category": "math", "turns": ["What is 2 + 2?"]},
        {"question_id": "q2", "category": "coding", "turns": ["Write fizzbuzz.", "Now in Go."]},
        {"question_id": "q3", "category": "coding", "turns": [PLACEHOLDER + " Use Python."]},
        {"question_id": "q4", "category": "coding", "turns": ["Trie une liste à l'envers."]},
    ]
}


class Dataset(list):
    def filter(self, function):
        return Dataset(row for row in self if function(row))


def load_dataset(path, name, split, revision):
    return Dataset(SPLITS[(path, name, split, revision)])


def _resolve_external_data(dataset, speed_config):
    def resolve(row):
        if row["category"] != "coding" or speed_config != "qualitative":
            raise RuntimeError(f"{row['question_id']} resolves from a gated dataset")
        turns = [turn.replace(PLACEHOLDER, "Solve problem 7.") for turn in row["turns"]]
        return {**row, "turns": turns}

    return Dataset(resolve(row) for row in dataset)
"""
CODING = ["Write fizzbuzz.", "Solve problem 7. Use Python.", "Trie une liste à l'envers."]
# sha256sum of CODING as one UTF-8 JSON array.
CODING_SHA256 = "cc3481dc93a4b3910188bb25a29709b7ee7f63cd2a3205425f6bfa4ee2e554ff"


@pytest.fixture
def prepare_py(tmp_path, monkeypatch):
    """The stand-in, served as the reviewed prepare.py; the golden prompts are CODING."""
    path = tmp_path / "prepare.py"
    path.write_text(FAKE_PREPARE_PY, encoding="utf-8")
    reviewed = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(speedbench, "PREPARE_PY", path.as_uri())
    monkeypatch.setattr(speedbench, "PREPARE_PY_SHA256", reviewed)
    monkeypatch.setattr(speedbench, "SPEED_BENCH_REVISION", "rev-under-test")
    monkeypatch.setattr(speedbench, "PROMPTS_SHA256", CODING_SHA256)
    return path


def test_the_coding_rows_first_user_messages_are_written_in_order(tmp_path, prepare_py, capsys):
    out = tmp_path / "out"

    assert speedbench.main(["--output-dir", str(out)]) == 0

    lines = (out / "speedbench_coding.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"question_id": "q2", "prompt": CODING[0]},
        {"question_id": "q3", "prompt": CODING[1]},
        {"question_id": "q4", "prompt": CODING[2]},
    ]
    # Gate containers read the same prompts back.
    assert speedbench.load(out / "speedbench_coding.jsonl") == CODING
    assert f"Wrote 3 SPEED-Bench prompts to {out / 'speedbench_coding.jsonl'}" in (
        capsys.readouterr().out
    )


def test_prompts_other_than_the_golden_ones_are_not_written(tmp_path, prepare_py, monkeypatch):
    # SPEED-Bench or a source it resolves from changed since the golden collections.
    monkeypatch.setattr(speedbench, "PROMPTS_SHA256", "0" * 64)

    with pytest.raises(SystemExit, match=f"3 prompts with sha256 {CODING_SHA256}, not 0+$"):
        speedbench.main(["--output-dir", str(tmp_path / "out")])

    assert not (tmp_path / "out").exists()


def test_only_the_reviewed_prepare_py_runs(tmp_path, prepare_py):
    ran = tmp_path / "ran"
    prepare_py.write_text(f"{FAKE_PREPARE_PY}open({str(ran)!r}, 'w').close()\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="prepare.py has sha256"):
        speedbench.main(["--output-dir", str(tmp_path / "out")])

    assert not ran.exists()
    assert not (tmp_path / "out").exists()
