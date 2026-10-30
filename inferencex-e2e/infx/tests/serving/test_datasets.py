import hashlib
import sys
import types

import pytest

from infx.serving.datasets import DatasetError, LockEntry, fetch, load_lock


def test_lock_pins_greenspec_and_speedbench_without_entries():
    lock = load_lock()
    assert set(lock) == {
        "tau2-cs",
        "shared-prefix-e2",
        "hidden-prefix-e3",
        "livekit-111868-v1",
        "livekit-111868-v2",
        "livekit-111868-v1.1-warm",
        "greenspec-v1.1.0",
        "speedbench-qualitative",
        "speedbench-throughput-1k",
        "speedbench-throughput-2k",
        "speedbench-throughput-8k",
        "speedbench-throughput-16k",
        "speedbench-throughput-32k",
    }
    assert {e.repo for n, e in lock.items() if n.startswith("speedbench-")} == {
        "Green-Lab/speed-bench-prepared"
    }
    assert (
        lock["speedbench-qualitative"].file == "qualitative.jsonl"
        and lock["speedbench-qualitative"].entries is None
    )
    assert (
        lock["greenspec-v1.1.0"].repo == "Green-Lab/GreenSpec"
        and lock["greenspec-v1.1.0"].entries is None
    )
    assert lock["livekit-111868-v1.1-warm"].entries == 671
    assert lock["livekit-111868-v1"].entries == 1217


@pytest.mark.parametrize(
    "body, message",
    [
        ("x: {repo: a/b}\n", "x needs exactly repo, revision, file, sha256"),
        ("x: {repo: a/b, revision: abc, file: f, sha256: " + "b" * 64 + "}\n", "x.revision"),
        # all-digit hashes would load as YAML ints, hence a/b
        (
            "x: {repo: a/b, revision: "
            + "a" * 40
            + ", file: f, sha256: "
            + "b" * 64
            + ", entries: 0}\n",
            "x.entries must be a positive integer",
        ),
    ],
)
def test_bad_lock_is_one_line_error(tmp_path, body, message):
    path = tmp_path / "lock.yaml"
    path.write_text(body)
    with pytest.raises(DatasetError, match=message):
        load_lock(path)


def _fake_hub(monkeypatch, path):
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: str(path)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


def test_fetch_verifies_sha256_and_records_the_dataset(tmp_path, monkeypatch):
    data = tmp_path / "trace.jsonl"
    data.write_text("{}\n")
    digest = hashlib.sha256(b"{}\n").hexdigest()
    _fake_hub(monkeypatch, data)
    monkeypatch.setenv("RESULT_DIR", str(tmp_path))
    lock = {"d": LockEntry("d", "o/r", "0" * 40, "trace.jsonl", digest, 3)}
    assert fetch("d", lock) == data
    assert (tmp_path / "dataset.txt").read_text().startswith("d o/r@")


def test_fetch_rejects_a_changed_file(tmp_path, monkeypatch):
    data = tmp_path / "trace.jsonl"
    data.write_text("tampered\n")
    _fake_hub(monkeypatch, data)
    lock = {"d": LockEntry("d", "o/r", "0" * 40, "trace.jsonl", "0" * 64, 3)}
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        fetch("d", lock)


def test_unknown_dataset(monkeypatch):
    with pytest.raises(DatasetError, match="dataset nope: not in lock.yaml"):
        fetch("nope", {})
