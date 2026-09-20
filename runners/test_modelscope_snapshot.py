"""Check snapshot provenance with synthetic files and a mocked hub download."""

import json
import sys
import types
from pathlib import Path

import pytest

from runners.modelscope_snapshot import record_empty_caches, record_snapshot


def test_empty_cache_requirement(tmp_path):
    cache, hf_home = tmp_path / "modelscope", tmp_path / "hf"
    cache.mkdir()
    hf_home.mkdir()
    report = tmp_path / "report.json"
    (cache / "old-file").touch()
    with pytest.raises(ValueError, match="Expected an empty cache"):
        record_empty_caches("test/model", cache, hf_home, report)
    assert not report.exists()


def test_snapshot_evidence_and_isolation(tmp_path, monkeypatch):
    cache, hf_home = tmp_path / "modelscope", tmp_path / "hf"
    cache.mkdir()
    hf_home.mkdir()
    report = tmp_path / "report.json"
    record_empty_caches("test/model", cache, hf_home, report)
    snapshot = cache / "snapshot"
    snapshot.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (snapshot / name).write_bytes(b"abc")
    downloader = types.ModuleType("tensorrt_llm.llmapi.utils")
    downloader.download_hf_model = lambda model: snapshot
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi.utils", downloader)

    record_snapshot("test/model", cache, hf_home, report)
    result = json.loads(report.read_text())
    assert Path(result["snapshot"]).samefile(snapshot)
    assert result["modelscope_initial_entries"] == []
    assert result["assets"]["model.safetensors"] == {
        "bytes": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }
    assert result["hf_files_after_startup"] == []

    (hf_home / "modules").mkdir()
    for name in ("modules/__init__.py", "modules/hf_remote_code.lock"):
        (hf_home / name).touch()
    record_snapshot("test/model", cache, hf_home, report)
    assert set(json.loads(report.read_text())["hf_files_after_startup"]) == {
        "modules/__init__.py",
        "modules/hf_remote_code.lock",
    }
    (hf_home / "modules/__init__.py").write_text("downloaded code")
    with pytest.raises(ValueError, match="Hugging Face fallback"):
        record_snapshot("test/model", cache, hf_home, report)
    (hf_home / "modules/__init__.py").write_text("")

    downloader.download_hf_model = lambda model: tmp_path
    with pytest.raises(ValueError, match="outside the fresh ModelScope cache"):
        record_snapshot("test/model", cache, hf_home, report)

    downloader.download_hf_model = lambda model: snapshot
    (hf_home / "tokenizer.json").touch()
    with pytest.raises(ValueError, match="Hugging Face fallback"):
        record_snapshot("test/model", cache, hf_home, report)
