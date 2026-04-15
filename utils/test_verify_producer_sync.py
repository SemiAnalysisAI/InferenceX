import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "verify_producer_sync.py"


RELEVANT_FILES = {
    "extension_131k/sglang/code_131k1k_qwen3.5.json": {"name": "e131k"},
    "preview/long_context_500k/manifest_qwen3.5.json": {"name": "500k"},
    "preview/long_context_1m/manifest.json": {"name": "1m"},
}


def _write_tree(root: Path, files: dict[str, dict]) -> None:
    for relative_path, payload in files.items():
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, sort_keys=True))


def _run_verify(producer_root: Path, consumer_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--producer-root",
            str(producer_root),
            "--consumer-root",
            str(consumer_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_verify_producer_sync_passes_for_identical_trees(tmp_path: Path) -> None:
    producer_root = tmp_path / "producer"
    consumer_root = tmp_path / "consumer"
    _write_tree(producer_root, RELEVANT_FILES)
    _write_tree(consumer_root, RELEVANT_FILES)

    result = _run_verify(producer_root, consumer_root)

    assert result.returncode == 0
    assert "sync check passed" in result.stdout


def test_verify_producer_sync_fails_on_content_mismatch(tmp_path: Path) -> None:
    producer_root = tmp_path / "producer"
    consumer_root = tmp_path / "consumer"
    _write_tree(producer_root, RELEVANT_FILES)
    _write_tree(consumer_root, RELEVANT_FILES)

    mismatched_path = consumer_root / "preview/long_context_500k/manifest_qwen3.5.json"
    mismatched_path.write_text(json.dumps({"name": "changed"}, sort_keys=True))

    result = _run_verify(producer_root, consumer_root)

    assert result.returncode == 1
    assert "content_mismatch" in result.stderr
    assert "preview/long_context_500k/manifest_qwen3.5.json" in result.stderr
