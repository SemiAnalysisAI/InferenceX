"""Exercise checkpoint staging against a fake Hub and real temporary cache files."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.tests.srt_slurm.test_setup_scripts import CONFIGS, MOUNT, ROOT

REVISION = "a" * 40
HUB = '''
import json, os, shutil
from pathlib import Path
from types import SimpleNamespace

root = Path(os.environ["HF_HUB_CACHE"]) / "models--org--model"
revision = os.environ["HF_CHECKPOINT_REVISION"]
manifest = json.loads(os.environ["MANIFEST"])
shutil.disk_usage = lambda path: SimpleNamespace(free=int(os.environ["AVAILABLE"]))

class HfApi:
    def model_info(self, repo, *, revision, files_metadata):
        assert repo == "org/model" and files_metadata
        assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "0"
        return SimpleNamespace(sha=os.environ.get("RETURNED_SHA", revision),
                               siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in manifest])

def try_to_load_from_cache(repo, filename, *, revision, cache_dir):
    assert repo == "org/model" and cache_dir == os.environ["HF_HUB_CACHE"]
    return str(root / "snapshots" / revision / filename)

def snapshot_download(repo, *, revision, cache_dir):
    assert repo == "org/model" and cache_dir == os.environ["HF_HUB_CACHE"]
    Path(os.environ["DOWNLOAD_TRACE"]).write_text(revision)
    for filename, size in manifest:
        path = root / "snapshots" / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    if os.environ.get("INCOMPLETE_DOWNLOAD"):
        path.unlink()
'''


@pytest.mark.parametrize(
    ("changes", "cached", "old_revision", "success", "downloaded"),
    [
        ({}, {}, False, True, True),
        ({"HF_CHECKPOINT_REVISION": "main"}, {}, False, False, False),
        ({"HF_CHECKPOINT_BUDGET_BYTES": "0"}, {}, False, False, False),
        ({"AVAILABLE": "0"}, {"config.json": 2, "weights": 8}, False, True, False),
        ({"AVAILABLE": "0"}, {}, True, False, False),
        ({"AVAILABLE": "0"}, {"weights": 7}, False, False, False),
        ({"AVAILABLE": "3"}, {"weights": 8}, False, False, False),
        ({"AVAILABLE": "4"}, {"weights": 8}, True, True, True),
        ({"MANIFEST": "[]"}, {}, False, False, False),
        ({"MANIFEST": '[["weights", null]]'}, {}, False, False, False),
        ({"MANIFEST": '[["weights", -1]]'}, {}, False, False, False),
        ({"MANIFEST": '[["weights", true]]'}, {}, False, False, False),
        ({"MANIFEST": '[["weights", 1.5]]'}, {}, False, False, False),
        ({"RETURNED_SHA": "b" * 40}, {}, False, False, False),
        ({"INCOMPLETE_DOWNLOAD": "1"}, {}, False, False, True),
        ({"AVAILABLE": "0", "BROKEN_LINK": "1"}, {}, False, False, False),
    ],
)
def test_pinned_checkpoint_staging(tmp_path, changes, cached, old_revision, success, downloaded):
    (tmp_path / "huggingface_hub.py").write_text(HUB)
    mount = tmp_path / "mount.sh"
    mount.write_text(MOUNT)
    cache = tmp_path / "cache"
    storage = cache / "models--org--model"
    snapshot = storage / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name, size in cached.items():
        blob = storage / "blobs" / name
        blob.parent.mkdir(exist_ok=True)
        blob.write_bytes(b"x" * size)
        (snapshot / name).symlink_to(blob)
    if old_revision:
        old = storage / "snapshots" / ("b" * 40)
        old.mkdir()
        (old / "weights").write_bytes(b"x" * 1000)
        (storage / "blobs").mkdir(exist_ok=True)
        (storage / "blobs" / "weights.incomplete").write_bytes(b"x" * 1000)
    if changes.get("BROKEN_LINK"):
        (snapshot / "weights").symlink_to(storage / "blobs" / "missing")
    trace = tmp_path / "download"
    parent_env = tmp_path / "parent-env"
    result = subprocess.run(
        ["bash", "-c", 'bash "$STAGING_SCRIPT"; status=$?; '
         'printf "%s %s" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" > "$PARENT_ENV"; exit "$status"'],
        env={
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(tmp_path),
            "BASH_ENV": str(mount),
            "STUB_WORKSPACE": str(ROOT),
            "STAGING_SCRIPT": str(CONFIGS / "stage-hf-checkpoint.sh"),
            "HF_HUB_CACHE": str(cache),
            "HF_CHECKPOINT_REPO": "org/model",
            "HF_CHECKPOINT_REVISION": REVISION,
            "HF_CHECKPOINT_BUDGET_BYTES": "12",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AVAILABLE": "20",
            "MANIFEST": '[["config.json", 2], ["weights", 8]]',
            "DOWNLOAD_TRACE": str(trace),
            "PARENT_ENV": str(parent_env),
            **changes,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert trace.exists() is downloaded
    assert parent_env.read_text() == "1 1"
    if downloaded:
        assert trace.read_text() == REVISION
        assert (cache / ".models--org--model.download.lock").exists()
