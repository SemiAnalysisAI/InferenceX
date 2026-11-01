"""Exercise checkpoint staging without downloading weights or using a real cache."""

import os
import subprocess

import pytest

from infx.tests.bench.stubs import executable
from infx.tests.srt_slurm.test_setup_scripts import CONFIGS, MOUNT, ROOT


@pytest.mark.parametrize(
    ("changes", "cached", "success"),
    [
        ({}, False, True),
        ({"HF_CHECKPOINT_REVISION": "main"}, False, False),
        ({"HF_CHECKPOINT_BUDGET_BYTES": "0"}, False, False),
        ({"HF_HUB_OFFLINE": "1"}, False, False),
        ({"TRANSFORMERS_OFFLINE": "1"}, False, False),
        ({"AVAILABLE": "10"}, False, False),
        ({"AVAILABLE": "10"}, True, True),
    ],
)
def test_staging_requires_pinned_online_download_and_sufficient_cache_space(
    tmp_path, changes, cached, success
):
    tools = tmp_path / "bin"
    tools.mkdir()
    executable(tools / "df", '#!/bin/bash\nprintf "header\\nfs 1000 0 %s\\n" "$AVAILABLE"\n')
    executable(tools / "du", '#!/bin/bash\nprintf "100 cached\\n"\n')
    executable(tools / "hf", '#!/bin/bash\nprintf "%s\\n" "$@" > "$DOWNLOAD_TRACE"\n')
    mount = tmp_path / "mount.sh"
    mount.write_text(MOUNT)
    cache = tmp_path / "cache"
    if cached:
        (cache / "models--org--model").mkdir(parents=True)
    trace = tmp_path / "download"
    revision = "a" * 40
    result = subprocess.run(
        ["bash", str(CONFIGS / "stage-hf-checkpoint.sh")],
        env={
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "BASH_ENV": str(mount),
            "STUB_WORKSPACE": str(ROOT),
            "HF_HUB_CACHE": str(cache),
            "HF_CHECKPOINT_REPO": "org/model",
            "HF_CHECKPOINT_REVISION": revision,
            "HF_CHECKPOINT_BUDGET_BYTES": "100",
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "AVAILABLE": "200",
            "DOWNLOAD_TRACE": str(trace),
            **changes,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert trace.exists() is success
    if success:
        assert trace.read_text().splitlines() == [
            "download",
            "org/model",
            "--revision",
            revision,
            "--cache-dir",
            str(cache),
        ]
        assert (cache / ".models--org--model.download.lock").exists()
