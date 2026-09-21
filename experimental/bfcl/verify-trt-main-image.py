"""Verify the fresh test container matches the package hashed at image build time."""

import hashlib
import json
import os
from pathlib import Path

import tensorrt_llm

manifest = json.loads(Path('/opt/inferencex-trt-main-build.json').read_text())
assert manifest['source_commit'] == os.environ['TRT_LLM_GIT_COMMIT']
assert manifest['runtime_source_patches'] is False
root = Path(tensorrt_llm.__file__).parent
for relative, expected in manifest['installed_files'].items():
    with (root / relative).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == expected, relative
print(f"Verified upstream {manifest['source_commit']}: "
      f"{len(manifest['installed_files'])} source/native files match the built image.")
