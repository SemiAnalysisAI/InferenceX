import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


output = Path(sys.argv[1])
image = json.loads((output / "image-inspect.json").read_text())[0]
repository = os.environ["IMAGE_REF"].rsplit(":", 1)[0]
digests = [
    value for value in image["RepoDigests"] if value.startswith(repository + "@sha256:")
]
if len(digests) != 1:
    raise ValueError("Expected exactly one published digest for the target image")
receipt = {
    "source_repository": "ROCm/device-metrics-exporter",
    "source_sha": os.environ["SOURCE_SHA"],
    "source_run_id": 36867107435,
    "source_artifact_id": 11169689175,
    "source_artifact_name": "build-dme-10.2.0a20261001",
    "gpu_agent_sha": os.environ["GPU_AGENT_SHA"],
    "cache_patch": {"sha256": sha256(output / "cache.patch"), "ttl": "0s"},
    "binary": {
        "name": "amd-metrics-exporter",
        "sha256": sha256(output / "amd-metrics-exporter"),
    },
    "archive": {
        "name": os.environ["ARCHIVE_NAME"],
        "sha256": os.environ["ARCHIVE_SHA256"],
    },
    "base_recovery": json.loads((output / "base-recovery.json").read_text()),
    "image": {
        "loaded_tag": os.environ["IMAGE_REF"],
        "config_digest": image["Id"],
        "reference": digests[0],
    },
    "squash": {
        "name": "amd-exporter.sqsh",
        "sha256": sha256(output / "amd-exporter.sqsh"),
    },
    "preparation": {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "workflow_sha": os.environ["GITHUB_SHA"],
    },
}
if receipt["cache_patch"]["sha256"] != os.environ["PATCH_SHA256"]:
    raise ValueError("Cache patch digest changed during preparation")
(output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(digests[0])
