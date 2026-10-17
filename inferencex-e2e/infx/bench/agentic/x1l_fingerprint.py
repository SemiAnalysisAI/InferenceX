"""Record the installed AgentX harness; exit 1 unless it is the vendored experiment arm.

Runs under the isolated runtime's python: ``x1l_fingerprint.py OUT ARM_JSON WHEEL UV``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def main(out: str, arm_file: str, wheel: str, uv_exe: str) -> int:
    arm = json.loads(Path(arm_file).read_text())
    dist = metadata.distribution("agentx-harness")
    checked, mismatched = 0, []
    for entry in dist.files or ():
        if entry.hash is None or entry.hash.mode != "sha256":
            continue
        checked += 1
        digest = hashlib.sha256(Path(entry.locate()).read_bytes()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
            mismatched.append(str(entry))
    from aiperf import _build_info

    uv = subprocess.run([uv_exe, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    record = {
        "arm": arm,
        "agentx_harness_version": dist.version,
        "build_commit": _build_info.COMMIT_SHA,
        "wheel_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        "record_files_checked": checked,
        "record_mismatches": mismatched,
        "python": sys.version.split()[0],
        "uv": uv,
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }
    record["matches_arm"] = (
        record["wheel_sha256"] == arm["wheel_sha256"]
        and record["build_commit"] == arm["harness_commit"]
        and checked > 0
        and not mismatched
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(
        f"AgentX harness arm {arm['arm']}: commit {record['build_commit']} "
        f"wheel {record['wheel_sha256'][:12]} matches={record['matches_arm']}",
        flush=True,
    )
    return 0 if record["matches_arm"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
