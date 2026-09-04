"""Check image references against the Enroot 3.x consumer grammar."""

import subprocess
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[1] / "runners" / "container_utils.sh"
DIGEST = "sha256:" + "a" * 64

# Consumer grammar from NVIDIA/enroot v3.5.0 src/docker.sh:259-269.
# `tag` is passed directly to /v2/<image>/manifests/<tag>, so sha256:<hex>
# remains an immutable manifest reference without using Docker's @ syntax.
PARSE_URI = r"""
source "$1"
uri=$(enroot_uri_for_image "$2") || exit
reg_user="[[:alnum:]_.!~*\'()%\;:\&=+$,-@]+"
reg_registry="[^#]+"
reg_image="[[:lower:][:digit:]/._-]+"
reg_tag="[[:alnum:]._:-]+"
if [[ "${uri}" =~ ^docker://((${reg_user})@)?((${reg_registry})#)?(${reg_image})(:(${reg_tag}))?$ ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}" "${BASH_REMATCH[4]}" "${BASH_REMATCH[5]}" "${BASH_REMATCH[7]}"
else
    exit 1
fi
"""


@pytest.mark.parametrize(
    ("image", "registry", "repository", "reference"),
    [
        (f"lmsysorg/sglang:v0.5.16-cu130@{DIGEST}", "registry-1.docker.io", "lmsysorg/sglang", DIGEST),
        (f"lmsysorg/sglang:v0.5.16-rocm720-mi35x@{DIGEST}", "registry-1.docker.io", "lmsysorg/sglang", DIGEST),
        (f"lmsysorg/sglang@{DIGEST}", "registry-1.docker.io", "lmsysorg/sglang", DIGEST),
        (f"ubuntu@{DIGEST}", "registry-1.docker.io", "library/ubuntu", DIGEST),
        (f"ghcr.io/org/image:release@{DIGEST}", "ghcr.io", "org/image", DIGEST),
        (f"localhost:5000/org/image:release@{DIGEST}", "localhost:5000", "org/image", DIGEST),
        ("lmsysorg/sglang:v0.5.16-cu130", "", "lmsysorg/sglang", "v0.5.16-cu130"),
        ("nvcr.io/nvidia/cuda:13.0", "nvcr.io", "nvidia/cuda", "13.0"),
    ],
)
def test_enroot_reference_preserves_manifest_identity(
    image: str, registry: str, repository: str, reference: str
) -> None:
    result = subprocess.run(
        ["bash", "-c", PARSE_URI, "bash", str(HELPER), image],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["", registry, repository, reference]
