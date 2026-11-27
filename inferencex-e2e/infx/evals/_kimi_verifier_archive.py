"""Fetch the pinned Kimi Vendor Verifier archive subset and the git-lfs objects it points to.

Run by ``infx.bench.eval.vendor``. GitHub source archives hold git-lfs pointer files, not the
objects, so the BEAM data files are fetched from the LFS media endpoint and verified against
the pointer's SHA256 and size before anything is written to the checkout.
"""

import re
import sys
import tarfile
import tempfile
import time
from hashlib import sha256
from http.client import IncompleteRead
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def archive_member_parts(name: str) -> tuple[str, ...]:
    """Split an archive member name into its path parts.

    Raises ValueError if the name is absolute, empty or escapes the archive root.
    """
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized = name.rstrip("/")
    parts = normalized.split("/")
    if not normalized or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return tuple(parts)


LFS_POINTER_VERSION = b"version https://git-lfs.github.com/spec/v1"
LFS_OBJECT_LIMIT = 64 * 1024 * 1024
LFS_OBJECTS_LIMIT = 128 * 1024 * 1024


def lfs_pointer(content: bytes) -> tuple[str, int] | None:
    """Return the ``(sha256, size)`` a git-lfs pointer file names, or ``None`` for other content."""
    if not content.startswith(LFS_POINTER_VERSION):
        return None
    fields = {}
    for line in content.decode("ascii", errors="replace").splitlines():
        key, separator, value = line.partition(" ")
        if not separator or key in fields:
            raise ValueError("malformed git-lfs pointer")
        fields[key] = value
    oid = fields.get("oid", "")
    size = fields.get("size", "")
    if not oid.startswith("sha256:") or not re.fullmatch(r"[0-9a-f]{64}", oid[7:]):
        raise ValueError("git-lfs pointer has no sha256 oid")
    if not re.fullmatch(r"[1-9][0-9]*", size) or int(size) > LFS_OBJECT_LIMIT:
        raise ValueError(f"git-lfs pointer size {size!r} is missing or exceeds the 64 MiB limit")
    return oid[7:], int(size)


def default_lfs_media_base_url(netloc: str, repo_path: str) -> str:
    """GitHub's git-lfs media endpoint for ``repo_path``; other hosts need an explicit base URL."""
    if netloc != "github.com":
        raise ValueError(f"git-lfs objects need a media base URL for {netloc}")
    return f"https://media.githubusercontent.com/media{repo_path}"


def _download(
    url: str, expected_sha256: str, *, label: str, limit: int, deadline_s: int, sink: BinaryIO
) -> int:
    """Stream ``url`` into ``sink`` with three attempts; return the byte count.

    Raises when the content never matches ``expected_sha256``, exceeds ``limit`` or an attempt
    outlives ``deadline_s``.
    """
    request = Request(url, headers={"User-Agent": "InferenceX-Kimi-Vendor-Verifier"})  # noqa: S310
    downloaded = 0
    for attempt in range(1, 4):
        sink.seek(0)
        sink.truncate()
        downloaded = 0
        digest = sha256()
        deadline = time.monotonic() + deadline_s
        try:
            with urlopen(request, timeout=deadline_s) as response:  # noqa: S310
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"download exceeded the {deadline_s}-second deadline")
                    sock = getattr(getattr(response, "fp", None), "raw", None)
                    sock = getattr(sock, "_sock", None)
                    if sock is not None:
                        sock.settimeout(max(0.001, remaining))
                    try:
                        chunk = response.read(1024 * 1024)
                    except TimeoutError as error:
                        raise TimeoutError(
                            f"download exceeded the {deadline_s}-second deadline"
                        ) from error
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > limit:
                        raise ValueError(
                            f"download exceeds the {limit // (1024 * 1024)} MiB safety limit"
                        )
                    sink.write(chunk)
                    digest.update(chunk)
        except HTTPError as error:
            if error.code not in (408, 429) and not 500 <= error.code < 600:
                raise
            if attempt == 3:
                raise
            print(
                f"WARN: {label} download attempt {attempt}/3 failed: {error}; retrying",
                file=sys.stderr,
            )
            time.sleep(attempt)
        except (TimeoutError, URLError, ConnectionError, IncompleteRead) as error:
            if attempt == 3:
                raise
            print(
                f"WARN: {label} download attempt {attempt}/3 failed: {error}; retrying",
                file=sys.stderr,
            )
            time.sleep(attempt)
        else:
            if digest.hexdigest() == expected_sha256 or attempt == 3:
                break
            print(
                f"WARN: {label} download attempt {attempt}/3 failed: SHA256 mismatch; retrying",
                file=sys.stderr,
            )
            time.sleep(attempt)
    if downloaded == 0:
        raise ValueError("downloaded content is empty")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return downloaded


def main() -> None:
    """Download the pinned verifier archive, check its SHA256 and extract the required files.

    Takes repo URL, commit SHA, archive SHA256 and checkout directory from argv, plus an
    optional git-lfs media base URL (default: GitHub's, derived from the repo URL). Exits 1
    after printing the failed stage.
    """
    repo_url, verifier_ref, expected_archive_sha256, checkout_dir_arg = sys.argv[1:5]
    lfs_media_base_url = sys.argv[5] if len(sys.argv) > 5 else None
    checkout_dir = Path(checkout_dir_arg)
    stage = "derive the pinned archive URL"

    try:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", verifier_ref):
            raise ValueError(f"expected a 40-character commit SHA, got {verifier_ref!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_archive_sha256):
            raise ValueError(
                f"expected a 64-character archive SHA256, got {expected_archive_sha256!r}"
            )
        expected_archive_sha256 = expected_archive_sha256.lower()

        parsed_repo_url = urlsplit(repo_url)
        if parsed_repo_url.scheme not in ("http", "https") or not parsed_repo_url.netloc:
            raise ValueError(f"unsupported repository URL: {repo_url!r}")
        if parsed_repo_url.query or parsed_repo_url.fragment:
            raise ValueError(f"repository URL must not contain a query or fragment: {repo_url!r}")
        repo_path = parsed_repo_url.path.rstrip("/")
        repo_path = repo_path.removesuffix(".git")
        if not repo_path:
            raise ValueError(f"repository URL has no repository path: {repo_url!r}")
        archive_path = f"{repo_path}/archive/{quote(verifier_ref, safe='')}.tar.gz"
        archive_url = urlunsplit(
            (parsed_repo_url.scheme, parsed_repo_url.netloc, archive_path, "", "")
        )

        stage = f"download {archive_url}"
        with tempfile.TemporaryFile() as archive_file:
            _download(
                archive_url,
                expected_archive_sha256,
                label="Kimi-Vendor-Verifier archive",
                limit=128 * 1024 * 1024,
                deadline_s=60,
                sink=archive_file,
            )
            archive_file.seek(0)

            stage = "validate the downloaded archive"
            required_files = {
                "pyproject.toml",
                "tests/conftest.py",
                "tests/__init__.py",
                "tests/tool_call_json_schema/conftest.py",
                "tests/tool_call_json_schema/__init__.py",
                "tests/tool_call_json_schema/test_tool_call_json_schema.py",
                "tests/tool_call_json_schema/validator.py",
                "testdata/walle_validator_cases/validator_cases/TestAdditionalProperties/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestAnyOf/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestBasicTypes/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestDefs/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestDescription/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestEnforcerCases/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestID/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestKeywordsValidation/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestNestedDefsDepth/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestNumberFormat/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestRangeConstraints/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestRefInProperties/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestReferences/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestRequired/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestSingleTypeInArray/valid.jsonl",
                "testdata/walle_validator_cases/validator_cases/TestTypeLocation/valid.jsonl",
                # BEAM (1M): the two scripts, the bundled Kimi-K2.6 tokenizer and the data pointers.
                "beam/beam_generate.py",
                "beam/beam_judge.py",
                "beam/tokenizer/config.json",
                "beam/tokenizer/tiktoken.model",
                "beam/tokenizer/tokenization_kimi.py",
                "beam/tokenizer/tokenizer_config.json",
                "beam/tokenizer/tool_declaration_ts.py",
                "beam/data/beam_1m_chats.jsonl.gz",
                "beam/data/beam_1m_questions.jsonl",
            }
            selected_files = {}
            archive_roots = set()
            member_count = 0
            archive_size = 0
            selected_size = 0

            with tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
                for member in archive:
                    member_count += 1
                    if member_count > 100_000:
                        raise ValueError("archive contains more than 100000 members")
                    if member.size < 0:
                        raise ValueError(f"archive member has a negative size: {member.name!r}")
                    archive_size += member.size
                    if archive_size > 512 * 1024 * 1024:
                        raise ValueError("expanded archive exceeds the 512 MiB safety limit")

                    parts = archive_member_parts(member.name)
                    archive_roots.add(parts[0])
                    if len(archive_roots) > 1:
                        roots = ", ".join(sorted(archive_roots))
                        raise ValueError(f"archive has multiple roots: {roots}")
                    if not (member.isdir() or member.isfile()):
                        raise ValueError(f"archive member has unsafe type: {member.name!r}")
                    if len(parts) == 1:
                        continue

                    relative_path = "/".join(parts[1:])
                    if relative_path not in required_files:
                        continue
                    if relative_path in selected_files:
                        raise ValueError(
                            f"archive contains duplicate selected path: {relative_path!r}"
                        )
                    if not member.isfile():
                        raise ValueError(f"required path is not a regular file: {relative_path}")
                    selected_size += member.size
                    if selected_size > 256 * 1024 * 1024:
                        raise ValueError("selected archive subset exceeds the 256 MiB safety limit")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"could not read archive member: {member.name!r}")
                    with source:
                        content = source.read(member.size + 1)
                    if len(content) != member.size:
                        raise ValueError(f"archive member size mismatch: {member.name!r}")
                    selected_files[relative_path] = content

            if member_count == 0:
                raise ValueError("archive contains no members")
            if len(archive_roots) != 1:
                raise ValueError("archive does not have exactly one root")
            missing_files = sorted(required_files - selected_files.keys())
            if missing_files:
                raise ValueError("archive is missing required files: " + ", ".join(missing_files))

            stage = "fetch the git-lfs objects the archive points to"
            pointers = {
                path: pointer
                for path, content in selected_files.items()
                if (pointer := lfs_pointer(content)) is not None
            }
            if pointers and lfs_media_base_url is None:
                lfs_media_base_url = default_lfs_media_base_url(parsed_repo_url.netloc, repo_path)
            objects_size = 0
            for relative_path, (oid, size) in pointers.items():
                objects_size += size
                if objects_size > LFS_OBJECTS_LIMIT:
                    raise ValueError("git-lfs objects exceed the 128 MiB safety limit")
                object_url = f"{lfs_media_base_url.rstrip('/')}/{quote(verifier_ref, safe='')}/{relative_path}"
                stage = f"download {object_url}"
                with tempfile.TemporaryFile() as object_file:
                    downloaded = _download(
                        object_url,
                        oid,
                        label=f"git-lfs object {relative_path}",
                        limit=LFS_OBJECT_LIMIT,
                        deadline_s=300,
                        sink=object_file,
                    )
                    if downloaded != size:
                        raise ValueError(
                            f"git-lfs object {relative_path} has {downloaded} bytes, "
                            f"its pointer says {size}"
                        )
                    object_file.seek(0)
                    selected_files[relative_path] = object_file.read()

            stage = "extract the verified archive subset"
            if any(checkout_dir.iterdir()):
                raise ValueError(f"checkout directory is not empty: {checkout_dir}")
            for relative_path, content in selected_files.items():
                destination = checkout_dir.joinpath(*relative_path.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(content)
    except Exception as error:  # noqa: BLE001
        print(
            f"ERROR: failed to {stage} for Kimi-Vendor-Verifier at {verifier_ref}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
