"""Internal pinned Kimi verifier archive preparation for benchmark_lib.sh."""

from hashlib import sha256
from pathlib import Path
import re
import socket
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def archive_member_parts(name):
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized = name.rstrip("/")
    parts = normalized.split("/")
    if not normalized or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return tuple(parts)



def main() -> None:
    repo_url, verifier_ref, expected_archive_sha256, checkout_dir_arg = sys.argv[1:]
    checkout_dir = Path(checkout_dir_arg)
    stage = "derive the pinned archive URL"


    try:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", verifier_ref):
            raise ValueError(f"expected a 40-character commit SHA, got {verifier_ref!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_archive_sha256):
            raise ValueError(
                "expected a 64-character archive SHA256, got "
                f"{expected_archive_sha256!r}"
            )

        parsed_repo_url = urlsplit(repo_url)
        if parsed_repo_url.scheme not in ("http", "https") or not parsed_repo_url.netloc:
            raise ValueError(f"unsupported repository URL: {repo_url!r}")
        if parsed_repo_url.query or parsed_repo_url.fragment:
            raise ValueError(f"repository URL must not contain a query or fragment: {repo_url!r}")
        repo_path = parsed_repo_url.path.rstrip("/")
        if repo_path.endswith(".git"):
            repo_path = repo_path[:-4]
        if not repo_path:
            raise ValueError(f"repository URL has no repository path: {repo_url!r}")
        archive_path = f"{repo_path}/archive/{quote(verifier_ref, safe='')}.tar.gz"
        archive_url = urlunsplit(
            (parsed_repo_url.scheme, parsed_repo_url.netloc, archive_path, "", "")
        )

        stage = f"download {archive_url}"
        request = Request(
            archive_url,
            headers={"User-Agent": "InferenceX-Kimi-Vendor-Verifier"},
        )
        with tempfile.TemporaryFile() as archive_file:
            downloaded = 0
            for attempt in range(1, 4):
                archive_file.seek(0)
                archive_file.truncate()
                downloaded = 0
                digest = sha256()
                deadline = time.monotonic() + 60
                try:
                    with urlopen(request, timeout=60) as response:
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(
                                    "archive download exceeded the 60-second deadline"
                                )
                            sock = getattr(getattr(response, "fp", None), "raw", None)
                            sock = getattr(sock, "_sock", None)
                            if sock is not None:
                                sock.settimeout(max(0.001, remaining))
                            try:
                                chunk = response.read(1024 * 1024)
                            except socket.timeout as error:
                                raise TimeoutError(
                                    "archive download exceeded the 60-second deadline"
                                ) from error
                            if not chunk:
                                break
                            downloaded += len(chunk)
                            if downloaded > 128 * 1024 * 1024:
                                raise ValueError(
                                    "archive download exceeds the 128 MiB safety limit"
                                )
                            archive_file.write(chunk)
                            digest.update(chunk)
                    break
                except HTTPError as error:
                    if error.code not in (408, 429) and not 500 <= error.code < 600:
                        raise
                    if attempt == 3:
                        raise
                    print(
                        f"WARN: Kimi-Vendor-Verifier archive download attempt "
                        f"{attempt}/3 failed: {error}; retrying",
                        file=sys.stderr,
                    )
                    time.sleep(attempt)
                except (TimeoutError, URLError, ConnectionError) as error:
                    if attempt == 3:
                        raise
                    print(
                        f"WARN: Kimi-Vendor-Verifier archive download attempt "
                        f"{attempt}/3 failed: {error}; retrying",
                        file=sys.stderr,
                    )
                    time.sleep(attempt)
            if downloaded == 0:
                raise ValueError("downloaded archive is empty")
            actual_archive_sha256 = digest.hexdigest()
            if actual_archive_sha256 != expected_archive_sha256:
                raise ValueError(
                    "Kimi-Vendor-Verifier archive SHA256 mismatch: expected "
                    f"{expected_archive_sha256}, got {actual_archive_sha256}"
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
                        raise ValueError(
                            f"archive member has a negative size: {member.name!r}"
                        )
                    archive_size += member.size
                    if archive_size > 512 * 1024 * 1024:
                        raise ValueError("expanded archive exceeds the 512 MiB safety limit")

                    parts = archive_member_parts(member.name)
                    archive_roots.add(parts[0])
                    if len(archive_roots) > 1:
                        roots = ", ".join(sorted(archive_roots))
                        raise ValueError(f"archive has multiple roots: {roots}")
                    if not (member.isdir() or member.isfile()):
                        raise ValueError(
                            f"archive member has unsafe type: {member.name!r}"
                        )
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
                        raise ValueError(
                            f"required path is not a regular file: {relative_path}"
                        )
                    selected_size += member.size
                    if selected_size > 256 * 1024 * 1024:
                        raise ValueError(
                            "selected archive subset exceeds the 256 MiB safety limit"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"could not read archive member: {member.name!r}")
                    with source:
                        content = source.read(member.size + 1)
                    if len(content) != member.size:
                        raise ValueError(
                            f"archive member size mismatch: {member.name!r}"
                        )
                    selected_files[relative_path] = content

            if member_count == 0:
                raise ValueError("archive contains no members")
            if len(archive_roots) != 1:
                raise ValueError("archive does not have exactly one root")
            missing_files = sorted(required_files - selected_files.keys())
            if missing_files:
                raise ValueError(
                    "archive is missing required files: " + ", ".join(missing_files)
                )

            stage = "extract the verified archive subset"
            if any(checkout_dir.iterdir()):
                raise ValueError(f"checkout directory is not empty: {checkout_dir}")
            for relative_path, content in selected_files.items():
                destination = checkout_dir.joinpath(*relative_path.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(content)
    except Exception as error:
        print(
            f"ERROR: failed to {stage} for Kimi-Vendor-Verifier "
            f"at {verifier_ref}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
