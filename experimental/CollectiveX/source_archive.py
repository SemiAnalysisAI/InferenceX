#!/usr/bin/env python3
"""Validate and extract one pinned backend from a shared source tar."""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Optional, Sequence


PathParts = tuple[str, ...]
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
MAX_ARCHIVE_MEMBERS = 20_000
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_HEADERS = 40_000
MAX_EXTENSION_BYTES = 64 * 1024 * 1024
MAX_EXTENSION_MEMBER_BYTES = 1024 * 1024
MAX_EXTENSION_CHAIN = 8
_TAR_BLOCK = 512
_EXTENSION_TYPES = {b"L", b"K", b"x", b"g", b"X"}


class SourceArchiveError(ValueError):
    """The backend source archive cannot be extracted safely."""


def _tar_size(field: bytes) -> int:
    if field[0] in (0o200, 0o377):
        value = int.from_bytes(field[1:], "big")
        if field[0] == 0o377:
            value -= 256 ** (len(field) - 1)
        return value
    try:
        text = field.split(b"\0", 1)[0].decode("ascii").strip()
        return int(text or "0", 8)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourceArchiveError("archive contains an invalid size field") from exc


def _preflight_archive(descriptor: int, archive_size: int) -> None:
    if archive_size <= 0 or archive_size > MAX_ARCHIVE_BYTES:
        raise SourceArchiveError("backend source archive exceeds the raw size limit")
    offset = headers = extension_bytes = extension_chain = 0
    while offset < archive_size:
        header = os.pread(descriptor, _TAR_BLOCK, offset)
        if len(header) != _TAR_BLOCK:
            raise SourceArchiveError("archive header is truncated")
        if not any(header):
            return
        headers += 1
        if headers > MAX_ARCHIVE_HEADERS:
            raise SourceArchiveError("archive has too many physical headers")
        size = _tar_size(header[124:136])
        if size < 0:
            raise SourceArchiveError("archive contains a negative payload size")
        type_flag = header[156:157]
        if type_flag in _EXTENSION_TYPES:
            extension_chain += 1
            extension_bytes += size
            if (
                extension_chain > MAX_EXTENSION_CHAIN
                or size > MAX_EXTENSION_MEMBER_BYTES
                or extension_bytes > MAX_EXTENSION_BYTES
            ):
                raise SourceArchiveError("archive extension metadata exceeds its limit")
            if type_flag in {b"x", b"g", b"X"}:
                payload = os.pread(descriptor, size, offset + _TAR_BLOCK)
                if len(payload) != size:
                    raise SourceArchiveError("archive extension metadata is truncated")
                if b"GNU.sparse." in payload:
                    raise SourceArchiveError("archive contains sparse extension metadata")
        else:
            extension_chain = 0
            if type_flag == b"S":
                raise SourceArchiveError("archive contains a sparse member")
        blocks = (size + _TAR_BLOCK - 1) // _TAR_BLOCK
        offset += _TAR_BLOCK + blocks * _TAR_BLOCK
        if offset > archive_size:
            raise SourceArchiveError("archive payload is truncated")


def _member_parts(name: str) -> PathParts:
    if not name or "\\" in name or "\0" in name:
        raise SourceArchiveError("archive contains a noncanonical member path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or not path.parts
        or path.parts[0] != ".cx_sources"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SourceArchiveError("archive contains a noncanonical member path")
    return path.parts


def _root_parts(root_basename: str) -> PathParts:
    path = PurePosixPath(root_basename)
    if (
        not root_basename
        or "\\" in root_basename
        or "\0" in root_basename
        or path.is_absolute()
        or path.as_posix() != root_basename
        or len(path.parts) != 1
        or path.parts[0] in {"", ".", ".."}
    ):
        raise SourceArchiveError("invalid backend source root")
    return (".cx_sources", root_basename)


def _read_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    for member in archive:
        if len(members) >= MAX_ARCHIVE_MEMBERS:
            raise SourceArchiveError("archive has an invalid member count")
        members.append(member)
    return members


def _validate_members(
    members: list[tarfile.TarInfo], selected_root: PathParts
) -> dict[PathParts, tarfile.TarInfo]:
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise SourceArchiveError("archive has an invalid member count")
    entries: dict[PathParts, tarfile.TarInfo] = {}
    expanded_bytes = 0
    for member in members:
        parts = _member_parts(member.name)
        if parts in entries:
            raise SourceArchiveError("archive contains duplicate member paths")
        if member.sparse is not None:
            raise SourceArchiveError("archive contains a sparse member")
        if member.isdir():
            if member.size != 0:
                raise SourceArchiveError("archive contains an invalid directory")
        elif member.isfile():
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise SourceArchiveError("archive member exceeds the size limit")
            expanded_bytes += member.size
            if expanded_bytes > MAX_EXPANDED_BYTES:
                raise SourceArchiveError("archive exceeds the expanded size limit")
        elif member.issym():
            if member.size != 0:
                raise SourceArchiveError("archive contains an invalid symbolic link")
        else:
            raise SourceArchiveError("archive contains a non-file member")
        entries[parts] = member

    source_parent = entries.get((".cx_sources",))
    selected = entries.get(selected_root)
    if source_parent is None or not source_parent.isdir():
        raise SourceArchiveError("archive is missing its source directory")
    if selected is None or not selected.isdir():
        raise SourceArchiveError("archive is missing the selected backend source")

    for parts in entries:
        for depth in range(1, len(parts)):
            parent = entries.get(parts[:depth])
            if parent is None or not parent.isdir():
                raise SourceArchiveError("archive member has an unsafe parent")

    for parts, member in entries.items():
        if not member.issym():
            continue
        target_name = member.linkname
        target_path = PurePosixPath(target_name)
        if (
            not target_name
            or "\\" in target_name
            or "\0" in target_name
            or target_path.is_absolute()
            or target_path.as_posix() != target_name
        ):
            raise SourceArchiveError("archive contains an unsafe symbolic link")
        target = list(parts[:-1])
        for component in target_path.parts:
            if component == "..":
                if len(target) <= 2:
                    raise SourceArchiveError("symbolic link escapes its backend source")
                target.pop()
            else:
                target.append(component)
        resolved = tuple(target)
        if resolved[:2] != parts[:2]:
            raise SourceArchiveError("symbolic link crosses backend sources")
        target_member = entries.get(resolved)
        if target_member is None or not target_member.isfile():
            raise SourceArchiveError("symbolic link target is not a regular archive file")
    return entries


def _open_directory(root_fd: int, parts: PathParts) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_directory(root_fd: int, parts: PathParts) -> None:
    parent_fd = _open_directory(root_fd, parts[:-1])
    try:
        os.mkdir(parts[-1], mode=0o700, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _extract_file(
    archive: tarfile.TarFile, root_fd: int, parts: PathParts, member: tarfile.TarInfo
) -> None:
    parent_fd = _open_directory(root_fd, parts[:-1])
    descriptor = -1
    source = None
    try:
        mode = 0o700 if member.mode & 0o111 else 0o600
        descriptor = os.open(parts[-1], _FILE_FLAGS, mode, dir_fd=parent_fd)
        source = archive.extractfile(member)
        if source is None:
            raise SourceArchiveError("archive file has no readable payload")
        remaining = member.size
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise SourceArchiveError("archive file payload is truncated")
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            remaining -= len(chunk)
        os.fchmod(descriptor, mode)
    finally:
        if source is not None:
            source.close()
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _extract_symlink(root_fd: int, parts: PathParts, member: tarfile.TarInfo) -> None:
    parent_fd = _open_directory(root_fd, parts[:-1])
    try:
        os.symlink(member.linkname, parts[-1], dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _extract_selected(
    archive: tarfile.TarFile,
    destination_fd: int,
    entries: dict[PathParts, tarfile.TarInfo],
    selected_root: PathParts,
) -> None:
    try:
        os.stat(".cx_sources", dir_fd=destination_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SourceArchiveError("backend source output already exists")

    selected = {
        parts: member
        for parts, member in entries.items()
        if parts[: len(selected_root)] == selected_root
    }
    _create_directory(destination_fd, (".cx_sources",))
    directories = sorted(
        (parts for parts, member in selected.items() if member.isdir()),
        key=lambda parts: (len(parts), parts),
    )
    for parts in directories:
        _create_directory(destination_fd, parts)
    for parts, member in sorted(selected.items()):
        if member.isfile():
            _extract_file(archive, destination_fd, parts, member)
    for parts, member in sorted(selected.items()):
        if member.issym():
            _extract_symlink(destination_fd, parts, member)


def extract_source_archive(
    archive_path: Path, destination: Path, root_basename: str
) -> None:
    """Validate the complete tar, then safely extract one backend source root."""
    selected_root = _root_parts(root_basename)
    archive_fd = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        metadata = os.fstat(archive_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise SourceArchiveError("backend source archive has unsafe metadata")
        _preflight_archive(archive_fd, metadata.st_size)
        with os.fdopen(os.dup(archive_fd), "rb") as stream:
            try:
                with tarfile.open(fileobj=stream, mode="r:") as archive:
                    entries = _validate_members(_read_members(archive), selected_root)
                    destination_fd = os.open(destination, _DIRECTORY_FLAGS)
                    try:
                        destination_metadata = os.fstat(destination_fd)
                        if (
                            destination_metadata.st_uid != os.getuid()
                            or stat.S_IMODE(destination_metadata.st_mode) != 0o700
                        ):
                            raise SourceArchiveError("backend source destination is unsafe")
                        previous_umask = os.umask(0o077)
                        try:
                            _extract_selected(
                                archive, destination_fd, entries, selected_root
                            )
                        finally:
                            os.umask(previous_umask)
                    finally:
                        os.close(destination_fd)
            except RecursionError as exc:
                raise SourceArchiveError("archive extension metadata is recursive") from exc
    finally:
        os.close(archive_fd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely install one pinned backend source archive"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("root_basename")
    args = parser.parse_args(argv)
    try:
        extract_source_archive(args.archive, args.destination, args.root_basename)
    except (OSError, SourceArchiveError, tarfile.TarError) as exc:
        parser.error(f"backend source archive rejected: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
