"""Expose research traces directly in CI artifacts instead of a nested log tarball."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def export_profiles(archives: list[Path], output: Path) -> int:
    count = 0
    for archive in archives:
        if not archive.is_file():
            continue
        with tarfile.open(archive) as bundle:
            selected = []
            for member in bundle:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                parts = path.parts
                if parts and parts[0] == "logs":
                    parts = parts[1:]
                if len(parts) < 2 or parts[0] != "research":
                    continue
                selected.append((member, parts[1:]))
            if not any(
                member.name.endswith((".trace.json.gz", ".trace.json", ".nsys-rep"))
                for member, _ in selected
            ):
                continue
            for member, relative in selected:
                destination = output.joinpath(*relative)
                if destination.exists():
                    raise ValueError(f"Duplicate or stale profile file: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable profile member: {member.name}")
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Exported {export_profiles(args.archive, args.output)} research files to {args.output}"
    )


if __name__ == "__main__":
    main()
