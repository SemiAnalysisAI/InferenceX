import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def container_path(root: Path, path: str) -> Path:
    pending = path.split("/")
    parts = []
    links = 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError("Path escapes the container root")
            parts.pop()
            continue
        candidate = root.joinpath(*parts, part)
        if candidate.is_symlink() and pending:
            links += 1
            if links > 40:
                raise ValueError("Too many symlinks in container path")
            target = os.readlink(candidate)
            if target.startswith("/"):
                parts = []
            pending = target.split("/") + pending
        else:
            parts.append(part)
    return root.joinpath(*parts)


def verify_rootfs(root: Path, files: dict) -> None:
    for name, expected in files.items():
        path = container_path(root, name)
        mode = path.lstat().st_mode
        if expected["type"] == "symlink":
            if not stat.S_ISLNK(mode) or os.readlink(path) != expected["target"]:
                raise ValueError(f"Runtime symlink changed: {name}")
        elif not stat.S_ISREG(mode) or sha256(path) != expected["sha256"]:
            raise ValueError(f"Runtime bytes changed: {name}")
        elif mode & 0o111 != expected["mode"] & 0o111:
            raise ValueError(f"Runtime executable bits changed: {name}")


def config_changes(config: dict) -> list[str]:
    changes = []
    for entry in config.get("Env", []):
        key, value = entry.split("=", 1)
        changes += ["--change", f"ENV {key}={json.dumps(value)}"]
    for key, value in config.get("Labels", {}).items():
        changes += ["--change", f"LABEL {key}={json.dumps(value)}"]
    for key in ("Entrypoint", "Cmd", "WorkingDir", "User"):
        if config.get(key):
            instruction = "WORKDIR" if key == "WorkingDir" else key.upper()
            changes += ["--change", f"{instruction} {json.dumps(config[key])}"]
    return changes


def verify_config(expected: dict, actual: dict) -> None:
    for key in ("Env", "Entrypoint", "Cmd", "WorkingDir", "User", "Labels"):
        if expected.get(key) != actual.get(key) and not (
            key in ("Cmd", "User") and not expected.get(key) and not actual.get(key)
        ):
            raise ValueError(f"Recovered Docker configuration changed: {key}")


def verify_provenance(source: dict, fixture: dict) -> None:
    checks = (
        (source["squash"]["sha256"], fixture["recovery"]["squash_sha256"]),
        (source["archive"]["sha256"], fixture["official_archive_sha256"]),
        (source["image"]["config_digest"], fixture["official_image_config_digest"]),
        (str(source["preparation"]["run_id"]), str(fixture["recovery"]["run_id"])),
        (source["preparation"]["repository"], fixture["recovery"]["repository"]),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("Prepared runtime provenance does not match the pinned base")


def main() -> None:
    root, source_path, fixture_path, image_ref, output_path = sys.argv[1:]
    root = Path(root)
    fixture = json.loads(Path(fixture_path).read_text())
    source = json.loads(Path(source_path).read_text())
    verify_provenance(source, fixture)
    verify_rootfs(root, fixture["files"])
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["tar", "--numeric-owner", "-C", str(root), "-cpf", "-", "."],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        subprocess.run(
            [
                "docker",
                "import",
                "--platform=linux/amd64",
                *config_changes(fixture["config"]),
                "-",
                image_ref,
            ],
            stdin=archive,
            check=True,
        )
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image_ref])
    )[0]
    verify_config(fixture["config"], image["Config"])
    receipt = {
        "kind": "reconstructed_from_enroot_squash",
        "recovery": fixture["recovery"],
        "official_image_config_digest": fixture["official_image_config_digest"],
        "official_archive_sha256": fixture["official_archive_sha256"],
        "reconstructed_image_config_digest": image["Id"],
        "reconstructed_layers": image["RootFS"]["Layers"],
        "critical_files_verified": len(fixture["files"]),
        "runtime_manifest_sha256": sha256(Path(fixture_path)),
        "conversion": "Enroot 4.2.0: flattened layers, root ownership, normalized permissions, generated /etc/{environment,fstab,rc,rc.local}; original Docker config restored",
        "source_provenance": source,
    }
    Path(output_path).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
