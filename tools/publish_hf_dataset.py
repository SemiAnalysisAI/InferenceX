#!/usr/bin/env python3
"""Publish the ISB-1 kv-cache-tester corpus to a Hugging Face dataset repo.

This script stages `datasets/isb1/converted/` together with the checked-in
Hugging Face dataset card (`datasets/isb1/hf_dataset_card.md`) and either:

- prints a dry-run plan without making network changes, or
- creates/uploads a dataset repository via `huggingface_hub`.

Examples:
    python3 tools/publish_hf_dataset.py \
        --source datasets/isb1/converted/ \
        --repo semianalysisai/isb1-cc-traces \
        --private \
        --dry-run

    python3 tools/publish_hf_dataset.py \
        --source datasets/isb1/converted/ \
        --repo semianalysisai/isb1-cc-traces \
        --public \
        --commit-message "Publish ISB-1 kv-cache-tester traces"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_SOURCE = Path("datasets/isb1/converted")
DEFAULT_CARD = Path("datasets/isb1/hf_dataset_card.md")
DEFAULT_COMMIT_MESSAGE = "Publish ISB-1 kv-cache-tester traces"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="publish_hf_dataset.py",
        description=(
            "Stage and publish the ISB-1 converted kv-cache-tester corpus to a "
            "Hugging Face dataset repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 tools/publish_hf_dataset.py --source datasets/isb1/converted/ "
            "--repo semianalysisai/isb1-cc-traces --private --dry-run"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="directory containing converted trace JSON files and manifest.json",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="destination dataset repo in <org>/<name> form",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="create/publish as a private dataset repo (default)",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="create/publish as a public dataset repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the staged upload plan without calling the Hugging Face Hub",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help=f"commit message for the upload commit (default: {DEFAULT_COMMIT_MESSAGE!r})",
    )
    return parser.parse_args(argv)


def fail(message: str, exit_code: int = 2) -> int:
    print(message, file=sys.stderr)
    return exit_code


def require_existing_dir(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {resolved}")
    return resolved


def require_existing_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a file: {resolved}")
    return resolved


def load_manifest(source_dir: Path) -> dict:
    manifest_path = source_dir / "manifest.json"
    require_existing_file(manifest_path, "manifest")
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {manifest_path}: {exc}") from exc


def infer_dataset_card(source_dir: Path) -> Path:
    candidate = source_dir.parent / "hf_dataset_card.md"
    return require_existing_file(candidate, "dataset card")


def iter_upload_files(stage_dir: Path) -> list[Path]:
    return sorted(path for path in stage_dir.rglob("*") if path.is_file())


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def summarize_manifest(manifest: dict) -> dict[str, object]:
    traces = manifest.get("traces", [])
    by_scale = Counter()
    for trace in traces:
        by_scale[str(trace.get("scale_band", "unknown"))] += 1
    return {
        "schema_version": manifest.get("schema_version", "unknown"),
        "generated_at": manifest.get("generated_at", "unknown"),
        "total_traces": manifest.get("total_traces", len(traces)),
        "total_requests": manifest.get("total_requests", "unknown"),
        "by_scale_band": dict(sorted(by_scale.items())),
    }


def print_summary(*, repo: str, private: bool, manifest_summary: dict[str, object], files: Iterable[Path], stage_dir: Path) -> None:
    files = list(files)
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"repo: {repo}")
    print(f"visibility: {'private' if private else 'public'}")
    print(f"dataset_uri: https://huggingface.co/datasets/{repo}")
    print(f"trace_dir_alias: hf_{repo.replace('/', '--')}")
    print(f"staged_dir: {stage_dir}")
    print("manifest:")
    for key, value in manifest_summary.items():
        print(f"  {key}: {value}")
    print(f"files_to_upload: {len(files)} files | {human_bytes(total_bytes)}")
    for file_path in files:
        rel = file_path.relative_to(stage_dir)
        print(f"  - {rel} ({human_bytes(file_path.stat().st_size)})")


def stage_upload_tree(source_dir: Path, dataset_card_path: Path, work_dir: Path) -> Path:
    stage_dir = work_dir / "hf_dataset_upload"
    shutil.copytree(source_dir, stage_dir)
    shutil.copy2(dataset_card_path, stage_dir / "README.md")
    return stage_dir


def load_hf_api() -> tuple[object, object]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for live publish operations. "
            "Install it with `python3 -m pip install huggingface_hub`."
        ) from exc
    return HfApi, snapshot_download


def verify_remote_snapshot(snapshot_download: object, repo: str) -> Path:
    verify_root = Path(tempfile.mkdtemp(prefix="isb1-hf-verify-"))
    snapshot_path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(verify_root),
        local_dir_use_symlinks=False,
    )
    return Path(snapshot_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if "/" not in args.repo or args.repo.count("/") != 1:
        return fail("--repo must be in <org>/<name> form")

    private = not args.public

    try:
        source_dir = require_existing_dir(args.source, "source directory")
        manifest = load_manifest(source_dir)
        dataset_card_path = infer_dataset_card(source_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return fail(str(exc))

    with tempfile.TemporaryDirectory(prefix="isb1-hf-stage-") as temp_root:
        stage_dir = stage_upload_tree(source_dir, dataset_card_path, Path(temp_root))
        manifest_summary = summarize_manifest(manifest)
        upload_files = iter_upload_files(stage_dir)
        print_summary(
            repo=args.repo,
            private=private,
            manifest_summary=manifest_summary,
            files=upload_files,
            stage_dir=stage_dir,
        )

        if args.dry_run:
            print("dry_run: true")
            print("remote_actions: skipped")
            print(
                "note: dry-run stages README.md + manifest + trace files without calling "
                "create_repo/upload_folder/snapshot_download"
            )
            return 0

        try:
            HfApi, snapshot_download = load_hf_api()
            from huggingface_hub.errors import RepositoryNotFoundError
        except RuntimeError as exc:
            return fail(str(exc))
        except ImportError as exc:
            return fail(f"huggingface_hub import failed: {exc}")

        api = HfApi()
        repo_exists = True
        try:
            api.repo_info(repo_id=args.repo, repo_type="dataset")
        except RepositoryNotFoundError:
            repo_exists = False

        if not repo_exists:
            api.create_repo(
                repo_id=args.repo,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            print(f"created_repo: {args.repo}")
        else:
            print(f"created_repo: skipped (already exists: {args.repo})")

        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=str(stage_dir),
            commit_message=args.commit_message,
        )
        print(f"uploaded_repo: {args.repo}")
        snapshot_path = verify_remote_snapshot(snapshot_download, args.repo)
        print(f"verified_snapshot: {snapshot_path}")
        print(f"publish_complete: https://huggingface.co/datasets/{args.repo}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
