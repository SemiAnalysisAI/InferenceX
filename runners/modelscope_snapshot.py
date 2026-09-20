"""Record cold-cache and snapshot evidence for ModelScope serving."""

import argparse
import hashlib
import json
from pathlib import Path


def record_empty_caches(model: str, cache: Path, hf_home: Path, report: Path) -> None:
    for directory in (cache, hf_home):
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(f"Expected an empty cache directory: {directory}")
    report.write_text(
        json.dumps(
            {
                "model": model,
                "modelscope_cache": str(cache),
                "hf_home": str(hf_home),
                "modelscope_initial_entries": [],
                "hf_initial_entries": [],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified empty ModelScope cache before server startup: {cache}")
    print(f"Verified empty Hugging Face home before server startup: {hf_home}")


def record_snapshot(model: str, cache: Path, hf_home: Path, report: Path) -> None:
    from tensorrt_llm.llmapi.utils import download_hf_model

    evidence = json.loads(report.read_text())
    if (evidence["model"], evidence["modelscope_cache"], evidence["hf_home"]) != (
        model,
        str(cache),
        str(hf_home),
    ):
        raise ValueError(
            "Snapshot verification does not match the cold-start configuration"
        )
    snapshot = download_hf_model(model).resolve()
    if not snapshot.is_relative_to(cache.resolve()):
        raise ValueError(f"Snapshot is outside the fresh ModelScope cache: {snapshot}")
    assets = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.suffix in (
            ".json",
            ".safetensors",
            ".txt",
            ".model",
            ".jinja",
        ):
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            assets[str(path.relative_to(snapshot))] = {
                "bytes": path.stat().st_size,
                "sha256": checksum,
            }
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= assets.keys():
        raise ValueError("The ModelScope snapshot is missing tokenizer/config files")
    if not any(name.endswith(".safetensors") for name in assets):
        raise ValueError("The ModelScope snapshot contains no model weights")
    hf_files = [
        str(path.relative_to(hf_home)) for path in hf_home.rglob("*") if path.is_file()
    ]
    # Transformers creates empty import scaffolding even without HF downloads.
    empty_scaffolding = {"modules/__init__.py", "modules/hf_remote_code.lock"}
    if any(
        Path(name).name != "version.txt"
        and not (
            name in empty_scaffolding
            and not (hf_home / name).is_symlink()
            and (hf_home / name).stat().st_size == 0
        )
        for name in hf_files
    ):
        raise ValueError(f"Unexpected Hugging Face fallback/cache files: {hf_files}")
    evidence.update(
        snapshot=str(snapshot), assets=assets, hf_files_after_startup=hf_files
    )
    report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    operation = record_empty_caches if args.phase == "before" else record_snapshot
    operation(args.model, args.cache, args.hf_home, args.report)
