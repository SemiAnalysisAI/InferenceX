"""Evidence collection for the isolated H100 hub parity experiment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest(root: Path) -> dict:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
    }
    files = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and (
            path.name in names
            or path.suffix == ".safetensors"
            or path.name.endswith(".safetensors.index.json")
        )
    }
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= files.keys():
        raise ValueError(f"Incomplete tokenizer/config snapshot: {root}")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"No safetensors weights in {root}")
    return {"path": str(root), "files": files}


def prepare(out: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    if importlib.metadata.version("tensorrt_llm") != "1.3.0rc27":
        raise ValueError("The experiment requires the source-matched 1.3.0rc27 image")
    model = os.environ["MODEL"]
    # Stage main so the subsequent offline remote-ID load uses this cached ref.
    revision = HfApi().model_info(model).sha
    hf_path = Path(
        snapshot_download(model, revision="main", ignore_patterns=["original/**/*"])
    )
    if hf_path.name != revision:
        raise ValueError("HF main changed while staging; retry with a stable snapshot")
    hf = manifest(hf_path)
    write_json(
        out / "snapshots.json",
        {
            "model": model,
            "hf_revision": revision,
            "ms_revision": "master",
            "hf": hf,
        },
    )
    (out / "hf_path").write_text(str(hf_path))
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "node": platform.node(),
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "settings": {
                key: os.environ.get(key)
                for key in (
                    "MODEL",
                    "IMAGE",
                    "TP",
                    "ISL",
                    "OSL",
                    "PARITY_MAX_BATCH_SIZE",
                    "PARITY_CONTEXT",
                    "PARITY_CONCURRENCIES",
                    "PARITY_REPEATS",
                    "PARITY_PR_SHA",
                    "SLURM_JOB_ID",
                    "SLURMD_NODENAME",
                    "GITHUB_SHA",
                    "GITHUB_RUN_ID",
                )
            },
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
                text=True,
            ),
        },
    )
    sha = os.environ["PARITY_PR_SHA"]
    url = f"https://raw.githubusercontent.com/SemiAnalysisAI/TensorRT-LLM/{sha}/tests/unittest/llmapi/test_utils.py"
    with urllib.request.urlopen(url, timeout=60) as response:
        original = response.read().decode()
    (out / "pr_test_utils.py").write_text(original)
    # Preserve the PR download tests verbatim, without unrelated newer-release
    # affinity imports/tests or repository-wide CUDA test fixtures.
    tests = original[
        original.index("def _stub_modelscope") : original.index(
            "def test_api_status_registry"
        )
    ]
    imports = (
        "import sys\nimport types\nimport pytest\n"
        "import tensorrt_llm.llmapi.utils as llmapi_utils\n"
        "from tensorrt_llm.llmapi.utils import download_hf_model, download_hf_partial\n\n"
    )
    (out / "test_pr_downloads.py").write_text(imports + tests)
    print("HF snapshot recorded; ModelScope will start with an empty cache", flush=True)


def cold_start(out: Path) -> None:
    cache = Path(os.environ["MODELSCOPE_CACHE"])
    hf_home = Path(os.environ["PARITY_COLD_HF_HOME"])
    for root in (cache, hf_home):
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError(f"Cold-start cache is not empty: {root}")
    write_json(
        out / "cold_start.json",
        {
            "modelscope_cache": str(cache),
            "hf_home": str(hf_home),
            "modelscope_initial_entries": [],
            "hf_initial_entries": [],
            "download_trigger": "trtllm-serve remote model ID; no predownload",
        },
    )
    print(f"Verified empty ModelScope cache: {cache}", flush=True)
    print(f"Verified empty isolated HF home: {hf_home}", flush=True)


def verify_cold(out: Path) -> None:
    from tensorrt_llm.llmapi.utils import download_hf_model

    cold = json.loads((out / "cold_start.json").read_text())
    snapshot = download_hf_model(os.environ["MODEL"])
    if not snapshot.resolve().is_relative_to(Path(cold["modelscope_cache"]).resolve()):
        raise ValueError(f"ModelScope reused an external cache: {snapshot}")
    before = json.loads((out / "snapshots.json").read_text())
    before["modelscope"] = manifest(snapshot)
    write_json(out / "snapshots.json", before)
    if before["hf"]["files"] != before["modelscope"]["files"]:
        raise ValueError("Fresh ModelScope assets differ from the HF baseline")
    hf_files = [
        str(f.relative_to(cold["hf_home"]))
        for f in Path(cold["hf_home"]).rglob("*")
        if f.is_file()
    ]
    cold["hf_files_after_server_start"] = hf_files
    write_json(out / "cold_start.json", cold)
    # Some Transformers versions create a cache-version marker on import.
    if any(Path(name).name != "version.txt" for name in hf_files):
        raise ValueError(f"Unexpected HF fallback/cache files: {hf_files}")
    (out / "modelscope_path").write_text(str(snapshot))
    print(
        "Fresh ModelScope download matches all HF inference assets; no HF model cache used",
        flush=True,
    )


def runtime(out: Path, label: str) -> None:
    spec = importlib.util.find_spec("tensorrt_llm")
    package = Path(next(iter(spec.submodule_search_locations)))
    dest = out / label
    dest.mkdir()
    hashes = {}
    for name in ("utils.py", "llm.py", "llm_args.py"):
        path = package / "llmapi" / name
        shutil.copy2(path, dest / name)
        hashes[name] = digest(path)
    write_json(dest / "hashes.json", hashes)


def record(out: Path, arm: str, repeat: int, concurrency: int) -> None:
    label = f"{arm}_r{repeat}_c{concurrency}"
    files = list((out / label).rglob("results*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one result for {label}, found {files}")
    result = json.loads(files[0].read_text())
    if result["n-samples"]["gsm8k"]["effective"] != 1319:
        raise ValueError("Incomplete GSM8K split")
    report_file = out / "progress.json"
    rows = json.loads(report_file.read_text()) if report_file.exists() else []
    rows.append(
        {
            "arm": arm,
            "repeat": repeat,
            "concurrency": concurrency,
            "scores": result["results"]["gsm8k"],
            "result": str(files[0].relative_to(out)),
        }
    )
    write_json(report_file, rows)
    # Expose one identified result to standard InferenceX collection. All 12
    # results remain in the authoritative, independently labelled archive.
    if arm == "patched_modelscope" and repeat == 2 and concurrency == 64:
        shutil.copy2(files[0], "/workspace/results_modelscope_parity_r2_conc64.json")
    print(json.dumps(rows[-1]), flush=True)


def verify(out: Path) -> None:
    before = json.loads((out / "snapshots.json").read_text())
    for hub in ("hf", "modelscope"):
        if manifest(Path(before[hub]["path"])) != before[hub]:
            raise ValueError(f"Snapshot changed during experiment: {hub}")
    write_json(out / "snapshots_verified_after.json", before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["prepare", "runtime", "record", "verify", "cold-start", "verify-cold"],
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "cold-start":
        cold_start(args.out)
    elif args.action == "verify-cold":
        verify_cold(args.out)
    elif args.action == "runtime":
        runtime(args.out, args.label)
    elif args.action == "record":
        record(args.out, args.label, args.repeat, args.concurrency)
    else:
        verify(args.out)
