from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from infx.bench.env import BenchError

SOURCE_HASHES = {
    "main.py": "4049a54c5ece274da5f56eeb5fb70cbe85e3c1f9ffe22c4876a13bcbaf1fa66b",
    "prepost.py": "0671f9c036ea40391a1f9d2f45ed07fe8870bc9861287ea8c83571e80b62b564",
    "vllm_processor.py": "9d469ea8d138541209b36057d04818ece22bf454daecdd8f8fd3cd41ee823571",
}


def install(source: Path, target: Path) -> None:
    """Adapt the pinned Dynamo Python frontend to the MUM image in a private package copy."""
    contents = {}
    for name, expected in SOURCE_HASHES.items():
        data = (source / "frontend" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise BenchError(f"Unsupported Dynamo frontend snapshot: {name}")
        contents[name] = data.decode()
    # MUM uses a local version string; semver cannot select its moved frontend APIs.
    old = """            if Version(importlib.metadata.version("vllm")).release >= (0, 29):
                from vllm.entrypoints.launchers.cli_args import FrontendArgs
            else:
                from vllm.entrypoints.openai.cli_args import FrontendArgs"""
    contents["main.py"] = contents["main.py"].replace(
        old, "            from vllm.entrypoints.common.cli_args import FrontendArgs"
    )
    for name in ("prepost.py", "vllm_processor.py"):
        contents[name] = contents[name].replace(
            "from vllm.entrypoints.chat_utils import", "from vllm.renderers.chat_utils import"
        )
    # The frontend owns no KV cache; honor its flag without changing worker cache settings.
    contents["vllm_processor.py"] = contents["vllm_processor.py"].replace(
        "cache_config=CacheConfig(),",
        "cache_config=CacheConfig(enable_prefix_caching=self.flags.enable_prefix_caching),",
    )
    shutil.copytree(source, target)
    for name, content in contents.items():
        (target / "frontend" / name).write_text(content)
    print(json.dumps({"dynamo_overlay": str(target), "source_sha256": SOURCE_HASHES}))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    install(args.source, args.target)
    return 0
