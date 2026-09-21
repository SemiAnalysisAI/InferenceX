"""Fail before GPU startup if the pinned DSpark checkpoint is incomplete."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
CONFIG_BLOB = "7fc3b0c42c815a0a7481b5ca15f42139f94ac4ab"
INDEX_SHA256 = "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a"
HEAD_KEYS = (
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
)


def check_checkpoint(model_path: Path, revision: str) -> dict:
    if revision != REVISION:
        raise ValueError(f"Unsupported checkpoint revision: {revision}")
    model_path = model_path.resolve(strict=True)
    if model_path.parent.name == "snapshots" and model_path.name != revision:
        raise ValueError(f"Snapshot path does not match revision: {model_path}")

    config_bytes = (model_path / "config.json").read_bytes()
    config_blob = hashlib.sha1(
        f"blob {len(config_bytes)}\0".encode() + config_bytes
    ).hexdigest()
    if config_blob != CONFIG_BLOB:
        raise ValueError(f"config.json does not match the pinned DSpark checkpoint: {config_blob}")
    config = json.loads(config_bytes)
    expected = {
        "dspark_block_size": 5, "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [58, 59, 60], "dspark_markov_rank": 512,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid {key}: {config.get(key)!r}, expected {value!r}")

    index_bytes = (model_path / "model.safetensors.index.json").read_bytes()
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if index_sha256 != INDEX_SHA256:
        raise ValueError(f"Weight index does not match the pinned checkpoint: {index_sha256}")
    weight_map = json.loads(index_bytes)["weight_map"]
    shard_keys: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if Path(filename).name != filename:
            raise ValueError(f"Invalid shard filename: {filename}")
        shard_keys.setdefault(filename, set()).add(key)
    if len(shard_keys) != 66 or any(key not in weight_map for key in HEAD_KEYS):
        raise ValueError("Expected 66 shards and all DSpark Markov/confidence head weights")

    shards = []
    heads = {}
    for filename, keys in sorted(shard_keys.items()):
        path = model_path / filename
        # Read actual safetensors headers and both payload boundaries, checking
        # file permissions, missing tensors and truncated payloads without
        # streaming the entire checkpoint twice before each benchmark.
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size <= 16 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header size: {filename}")
            header = json.loads(stream.read(header_size))
            missing = keys - header.keys()
            if missing:
                raise ValueError(f"Missing indexed tensors in {filename}: {sorted(missing)[:5]}")
            payload_size = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            expected_size = 8 + header_size + payload_size
            if path.stat().st_size != expected_size:
                raise ValueError(f"Truncated or invalid shard: {filename}")
            if not stream.read(1):
                raise ValueError(f"Unreadable shard payload: {filename}")
            stream.seek(expected_size - 1)
            if not stream.read(1):
                raise ValueError(f"Unreadable shard tail: {filename}")
        for key in HEAD_KEYS:
            if key in keys:
                heads[key] = {"shard": filename, **header[key]}
        shards.append({"name": filename, "bytes": expected_size, "indexed_tensors": len(keys)})

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokens = tokenizer.encode("DSpark checkpoint readability check: 2 + 2 = 4.")
    if not tokens or not tokenizer.decode(tokens):
        raise ValueError("Tokenizer encode/decode check failed")
    if config["dspark_noise_token_id"] >= len(tokenizer):
        raise ValueError("DSpark noise token is outside the tokenizer vocabulary")
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        data = (model_path / name).read_bytes()
        tokenizer_files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813", "requested_revision": revision,
        "model_path": str(model_path), "config_git_blob": config_blob,
        "index_sha256": index_sha256, "dspark_config": expected,
        "dspark_heads": heads, "shard_count": len(shards), "shards": shards,
        "tokenizer_path": str(model_path), "tokenizer_files": tokenizer_files,
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_size": len(tokenizer),
        "readability_check": "All headers, indexed tensor names, payload sizes and boundary reads; not full weight hashes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = check_checkpoint(args.model_path, args.revision)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DSpark checkpoint preflight passed: {manifest['shard_count']} shards, tokenizer and heads readable")


if __name__ == "__main__":
    main()
