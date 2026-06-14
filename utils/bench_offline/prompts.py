"""Build a reproducible CANN-shaped DeepSeek-V4 prompt corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable

from io_utils import read_json, write_json
from trt_config import INPUT_TOKENS


INFINITEBENCH_REPO = "xinrongzhang2022/InfiniteBench"
INFINITEBENCH_REVISION = "90f0394333616266d9fe85824ceaf505093cbaa5"
INFINITEBENCH_FILE = "longbook_qa_eng.jsonl"
INFINITEBENCH_PREFIX = (
    "Please read a part of the book below, and then give me the summary.\n"
    "[start of the book]\n"
)
INFINITEBENCH_SUFFIX = (
    "\n[end of the book]\n\n"
    "Now you have read it. Please summarize it for me. First, tell me the "
    "title and the author, and then tell the story in 256 words.\n\n "
)
BOUNDARY_ADJUSTMENTS = (
    ("space", " "),
    ("double-space", "  "),
    ("newline", "\n"),
    ("space-newline", " \n"),
    ("newline-space", "\n "),
    ("double-newline", "\n\n"),
    ("tab", "\t"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contexts(dataset_path: Path, count: int) -> list[str]:
    contexts: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get("context", row.get("content"))
            if not isinstance(context, str) or not context:
                raise ValueError(
                    f"{dataset_path}:{line_number} has no context/content text"
                )
            contexts.append(context)
            if len(contexts) >= count:
                break
    if not contexts:
        raise ValueError(f"No contexts found in {dataset_path}")
    if len(contexts) < count:
        repeats = (count + len(contexts) - 1) // len(contexts)
        contexts = (contexts * repeats)[:count]
    return contexts


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Tokenizer returned unsupported IDs: {type(value)}")
    token_ids = [int(item) for item in value]
    if any(item < 0 or item > 0xFFFFFFFF for item in token_ids):
        raise ValueError("Token IDs must fit in an unsigned 32-bit integer")
    return token_ids


def _encode(tokenizer: Any, text: str) -> list[int]:
    return _token_ids(tokenizer.encode(text, add_special_tokens=False))


def _render_prompt_text_ids(
    tokenizer: Any,
    context: str,
    prefix: str,
    suffix: str,
) -> list[int]:
    raw_prompt = prefix + context + suffix
    return _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _render_prompt_ids(
    tokenizer: Any,
    context_ids: list[int],
    context_tokens: int,
    prefix: str,
    suffix: str,
) -> list[int]:
    context = tokenizer.decode(
        context_ids[:context_tokens],
        skip_special_tokens=True,
    )
    return _render_prompt_text_ids(tokenizer, context, prefix, suffix)


def _prompt_metadata(
    *,
    wrapper_tokens: int,
    source_context_tokens: int,
    used_context_tokens: int,
    estimate: int,
    boundary_adjustment: str = "none",
    boundary_adjustment_characters: int = 0,
    context_tail_trimmed_characters: int = 0,
) -> dict[str, Any]:
    return {
        "wrapper_tokens": wrapper_tokens,
        "source_context_tokens": source_context_tokens,
        "used_context_tokens": used_context_tokens,
        "adjustment_from_initial_estimate": used_context_tokens - estimate,
        "boundary_adjustment": boundary_adjustment,
        "boundary_adjustment_characters": boundary_adjustment_characters,
        "context_tail_trimmed_characters": context_tail_trimmed_characters,
    }


def build_exact_prompt_ids(
    context: str,
    tokenizer: Any,
    target_tokens: int = INPUT_TOKENS,
    prefix: str = INFINITEBENCH_PREFIX,
    suffix: str = INFINITEBENCH_SUFFIX,
) -> tuple[list[int], dict[str, Any]]:
    """Render one prompt with exactly ``target_tokens`` real token IDs."""
    wrapper_ids = _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + suffix}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    context_ids = _encode(tokenizer, context)
    estimate = min(len(context_ids), target_tokens - len(wrapper_ids))
    if estimate < 0:
        raise ValueError(
            f"Rendered wrapper is {len(wrapper_ids)} tokens, above target "
            f"{target_tokens}"
        )

    attempted: set[int] = set()
    current = estimate
    for _ in range(64):
        if current in attempted:
            break
        attempted.add(current)
        prompt_ids = _render_prompt_ids(
            tokenizer,
            context_ids,
            current,
            prefix,
            suffix,
        )
        delta = target_tokens - len(prompt_ids)
        if delta == 0:
            return prompt_ids, _prompt_metadata(
                wrapper_tokens=len(wrapper_ids),
                source_context_tokens=len(context_ids),
                used_context_tokens=current,
                estimate=estimate,
            )
        current = max(0, min(len(context_ids), current + delta))

    lower = max(0, min(attempted, default=estimate) - 256)
    upper = min(len(context_ids), max(attempted, default=estimate) + 256)
    for current in range(lower, upper + 1):
        if current in attempted:
            continue
        prompt_ids = _render_prompt_ids(
            tokenizer,
            context_ids,
            current,
            prefix,
            suffix,
        )
        if len(prompt_ids) == target_tokens:
            return prompt_ids, _prompt_metadata(
                wrapper_tokens=len(wrapper_ids),
                source_context_tokens=len(context_ids),
                used_context_tokens=current,
                estimate=estimate,
            )

    boundary_counts = {
        max(0, min(len(context_ids), base + offset))
        for base in attempted
        for offset in range(-16, 17)
    }
    over_target_contexts: list[tuple[int, int, str]] = []
    for current in sorted(
        boundary_counts,
        key=lambda value: (abs(value - estimate), -value),
    ):
        context_text = tokenizer.decode(
            context_ids[:current],
            skip_special_tokens=True,
        )
        plain_ids = _render_prompt_text_ids(
            tokenizer,
            context_text,
            prefix,
            suffix,
        )
        plain_delta = len(plain_ids) - target_tokens
        if 0 < plain_delta <= 8:
            over_target_contexts.append(
                (plain_delta, current, context_text)
            )
        for adjustment_name, adjustment_text in BOUNDARY_ADJUSTMENTS:
            prompt_ids = _render_prompt_text_ids(
                tokenizer,
                context_text + adjustment_text,
                prefix,
                suffix,
            )
            if len(prompt_ids) == target_tokens:
                return prompt_ids, _prompt_metadata(
                    wrapper_tokens=len(wrapper_ids),
                    source_context_tokens=len(context_ids),
                    used_context_tokens=current,
                    estimate=estimate,
                    boundary_adjustment=adjustment_name,
                    boundary_adjustment_characters=len(adjustment_text),
                )

    for _, current, context_text in sorted(over_target_contexts)[:8]:
        max_trim = min(128, len(context_text))
        for trimmed_characters in range(1, max_trim + 1):
            adjusted_context = context_text[:-trimmed_characters]
            prompt_ids = _render_prompt_text_ids(
                tokenizer,
                adjusted_context,
                prefix,
                suffix,
            )
            if len(prompt_ids) == target_tokens:
                return prompt_ids, _prompt_metadata(
                    wrapper_tokens=len(wrapper_ids),
                    source_context_tokens=len(context_ids),
                    used_context_tokens=len(
                        _encode(tokenizer, adjusted_context)
                    ),
                    estimate=estimate,
                    context_tail_trimmed_characters=trimmed_characters,
                )

    observed = sorted(
        {
            len(
                _render_prompt_ids(
                    tokenizer,
                    context_ids,
                    current,
                    prefix,
                    suffix,
                )
            )
            for current in attempted
        }
    )
    raise RuntimeError(
        f"Could not render an exact {target_tokens}-token prompt; "
        f"observed lengths near the estimate: {observed[:12]}"
    )


def _uint32_bytes(token_ids: Iterable[int]) -> bytes:
    values = array("I", token_ids)
    if values.itemsize != 4:
        raise RuntimeError("This platform does not use 32-bit unsigned ints")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _write_uint32(path: Path, token_ids: Iterable[int]) -> None:
    with path.open("wb") as stream:
        stream.write(_uint32_bytes(token_ids))


def prepare_corpus(
    dataset_path: Path,
    model_path: str,
    global_batch_size: int,
    output_dir: Path,
    dataset_revision: str = INFINITEBENCH_REVISION,
) -> dict[str, Any]:
    """Build the fixed binary corpus and return its manifest."""
    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.bin"
    manifest_path = output_dir / "corpus_manifest.json"
    tokenizer = DeepseekV4Tokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    contexts = load_contexts(dataset_path, global_batch_size)
    cache: dict[str, tuple[list[int], dict[str, Any]]] = {}
    prompt_ids: list[list[int]] = []
    prompt_metadata: list[dict[str, Any]] = []
    for context in contexts:
        built = cache.get(context)
        if built is None:
            built = build_exact_prompt_ids(context, tokenizer)
            cache[context] = built
        ids, metadata = built
        prompt_ids.append(ids)
        prompt_metadata.append(metadata)

    flat_ids = [token_id for prompt in prompt_ids for token_id in prompt]
    _write_uint32(corpus_path, flat_ids)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "repo": INFINITEBENCH_REPO,
            "revision": dataset_revision,
            "file": INFINITEBENCH_FILE,
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "model_path": model_path,
        "global_batch_size": global_batch_size,
        "concurrency": global_batch_size,
        "prompt_tokens": INPUT_TOKENS,
        "prompt_count": len(prompt_ids),
        "unique_contexts": len(cache),
        "corpus_format": "little-endian uint32 prompt-major",
        "corpus_bytes": corpus_path.stat().st_size,
        "corpus_sha256": sha256_file(corpus_path),
        "prompt_sha256": [
            hashlib.sha256(_uint32_bytes(prompt)).hexdigest()
            for prompt in prompt_ids
        ],
        "context_tokenization": prompt_metadata,
        "prompt_text": {
            "prefix": INFINITEBENCH_PREFIX,
            "suffix": INFINITEBENCH_SUFFIX,
            "suffix_word_count_literal": 256,
            "chat_mode": "DeepSeek-V4 chat, non-thinking",
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def load_corpus(
    corpus_path: Path,
    manifest_path: Path,
) -> tuple[list[list[int]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    if sha256_file(corpus_path) != manifest["corpus_sha256"]:
        raise RuntimeError("Corpus checksum does not match its manifest")
    if corpus_path.stat().st_size % 4:
        raise RuntimeError("Corpus byte size is not divisible by four")
    values = array("I")
    with corpus_path.open("rb") as stream:
        values.fromfile(stream, corpus_path.stat().st_size // 4)
    if sys.byteorder != "little":
        values.byteswap()
    prompt_tokens = int(manifest["prompt_tokens"])
    prompt_count = int(manifest["prompt_count"])
    if len(values) != prompt_tokens * prompt_count:
        raise RuntimeError(
            f"Corpus has {len(values)} IDs; expected "
            f"{prompt_tokens * prompt_count}"
        )
    prompts = [
        list(values[index * prompt_tokens : (index + 1) * prompt_tokens])
        for index in range(prompt_count)
    ]
    return prompts, manifest
