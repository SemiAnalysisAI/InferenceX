"""InfiniteBench prompt builders for offline benchmarking.

Mirrors cann-recipes-infer's executor/utils/data_utils.py::build_dataset_input
and models/deepseek-v4/models/model_infer.py::get_inputs flow:
  1. Load longbook_qa_eng contexts.
  2. Compute system_prompt_len from chat-templated `prefix + suffix`.
  3. Tokenize each raw context with truncation to context_budget.
  4. Decode, assemble `prefix + ctx + suffix`, apply chat template (DSV4 chat
     mode), return as the final string sent to the engine.

Ported from utils/bench_serving/benchmark_serving.py so the offline harness
can build prompts without importing the async serving stack.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

INFINITEBENCH_REPO_ID = "xinrongzhang2022/InfiniteBench"
INFINITEBENCH_PREFIX = (
    "Please read a part of the book below, and then give me the summary.\n"
    "[start of the book]\n"
)


def infinitebench_suffix(max_new_tokens: int) -> str:
    return (
        "\n[end of the book]\n\n"
        "Now you have read it. Please summarize it for me. "
        "First, tell me the title and the author, and then tell the story "
        f"in {max_new_tokens} words.\n\n "
    )


def _task_file(task: str) -> str:
    if task.endswith(".jsonl"):
        return task
    return f"{task}.jsonl"


def _resolve_jsonl(dataset_path: Optional[str], task_file: str) -> Optional[Path]:
    candidates: List[Path] = []
    if dataset_path:
        root = Path(dataset_path)
        if root.is_file():
            candidates.append(root)
        else:
            candidates.extend([root / task_file, root / "InfiniteBench" / task_file])
    else:
        candidates.extend([
            Path.cwd() / "dataset" / "InfiniteBench" / task_file,
            Path.cwd() / "data" / "InfiniteBench" / task_file,
            Path("/workspace/dataset/InfiniteBench") / task_file,
            Path("/workspace/data/InfiniteBench") / task_file,
        ])
    for c in candidates:
        if c.is_file():
            return c
    return None


def _download_jsonl(task_file: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(
        hf_hub_download(
            repo_id=INFINITEBENCH_REPO_ID,
            repo_type="dataset",
            filename=task_file,
        ))


def load_infinitebench_contexts(
    dataset_path: Optional[str],
    task: str,
    num_prompts: int,
) -> List[str]:
    task_file = _task_file(task)
    jsonl_path = _resolve_jsonl(dataset_path, task_file) or _download_jsonl(task_file)
    contexts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            ctx = row.get("context", row.get("content"))
            if ctx is None:
                raise ValueError(
                    f"Expected 'context' or 'content' in {jsonl_path}, got "
                    f"{sorted(row.keys())}")
            contexts.append(ctx)
            if len(contexts) >= num_prompts:
                break
    if not contexts:
        raise ValueError(f"No InfiniteBench contexts loaded from {jsonl_path}")
    if len(contexts) < num_prompts:
        contexts = (contexts * (num_prompts // len(contexts) + 1))[:num_prompts]
    return contexts


def _apply_chat(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    dsv4: bool,
    dsv4_thinking_mode: str,
) -> str:
    if dsv4:
        # Reuse the same encoder our serving path uses.
        from encoding_dsv4 import encode_messages
        return encode_messages(
            [{"role": "user", "content": prompt}],
            thinking_mode=dsv4_thinking_mode,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _format_prompt(
    raw_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool,
    dsv4: bool,
    dsv4_thinking_mode: str,
) -> str:
    if use_chat_template:
        return _apply_chat(raw_prompt, tokenizer, dsv4, dsv4_thinking_mode)
    return raw_prompt


def _token_count(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_infinitebench_prompts(
    dataset_path: Optional[str],
    task: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool = True,
    dsv4: bool = True,
    dsv4_thinking_mode: str = "chat",
) -> List[Tuple[str, int, int]]:
    """Returns a list of (prompt_str, prompt_len, output_len) tuples ready to
    submit to an offline engine. One-shot trim mirroring cann-recipes-infer."""
    if dsv4 and not use_chat_template:
        raise ValueError("dsv4=True requires use_chat_template=True.")

    suffix = infinitebench_suffix(output_len)
    wrapper = INFINITEBENCH_PREFIX + suffix
    rendered = _format_prompt(
        wrapper, tokenizer, use_chat_template, dsv4, dsv4_thinking_mode)
    system_prompt_len = _token_count(tokenizer, rendered)
    context_budget = input_len - system_prompt_len
    if context_budget <= 0:
        raise ValueError(
            f"input_len {input_len} too short for rendered wrapper "
            f"({system_prompt_len} tokens).")

    contexts = load_infinitebench_contexts(dataset_path, task, num_prompts)
    out: List[Tuple[str, int, int]] = []
    mismatches: List[int] = []
    t0 = time.perf_counter()
    for ctx in contexts:
        ctx_ids = tokenizer.encode(
            ctx, add_special_tokens=False,
            truncation=True, max_length=context_budget)
        trimmed = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        raw = INFINITEBENCH_PREFIX + trimmed + suffix
        prompt = _format_prompt(
            raw, tokenizer, use_chat_template, dsv4, dsv4_thinking_mode)
        plen = _token_count(tokenizer, prompt)
        mismatches.append(plen - input_len)
        out.append((prompt, plen, output_len))

    elapsed = time.perf_counter() - t0
    header = f'{"-"*16}  InfiniteBench Input/Output Statistics  {"-"*16}'
    print(header)
    print(f" prompt_build_time_s: {elapsed:.2f}")
    print(
        f" input_lens : "
        f"min={min(r[1] for r in out):<4d}  "
        f"max={max(r[1] for r in out):<4d}  "
        f"mean={np.mean([r[1] for r in out]):<7.2f}  "
        f"avg_token_mismatch={np.mean(mismatches):<5.2f}")
    print(
        f" output_lens: "
        f"min={min(r[2] for r in out):<4d}  "
        f"max={max(r[2] for r in out):<4d}  "
        f"mean={np.mean([r[2] for r in out]):<7.2f}")
    print('-' * len(header), '\n')
    return out
