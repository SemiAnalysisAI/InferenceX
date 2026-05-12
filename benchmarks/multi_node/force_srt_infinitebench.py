#!/usr/bin/env python3
"""Force a cloned srt-slurm sa-bench checkout to run InfiniteBench 8k/256.

This branch is an experiment and is not intended for main. The launcher keeps
the existing srt-slurm recipes, including their concurrency and parallelism
fields, then applies this patch to the runtime copy of the sa-bench client.
"""

from __future__ import annotations

import sys
from pathlib import Path


INFINITEBENCH_HELPERS = r'''

INFINITEBENCH_REPO_ID = "xinrongzhang2022/InfiniteBench"
DEFAULT_INFINITEBENCH_TASK = "infinitebench"
DEFAULT_INFINITEBENCH_TASK_FILE = "longbook_qa_eng.jsonl"
INFINITEBENCH_PREFIX = (
    "Please read a part of the book below, and then give me the summary.\n"
    "[start of the book]\n"
)


def _infinitebench_suffix(max_new_tokens: int) -> str:
    return (
        "\n[end of the book]\n\n"
        "Now you have read it. Please summarize it for me. "
        "First, tell me the title and the author, and then tell the story "
        f"in {max_new_tokens} words.\n\n "
    )


def _infinitebench_task_file(task: str) -> str:
    if task == DEFAULT_INFINITEBENCH_TASK:
        return DEFAULT_INFINITEBENCH_TASK_FILE
    if task.endswith(".jsonl"):
        return task
    return f"{task}.jsonl"


def _resolve_infinitebench_jsonl(dataset_path: str | None, task_file: str) -> Path | None:
    candidates: list[Path] = []
    if dataset_path:
        root = Path(dataset_path)
        if root.is_file():
            candidates.append(root)
        else:
            candidates.extend([
                root / task_file,
                root / "InfiniteBench" / task_file,
            ])
    else:
        candidates.extend([
            Path.cwd() / "dataset" / "InfiniteBench" / task_file,
            Path.cwd() / "data" / "InfiniteBench" / task_file,
            Path("/workspace/dataset/InfiniteBench") / task_file,
            Path("/workspace/data/InfiniteBench") / task_file,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _download_infinitebench_jsonl(task_file: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download InfiniteBench. "
            "Install it or pass --dataset-path pointing at an InfiniteBench JSONL."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=INFINITEBENCH_REPO_ID,
            repo_type="dataset",
            filename=task_file,
        )
    )


def _load_infinitebench_contexts(dataset_path: str | None, task: str) -> list[str]:
    task_file = _infinitebench_task_file(task)
    jsonl_path = _resolve_infinitebench_jsonl(dataset_path, task_file)
    if jsonl_path is None:
        jsonl_path = _download_infinitebench_jsonl(task_file)

    contexts: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get("context", row.get("content"))
            if context is None:
                raise ValueError(
                    f"Expected 'context' or 'content' in {jsonl_path}, got keys "
                    f"{sorted(row.keys())}"
                )
            contexts.append(context)

    if not contexts:
        raise ValueError(f"No InfiniteBench contexts loaded from {jsonl_path}")
    return contexts


def _format_infinitebench_prompt(
    raw_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool,
) -> str:
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return raw_prompt


def _token_count(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_infinitebench_requests(
    dataset_path: str | None,
    task: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool = False,
) -> list[tuple[str, int, int, None]]:
    """Build CANN-style InfiniteBench summary requests."""
    suffix = _infinitebench_suffix(output_len)
    wrapper_prompt = INFINITEBENCH_PREFIX + suffix
    rendered_wrapper = _format_infinitebench_prompt(
        wrapper_prompt, tokenizer, use_chat_template
    )
    system_prompt_len = _token_count(tokenizer, rendered_wrapper)
    context_budget = input_len - system_prompt_len
    if context_budget <= 0:
        raise ValueError(
            f"InfiniteBench input length {input_len} is too short for the "
            f"rendered prompt wrapper ({system_prompt_len} tokens)."
        )

    contexts = _load_infinitebench_contexts(dataset_path, task)
    prompt_cache: dict[int, tuple[str, int, int, None]] = {}
    input_requests: list[tuple[str, int, int, None]] = []
    mismatches: list[int] = []

    print(
        "Building InfiniteBench requests: "
        f"task={task}, input_len={input_len}, output_len={output_len}, "
        f"num_prompts={num_prompts}, unique_contexts={len(contexts)}, "
        f"use_chat_template={use_chat_template}"
    )
    t0 = time.perf_counter()
    for i in range(num_prompts):
        context_index = i % len(contexts)
        cached = prompt_cache.get(context_index)
        if cached is None:
            # Mirror cann-recipes-infer build_dataset_input: tokenize the raw
            # context with truncation to the system-prompt-adjusted budget,
            # decode, then assemble — no iterative re-encode loop.
            context_ids = tokenizer.encode(
                contexts[context_index],
                add_special_tokens=False,
                truncation=True,
                max_length=context_budget,
            )
            trimmed_context = tokenizer.decode(
                context_ids, skip_special_tokens=True
            )
            raw_prompt = INFINITEBENCH_PREFIX + trimmed_context + suffix
            prompt = _format_infinitebench_prompt(
                raw_prompt, tokenizer, use_chat_template
            )
            prompt_len = _token_count(tokenizer, prompt)

            cached = (prompt, prompt_len, output_len, None)
            prompt_cache[context_index] = cached

        mismatches.append(cached[1] - input_len)
        input_requests.append(cached)

    elapsed = time.perf_counter() - t0
    header_str = f'{"-"*16}  InfiniteBench Input/Output Statistics  {"-"*16}'
    print(header_str)
    print(f" prompt_build_time_s: {elapsed:.2f}")
    print(
        f' input_lens : '
        f'min={min(r[1] for r in input_requests):<4d}  '
        f'max={max(r[1] for r in input_requests):<4d}  '
        f'mean={np.mean([r[1] for r in input_requests]):<7.2f}  '
        f'avg_token_mismatch={np.mean(mismatches):<5.2f} '
    )
    print(
        f' output_lens: '
        f'min={min(r[2] for r in input_requests):<4d}  '
        f'max={max(r[2] for r in input_requests):<4d}  '
        f'mean={np.mean([r[2] for r in input_requests]):<7.2f} '
    )
    print("-" * len(header_str), "\n")

    return input_requests
'''


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_benchmark_serving(path: Path) -> None:
    text = path.read_text()

    if "from pathlib import Path" not in text:
        text = text.replace("import os\n", "import os\nfrom pathlib import Path\n", 1)

    if "INFINITEBENCH_REPO_ID" not in text:
        text = replace_once(text, "\nasync def get_request(", INFINITEBENCH_HELPERS + "\n\nasync def get_request(", path)

    if 'args.dataset_name == "infinitebench"' not in text:
        random_branch = '''    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
        )
'''
        indent = "    "
        if random_branch not in text:
            indented_random_branch = "\n".join(
                f"    {line}" if line else line
                for line in random_branch.splitlines()
            ) + "\n"
            if indented_random_branch in text:
                random_branch = indented_random_branch
                indent = "        "
            else:
                random_marker = '        elif args.dataset_name == "random":\n'
                start = text.find(random_marker)
                end = text.find('        else:\n', start)
                if start == -1 or end == -1:
                    raise RuntimeError(
                        f"Expected random dataset branch not found in {path}"
                    )
                random_branch = text[start:end]
                indent = "        "

        infinite_branch = f'''{indent}elif args.dataset_name == "infinitebench":
{indent}    input_requests = sample_infinitebench_requests(
{indent}        dataset_path=args.dataset_path,
{indent}        task=args.infinitebench_task,
{indent}        input_len=args.infinitebench_input_len,
{indent}        output_len=args.infinitebench_output_len,
{indent}        num_prompts=args.num_prompts,
{indent}        tokenizer=tokenizer,
{indent}        use_chat_template=args.use_chat_template,
{indent}    )

''' + random_branch
        text = replace_once(text, random_branch, infinite_branch, path)

    text = text.replace(
        'choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],',
        'choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "infinitebench"],',
    )
    text = text.replace(
        'choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "custom"],',
        'choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "custom", "infinitebench"],',
    )

    if "--infinitebench-task" not in text:
        infinitebench_parser = '''    infinitebench_group = parser.add_argument_group("InfiniteBench options")
    infinitebench_group.add_argument(
        "--infinitebench-task",
        type=str,
        default="infinitebench",
        help="InfiniteBench JSONL task/file to load.",
    )
    infinitebench_group.add_argument(
        "--infinitebench-input-len",
        type=int,
        default=8192,
        help="Maximum rendered prompt tokens for InfiniteBench.",
    )
    infinitebench_group.add_argument(
        "--infinitebench-output-len",
        type=int,
        default=256,
        help="Fixed max_tokens value for each InfiniteBench request.",
    )
    infinitebench_group.add_argument(
        "--num-chips",
        type=int,
        default=None,
        help="Chip/GPU count used to compute guide-style decode throughput.",
    )

'''
        text = replace_once(
            text,
            '    hf_group = parser.add_argument_group("hf dataset options")\n',
            infinitebench_parser + '    hf_group = parser.add_argument_group("hf dataset options")\n',
            path,
        )

    if 'result_json["dataset_name"] = args.dataset_name' not in text:
        text = replace_once(
            text,
            '        result_json["num_prompts"] = args.num_prompts\n',
            '''        result_json["num_prompts"] = args.num_prompts
        result_json["dataset_name"] = args.dataset_name
        result_json["temperature"] = float(os.environ.get("SA_BENCH_TEMPERATURE", "0.0"))
        if args.dataset_name == "infinitebench":
            result_json["infinitebench_task"] = args.infinitebench_task
            result_json["infinitebench_input_len"] = args.infinitebench_input_len
            result_json["infinitebench_output_len"] = args.infinitebench_output_len

''',
            path,
        )

    if 'decode_throughput_from_mean_tpot' not in text:
        text = replace_once(
            text,
            "        result_json = {**result_json, **benchmark_result}\n",
            '''        result_json = {**result_json, **benchmark_result}

        if args.num_chips is not None:
            result_json["num_chips"] = args.num_chips
            mean_tpot_ms = result_json.get("mean_tpot_ms")
            if args.max_concurrency and mean_tpot_ms:
                decode_tput = args.max_concurrency * 1000.0 / mean_tpot_ms
                result_json["decode_throughput_from_mean_tpot"] = decode_tput
                result_json["decode_throughput_per_chip_from_mean_tpot"] = (
                    decode_tput / args.num_chips
                )
''',
            path,
        )

    path.write_text(text)


def patch_backend_request_func(path: Path) -> None:
    text = path.read_text()
    text = text.replace(
        '"temperature": 0.0,',
        '"temperature": float(os.environ.get("SA_BENCH_TEMPERATURE", "0.0")),',
    )
    path.write_text(text)


def patch_bench_sh(path: Path) -> None:
    text = path.read_text()

    if "InferenceX InfiniteBench experiment override" not in text:
        marker = "# Build optional custom tokenizer args\n"
        override = '''# InferenceX InfiniteBench experiment override.
# Keep recipe concurrency and parallelism intact, but ignore recipe ISL/OSL
# for the client workload on this temporary branch.
ISL=8192
OSL=256
RANDOM_RANGE_RATIO=1.0
DATASET_NAME="infinitebench"
export SA_BENCH_TEMPERATURE=1.0

'''
        text = replace_once(text, marker, override + marker, path)

    text = text.replace(
        "        --dataset-name random \\\n",
        '''        --dataset-name infinitebench \\
        --infinitebench-task infinitebench \\
        --infinitebench-input-len "$ISL" \\
        --infinitebench-output-len "$OSL" \\
        --num-chips "$TOTAL_GPUS" \\
''',
    )
    text = text.replace(
        '        "${DATASET_ARGS[@]}" \\\n',
        '''        "${DATASET_ARGS[@]}" \\
        --infinitebench-task infinitebench \\
        --infinitebench-input-len "$ISL" \\
        --infinitebench-output-len "$OSL" \\
        --num-chips "$TOTAL_GPUS" \\
''',
    )

    path.write_text(text)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: force_srt_infinitebench.py /path/to/srt-slurm", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    bench_dir = root / "src" / "srtctl" / "benchmarks" / "scripts" / "sa-bench"
    benchmark_serving = bench_dir / "benchmark_serving.py"
    backend_request_func = bench_dir / "backend_request_func.py"
    bench_sh = bench_dir / "bench.sh"

    for path in [benchmark_serving, backend_request_func, bench_sh]:
        if not path.exists():
            raise FileNotFoundError(path)

    patch_benchmark_serving(benchmark_serving)
    patch_backend_request_func(backend_request_func)
    patch_bench_sh(bench_sh)

    print(f"Patched srt-slurm sa-bench for InfiniteBench 8192/256 at {bench_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
