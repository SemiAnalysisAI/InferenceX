#!/usr/bin/env python3
"""
Offline vLLM benchmark helper.

Reads required parameters from environment and performs a batch generate()
call with vLLM.LLM, then writes a result JSON compatible with
utils/process_result.py expectations.

Required env vars:
- MODEL, TP, ISL, OSL, RESULT_FILENAME

Optional env vars:
- CONC (used for max_concurrency field)
- NUM_PROMPTS (default: CONC*10)
- CALCULATED_MAX_MODEL_LEN (fallback: ISL+OSL+128)
- KV_CACHE_DTYPE (if set, passes to vLLM LLM(..., kv_cache_dtype=...))
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import ceil


def getenv_required(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return v


def main() -> None:
    model = getenv_required("MODEL")
    tp = int(getenv_required("TP"))
    isl = int(getenv_required("ISL"))
    osl = int(getenv_required("OSL"))
    result_filename = getenv_required("RESULT_FILENAME")

    conc = int(os.environ.get("CONC", "1"))
    num_prompts = int(os.environ.get("NUM_PROMPTS", str(conc * 10)))
    batch_size_env = os.environ.get("BATCH_SIZE")
    if batch_size_env and batch_size_env.isdigit():
        num_prompts = int(batch_size_env)
    max_len_env = os.environ.get("CALCULATED_MAX_MODEL_LEN", "")
    max_model_len = int(max_len_env) if (max_len_env and max_len_env.isdigit()) else isl + osl + 128
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE")

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print(f"Failed to import vLLM: {e}", file=sys.stderr)
        sys.exit(2)

    # Build synthetic prompts. Exact token length may vary by tokenizer.
    base = "The quick brown fox jumps over the lazy dog. "
    repeat = max(1, ceil(isl / 8))
    prompts = [(base * repeat)[:1024] for _ in range(num_prompts)]

    sp = SamplingParams(max_tokens=osl, temperature=0.0, top_p=1.0)

    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    if kv_cache_dtype:
        # Only pass if provided; older vLLM may not support it
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

    # Prefer multiprocessing backend to avoid Ray import/init on ROCm
    try:
        llm = LLM(distributed_executor_backend="mp", **llm_kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Fallback for older vLLM without the arg
        llm = LLM(**llm_kwargs)

    start = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dur_s = max(time.perf_counter() - start, 1e-6)

    total_output_tokens = 0
    total_input_tokens = 0
    for o in outs:
        # Output tokens
        if getattr(o, "outputs", None):
            total_output_tokens += len(o.outputs[0].token_ids)
        else:
            # Fallback to text word count if token ids unavailable
            txt = o.outputs[0].text if getattr(o, "outputs", None) else ""
            total_output_tokens += max(len(txt.split()), osl)

        # Input tokens (measured)
        if hasattr(o, "prompt_token_ids") and o.prompt_token_ids is not None:
            total_input_tokens += len(o.prompt_token_ids)
        else:
            total_input_tokens += isl

    total_token_throughput = (total_input_tokens + total_output_tokens) / dur_s
    output_throughput = total_output_tokens / dur_s

    result = {
        "model_id": model,
        "max_concurrency": conc,
        "num_prompts": num_prompts,
        "duration_s": dur_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_token_throughput": total_token_throughput,
        "output_throughput": output_throughput,
    }

    with open(f"{result_filename}.json", "w") as f:
        json.dump(result, f)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
