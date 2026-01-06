#!/usr/bin/env python3
"""
Offline vLLM benchmark: utils/offline_benchmark_vllm.py

Reads required parameters from environment and performs a batch generate()
call with vLLM.LLM, then writes a result JSON compatible with
utils/process_result.py expectations.

Required env vars (single-run mode):
- MODEL, TP, ISL, OSL, RESULT_FILENAME

Single-session batch mode (reuse one LLM across many runs):
- Set env `SINGLE_SESSION=1`
- Provide one of:
  - `TEST_MATRIX` as comma-separated pairs like "1024:1024,1024:8192,8192:1024"
  - or `ISL_LIST` and `OSL_LIST` as space-separated lists (cartesian product)
- Optional: `BATCH_LIST` (space-separated, default: "4 16 64")
- Optional: `RESULT_PREFIX` to name files like
  "${RESULT_PREFIX}_isl_${isl}_osl_${osl}_bs${bs}_tp${TP}.json"

Optional env vars:
- CONC (used for max_concurrency field)
- NUM_PROMPTS (default: CONC*10)
- CALCULATED_MAX_MODEL_LEN (fallback: ISL+OSL+128)
- KV_CACHE_DTYPE (if set, passes to vLLM LLM(..., kv_cache_dtype=...))
- BLOCK_SIZE (if set, passes to vLLM LLM(..., block_size=...))
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


def _calc_max_model_len_for_pair(isl: int, osl: int) -> int:
    # Keep parity with shell scripts' logic for safety margin.
    if isl == 1024 and osl == 1024:
        return isl + osl + 20
    elif isl == 8192 or osl == 8192:
        return isl + osl + 200
    else:
        return isl + osl + 128

# Replace the prompt generation section with this:

def _generate_prompts_for_isl(llm, isl: int, num_prompts: int) -> list[str]:
    """Generate prompts that will tokenize to approximately ISL tokens."""
    tokenizer = llm.get_tokenizer()
    
    # Base text to repeat
    base = "The quick brown fox jumps over the lazy dog. This is filler text for benchmarking. "
    
    # Estimate tokens per char ratio (roughly 4 chars per token for English)
    chars_per_token = 4
    target_chars = isl * chars_per_token
    
    # Build a long enough string
    long_text = base * (target_chars // len(base) + 1)
    
    prompts = []
    for _ in range(num_prompts):
        # Binary search to find the right length
        low, high = 0, len(long_text)
        while low < high:
            mid = (low + high + 1) // 2
            tokens = tokenizer.encode(long_text[:mid])
            if len(tokens) <= isl:
                low = mid
            else:
                high = mid - 1
        
        prompt = long_text[:low]
        prompts.append(prompt)
    
    return prompts

def _run_single(llm_ctor_kwargs: dict, llm_runtime_kwargs: dict) -> dict:
    # Instantiate LLM once and run a single workload, returning the result dict.
    from vllm import LLM, SamplingParams

    isl = llm_runtime_kwargs["isl"]
    osl = llm_runtime_kwargs["osl"]
    num_prompts = llm_runtime_kwargs["num_prompts"]

    sp = SamplingParams(max_tokens=osl, temperature=0.0, top_p=1.0)

    # Prefer multiprocessing backend; be robust to older vLLM versions.
    def try_construct(use_mp: bool, allow_block_size: bool):
        kwargs = dict(llm_ctor_kwargs)
        if not allow_block_size:
            kwargs.pop("block_size", None)
        if use_mp:
            try:
                return LLM(distributed_executor_backend="mp", **kwargs)  # type: ignore[arg-type]
            except TypeError:
                return None
        else:
            try:
                return LLM(**kwargs)
            except TypeError:
                return None

    llm = (
        try_construct(use_mp=True, allow_block_size=True)
        or try_construct(use_mp=True, allow_block_size=False)
        or try_construct(use_mp=False, allow_block_size=True)
        or try_construct(use_mp=False, allow_block_size=False)
    )
    if llm is None:
        print("Failed to construct vLLM LLM with provided arguments", file=sys.stderr)
        sys.exit(3)

    prompts = _generate_prompts_for_isl(llm, isl, num_prompts)

    start = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dur_s = max(time.perf_counter() - start, 1e-6)

    total_output_tokens = 0
    total_input_tokens = 0
    for o in outs:
        if getattr(o, "outputs", None):
            total_output_tokens += len(o.outputs[0].token_ids)
        else:
            txt = o.outputs[0].text if getattr(o, "outputs", None) else ""
            total_output_tokens += max(len(txt.split()), osl)
        if hasattr(o, "prompt_token_ids") and o.prompt_token_ids is not None:
            total_input_tokens += len(o.prompt_token_ids)
        else:
            total_input_tokens += isl

    total_token_throughput = (total_input_tokens + total_output_tokens) / dur_s
    output_throughput = total_output_tokens / dur_s

    result = {
        "model_id": llm_ctor_kwargs["model"],
        "max_concurrency": llm_runtime_kwargs["conc"],
        "num_prompts": num_prompts,
        "duration_s": dur_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_token_throughput": total_token_throughput,
        "output_throughput": output_throughput,
    }
    return result


def main() -> None:
    model = getenv_required("MODEL")
    tp = int(getenv_required("TP"))
    single_session = os.environ.get("SINGLE_SESSION", "0") in {"1", "true", "True"}

    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE")
    block_size_env = os.environ.get("BLOCK_SIZE")
    block_size = int(block_size_env) if (block_size_env and block_size_env.isdigit()) else None

    if single_session:
        # Parse matrix of runs from env.
        test_matrix = os.environ.get("TEST_MATRIX")
        isl_list_env = os.environ.get("ISL_LIST", "")
        osl_list_env = os.environ.get("OSL_LIST", "")
        if test_matrix:
            pairs = []
            for item in test_matrix.split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    isl_s, osl_s = item.split(":", 1)
                    pairs.append((int(isl_s), int(osl_s)))
                except Exception:
                    print(f"Invalid TEST_MATRIX pair: {item}", file=sys.stderr)
                    sys.exit(1)
        else:
            isl_list = [int(x) for x in isl_list_env.split() if x.strip()]
            osl_list = [int(x) for x in osl_list_env.split() if x.strip()]
            if not isl_list and not osl_list:
                # Match shell scripts' default sweep
                pairs = [(1024, 1024), (1024, 8192), (8192, 1024)]
            else:
                isl_list = isl_list or [1024]
                osl_list = osl_list or [1024]
                pairs = [(i, o) for i in isl_list for o in osl_list]

        batch_list_env = os.environ.get("BATCH_LIST", "4 16 64")
        batch_list = [int(x) for x in batch_list_env.split() if x.strip()]
        result_prefix = os.environ.get("RESULT_PREFIX", "result")

        # Compute a global max_model_len so we can reuse a single LLM.
        global_max_model_len = 0
        for isl, osl in pairs:
            global_max_model_len = max(global_max_model_len, _calc_max_model_len_for_pair(isl, osl))

        llm_ctor_kwargs = dict(
            model=model,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            max_model_len=global_max_model_len,
        )
        if kv_cache_dtype:
            llm_ctor_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if block_size is not None:
            llm_ctor_kwargs["block_size"] = block_size

        # Build once and run sequentially; instantiate inside helper to reuse graphs/weights.
        # We create a temporary LLM inside _run_single which constructs once per call.
        # To actually reuse a single object, we inline the generation here instead.
        # However, to minimize code churn we reuse _run_single by passing ctor kwargs each time.
        # Note: This still constructs LLM for each call; to truly reuse, we implement inline here.

        # Implement true reuse: construct llm once here and run multiple generates.
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            print(f"Failed to import vLLM: {e}", file=sys.stderr)
            sys.exit(2)

        def try_construct(use_mp: bool, allow_block_size: bool):
            kwargs = dict(llm_ctor_kwargs)
            if not allow_block_size:
                kwargs.pop("block_size", None)
            if use_mp:
                try:
                    return LLM(distributed_executor_backend="mp", **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    return None
            else:
                try:
                    return LLM(**kwargs)
                except TypeError:
                    return None

        llm = (
            try_construct(use_mp=True, allow_block_size=True)
            or try_construct(use_mp=True, allow_block_size=False)
            or try_construct(use_mp=False, allow_block_size=True)
            or try_construct(use_mp=False, allow_block_size=False)
        )
        if llm is None:
            print("Failed to construct vLLM LLM with provided arguments", file=sys.stderr)
            sys.exit(3)

        for isl, osl in pairs:
            for bs in batch_list:
                conc = bs
                num_prompts = bs

                # Build prompts
                prompts = _generate_prompts_for_isl(llm, isl, num_prompts)
                sp = SamplingParams(max_tokens=osl, temperature=0.0, top_p=1.0)

                start = time.perf_counter()
                outs = llm.generate(prompts, sp)
                dur_s = max(time.perf_counter() - start, 1e-6)

                total_output_tokens = 0
                total_input_tokens = 0
                for o in outs:
                    if getattr(o, "outputs", None):
                        total_output_tokens += len(o.outputs[0].token_ids)
                    else:
                        txt = o.outputs[0].text if getattr(o, "outputs", None) else ""
                        total_output_tokens += max(len(txt.split()), osl)
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
                result_filename = f"{result_prefix}_isl_{isl}_osl_{osl}_bs{bs}_tp{tp}"
                with open(f"{result_filename}.json", "w") as f:
                    json.dump(result, f)
                print(json.dumps({"result_file": f"{result_filename}.json", **result}, indent=2))
        return

    # Single-run legacy behavior
    isl = int(getenv_required("ISL"))
    osl = int(getenv_required("OSL"))
    result_filename = getenv_required("RESULT_FILENAME")

    conc = int(os.environ.get("CONC", "1"))
    num_prompts = int(os.environ.get("NUM_PROMPTS", str(conc * 10)))
    batch_size_env = os.environ.get("BATCH_SIZE")
    if batch_size_env and batch_size_env.isdigit():
        num_prompts = int(batch_size_env)
    max_len_env = os.environ.get("CALCULATED_MAX_MODEL_LEN", "")
    max_model_len = int(max_len_env) if (max_len_env and max_len_env.isdigit()) else _calc_max_model_len_for_pair(isl, osl)

    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    if kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if block_size is not None:
        llm_kwargs["block_size"] = block_size

    # Run single workload using helper
    result = _run_single(
        llm_ctor_kwargs=llm_kwargs,
        llm_runtime_kwargs=dict(isl=isl, osl=osl, num_prompts=num_prompts, conc=conc),
    )

    with open(f"{result_filename}.json", "w") as f:
        json.dump(result, f)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
