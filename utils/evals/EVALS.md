# Evals

## What?
Quick graded QnA which measures model performance. Examples of test suites:
- **gsm8k**: Grade school math questions
- **gpqa**: Graduate level, Google-Proof multiple choice questions

## When?
At the highest and median concurrency levels (all TPs), per (model, runner, framework, precision, ISL, OSL, spec-decoding, dp-attn), only for 8k1k. In eval-only mode, the server starts with expanded context length. In combined mode (RUN_EVAL=true), evals run against the same server used for throughput benchmarks. Logic is defined in `mark_eval_entries` of `utils/matrix_logic/generate_sweep_configs.py`

## Why?
To verify how model outputs are affected by throughput optimizations. 
- TP/Conc might affect model outputs
- Check kernel implementations for correctness
- If there was a tradeoff in accuracy for performance

## How?
- `run_eval`, defined in `benchmarks/benchmark_lib.sh`, is called in `benchmarks/*`. It runs EleutherAI/lm-evaluation-harness (lmeval) against the running server's OpenAI-compatible endpoint. In eval-only mode (`EVAL_ONLY=true`), the server is started once with expanded context length (up to 5x benchmark context, capped at model native max). JSON results are processed and converted to a table with `utils/collect_eval_results.py`.

## Misc
Following files are task definitions from lmeval, more info on changes within the files
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`



