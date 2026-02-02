# Evals

## What?
Quick graded QnA which measures model performance. Examples of test suites:
- **gsm8k**: Grade school math questions
- **gpqa**: Graduate level, Google-Proof multiple choice questions

## When?
At highest concurrency for highest TP and lowest TP, per GPU per model only for 1k8k. Logic is defined in `mark_eval_entries` of `utils/matrix-logic/generate_sweep_configs.py`

## Why?
To verify how model outputs are affected by throughput optimizations. 
- TP/Conc might affect model outputs
- Check kernel implementations for correctness
- If there was a tradeoff in accuracy for performance

## How?
- `run_eval`, definined in `benchmarks/benchmark_lib.sh`, is called in `benchmarks/*`. EleutherAI/lm-evaluation-harness(lmeval), using the same endpoint as the throughput benchmark. JSON results are processed and converted to a table with `utils/collect_eval_results.py`.

## Misc
Following files are task definitions from lmeval, more info on changes within the files
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`



