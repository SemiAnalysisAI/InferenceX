# Trace replayer — model coverage tests

Smoke-test infrastructure on `chore/agentx-v0.1-testing` for verifying that
`utils/trace-replay/trace_replay_tester.py` works against every active
model family in this repo.

## How to dispatch

```bash
gh workflow run e2e-tests.yml --ref chore/agentx-v0.1-testing \
  -f generate-cli-command="full-sweep --runner-type b200 \
    --model-prefix <FAMILY> --precision <PREC> --framework <vllm|sglang> \
    --scenario-type agentic-coding --single-node --no-evals \
    --min-conc 4 --max-conc 4 --max-tp 4 \
    --config-files .github/configs/nvidia-master.yaml" \
  -f test-name="DEBUG: <MODEL> agentic" \
  -f duration-override=60
```

`duration-override=60` keeps the actual replay benchmark at 60 seconds;
the bulk of wall-clock time is the model load + cudagraph capture.

## Coverage matrix

Each agentic launcher lives at `benchmarks/single_node/agentic/<prefix>_<precision>_<hw>.sh`.
All sourced from `benchmarks/benchmark_lib.sh` for `build_replay_cmd` /
`write_agentic_result_json` / `resolve_trace_source` / `install_agentic_deps`.

| Family | NVIDIA launchers | AMD launchers |
|---|---|---|
| dsr1 | `dsr1_fp4_b200.sh` | `dsr1_fp4_mi355x.sh` |
| gpt-oss | `gptoss_fp4_b200.sh`, `gptoss_fp4_h100.sh`, `gptoss_fp4_h200.sh` | `gptoss_fp4_mi300x.sh`, `gptoss_fp4_mi325x.sh` |
| minimaxm2.5 | `minimaxm2.5_fp8_b200.sh`, `minimaxm2.5_fp4_b200.sh` | `minimaxm2.5_fp8_mi355x.sh` |
| qwen3.5 | `qwen3.5_bf16_b200.sh`, `qwen3.5_fp8_b200.sh` ¹ | `qwen3.5_fp8_mi355x.sh` |
| glm5 / glm5.1 | `glm5_fp8_b200.sh` | `glm5.1_fp4_mi355x.sh` |
| dsv4 | `dsv4_fp8_h200.sh` ² | (skipped — bespoke vLLM rebuild) |
| kimik2.5 | `kimik2.5_fp4_b200.sh`, `kimik2.5_int4_b200.sh` | `kimik2.5_fp4_mi355x.sh` |

¹ Both qwen3.5 NVIDIA images currently fail server start with PyTorch 2.9.1
+ CuDNN 9.13 incompatibility (pytorch/pytorch#168167). Replayer test pending
a working sglang image with CuDNN 9.15+.

² `dsv4-fp4-b200-sglang` uses `runner: b200-dsv4` which isn't registered in
runners.yaml; left unconfigured. Use `dsv4-fp8-h200-vllm` instead.

## Verifying a run

`agg_<RESULT_FILENAME>.json` under the `bmk_agentic_*` artifact contains:
- `num_requests_successful` / `num_requests_total`
- `total_generation_tokens` (output) / `total_prompt_tokens` (input)
- `mean_output_tokens_actual`
- `median_ttft` / `median_tpot` (seconds)
- `total_tput_tps` / `output_tput_tps`

Sanity thresholds: any of these being zero or absent indicates the
trace replayer failed to drive the server end-to-end.
