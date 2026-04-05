# MXFP4 Qwen3.5 Eval Failure: `srun --pty` vs `srun` (no pty)

## Summary

Running `amd/Qwen3.5-397B-A17B-MXFP4` on MI355X with SGLang (`rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260402`) produces **~15% GSM8K accuracy when launched via `srun bash -c '...'`**, but **96-98% accuracy when launched via `srun --pty bash`** (interactive). The commands, container image, model weights, environment variables, and hardware are identical. The only difference is whether `--pty` is passed to `srun`.

FP8 models are unaffected — `Qwen/Qwen3.5-397B-A17B-FP8` scores 97% via CI on the same infrastructure.

## Reproduction

### Works (~96-98% accuracy)

```bash
# Step 1: Allocate node
salloc --partition=compute --gres=gpu:4 --exclusive --cpus-per-task=128 --time=180

# Step 2: Enter container interactively (--pty)
srun --jobid=$JOB_ID \
    --container-image=/var/lib/squash/rocm_sgl-dev_v0.5.10rc0-rocm720-mi35x-20260402.sqsh \
    --container-mounts=/var/lib/hf-hub-cache/:/mnt/hf_hub_cache/ \
    --container-mount-home --container-writable \
    --container-workdir=/workspace/ \
    --no-container-entrypoint --export=ALL \
    --pty bash

# Step 3: Inside the container, run server + eval
export SGLANG_USE_AITER=1
export HF_HUB_CACHE=/mnt/hf_hub_cache/
python3 -m sglang.launch_server --model-path amd/Qwen3.5-397B-A17B-MXFP4 \
    --trust-remote-code --host 0.0.0.0 --port 9000 \
    --tensor-parallel-size 4 --attention-backend aiter \
    --mem-fraction-static 0.9 \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --watchdog-timeout 1200 --context-length 9416 > /tmp/server.log 2>&1 &
# (wait for server ready)

export OPENAI_API_KEY=EMPTY
python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks utils/evals/gsm8k.yaml --output_path /tmp/eval_results --log_samples \
    --model_args "model=amd/Qwen3.5-397B-A17B-MXFP4,base_url=http://0.0.0.0:9000/v1/chat/completions,api_key=EMPTY,eos_string=,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=9416" \
    --gen_kwargs "max_tokens=5320,temperature=0,top_p=1"
# Result: 96-98% on GSM8K
```

### Also works (~98% accuracy)

Running `bash -c '...'` **inside** the already-running interactive container also produces correct results. This rules out `bash -c` shell initialization as the cause:

```bash
# From within the interactive container (step 3 above), run:
bash -c '
export SGLANG_USE_AITER=1
export HF_HUB_CACHE=/mnt/hf_hub_cache/
python3 -m sglang.launch_server ... > /tmp/server2.log 2>&1 &
# (wait for server)
python3 -m lm_eval ...
'
# Result: 98% on GSM8K (100 samples)
```

### Fails (~15% accuracy)

The exact same commands, but launched via `srun bash -c '...'` (no `--pty`):

```bash
srun --jobid=$JOB_ID \
    --container-image=/var/lib/squash/rocm_sgl-dev_v0.5.10rc0-rocm720-mi35x-20260402.sqsh \
    --container-mounts=/path/to/workspace:/workspace/,/var/lib/hf-hub-cache/:/mnt/hf_hub_cache/ \
    --container-mount-home --container-writable \
    --container-workdir=/workspace/ \
    --no-container-entrypoint --export=ALL \
    bash -c '
export SGLANG_USE_AITER=1
export HF_HUB_CACHE=/mnt/hf_hub_cache/
python3 -m sglang.launch_server --model-path amd/Qwen3.5-397B-A17B-MXFP4 \
    --trust-remote-code --host 0.0.0.0 --port 9000 \
    --tensor-parallel-size 4 --attention-backend aiter \
    --mem-fraction-static 0.9 \
    --model-loader-extra-config '"'"'{"enable_multithread_load": true}'"'"' \
    --watchdog-timeout 1200 --context-length 9416 > /tmp/server.log 2>&1 &
# (wait for server)
python3 -m lm_eval ...
'
# Result: ~15% on GSM8K
```

## What the bad outputs look like

Analysis of 2,638 sample outputs from the failing CI run:

- **82% of responses** start with verbose plain-text `Thinking Process:` analysis (not native `<think>` blocks)
- **58% of responses** are >10,000 characters (degenerate, repetitive)
- **Short responses (<5k chars)**: 37.5% correct
- **Long responses (≥5k chars)**: 12.6% correct

Failure modes observed:
1. **Degenerate repetition**: Model loops a phrase ("She sells the eggs for $2 each.") for 16,000+ chars, never producing a `#### [number]` answer
2. **Topic drift**: Model starts reasoning about the correct question, but mid-thinking drifts to answering a completely different question from the few-shot examples
3. **Correct but short**: Simple problems that require brief reasoning are answered correctly

This pattern (coherent start, degeneration over long sequences) is consistent with a numerical precision or GPU computation issue specific to MXFP4.

## What we've ruled out

| Hypothesis | Test | Result |
|---|---|---|
| `eos_string=</s>` in lm-eval model_args | Removed it | Still ~15% |
| Corrupted squash file | Deleted and rebuilt | Still ~15% |
| CI environment variables (`--export=ALL`) | Wiped all CI env vars inside container | Still ~15% |
| `--export=ALL` passing bad env vars | Removed `--export=ALL` entirely | Still ~15% (also broke aiter import until fixed) |
| `--container-mount-home` leaking runner's Python packages | Removed it | Still ~15% |
| `sitecustomize.py` lm-eval monkey-patch | Removed it entirely | Still ~15% |
| Shell initialization files (`.bashrc`, `/etc/profile`) | Used `bash --login -c` | Still ~15% |
| CPU binding / NUMA affinity | Added `--cpu-bind=none` | Still ~15% |
| Docker image / sglang version | Confirmed container sglang commit `ba6d54d0f` produces 96-98% when run interactively | Image is fine |
| Different nodes | Tested on mia1-p01-g10, g16, g17 | Same results on all nodes |
| Sample selection bias | Checked first 100 samples from CI run: 17% accuracy | Same samples score 97% interactively |

## What we know

1. **`srun --pty bash` + commands → 96-98%** (interactive shell in container)
2. **`bash -c '...'` inside interactive container → 98%** (non-interactive shell, but within pty session)
3. **`srun bash -c '...'` → ~15%** (non-interactive, no pty)
4. **`srun bash --login -c '...'` → ~18%** (login shell, no pty)
5. **`srun --cpu-bind=none bash -c '...'` → ~16%** (no CPU binding, no pty)
6. **FP8 models are unaffected** — only MXFP4 shows this behavior
7. **The server starts and serves requests in both cases** — it's not a crash, it's degraded inference quality
8. **Results are consistent** across multiple runs and multiple nodes (~15% every time without pty)

## Environment

- **Hardware**: AMD Instinct MI355X (8 GPUs per node, TP=4)
- **Container**: `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260402`
- **SGLang commit**: `ba6d54d0f`
- **Model**: `amd/Qwen3.5-397B-A17B-MXFP4` (MXFP4 quantized, 397B MoE, 17B active params)
- **Slurm**: `salloc --gres=gpu:4 --exclusive --cpus-per-task=128`
- **Container runtime**: Pyxis/enroot via Slurm

## Hypothesis

The `--pty` flag in `srun` changes how the process is managed at the Slurm/kernel level. Without `--pty`, some aspect of the process environment — potentially related to:

- Process group / session management
- File descriptor inheritance (stdin/stdout/stderr buffering)
- Terminal control / signal handling
- cgroup resource limits applied by Slurm for batch vs interactive tasks
- GPU context initialization behavior when no controlling terminal is present
- Memory mapping or hugepage allocation differences

— causes the MXFP4 quantized model's inference to produce degraded results. This does NOT affect FP8 models, suggesting the issue is in a code path specific to MXFP4 dequantization or the aiter attention backend's MXFP4 kernels.

## Smoking gun: MXFP4 non-determinism at temperature=0

Smoke test: same prompt sent 3 times at `temperature=0, top_p=1` — should produce identical outputs.

### Interactive session (srun --pty, ~97% GSM8K)
```
mul (17*19, ~598 chars): 3/3 IDENTICAL hashes (14af266431e79e60) ← deterministic
bags (word problem, ~820 chars): 3/3 DIFFERENT hashes ← non-deterministic even here!
  7f93570773307baf, 9d7b3a60c07da5d9, 997ce1220a7b610c
```

### CI (srun, no pty, ~15% GSM8K)
```
mul (17*19, ~580 chars): 3/3 DIFFERENT hashes ← non-deterministic even for simple prompts!
  00fb7aad92c0533f, d12bd28cab7f6ce1, 2e717b3d88d40a6b
bags (word problem, ~760 chars): 3/3 DIFFERENT hashes
  28aabf5a06d28df2, 7750dff3b5a4c446, c6df81ff26610c3e
```

### Interpretation

The MXFP4/aiter non-determinism exists in BOTH environments — even the interactive session produces different outputs for longer prompts. But the CI environment (no pty) makes it significantly worse, causing non-determinism even for short ~600 char responses. This points to a fundamental MXFP4/aiter computation correctness bug that is exacerbated by the process launch environment.

## Root cause

This is an **MXFP4/aiter correctness bug** — non-deterministic inference at temperature=0. The pty vs no-pty launch path changes the severity but does not cause the bug. FP8 models are unaffected, confirming the issue is in MXFP4-specific code paths.

Relevant upstream issues (per expert analysis):
- AITER open FP4 issue: silent, non-deterministic incorrectness depending on tensor storage/layout
- AITER MI355x/ROCm 7.2 issue: fused_moe using wrong MXFP4 quantization internally
- Earlier MXFP4 rounding bug (fixed separately)

## Key question

**What does `srun --pty` change at the process/kernel level that reduces (but does not eliminate) MXFP4 non-determinism on MI355X?** The only observable differences are fd0/fd2 being PTY vs pipe. CPU affinity, task count, env vars, and container config are all identical.

## Related links

- SGLang issue: https://github.com/sgl-project/sglang/issues/21919
- SGLang MXFP4 support PR: https://github.com/sgl-project/sglang/pull/21234
- InferenceX PR #994 (original failure): https://github.com/SemiAnalysisAI/InferenceX/pull/994
- Failing CI run (with `--cpu-bind=none`): https://github.com/SemiAnalysisAI/InferenceX/actions/runs/23997512753/job/69988001403
- Passing interactive test (500 samples, 96.2%): run manually on mia1-p01-g16
