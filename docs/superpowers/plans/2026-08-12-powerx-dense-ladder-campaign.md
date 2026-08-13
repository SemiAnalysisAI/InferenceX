# PowerX Dense Ladder Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Queue three independent measured-power replicates for the Qwen3.5 FP8 8k/1k c1–128 ladder on H200, B200, and MI355X, then add a matched B300-only follow-up for the B200/B300 comparison, without changing default master sweeps.

**Architecture:** Keep all article-only lanes in one campaign config on the branch rooted at frozen commit `64c03138a`. The original matrix contains H200 TP8/EP8, B200 TP4/EP1, and MI355X TP4/EP1 STP lanes. The follow-up adds B300 TP4/EP1 with the same Qwen checkpoint, SGLang image, 8k/1k workload, and concurrency ladder as B200. Use `test-config` with an explicit concurrency list: the original matrix emits 24 fixed-sequence jobs per dispatch, while the B300-only supplement emits eight. Dispatch three independent repetitions with fail-closed power validation.

**Tech Stack:** InferenceX YAML matrix generation, GitHub Actions `e2e-tests.yml`, `uv`, Pydantic, PyYAML.

---

### Task 1: Define and validate the campaign matrix

**Files:**
- Create: `configs/powerx-dense-ladder.yaml`

- [x] Add the three frozen-snapshot STP configurations with `conc-start: 1` and `conc-end: 128`.
- [x] Run `generate_sweep_configs.py test-config` with `--conc 1 2 4 8 16 32 64 128 --seq-lens 8k1k --no-evals`.
- [x] Assert exactly eight rows per platform, 24 total, all at 8192/1024 with the intended topology and `spec-decoding=none`.
- [x] Run the matrix-logic test suite and `git diff --check`.

### Task 2: Queue three independent dispatches

**Files:**
- No additional file changes.

- [x] Commit the campaign config with an English subject and Simplified Chinese body.
- [x] Push `codex/powerx-dense-ladder`.
- [x] Dispatch `e2e-tests.yml` from `main` three times, checking out the campaign branch, with `require-power=true` and no duration override.
- [x] Record and verify the three original dense-ladder runs: `31672765610`, `31672784242`, and `31672785531`.

### Task 3: Add the matched B300 supplement

**Files:**
- Modify: `configs/powerx-dense-ladder.yaml`
- Modify: `benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_b300.sh`
- Add: `utils/test_powerx_b300_launcher_contract.py`
- Modify: `docs/superpowers/plans/2026-08-12-powerx-dense-ladder-campaign.md`

**Comparison contract:**

- Compare B300 only against the existing dense B200 lane: `Qwen/Qwen3.5-397B-A17B-FP8`, `lmsysorg/sglang:v0.5.14-cu130`, single-node TP4/EP1 STP, 8192 input tokens, 1024 output tokens, and concurrency 1/2/4/8/16/32/64/128.
- The B300 request and server tuning semantics are harmonized with the frozen B200 lane: neither uses the chat template, `--max-running-requests`, or a multimodal-attention override, and both use `--scheduler-recv-interval 10` through c4 and 30 above c4. The only retained script differences are B300 cluster path plumbing: the runner derives `MODEL_PATH` from the exact campaign checkpoint basename `Qwen3.5-397B-A17B-FP8`, the server loads weights and its tokenizer from that local path, and `--served-model-name` preserves the campaign's Hugging Face model ID for the request. B200 instead rewrites `MODEL` to its pre-staged local path before entering the benchmark script.
- Scope any result as a comparison of the matched deployed platform recipes, not isolated silicon. Hardware, firmware, host, and cluster environment remain platform-specific even after the benchmark launch contract is harmonized.
- Treat the device limits as capacity references, not measured consumption. The relevant HGX/DGX per-GPU limits are 1,000 W for B200 and 1,100 W for B300. Do not use 1,200 W for this B300 comparison. NVIDIA's [DGX B200 guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) records the 1,000 W maximum per GPU, and the [Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-gpu-resources/blackwell-ultra-datasheet) lists HGX B300 up to 1,100 W per GPU.
- Keep the system boundary separate. The [DGX B300 guide](https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html) lists 14.5 kW power consumption for the complete eight-GPU DGX system. That chassis figure includes non-GPU loads and is not a per-GPU TDP/TGP value; it must not be plotted as if it were the 1,100 W device limit.
- Publish a B300/B200 measured-point verdict only after all eight B300 cells have three valid independent dispatches (`power_valid=true`) from the same pinned campaign commit. A rating-envelope curve may use 1,100 W versus 1,000 W, but it must remain visually and verbally separate from measured GPU-board energy.

- [x] Add `qwen3.5-fp8-b300-sglang` to the campaign-only config with the matched contract above.
- [x] Generate the explicit ladder with `--conc 1 2 4 8 16 32 64 128 --seq-lens 8k1k --no-evals` and assert exactly eight jobs, all TP4/EP1 STP on `runner: b300`.
- [x] Harmonize B300 request/server semantics with the frozen B200 launcher and lock the allowed local-path differences with focused contract tests.
- [ ] Commit and push the campaign supplement. Record the resulting immutable commit as `POWERX_B300_COMMIT`; dispatch from that SHA rather than the moving branch name.
- [ ] Dispatch each command below once (three workflow runs total). These commands are prepared only; they must not be run as part of this plan edit.

```bash
POWERX_B300_COMMIT="$(git rev-parse HEAD)"
test "$POWERX_B300_COMMIT" = "$(git rev-parse origin/codex/powerx-dense-ladder)"

gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f "inputs[ref]=$POWERX_B300_COMMIT" \
  -f 'inputs[test-name]=PowerX B300 dense ladder r1' \
  -f 'inputs[generate-cli-command]=test-config --config-files configs/powerx-dense-ladder.yaml --config-keys qwen3.5-fp8-b300-sglang --conc 1 2 4 8 16 32 64 128 --seq-lens 8k1k --no-evals' \
  -f 'inputs[duration-override]=' \
  -F 'inputs[require-power]=true'

gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f "inputs[ref]=$POWERX_B300_COMMIT" \
  -f 'inputs[test-name]=PowerX B300 dense ladder r2' \
  -f 'inputs[generate-cli-command]=test-config --config-files configs/powerx-dense-ladder.yaml --config-keys qwen3.5-fp8-b300-sglang --conc 1 2 4 8 16 32 64 128 --seq-lens 8k1k --no-evals' \
  -f 'inputs[duration-override]=' \
  -F 'inputs[require-power]=true'

gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f "inputs[ref]=$POWERX_B300_COMMIT" \
  -f 'inputs[test-name]=PowerX B300 dense ladder r3' \
  -f 'inputs[generate-cli-command]=test-config --config-files configs/powerx-dense-ladder.yaml --config-keys qwen3.5-fp8-b300-sglang --conc 1 2 4 8 16 32 64 128 --seq-lens 8k1k --no-evals' \
  -f 'inputs[duration-override]=' \
  -F 'inputs[require-power]=true'
```
