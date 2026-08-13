# PowerX Dense Ladder Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Queue three independent measured-power replicates for the Qwen3.5 FP8 8k/1k c1–128 ladder on H200, B200, and MI355X without changing default master sweeps.

**Architecture:** Add one campaign-only config at frozen commit `64c03138a`. The file contains only H200 TP8/EP8, B200 TP4, and MI355X TP4 STP lanes. Use `test-config` with an explicit concurrency list so each workflow emits exactly 24 fixed-sequence jobs, then dispatch the same matrix three times with fail-closed power validation.

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

- [ ] Commit the campaign config with an English subject and Simplified Chinese body.
- [ ] Push `codex/powerx-dense-ladder`.
- [ ] Dispatch `e2e-tests.yml` from `main` three times, checking out the campaign branch, with `require-power=true` and no duration override.
- [ ] Record all three run IDs and verify that each run enters the Actions queue with the expected PowerX dense-ladder name.
