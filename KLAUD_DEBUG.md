# KLAUD_DEBUG.md — Operational Knowledge for Recipe-Bump PRs

A running playbook of failures the Klaud-Cold image-bump cron has hit, the diagnoses, and the fixes/workarounds applied. **Read this first** before debugging a new failing claude/* PR — most failure modes here recur.

When you fix something not yet listed, add it here so the next session doesn't re-learn it.

---

## 1. PR setup-stage failures

### 1.1 `perf-changelog.yaml`: deletion-not-allowed
**Symptom:** the `setup` job fails before any sweep runs with
```
ValueError: Deletions are not allowed in /home/runner/work/InferenceX/InferenceX/perf-changelog.yaml.
Only additions to the changelog are permitted. Found deleted line: ...
```
**Root cause:** Cron-PR branches go stale; when main merges new changelog entries, the PR's local snapshot of `perf-changelog.yaml` no longer covers them, so the validator sees the missing lines as deletions. A naive rebase can also strip trailing whitespace from unrelated entries — same effect (e.g. `pr-link: ...1311  ` → `pr-link: ...1311`).

**Fix (canonical):**
```bash
# In the PR's worktree, after `git merge origin/main` conflicts on perf-changelog.yaml:
git checkout origin/main -- perf-changelog.yaml          # take main's bytes verbatim
cat >> perf-changelog.yaml <<EOF                          # then append THIS PR's entry at tail

- config-keys:
    - <recipe-key>
  description:
    - "<one-line summary>"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/<N>
EOF
python3 -c "import yaml; yaml.safe_load(open('perf-changelog.yaml'))"
```

Do **not** try a 3-way merge of `perf-changelog.yaml` — whitespace edits will silently re-trigger the deletion check.

---

## 2. vLLM v0.21.x / v0.20.x: GPU OOM at model-load

**Symptom:** vLLM workers OOM during weight loading or right after warmup:
- `HSA_STATUS_ERROR_OUT_OF_RESOURCES: Available Free mem : 0 MB` (AMD)
- `torch.OutOfMemoryError: CUDA out of memory. ... GPU N has X GiB of which Y MiB is free` (NVIDIA)
- vLLM may also log `_check_enough_kv_cache_memory` failing with **negative** available bytes (e.g. `-25.24 GiB`).

**Root cause:** v0.21.0 (and v0.20.2+) enabled an aggressive CUDA-graph memory profiler that pre-reserves a large chunk of VRAM up front (~30% on B200), shrinking effective `--gpu-memory-utilization` well below what the flag says. Old SHA-pinned custom images had a smaller footprint, so the recipe's existing `0.95` setting now starves the KV cache.

**Fix:** in `benchmarks/single_node/<recipe>.sh`, either:
1. **Lower `--gpu-memory-utilization`** (`0.95 → 0.90`, sometimes 0.85). Matches the H100/H200/B200 NVIDIA pattern. Smallest blast radius.
2. **Disable the profiler entirely** for cases where lowering isn't enough: `export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` before `vllm serve`. Matches `benchmarks/single_node/agentic/kimik2.5_fp4_b200.sh:65`.

Seen on: #1395 (kimik2.5-fp4-b200-vllm — needed env var), #1403 (gptoss-fp4-mi300x-vllm — needed 0.90), #1461 (dsv4-fp8-h200-vllm — needed 0.90).

---

## 3. Custom DSV4 image → generic v0.5.12 OOMs

**Symptom:** DSV4 recipes work on their SHA-pinned `lmsysorg/sglang:deepseek-v4-hopper@sha256:...` (or `deepseek-v4-b300`, `deepseek-v4-blackwell`) custom builds, but OOM on weights load when bumped to the generic `v0.5.12-cu130` release tag. Example: DSV4-Pro FP8+MTP weights consume ~125.43 GB / 141 GB per H200, leaving `-4.05 GB` for KV cache.

**Root cause:** The custom DSV4 images use a different weight layout / EAGLE draft handling that fits in less memory than the generic release. The release tag isn't a drop-in replacement.

**Fix:** keep DSV4 recipes pinned to their custom SHA-pinned image until upstream sglang gains the same DSV4-specific weight handling. Bumping to the generic tag is currently NOT viable.

Seen on: #1460 (dsv4-fp8-h200-sglang+mtp).

---

## 4. Upstream sglang v0.5.12 B300 regressions

Three distinct upstream regressions on NVIDIA B300 (Blackwell Ultra, `sm_103` — compute capability 10.3) shipped in `lmsysorg/sglang:v0.5.12-cu130`. (sm_120 is for *consumer* Blackwell / RTX 50 series, not B300 — don't propagate that.)

### 4a. DeepGemm TMA-descriptor crash (GLM-5-FP8)
**Symptom:** CUDA graph capture aborts with `CUDA_ERROR_ILLEGAL_ADDRESS (700)` at `/deepgemm/csrc/.../runtime_utils.hpp:143` on the **first batch size** for **every TP rank**. Server never serves a prompt.

**Workarounds (any one):**
1. `--fp8-gemm-runner-backend cutlass` to bypass DeepGemm via CUTLASS.
2. `export SGL_ENABLE_JIT_DEEPGEMM=0` before `python -m sglang.launch_server` to skip JIT DeepGemm.
3. Pin recipe to `lmsysorg/sglang:v0.5.11-cu130`.

Filed upstream: sgl-project/sglang#25551. Seen on #1421.

### 4b. trtllm GEMM bug at bs=128 + MTP / EAGLE (GLM-5-NVFP4)
**Symptom:** EAGLE draft CUDA graph capture crashes immediately at the largest batch size with `RuntimeError ... trtllm_batched_gemm_runner.cu:276 ... numBatches=256, GemmMNK 128x1024x6144`. The target model captures fine; only the draft model crashes.

**Workarounds:**
1. Cap `--cuda-graph-max-bs` and `--max-running-requests` to 64 in the launch script to avoid the bs=128 trigger.
2. Comment out the MTP/EAGLE scenarios on B300 in the recipe.
3. Pin to v0.5.11-cu130.

Filed upstream: sgl-project/sglang#25563. Seen on #1420.

### 4c. flash_attn SM-arch assertion (qwen3.5-bf16)
**Symptom:** All 4 TP workers AssertionError on first forward pass:
```
File "/opt/venv/.../sglang/srt/layers/attention/flashattention_backend.py:..."
  assert sm_100 <= arch <= sm_110f
```
B300 is `sm_103` (compute capability 10.3, Blackwell Ultra) — which is *nominally inside* the asserted `sm_100..sm_110f` range, yet the assertion still fires. Best guess is the cute kernel's `Arch.sm_110f` set only matches the architecture-specific feature-flag variants it was compiled for (e.g. `sm_100`, `sm_100f`, `sm_110`, `sm_110f`) and `sm_103` / `sm_103a` isn't in that explicit list. Server never becomes healthy; warmup times out at 600s.

**Fix:** Needs an sglang image with `flash_attn` that recognises `sm_103` / `sm_103a` — no local workaround. Pin to `v0.5.11-cu130` in the meantime.

Seen on #1422.

---

## 5. Cluster infrastructure (AMD MI355X / MI300X / MI325X)

### 5.1 `mia1-p01-g09 / g19 / g37` (amd-tw-mi355) — persistently drained
- **g09**: `pyxis is broken`
- **g19**: `Kill task failed (JobId=N StepId=N)`
- **g37**: `permission issues with GHA runner workflows : Not responding` (down since Mar 2026)

If a sweep job lands on any of these, it'll never start. Nothing to do at the recipe level — these stay drained until ops fixes them.

### 5.2 `mia1-p01-g11 / g12 / g31` — docker socket perms
**Symptom:** mi355x jobs fail with `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock` during the `docker stop $(docker ps -a -q)` cleanup step, cascading into SLURM job expiration.
**Fix:** ops needs to fix docker group / socket perms on these nodes. Recipe-level workaround: none.

### 5.3 `chi-mi300x-049` — `/nvme_home` disk-full
**Symptom:** pyxis container extraction fails with `No space left on device` writing to `/nvme_home/gharunner/.local/share/enroot/pyxis_*/opt/rocm-*/...`. The `/nvme_home` partition is hosted under `/` on this node and has been chronically near-full.

**Fix already landed:** `runners/launch_mi300x-amds.sh` now pins salloc to only known-good mi300x nodes (`chi-mi300x-[034-036,054,057-058]`) — see PR #1462. `chi-mi300x-049` is held in `State=DOWN` by a watchdog on the controller (`/home/gharunner/_audit/drain_049_watchdog.sh`) that re-applies the drain every 10s if SLURM auto-clears it (which it does on dynamic-norm nodes).

### 5.4 `chi-mi325x-pod1-017` — orphaned port-8888 process
**Symptom:** sglang server bind fails with `[Errno 98] Address already in use` on port 8888. Held by an MLPerf accuracy run started outside SLURM.
**Fix:** SSH to controller, find the holder via `ss -tlnp | grep :8888`, `kill` the PID. If recurring, file with the team running MLPerf experiments.

### 5.5 Cluster controller layout
- **amd-vultr-mi300**: SLURM controller for 7 mi300x nodes (3 down, see 5.3).
- **amd-vultr-mi325**: SLURM controller for 6 mi325x nodes.
- **amd-tw-mi355**: jumpbox → ssh further to compute nodes (`mia1-p01-gNN`). 12 nodes (3 drained, see 5.1).
- `/home` is NFS-mounted across clusters from `chi-mi325x-pod1-001:/nfs/homes`, **root-writable**.
- `/tmp` and `/nvme_home` are per-node local; HF cache lives at node-local `/raid/hf-hub-cache/` (2.7T per mi300x node).
- Use `srun -w <FQDN>` (with the **full FQDN**, not the short hostname) from the controller to run admin commands on a compute node.

### 5.6 Drain watchdog pattern
SLURM auto-clears `State=DRAIN` on `DYNAMIC_NORM` nodes when they re-register. To keep a node out of the pool sticky-style, use `State=DOWN` AND start a watchdog:
```bash
# on the controller, as root
nohup bash -c '
  while true; do
    s=$(scontrol show node <FQDN> 2>/dev/null | grep -oE "State=[A-Z+_]+")
    if ! echo "$s" | grep -qE "DOWN|DRAIN"; then
      scontrol update NodeName=<FQDN> State=DOWN Reason="watchdog" >/dev/null 2>&1
    fi
    sleep 10
  done
' > /home/gharunner/_audit/drain_<node>_watchdog.log 2>&1 &
```
Doesn't survive controller reboots — for permanent removal a SLURM admin should edit `slurm.conf`.

---

## 6. Docker image tag gotchas

**Don't invent a "release" tag pattern from a date-suffixed nightly.** `lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x` does **not** exist — only the dated `v0.5.12-rocm720-mi35x-20260517` does. All MI355X `sglang-rocm:rocm720` tags follow the dated-nightly pattern.

Before bumping an image, verify the target tag exists:
```bash
curl -sI "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags/v0.5.12-rocm720-mi35x"
# 200 → exists; 404 → doesn't
```

Or check whether any other recipe on main uses the proposed tag — if zero uses, suspect.

---

## 7. CI: rerun mechanics

- `gh run rerun <id> --failed` only works when the workflow run is **completed** with `conclusion=failure`. If the run is still `queued`/`in_progress`, the call returns "cannot be rerun".
- To abandon an in-flight run and start fresh, push an **empty commit** to the PR branch:
  ```bash
  git commit --allow-empty -m "Re-trigger sweep"
  git push
  ```
  The old run will be auto-cancelled by `workflow/cancel-sweep-on-merge` (provided the head SHA changed).
- For a `cancelled` run (not `failure`), use `gh run rerun <id>` without `--failed` to re-run everything.

---

## 8. gh CLI gotchas

- **`gh pr edit` silently aborts** on a Projects-classic deprecation GraphQL error. Title/body updates won't apply. Use `gh api -X PATCH "repos/<org>/<repo>/pulls/<N>" -f title="..." -F body=@file.md` instead.
- Same issue for adding labels — use `gh api -X POST "repos/<org>/<repo>/issues/<N>/labels" -f "labels[]=<name>"`.
- `gh pr view ... --jq .headRefName` output can have a trailing `\r`. Strip it: `gh pr view <N> --json headRefName --jq .headRefName | tr -d '\r\n'`. Otherwise shell concatenation produces `branchunners/launch_mi300x-amds.sh`-style corruption.
- `gh pr list --json statusCheckRollup` **truncates** each PR's rollup — never trust it for per-check filters. Re-query each PR individually with `gh pr view <N> --json statusCheckRollup`.
- `gh` and the GitHub Actions API: `conclusion` is `""` (empty string, not `null`) for in-flight checks, so `jq`'s `// .status` fallback doesn't trigger. Use:
  ```jq
  def state: if (.conclusion // "") != "" then .conclusion else .status end;
  ```

---

## 9. PR conventions for this repo

- Image-bump / new-recipe PRs I open on behalf of the user (or that the user creates) get the **`[Klaud Cold]`** title prefix.
- Add the `full-sweep-enabled` label so a canary-gated full sweep actually runs (`gh api -X POST ... labels[]=full-sweep-enabled`). Use `non-canary-full-sweep-enabled` instead only when the single-node canary is flaky or unrepresentative; it runs the full sweep without the canary gate. Without one of the sweep labels, the sweep is mostly SKIPPED.
- After any code change that shifts a PR's scope (drops a recipe, changes an image tag), **update the PR title AND body in the same step** and **verify** with `gh pr view <N> --json title,body` — `gh pr edit` silently fails (see §8).
- `utils/merge_with_reuse.sh <N>` is the merge entrypoint; it handles the `perf-changelog.yaml` auto-append.

---

## 10. Useful slash commands (defined in `.claude/commands/`)

- `/find-mergeable-claude-prs` — lists `claude/*` PRs whose full sweep finished all-green.
- `/list-claude-pr-status` — lists READY/RUNNING (and optionally FAILED) state per `claude/*` PR.
- `/fix-klaud-cron-prs` — diagnoses failing `claude/*` PRs by reading their failed job logs.
- `/merge-prs <N> [<N>...]` — sequential merge via `utils/merge_with_reuse.sh`.

Each command file is self-contained; read them to understand the exact jq filters they use.
