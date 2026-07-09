---
description: Add a new model+hardware single-node benchmark recipe (script + master-config entry + perf-changelog + launcher routing), open a [Klaud Cold] PR, label full-sweep-fail-fast, and monitor CI
argument-hint: <model-link> <gpu-sku> [recipes-link] [draft-model-link] [mtp]
---

Add a new single-node benchmark recipe for a `model × hardware` combination, basing it on the
closest existing sibling recipe, then ship it as one `[Klaud Cold]` PR with a full GPU sweep.

Inputs from `$ARGUMENTS` (links first, then SKU; the rest optional and order-tolerant):

- **model-link** (required) — HuggingFace URL of the exact checkpoint to benchmark, e.g.
  `https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8`. Derive from it: the `model:` id
  (`org/repo`), the **precision** (from the repo name — `MXFP8`/`FP8`→`fp8`, `NVFP4`/`FP4`→`fp4`,
  `INT4`→`int4`, else `bf16`), and the **model-prefix** (e.g. `minimaxm3`).
- **gpu-sku** (required) — `b200 | b300 | h100 | h200 | gb200 | mi300x | mi325x | mi355x`.
  Determines the master config (`mi*`→`amd-master.yaml`, else `nvidia-master.yaml`), the image
  repo (`vllm/vllm-openai` vs `vllm/vllm-openai-rocm`), and the launcher.
- **recipes-link** (optional) — the model's `recipes.vllm.ai` page (or a vendor recipe commit).
  Consult it for the authoritative serve flags (block size, parsers, attention backend,
  parallelism guidance). If omitted, copy the sibling recipe's flags.
- **draft-model-link** (optional) — HF URL of a speculative-decoding draft (e.g.
  `https://huggingface.co/Inferact/MiniMax-M3-EAGLE3`). Its presence means **build the MTP
  variant** (EAGLE3 with this draft); see the MTP appendix.
- **mtp** (optional) — force the `spec-decoding: mtp` variant even without a draft link (use
  native MTP if the checkpoint ships `num_mtp_modules > 0`).

**engine** defaults to `vllm`; infer otherwise from the sibling / recipes page.

Standing prefs: PR title prefixed `[Klaud Cold]`; add `full-sweep-fail-fast` (strongly recommended over `full-sweep-enabled`) via the REST API
(`gh pr edit` hits the projects-classic GraphQL bug); fill the perf-changelog `pr-link` after
the PR exists; then monitor the sweep to a fail/success conclusion and report the job
breakdown. Do **not** invent image tags — verify on the registry first.

## Step 0 — deep-research the recipe (do this thoroughly before writing anything)

Don't guess flags or concurrencies — **deep-research the InferenceX codebase first**, then
the external sources. Read *several* similar files, not just one, and copy what actually runs.

**A. In-codebase research (primary — the repo is the source of truth):**
```bash
# similar benchmark scripts: same model on other SKUs, AND same SKU on other models
ls benchmarks/single_node/fixed_seq_len/<model>_*.sh benchmarks/single_node/fixed_seq_len/*_<sku>*.sh
# similar master-config entries (search spaces, image, parallelism), this model + analogues
grep -nE "<model>-|.*-<sku>-" configs/{nvidia,amd}-master.yaml
# the runner launcher for this SKU (script-name routing, env, mounts, MODEL_PATH rewrite)
sed -n '1,80p' runners/launch_<sku>*.sh
# shared helpers the scripts rely on
grep -nE "run_benchmark_serving|setup_eval_context|wait_for_server_ready|start_gpu_monitor" benchmarks/benchmark_lib.sh
```
- **Read multiple sibling scripts** end-to-end for the exact env vars and serve shape (`VLLM_*`,
  `SGLANG_*`, device mapping, download/cache handling, `--enforce-eager` vs graph capture,
  KV-cache dtype, attention/MoE backend, parsers). These are the truth for each runner.
- **Compare several master-config search spaces** (e.g. `dsv4`, `glm5`, the same model on a
  sibling SKU) to choose `{tp, ep, dp-attn} × concurrency` combos that fit *this* hardware's
  memory (small-mem SKUs like h100/mi300x go TP8-only; bigger SKUs add tp4/tp2/DEP).
- **Internalize the fixed-seq-len nuances from the existing configs**: `8k1k`/`1k8k` do **not**
  need the full `MAX_MODEL_LEN` (the matrix supplies `isl + osl + slack`), and graph-capture
  batch sizes are scaled to concurrency/scenario (and spec-token count for MTP), not maxed —
  copy how sibling scripts/configs already do it.

**B. External research (confirm against upstream guidance):**
- **`WebFetch` the model-link card + its `config.json`** → confirm `model:` id, precision, max
  context, architecture, spec-decode fields (`num_mtp_modules`, etc.).
- **`WebFetch` the recipes-link** (if given) → canonical `vllm serve` flags + troubleshooting;
  reconcile with what the sibling scripts do (the repo wins if they conflict — note why).
- If a **draft-model-link** is given, note its id for `--speculative-config` and check the card
  for method (`eagle3` vs native `mtp`) and recommended token count.
- Pick the **image tag** from the sibling's master-config entry (or recipes page) and **verify
  it exists** on the registry before using it.

This research directly feeds Step 2 (script flags/env) and Step 3 (search space).

## What you're producing (4–5 files)

1. `benchmarks/single_node/fixed_seq_len/<model>_<precision>_<sku>[_<engine>][_mtp].sh`
2. an entry in the master config — **`configs/nvidia-master.yaml`** (b*/h*/gb* SKUs) or
   **`configs/amd-master.yaml`** (mi* SKUs)
3. a `perf-changelog.yaml` entry (this diff vs main is what selects the sweep)
4. (if missing) `SPEC_SUFFIX`/framework-suffix routing in `runners/launch_<sku>*.sh`

## Step 1 — branch + find the sibling to copy

```bash
git checkout main && git pull origin main
git checkout -b feat/<model>-<sku>[-mtp]-dayzero
# nearest sibling: same model other SKU, or same SKU other model
ls benchmarks/single_node/fixed_seq_len/<model>_*           # same model, other hardware
ls benchmarks/single_node/fixed_seq_len/*_<sku>*.sh         # same hardware, other model
grep -n "<model>-<precision>-<sku>" configs/{nvidia,amd}-master.yaml
```
Read the closest sibling script **and** its master-config entry — copy their flag shapes and
search-space structure rather than inventing. The right model is "same model on a sibling SKU,
adjusted for this hardware's quirks."

## Step 2 — write the benchmark script

Copy the sibling script and adjust. Things that vary and must be checked against the sibling /
the model's `recipes.vllm.ai` page:
- **Mandatory model flags** (carry from the sibling): block size, parser flags
  (`--tool-call-parser` / `--reasoning-parser`), `--language-model-only` for text-only sweeps,
  `--trust-remote-code` where the model needs it.
- **Per-hardware deltas** — KV cache dtype (e.g. mi300x/gfx942 keeps **BF16**: no calibrated
  ROCm FP8 attn scales; most others use `fp8`), attention backend (CUDA: FlashInfer default;
  ROCm: `--attention-backend TRITON_ATTN`), and graph capture vs `--enforce-eager` (several
  AMD recipes use eager).
- **Capture sizing** — fixed-seq-len runs don't need graphs past the request concurrency:
  capture up to the next power of two ≥ `CONC` (≥ `CONC * (1 + NUM_SPEC_TOKENS)` with spec
  decoding), capped at vLLM's 2048.
- **`MAX_MODEL_LEN`** is the matrix-supplied scenario value (`isl + osl + slack`) — never
  hardcode the full context for 8k1k / 1k8k.
- **Memory headroom** — bigger checkpoints constrain TP/EP; if the sibling on a smaller-memory
  SKU is TP8-only (e.g. h100), match that.

Validate as you go: `bash -n <script>`.

## Step 3 — master-config entry + search space

Append `<model>-<precision>-<sku>[-<engine>][-mtp]` after the sibling, with the correct
`image`, `model`, `model-prefix`, `runner`, `precision`, `framework`. The **search space** is
`{tp, ep, dp-attn} × concurrency` per scenario (1k1k, 8k1k):
- Mirror a sibling's parallelism layouts; trim concurrency ranges to what the SKU's memory
  supports (small-mem SKUs → TP8-only, drop tp2/tp4 and DEP).
- Latency (TP-only) rows should start at conc 1; TEP/DEP rows start higher (they only pay off
  at scale).

Confirm which master file by SKU: `mi*` → `amd-master.yaml`, everything else → `nvidia-master.yaml`.

## Step 4 — launcher routing

The runner's launcher must resolve your script name. Most build it as
`<model>_<precision>_<sku>[_<framework>][_mtp].sh`. h200 launchers already carry the framework
+ `SPEC_SUFFIX`; **h100 and mi300x/mi355x launchers have historically hardcoded the bare
`_<sku>.sh`** — check and fix if you added a framework-tagged or `_mtp` script:
```bash
grep -n 'SPEC_SUFFIX\|FRAMEWORK_SUFFIX\|bash benchmarks\|EXP_NAME%%' runners/launch_<sku>*.sh
```
If needed, add `SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')`
(and/or the framework suffix) near the top and splice it into the bench-script path. Simulate
both `none`/`mtp` to confirm the resolved filename exists.

## Step 5 — perf-changelog

Append a `- config-keys: [<key>]` block with a clear `description` and `pr-link: TBD`. The
changelog diff vs `origin/main` is what `process_changelog.py` uses to select the sweep, so a
new entry is **required** for CI to run your config.

## Step 6 — validate locally

```bash
bash -n benchmarks/single_node/fixed_seq_len/<script>
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/<nvidia|amd>-master.yaml --config-keys <key>
```
Sanity-check the generated matrix: expected layouts/concurrencies, `max-model-len` = scenario
values, `spec-decoding` set where intended. Ensure both yaml files keep a trailing newline.

## Step 7 — PR + label + monitor

```bash
git add -A && git commit -m "<key>: <one-line>" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin feat/<model>-<sku>[-mtp]-dayzero
gh pr create --repo SemiAnalysisAI/InferenceX --base main \
  --title "[Klaud Cold] <key>: day-zero <MODEL> <SKU> recipe" --body "<summary>"
# fill perf-changelog pr-link with the real URL → commit → push
gh api -X POST repos/SemiAnalysisAI/InferenceX/issues/<PR>/labels -f "labels[]=full-sweep-fail-fast" --jq '.[].name'
```
Wait for the sweep run to register on the head SHA, then monitor to a conclusion and report
the job breakdown (e.g. 24 success / 6 skipped / 0 fail). If the **canary** fails, pull its log
(`gh api repos/.../actions/jobs/<id>/logs`), diagnose, fix, and re-push before iterating.
Finish on a clean `main`.

---

## Appendix — MTP / EAGLE3 spec-decoding variant

When a **draft-model-link** is given (or `mtp` is forced), build the `spec-decoding: mtp`
sibling of the base recipe. Use the provided draft id as `--speculative-config.model`. The
proven setup for **MiniMax-M3** (merged for b300/b200/h100/h200/mi355x/mi300x) uses the
external **`Inferact/MiniMax-M3-EAGLE3`** draft, `method: eagle3`, **3 speculative tokens**:
```
--speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 3<CUDA: , \"attention_backend\": \"FLASH_ATTN\">}"
```
- **CUDA (b*/h*)**: pin the drafter to `FLASH_ATTN` — FlashInfer can't run the MHA EAGLE3 head
  at the mandatory page-size 128. Scale cudagraph capture to `CONC * (1 + NUM_SPEC_TOKENS)`.
- **ROCm (mi*)**: no backend pin (server runs `TRITON_ATTN`). The shipped
  `vllm/vllm-openai-rocm:minimax-m3` image's AMD model lacks `SupportsEagle3`, so until the
  upstream fix (`vllm-project/vllm#45546`) is in the image, patch the installed
  `models/minimax_m3/amd/model.py` in-place before serving — copy the idempotent, drift-checked
  `python3 - <<'PYEOF' ... PYEOF` block verbatim from `minimaxm3_fp8_mi355x_mtp.sh` (adds
  `EagleModelMixin` + aux-hidden-state emission + `SupportsEagle3` on the two outer classes).
- **All**: route benchmark prompts through `--use-chat-template` (+ `pip install -q datasets
  pandas`) — raw random tokens tank spec-decode acceptance. Search space mirrors the non-MTP
  entry trimmed at the extreme-conc end, latency rows starting at conc 1, `tp2-ep2` dropped.
- Other models may instead use **native MTP** (`method: mtp`, no external draft) when the
  checkpoint ships MTP modules (`num_mtp_modules > 0`), e.g. the DeepSeek-V4 recipes.
