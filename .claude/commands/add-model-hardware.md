---
description: Add a new model+hardware single-node benchmark recipe (srt-slurm recipe + master-config entry + perf-changelog), open a [Klaud Cold] PR, label full-sweep-fail-fast, and monitor CI
argument-hint: <model-link> <gpu-sku> [recipes-link] [draft-model-link] [mtp]
---

Add a new single-node benchmark recipe for a `model × hardware` combination, basing it on the
closest existing sibling recipe, then ship it as one `[Klaud Cold]` PR with a full GPU sweep.

Inputs from `$ARGUMENTS` (links first, then SKU. The rest are optional and order-tolerant):

- **model-link** (required). HuggingFace URL of the exact checkpoint to benchmark, e.g.
  `https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8`. Derive from it: the `model:` id
  (`org/repo`), the **precision** (from the repo name, where `MXFP8`/`FP8`→`fp8`, `NVFP4`/`FP4`→`fp4`,
  `INT4`→`int4`, else `bf16`), and the **model-prefix** (e.g. `minimaxm3`).
- **gpu-sku** (required). Choose from `b200 | b300 | h100 | h200 | gb200 | mi300x | mi325x | mi355x`.
  Determines the master config (`mi*`→`amd-master.yaml`, else `nvidia-master.yaml`), the image
  repo (`vllm/vllm-openai` vs `vllm/vllm-openai-rocm`), and the launcher.
- **recipes-link** (optional). The model's `recipes.vllm.ai` page (or a vendor recipe commit).
  Consult it for the authoritative serve flags (block size, parsers, attention backend,
  parallelism guidance). If omitted, copy the sibling recipe's flags.
- **draft-model-link** (optional). HF URL of a speculative-decoding draft (e.g.
  `https://huggingface.co/Inferact/MiniMax-M3-EAGLE3`). Its presence means **build the MTP
  variant** (EAGLE3 with this draft). See the MTP appendix.
- **mtp** (optional). Force the `spec-decoding: mtp` variant even without a draft link (use
  native MTP if the checkpoint ships `num_mtp_modules > 0`).

**engine** defaults to `vllm`. Infer otherwise from the sibling / recipes page.

Standing prefs: Prefix the PR title with `[Klaud Cold]`. Add `full-sweep-fail-fast` (strongly recommended over `full-sweep-enabled`) through the REST API
because `gh pr edit` hits the projects-classic GraphQL bug. Fill the perf-changelog `pr-link` after
the PR exists. Then monitor the sweep to a fail/success conclusion and report the job
breakdown. Do **not** invent image tags. Verify them on the registry first.

## Step 0 — deep-research the recipe (do this thoroughly before writing anything)

Don't guess flags or concurrencies. **Deep-research the InferenceX codebase first**, then
the external sources. Read *several* similar files, not just one, and copy what actually runs.

Check `MODELS.md` before choosing a model, scenario, or precision. Do not reintroduce retired coverage; preserve only explicitly documented exceptions. Use active siblings, not files under `deprecated/`.

**A. In-codebase research (primary because this repo is the source of truth):**
```bash
# similar srt-slurm recipes: same model on other SKUs, AND same SKU on other models
ls -d benchmarks/single_node/srt-slurm-recipes/<model>/*/* benchmarks/single_node/srt-slurm-recipes/*/*/<sku>-*
# similar master-config entries (search spaces, image, parallelism), this model + analogues
grep -nE "<model>-|.*-<sku>-" configs/{nvidia,amd}-master.yaml
# how a matrix point selects exactly one recipe variant (TP, GPUs, CONC, KV_OFFLOADING, image)
sed -n '/def select_recipe/,/^def runtime_arguments/p' infx/srt_slurm/single_node.py
```
- **Read multiple sibling recipes** end-to-end for the exact engine args, env vars and serve shape (`VLLM_*`,
  `SGLANG_*`, device mapping, download/cache handling, `--enforce-eager` vs graph capture,
  KV-cache dtype, attention/MoE backend, parsers, `setup_script`). These are the truth for each runner.
- **Compare several master-config search spaces** (e.g. `dsr1`, `qwen3.5`, the same model on a
  sibling SKU) to choose `{tp, ep, dp-attn} × concurrency` combos that fit *this* hardware's
  memory. Small-memory SKUs like h100/mi300x go TP8-only, while bigger SKUs add tp4/tp2/DEP.
- **Internalize the fixed-seq-len nuances from the existing configs**: `8k1k` runs do **not**
  need the full `MAX_MODEL_LEN` (the matrix supplies `isl + osl + slack`), and graph-capture
  batch sizes are scaled to concurrency/scenario (and spec-token count for MTP), not maxed.
  Copy how sibling recipes/configs already do it.

**B. External research (confirm against upstream guidance):**
- **`WebFetch` the model-link card + its `config.json`** → confirm `model:` id, precision, max
  context, architecture, spec-decode fields (`num_mtp_modules`, etc.).
- **`WebFetch` the recipes-link** (if given) → canonical `vllm serve` flags + troubleshooting.
  Reconcile with what the sibling recipes do. If they conflict, follow the repo and note why.
- If a **draft-model-link** is given, note its id for `--speculative-config` and check the card
  for method (`eagle3` vs native `mtp`) and recommended token count.
- Pick the **image tag** from the sibling's master-config entry (or recipes page) and **verify
  it exists** on the registry before using it.

This research directly feeds Step 2 (recipe args/env) and Step 3 (search space).

## What you're producing (3 files)

1. `benchmarks/single_node/srt-slurm-recipes/<model-prefix>/<engine>/<sku>-<precision>[-mtp]/8k1k.yaml`
   (an `agentic.yaml` beside it for AgentX)
2. an entry in either master config, **`configs/nvidia-master.yaml`** (b*/h*/gb* SKUs) or
   **`configs/amd-master.yaml`** (mi* SKUs), with `srt-recipe:` on every search-space row
3. a `perf-changelog.yaml` entry (this diff vs main is what selects the sweep)

## Step 1 — branch + find the sibling to copy

```bash
git checkout main && git pull origin main
git checkout -b feat/<model>-<sku>[-mtp]-dayzero
# nearest sibling: same model other SKU, or same SKU other model
ls -d benchmarks/single_node/srt-slurm-recipes/<model>/*/*      # same model, other hardware
ls -d benchmarks/single_node/srt-slurm-recipes/*/*/<sku>-*       # same hardware, other model
grep -n "<model>-<precision>-<sku>" configs/{nvidia,amd}-master.yaml
```
Read the closest sibling recipe **and** its master-config entry. Copy their flag shapes and
search-space structure rather than inventing. The right model is "same model on a sibling SKU,
adjusted for this hardware's quirks."

## Step 2 — write the srt-slurm recipe

Copy the sibling recipe (`base:` plus one `override_*` variant per matrix point) and adjust.
Engine flags are `roles.agg.args` keys without the leading `--`; env vars go in
`roles.agg.env`; each variant names its `CONC` (and `KV_OFFLOADING` for AgentX) in
`benchmark.env`, and `model.container` must equal the master-config `image`. Things that vary and must be checked against the sibling /
the model's `recipes.vllm.ai` page:
- **Mandatory model flags** (carry from the sibling): block size, parser flags
  (`--tool-call-parser` / `--reasoning-parser`), `--language-model-only` for text-only sweeps,
  `--trust-remote-code` where the model needs it.
- **Per-hardware deltas.** KV cache dtype (e.g. mi300x/gfx942 keeps **BF16** because it has no calibrated
  ROCm FP8 attn scales, while most others use `fp8`), attention backend (CUDA uses FlashInfer by default,
  while ROCm uses `--attention-backend TRITON_ATTN`), and graph capture vs `--enforce-eager` (several
  AMD recipes use eager).
- **Capture sizing.** Fixed-seq-len runs don't need graphs past the request concurrency.
  Capture up to the next power of two ≥ `CONC` (≥ `CONC * (1 + NUM_SPEC_TOKENS)` with spec
  decoding), capped at vLLM's 2048.
- **`MAX_MODEL_LEN`** is the matrix-supplied scenario value (`isl + osl + slack`). Never
  hardcode the full context for 8k1k.
- **Memory headroom.** Bigger checkpoints constrain TP/EP. If the sibling on a smaller-memory
  SKU is TP8-only (e.g. h100), match that.

Validate as you go: `python3 -c "import yaml; yaml.safe_load(open('<recipe>'))"`.

## Step 3 — master-config entry + search space

Append `<model>-<precision>-<sku>[-<engine>][-mtp]` after the sibling, with the correct
`image`, `model`, `model-prefix`, `runner`, `precision`, `framework`. The **search space** is
`{tp, ep, dp-attn} × concurrency` per supported scenario from `MODELS.md` (8k1k or AgentX as applicable; 1k1k is only retained for GLM-5.1 B200 TileRT):
- Mirror a sibling's parallelism layouts. Trim concurrency ranges to what the SKU's memory
  supports (small-mem SKUs → TP8-only, drop tp2/tp4 and DEP).
- Latency (TP-only) rows should start at conc 1. TEP/DEP rows start higher (they only pay off
  at scale).

Confirm which master file by SKU: `mi*` → `amd-master.yaml`, everything else → `nvidia-master.yaml`.

## Step 4 — no launcher routing

Single-node points with an `srt-recipe:` go through `launch_srt_single_node`, which picks the
one recipe variant whose TP/GPU count, `CONC`, `KV_OFFLOADING` and image match the matrix
point (`infx/srt_slurm/single_node.py::select_recipe`). No per-script launcher routing is
needed; if a point matches zero or several variants, fix the recipe, not the launcher.

## Step 5 — perf-changelog

Append a `- config-keys: [<key>]` block with a clear `description` and `pr-link: TBD`. The
changelog diff vs `origin/main` is what `process_changelog.py` uses to select the sweep, so a
new entry is **required** for CI to run your config.

## Step 6 — validate locally

```bash
python3 -c "import yaml; yaml.safe_load(open('benchmarks/single_node/srt-slurm-recipes/<recipe>'))"
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/<nvidia|amd>-master.yaml --config-keys <key>
```
Sanity-check the generated matrix: expected layouts/concurrencies, `max-model-len` = scenario
values, `spec-decoding` set where intended. Ensure both yaml files keep a trailing newline.

## Step 7 — PR + label + monitor

```bash
git add -A && git commit -m "<key>: <one-line>"
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
- **CUDA (b*/h*).** Pin the drafter to `FLASH_ATTN` because FlashInfer can't run the MHA EAGLE3 head
  at the mandatory page-size 128. Scale cudagraph capture to `CONC * (1 + NUM_SPEC_TOKENS)`.
- **ROCm (mi*)**: no backend pin (server runs `TRITON_ATTN`). Use an image that already carries
  the upstream `SupportsEagle3` fix (`vllm-project/vllm#45546`); the old in-place model patch was
  retired with the legacy bash scripts. Copy the sibling `minimaxm3/vllm/mi*-mtp` recipe.
- **All**: set `USE_CHAT_TEMPLATE: 'true'` in the recipe's `benchmark.env` (the single-node
  adapter requires it whenever speculation is on). Raw random tokens tank spec-decode acceptance. Search space mirrors the non-MTP
  entry trimmed at the extreme-conc end, latency rows starting at conc 1, `tp2-ep2` dropped.
- Other models may instead use **native MTP** (`method: mtp`, no external draft) when the
  checkpoint ships MTP modules (`num_mtp_modules > 0`), e.g. the DeepSeek-V4 recipes.
