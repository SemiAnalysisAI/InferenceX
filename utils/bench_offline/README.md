# DeepSeek-V4 offline decode-step benchmark

> Scope: `offline-bench` branch only — a throwaway experiment, not for merge to `main`.

## What this is

An **offline**, in-process DeepSeek-V4-Pro benchmark for NVIDIA SKUs
(H200, B200, B300, GB200, GB300). It keeps one fixed-shape lockstep batch and
reports per-decode-step latency/throughput so engine, SKU, and parallelism
variants can be compared without HTTP serving scheduler effects.

"Offline" means there is no server, no HTTP, no request scheduler. The harness
calls the engine's in-process `engine.generate()` once for a **warmup** batch
and once for a **timed** batch of identical-shape prompts, then reports
decode-step latency and throughput. This follows an offline in-process harness
shape (one lockstep batch advanced N decode steps) rather than our normal
client/server `aiperf` path.

The whole thing lives in two places:

| Piece | Path |
|---|---|
| Harness (engine-agnostic driver) | `utils/bench_offline/run_offline.py` |
| Per-engine adapters | `utils/bench_offline/engines/{vllm,sglang,trt}_offline.py` |
| Prompt builder (InfiniteBench) | `utils/bench_offline/prompts.py` |
| Bench scripts (one per SKU×engine) | `benchmarks/single_node/fixed_seq_len/dsv4_*_offline.sh` |
| Sweep configs | `.github/configs/nvidia-master.yaml` (keys `dsv4-*-offline*`) |
| Launchers | `runners/launch_{h200-dgxc-slurm,b300-nv,gb200-nv,gb300-nv,...}.sh` |

## Methodology (what each run does)

- **ISL = 8192** input tokens per prompt, built from InfiniteBench
  (`build_infinitebench_prompts`, `prompts.py`).
- **OSL = 256 decode *steps*** — main-model forward passes. This is **not**
  256 output tokens when MTP is on.
- **Batch = CONC** prompts (`num_prompts = args.batch_size`). One warmup batch,
  then one timed batch.
- **MTP draft tokens default to 3** (override with
  `DSV4_OFFLINE_MTP_SPEC_TOKENS`). MTP **bonus** tokens are excluded from the
  headline metrics by construction; we count main-model steps and record
  observed acceptance separately.
- **MoE routing**: vLLM forces simulated `uniform_random`
  (`VLLM_MOE_ROUTING_SIMULATION_STRATEGY`) for an idealized expert-balance run.
  Override with `DSV4_OFFLINE_ROUTING_SIM=none` for honest routing. SGLang/TRT
  have no such knob and use real routing; every result records what actually ran
  in `moe_routing`.

## Metrics

- **TPOT is per decode step** (`tpot_unit=decode_step`), so `mean/median_tpot_ms`
  is comparable only across rows with the same unit. If the engine can't report
  per-step times the harness falls back and loudly tags the unit
  (`output_token_fallback` / `wall_clock_fallback`) — do not compare across units.
- **Output throughput is decode-steps/s**: `decode_step_throughput =
  steps_per_user * batch_size`. Per-GPU normalization divides by `num_chips`
  (= TP), so `step_tput_per_gpu = (CONC / num_chips) / (tpot_s)`.
- Because throughput normalizes by `num_chips`, **per-GPU numbers are
  comparable across DEP widths** as long as the *per-rank* batch matches
  (see "Concurrency" below).

## The single-node bench script — example

`benchmarks/single_node/fixed_seq_len/dsv4_fp8_h200_offline.sh` is the canonical
example. It is deliberately thin: resolve the model, start a GPU monitor, then
hand everything to `run_offline.py`.

```bash
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL MAX_MODEL_LEN RESULT_FILENAME
# ... resolve MODEL_PATH (hf download if not pre-staged) ...

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
DPA_FLAG=(); [[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)

# Multi-node block: no-op on a single node (SLURM_NNODES defaults to 1).
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    export MASTER_PORT=${MASTER_PORT:-29501}
fi
GPU_METRICS_OUT="$PWD/gpu_metrics.csv"
[[ "${SLURM_PROCID:-0}" -gt 0 ]] && GPU_METRICS_OUT="$PWD/gpu_metrics_node${SLURM_PROCID}.csv"
start_gpu_monitor --output "$GPU_METRICS_OUT"

python3 utils/bench_offline/run_offline.py \
    --engine vllm --model "$MODEL_PATH" \
    --tp "$TP" --ep "$EP_SIZE" --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" --mtp "$NUM_SPEC_TOKENS" \
    --infinitebench-input-len "$ISL" --decode-steps "$OSL" \
    --routing-sim-strategy "${DSV4_OFFLINE_ROUTING_SIM:-uniform_random}" \
    --nnodes "${SLURM_NNODES:-1}" --node-rank "${SLURM_PROCID:-0}" \
    --batch-size "$CONC" --result-dir "$PWD/" --result-filename "$RESULT_FILENAME" \
    "${DPA_FLAG[@]}"
```

Everything (`MODEL`, `TP`, `EP_SIZE`, `CONC`, …) arrives via env from the
benchmark template; the script body has no SKU- or precision-specific logic.

## Standardized scripts

**Every `dsv4_*_offline.sh` is byte-identical to the H200 example except two
things: the SKU label on the comment line, and the `--engine` value.** They are
generated from the H200 file by substituting `H200 → <SKU>` and `vllm → <engine>`.
If you change one, regenerate the rest the same way so they stay in lock-step:

```bash
cd benchmarks/single_node/fixed_seq_len
base=dsv4_fp8_h200_offline.sh
gen() { sed "s/H200/$2/g; s/vllm/$3/g" "$base" > "$1"; }
gen dsv4_fp4_b300_sglang_offline.sh B300 sglang   # etc.
```

The 10 scripts: `{b200,b300}×{vllm,sglang,trt}`, `{gb200,gb300}×{vllm}`,
`h200×{vllm,sglang}`.

## DEP and multi-node scaling

The DEP shape is data-parallel attention + expert parallelism: each chip is one
DP-attention rank (`tensor_parallel_size=1`) and the
experts are sharded across all ranks. In the harness this is `--dp-attn`, driven
by `_run_dp()` in `engines/vllm_offline.py`, which spawns one worker per rank and
sets the `VLLM_DP_*` env vars (the offline SPMD path).

A DEP width larger than one node's GPU count spans multiple nodes:

- **One Slurm node = one tray**: H200/B200/B300 = 8 GPUs, GB200/GB300 NVL72 = 4.
- The launcher detects `TP > GPUS_PER_NODE`, allocates `TP / GPUS_PER_NODE`
  nodes, and runs the bench script **once per node** via
  `srun --ntasks-per-node=1`. `SLURM_PROCID` becomes the DP `node_rank`,
  `SLURM_NNODES` the node count — exactly the args the script forwards.
- `_run_dp()` splits the global batch across all `total_dp_size = TP` ranks,
  each node spawning `TP/nnodes` local workers at `rank_offset = node_rank *
  local_dp_size`. Followers write `.dp_partial_node{N}.json` to the shared
  workspace; the leader (node 0) merges them and writes the final result.
- **`MASTER_ADDR`** is resolved by the launcher via `scontrol show hostname`
  (compute nodes don't have `scontrol`) and exported into the container; the DP
  TCP rendezvous uses `MASTER_PORT` (default 29501).
- On **NVL72** (GB200/GB300) the trays share one NVLink domain, so cross-tray
  DP/EP NCCL rides the rack-scale fabric. Only the initial TCP rendezvous uses
  the inter-node IP network.

### Concurrency must be a multiple of the DEP width

`CONC` is the **global** batch, split evenly across all `TP` ranks. If
`CONC < TP`, some ranks get a `"Placeholder"` prompt and per-rank metrics skew.
Use multiples of the DEP width so every rank gets an equal batch and the
*per-rank* batch matches across widths. E.g. DEP=4 uses `conc 8..128`
(per-rank 2..32); the matching DEP=16 sweep is `conc 32..512` (per-rank 2..32),
so per-GPU TPOT/throughput compares directly between matching per-rank batches.

## Config inventory (`.github/configs/nvidia-master.yaml`)

| Key | Runner | Prec | Engine | Shape | Conc |
|---|---|---|---|---|---|
| `dsv4-fp4-b200-{vllm,sglang,trt}-offline` | b200-dsv4 | fp4 | each | TP8 **and** DEP8 | 8–128 |
| `dsv4-fp4-b300-{vllm,sglang,trt}-offline` | b300 | fp4 | each | TP8 **and** DEP8 | 8–128 |
| `dsv4-fp8-h200-vllm-offline` | h200 | fp8 | vllm | TP8 **and** DEP8 | 8–128 |
| `dsv4-fp8-h200-sglang-offline` | h200-dgxc | fp8 | sglang | TP8 | 8–128 |
| `dsv4-fp8-h200-vllm-offline-2n` | h200-dgxc | fp8 | vllm | DEP16 (2 nodes) | 8–128 ⚠ |
| `dsv4-fp4-gb200-vllm-offline` | gb200 | fp4 | vllm | DEP4 (1 tray) | 8–128 |
| `dsv4-fp4-gb300-vllm-offline` | gb300-nv | fp4 | vllm | DEP4 (1 tray) | 8–128 |
| `dsv4-fp4-gb300-vllm-offline-16chip` | gb300-nv | fp4 | vllm | DEP16 (4 trays) | 32–512 |

⚠ the H200 2-node config predates the "conc must be a multiple of DEP width"
rule and is left as-is (it never got past rendezvous — see history).

## Running / dispatching

Single config via `workflow_dispatch` against `e2e-tests.yml`:

```bash
gh api -X POST /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' -f 'inputs[ref]=offline-bench' \
  -f 'inputs[test-name]=DSV4 GB300 offline 16-chip' \
  -f 'inputs[generate-cli-command]=test-config --config-files .github/configs/nvidia-master.yaml --config-keys dsv4-fp4-gb300-vllm-offline-16chip'
```

`test-config --config-keys <key>` isolates one config; `full-sweep
--model-prefix dsv4 --framework vllm --runner-type <r>` runs all matching.
(`full-sweep` has no `--config-keys` flag — that's `test-config` only.)

## Status & history (per SKU)

- **GB300-NV DEP=4** — ✅ works (run 27449751617, all 5 conc points). The proven
  baseline; uses the node-local `/scratch/models/DeepSeek-V4-Pro` checkpoint
  (loads as `deepseek_v4_fp8`).
- **GB300-NV DEP=16 (16-chip, 4 trays)** — dispatched; first multi-tray NVL72
  offline run. Watch the cross-tray rendezvous + NVLink NCCL bring-up.
- **B300 DEP=8 (1 node, 8 GPUs)** — config + script ready; same launcher path.
- **GB200 DEP=4** — ❌ OOM: the native FP8 checkpoint needs >184 GB/rank at DEP=4,
  doesn't fit a 184 GB GB200 GPU (run 27449109824). Kept for reference; use GB300.
- **H200 DEP=16 (2 nodes)** — ❌ cross-node TCP rendezvous timeout on the
  H200 DGXC-slurm cluster (port 29501 blocked between arbitrary node pairs).
  The multi-node DP code itself is correct and is what GB300 reuses; the
  failure was cluster networking, not the harness.

## Gotchas

- **GB200/GB300 images must be `-ubuntu2404`** (aarch64 Grace); the plain
  `v0.21.0` tag is x86-only. B200/B300/H200 use the plain tag.
- **Model staging is node-local** on `/scratch` (GB300) — a multi-tray run needs
  the checkpoint on *every* tray; `launch_gb300-nv.sh` pre-flight-checks this and
  fails fast with the offending hostname.
- **`precision: fp4` is a label** — the `/scratch/models/DeepSeek-V4-Pro`
  checkpoint loads with DeepSeek-V4's own mixed `deepseek_v4_fp8` quantization.
- Headline TPOT is only valid when `tpot_unit=decode_step`; a fallback unit in
  the result JSON means the engine didn't report per-step times.
