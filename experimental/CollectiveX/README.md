# CollectiveX

Cross-vendor collective / EP-library benchmark (see `plan.md`). Per-SKU **launch
adapters** (InferenceX-style `launch_<sku>.sh`) run **any benchmark** — selected
by `CX_BENCH` — through a shared in-container runner, and a GitHub Actions
workflow triggers runs on `push` (no merge to main needed). Milestone-0 headline
already ran for real on both B200 (8× NVLink island) and GB200 (4× NVL72 MNNVL).

> Experimental: WIP, not an official InferenceMAX result. All logic stays under
> `experimental/CollectiveX/`; the only file outside is the orchestration-only
> workflow.

## Files

| File | Role |
|---|---|
| `env_capture.py` | Layer-0 environment + topology fingerprint → JSON (stdlib only) |
| `run_nccl.py` | run stock `nccl-tests`, parse the text table, emit flat JSON (stdlib only) |
| `run_deepep.py` | DeepEP dispatch+combine, normal mode, correctness-gated (torch + DeepEP) |
| `run_mori.py` | MoRI (AMD) dispatch+combine, normal mode, correctness-gated (torch + MoRI) |
| `plot.py` | latency/bus-bw curves, B200-vs-GB200 overlay with a comparison guard (matplotlib) |
| `launchers/common.sh` | shared helpers: image resolve, enroot squash, staging, nccl-tests build |
| `launchers/run_in_container.sh` | generic in-container dispatcher — runs `CX_BENCH` (nccl/deepep/mori/all) |
| `launchers/launch_<sku>.sh` | per-SKU adapters: `launch_b200-dgxc.sh` (8× NVLink), `launch_b200-dgxc-slurm.sh` (2-node IB), `launch_gb200-nv.sh` (NVL72 MNNVL), `launch_mi355x-amds.sh` (8× XGMI, AMD MoRI) |
| `CONTAINERS.md` | the pinned multi-arch container + audited library versions |
| `results/` | flat JSON artifacts (+ `plots/`, raw captures) |
| `tests/fixtures/` | captured nccl-tests output for offline parser checks |

## Run

### Via GitHub Actions (`.github/workflows/collectivex-experimental.yml`)

- **push** to `experimental/CollectiveX/**` → the **MI355X MoRI** dispatch/combine
  run (the "CollectiveX Experimental" job; lands on a free `mi355x-amds` runner).
- **workflow_dispatch** → pick `sku` (gb200 / b200-dgxc / b200-multinode /
  mi355x), `benchmark` (nccl / deepep / mori / all — `mori` is AMD-only), ops,
  sizes, ngpus. Lands on that SKU's self-hosted runner and runs
  `launch_${RUNNER_NAME%%_*}.sh`.

Each job renders a results table to the **GitHub Actions job summary** (via
`summarize.py --markdown` → `$GITHUB_STEP_SUMMARY`) and uploads the result JSONs
as an artifact. (The workflow only fires once the branch is pushed to GitHub.)

### Directly on a cluster login node

```bash
# benchmark is selected by CX_BENCH (default nccl)
bash experimental/CollectiveX/launchers/launch_gb200-nv.sh                 # GB200, NCCL primitives
CX_BENCH=deepep bash experimental/CollectiveX/launchers/launch_gb200-nv.sh # GB200, DeepEP (rebuild)
bash experimental/CollectiveX/launchers/launch_b200-dgxc.sh               # B200 8× NVLink
bash experimental/CollectiveX/launchers/launch_b200-dgxc-slurm.sh         # B200 2-node, cross-IB
bash experimental/CollectiveX/launchers/launch_mi355x-amds.sh             # MI355X 8× XGMI, MoRI EP (AMD; forces CX_BENCH=mori)
```

Knobs: `CX_BENCH` (nccl|deepep|mori|all), `CX_OPS`, `CX_MIN_BYTES`/`CX_MAX_BYTES`,
`CX_NGPUS`, `CX_TIME`, `CX_IMAGE`, `CX_SQUASH_DIR`, `CX_STAGE_DIR` (compute-visible
staging — needed on GB200/watchtower), `CX_DRYRUN=1` (print plan, allocate
nothing). Results land in `experimental/CollectiveX/results/`.

### Offline (no GPU) — verify the parser/JSON pipeline

```bash
python3 run_nccl.py --op all_reduce --parse-only tests/fixtures/all_reduce_perf_b200_8gpu.txt \
  --world-size 8 --nodes 1 --runner b200-dgxc --topology-class b200-nvlink-island --out /tmp/parsed.json
python3 env_capture.py            # prints a (degraded, off-GPU) env record
python3 plot.py --results-dir results --out-dir results/plots   # needs matplotlib
```

## Container

One **multi-arch** image for all NVIDIA SKUs, imported by tag
`lmsysorg/sglang:v0.5.11-cu130` (amd64 + arm64; index digest `sha256:061fb71f…`
recorded for provenance). Imported by tag, not digest — enroot's anonymous
Docker Hub auth needs a tag, and a bare digest ref hangs in CI. See
`CONTAINERS.md` for versions, the DeepEP-rebuild note, and the bundled-DeepEP
DeepSeek-V4 fallback images.

## How it runs (confirmed against the live clusters)

- Adapters mirror `runners/launch_*.sh`: `salloc` → enroot squash (import only if
  missing) → `srun --container-image=… --container-mounts=<repo>:/ix` → in-container
  `run_in_container.sh`. B200 partition `gpu-2`, GB200 partition `batch`, account
  `benchmark`.
- **AMD MI355X** (`launch_mi355x-amds.sh`, MoRI / `CX_BENCH=mori`) diverges: partition
  `compute`, no account, pyxis `--container-writable --container-remap-root`, and a
  **node-local** squash (`/var/lib/squash`) imported via `srun` on the allocated node
  (not the login node). Workspace is bind-mounted directly (no `CX_STAGE_DIR`).
- Login nodes have no `nvcc`, so `nccl-tests` is **built in-container** (cached in
  `.nccl-tests/`, `CX_NCCL_HOME=/usr`). Single-node uses `-g N`; the 2-node
  adapter builds `MPI=1` and launches one rank per GPU (`srun --mpi=pmix`).
- The sglang image installs editable under `/workspace`, so the repo is mounted at
  **`/ix`**. GB200 compute nodes don't see the runner workspace → `CX_STAGE_DIR`
  rsyncs the tree to Lustre first.
- Every result embeds an `env_capture` record and a `comparison_key`; topology
  class is part of the key, so B200(IB/NVLink) and GB200(MNNVL) stay labelled
  distinct, never silently overlaid.

## Status & known risks

- **Spike done on real hardware** (both SKUs, 4 NCCL primitives, correctness-passed)
  — on the DeepSeek-V4 images. Now standardizing on the **multi-arch** default;
  validate it on first run and refresh `CONTAINERS.md` (expect CUDA 13 / NCCL 2.28 / torch 2.9).
- **DeepEP** is not bundled in the multi-arch image → `run_in_container.sh` builds
  it via `rebuild-deepep` (CX_BENCH=deepep). Its Python API is version-sensitive;
  `run_deepep.py` marks the dispatch/combine block `ADAPT HERE` — validate against
  the built commit. B200 (x86_64) first; GB200 (aarch64) follows.
- **MoRI / MI355X** (`run_mori.py` + `launch_mi355x-amds.sh`) is **scaffolded, not yet
  run on hardware** (no MI355X access). It mirrors `ROCm/mori`'s dispatch/combine
  example — config + the `get_registered_combine_input_buffer` zero-copy path,
  correctness `expected = input × (#unique destination ranks)`. The API is
  version-sensitive (`ADAPT HERE`), so the first runner job is the validation, like
  GB200 was for DeepEP; the AMD ROCm image isn't digest-pinned yet.
- **Multi-node** (`launch_b200-dgxc-slurm.sh`) assumes `srun --mpi=pmix` + a
  compute-visible checkout (`CX_STAGE_DIR`); else fall back to mpirun-in-container
  or srt-slurm. CX_BENCH=nccl only for now.
- **B200 QOS:** account `benchmark` has only `gpu-2_qos` (the serving-sweep
  partition); idle `gpu-1` needs a QOS grant. GB200 `batch` is open.

Once the multi-arch image is validated end-to-end, freeze the schema from the
artifacts (plan: "Freeze the contract").
