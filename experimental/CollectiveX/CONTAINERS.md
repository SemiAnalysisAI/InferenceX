# CollectiveX — container & library versions

One **multi-arch, digest-pinned** container is used for all NVIDIA SKUs, so B200
(x86_64) and GB200 (aarch64) share a single reference and the cross-vendor
comparison is truly same-image. Set in `launchers/common.sh` (`cx_default_image`).

## Default container (all NVIDIA SKUs)

- **Image:** import by tag **`lmsysorg/sglang:v0.5.11-cu130`** (multi-arch OCI index). Expected index digest, recorded for provenance/verification: `sha256:061fb71f838e82000a1768c159654d526c2f17ebe751c21e7fc48ca53c8ef975`.
- **Multi-arch manifest list:** linux/amd64 + linux/arm64; `enroot import` on each host pulls the matching arch.
- **Import by TAG, not digest.** enroot builds its anonymous Docker Hub token scope from the *tag* and succeeds (no creds needed — same as the serving launchers). A bare `repo@sha256:` ref makes enroot prompt for a password and **hang** in non-interactive CI; a combined `tag@sha256:` ref 400s. `cx_ensure_squash` therefore imports by tag with `</dev/null` (a missing token fails fast instead of hanging). First import is multi-GB (~minutes); subsequent runs reuse the staged squash.
- **Why v0.5.11-cu130 (chosen):** it's the newest cu130 release **pre-staged on BOTH clusters** — B200 `/home/sa-shared/containers/` (amd64 squash) and GB200 `/mnt/lustre01/users-public/sa-shared/` (arm64 squash), same filename — so neither side imports at all. (Shared cu130 multi-arch squashes across both clusters: v0.5.8.post1, v0.5.9, v0.5.11 — v0.5.11 is newest.) `v0.5.12-cu130` is staged on B200 but **not** GB200: its 62 layers overflow enroot's overlay-based squash creation on the GB200 kernel (`enroot-mksquashovlfs: failed to mount overlay … Invalid argument`), so it can't be the shared default.
- **DeepEP: NOT bundled** here → `run_in_container.sh` builds it via `rebuild-deepep` at job setup (CX_BENCH=deepep). The NCCL path needs no DeepEP.
- **nccl-tests build:** in-container (login nodes have no `nvcc`), `CX_NCCL_HOME=/usr` (system `nccl.h` in `/usr/include`), `CX_CUDA_HOME=/usr/local/cuda`. cu130 lineage ⇒ CUDA 13; confirm exact NCCL/torch on first run and append below.

## Audited reference (cu130 lineage)

Live audit of the sibling DeepSeek-V4 image `lmsysorg/sglang:deepseek-v4-grace-blackwell` (aarch64) on GB200, 2026-06-23 — the multi-arch `v0.5.11-cu130` should match closely (same cu130 base); reconfirm on first run:

| Component | Version |
|---|---|
| OS / arch | Ubuntu 24.04.3, aarch64 |
| CUDA (`nvcc`) | 13.0 (V13.0.88) |
| NCCL (system `/usr/include/nccl.h`) | 2.28.3; torch-bundled 2.27.7 |
| PyTorch | 2.9.1+cu130 |
| DeepEP | bundled in *that* image; **not** in the multi-arch default |
| NVSHMEM | `libnvshmem_host.so.3` present |
| OpenMPI / gcc / make | 4.1.6 / 13.3.0 / 4.3 |
| GPU / driver | GB200, 580.126.20 |

**Version caveat:** the nccl-tests binary links **system NCCL** (2.28.x), while torch/DeepEP use the **bundled** NCCL (2.27.x). Record both in provenance (env_capture does); don't compare an nccl-tests curve against a DeepEP run as if NCCL were identical.

## Bundled-DeepEP reference images (not the default)

If a bundled DeepEP is needed before `rebuild-deepep` is wired on the multi-arch image, these arch-specific images bundle it (pin by digest):

- B200 (amd64): `lmsysorg/sglang:deepseek-v4-blackwell@sha256:df18bfc4aa9ecf59451002b49ba00cae58042de9e2a96378bbd21b404dd62c7b` (pre-staged on B200)
- GB200 (arm64): `lmsysorg/sglang:deepseek-v4-grace-blackwell@sha256:4f583347d7ff08aef7e16dbb4985b2a7c147ff49a0c261d5e27b8f5f41719368` (staged on GB200 Lustre)

Select via `CX_IMAGE=…@sha256:…` on the launch script.

## Cluster access / QOS

- **B200** (`slurm-login-slinky`): account `benchmark`, **only `gpu-2_qos`** → partition `gpu-2` only (shared with the serving sweep). `gpu-1`/`all` (idle) need `gpu-1_qos`/`all_qos`, not associated with this account.
- **GB200** (`watchtower`): account `benchmark`, qos `normal`, partition `batch` (`AllowQos=ALL`); idle capacity available. Runner workspace is **not** compute-visible → set `CX_STAGE_DIR` to a Lustre path (the launcher rsyncs there).

## First real results (Milestone-0 spike, on the DeepSeek-V4 images)

nccl-tests (system NCCL 2.28.3), all correctness-passed, peak bus-bw:

| op | B200 8× (NVLink island, x86_64) | GB200 4× (NVL72 MNNVL, aarch64) |
|---|---|---|
| all_reduce | 835 GB/s | 689 GB/s |
| all_gather | 653 | 658 |
| reduce_scatter | 667 | 661 |
| alltoall | 638 | 666 |

(B200 vs GB200 carry distinct `comparison_key`s by topology-class, so they are labelled-distinct, not silently merged. Re-run on the multi-arch default to refresh under one image.)
