# CollectiveX — container & library versions

One **multi-arch, digest-pinned** container is used for all NVIDIA SKUs, so B200
(x86_64) and GB200 (aarch64) share a single reference and the cross-vendor
comparison is truly same-image. Set in `launchers/common.sh` (`cx_default_image`).

## Default container (all NVIDIA SKUs)

- **Image (pin by digest):** `lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356` — the OCI image index for tag `v0.5.12-cu130`.
- **Multi-arch manifest list:** linux/amd64 (`sha256:015f39a4…`) + linux/arm64 (`sha256:7a76819e…`). One digest; `enroot import` on each host pulls the matching arch. **Use the digest-only ref** (`repo@sha256:`) in `common.sh` — enroot 400s on a combined `tag@sha256:` reference.
- **Importing needs registry creds:** anonymous Docker Hub pulls return 401 in ad-hoc SSH sessions; the CI runners import with their configured credentials (the serving sweeps pull images routinely), and already-staged squashes need no import. The refactored launcher path was validated on the already-staged `v0.5.11-cu130` (same multi-arch cu130 line).
- **DeepEP: NOT bundled** here → `run_in_container.sh` builds it via `rebuild-deepep` at job setup (CX_BENCH=deepep). The NCCL path needs no DeepEP.
- **nccl-tests build:** in-container (login nodes have no `nvcc`), `CX_NCCL_HOME=/usr` (system `nccl.h` in `/usr/include`), `CX_CUDA_HOME=/usr/local/cuda`. cu130 lineage ⇒ CUDA 13; confirm exact NCCL/torch on first run and append below.

## Audited reference (cu130 lineage)

Live audit of the sibling DeepSeek-V4 image `lmsysorg/sglang:deepseek-v4-grace-blackwell` (aarch64) on GB200, 2026-06-23 — the multi-arch `v0.5.12-cu130` should match closely (same cu130 base); reconfirm on first run:

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
