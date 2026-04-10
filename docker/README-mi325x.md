# MI325X Container Image Build

## Image: `semianalysiswork/sgl-bnxt-cdna3:latest-bnxt-mori`

SGLang (latest main) container for AMD Instinct MI325X/MI300X (gfx942 CDNA3) with:
- Broadcom Thor 2 RDMA support (bnxt_rocelib for RoCEv2 IBGDA)
- MoRI disaggregated inference (KV cache transfer)
- Qwen3.5 MoE (`qwen3_5_moe`), GLM-5 (`glm_moe_dsa`), DeepSeek-R1 model support
- AITER optimized kernels, TileLang NSA backends

## Prerequisites

1. **Broadcom BCM driver**: Download `bcm5760x_231.2.63.0a.zip` from [Broadcom support portal](https://www.broadcom.com/support) and place in `docker/` directory.

2. **Docker**: Must build on a node with Docker and GPU access. Use the sbatch script on the MI325X cluster.

3. **Docker Hub access**: Push credentials for `semianalysiswork` org. PAT is in `/nfsdata/sa/.j9s/InferenceX/.env.local` as `DOCKER_HUB_PAT`, login user `clustermax`.

## Build

```bash
# Option 1: Direct build (on a node with Docker)
cd docker/
bash build-sglang-bnxt-mi325x.sh

# Option 2: Submit as Slurm job
cd docker/
sbatch build-sglang-bnxt-mi325x.sbatch
```

## Build process

The script:
1. Clones [JordanNanos/sglang](https://github.com/JordanNanos/sglang) which contains the ROCm Dockerfile with bnxt patches
2. Copies the BCM driver into the build context
3. Builds with `SGL_BRANCH=main` (latest, supports all model types), `GPU_ARCH=gfx942`, `ENABLE_MORI=1`, `NIC_BACKEND=ibgda`
4. Pushes to `docker.io/semianalysiswork/sgl-bnxt-cdna3:latest-bnxt-mori`

Override defaults: `SGL_BRANCH=v0.5.10 IMAGE_TAG=semianalysiswork/sgl-bnxt-cdna3:v0.5.10-bnxt-mori bash build-sglang-bnxt-mi325x.sh`

### What the Dockerfile builds

- **Base**: `rocm/sgl-dev:rocm7-vllm-20250904` (ROCm 7.0 for gfx942)
- **AITER**: v0.1.10.post3 (AMD optimized kernels)
- **TileLang**: ML compiler for NSA backends (GLM-5)
- **Mooncake**: Distributed training framework
- **SGLang**: v0.5.10 (inference runtime)
- **MoRI**: AMD MoRI networking with bnxt_rocelib for Broadcom Thor 2 IBGDA
- **Broadcom bnxt_rocelib**: Compiled from BCM driver package

### Build args reference

| Arg | Default | Description |
|-----|---------|-------------|
| `SGL_BRANCH` | `v0.5.9` | SGLang git ref to build |
| `GPU_ARCH` | `gfx950` | GPU arch: `gfx942` (MI300X/MI325X) or `gfx950` (MI355X) |
| `ENABLE_MORI` | `0` | Set to `1` to build MoRI networking |
| `NIC_BACKEND` | `none` | `ainic` (Pensando), `ibgda` (Broadcom), or `none` |
| `BCM_DRIVER` | `bcm5760x_231.2.63.0a.zip` | BCM driver filename in build context |

## Usage in InferenceX configs

```yaml
# .github/configs/amd-master.yaml
dsr1-fp8-mi325x-sglang-disagg:
  image: semianalysiswork/sgl-bnxt-cdna3:v0.5.10-bnxt-mori
  ...
```

## Compatibility

- **MI325X** (gfx942, CDNA3, Broadcom Thor 2 NICs) — primary target
- **MI300X** (gfx942, CDNA3) — same architecture, works if NICs are compatible
- **MI355X** (gfx950, CDNA4) — NOT compatible, use upstream `rocm/sgl-dev` images

## Known issues

- EP8/DP with `--moe-a2a-backend mori` hangs on bnxt_re — use default a2a kernels (see sgl-project/sglang#22072)
- RDMA SQ overflow at high concurrency with EP8 — cap `MORI_IO_QP_MAX_SEND_WR=4096`
- Non-MLA models (Qwen3.5, GLM-5) need matched TP sizes between prefill and decode (see sgl-project/sglang#15674)
