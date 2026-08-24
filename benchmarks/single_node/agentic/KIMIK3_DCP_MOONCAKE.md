# Kimi-K3 DSpark DCP8 + Mooncake reproduction

## Pinned stack

- Image: `yukiozzz/vllm-openai-rocm:k3-dspark-dcp-cache-eb61e4c76-mooncake4c6d23c8`
- Runtime reference: `eb61e4c76e086bf8ce5d9c474d1292e9c6902178`
- Mooncake client/master: `4c6d23c8f77230dd5974cf9bc87344dcc946ee77`
- Shape: TP8/DCP8, DSpark block rejection, FP8 KV, eager, rank-local RDMA

The image verifier is `/opt/k3-runtime/verify-runtime.sh`; the file manifest is
`/opt/k3-runtime/manifest.txt`.

## Host setup

Mooncake uses 2 MiB hugepages because registering TB-scale 4 KiB memory exhausts
the ionic RNIC page table. Do not convert all available DRAM to hugepages:
fastsafetensors, ROCm, and warmup still require ordinary host memory.

```bash
export MOONCAKE_HOST_HEADROOM_GB=768
bash benchmarks/single_node/agentic/setup_k3_mooncake_host.sh 1500
```

The setup script includes registration slack, preserves existing hugepage use,
and refuses any plan that leaves less than 768 GB ordinary host memory. The
launcher also:

1. mounts `/dev/infiniband` and `/etc/libibverbs.d`;
2. dereferences the host ionic plugin symlinks into the workspace;
3. propagates unlimited `RLIMIT_MEMLOCK`.

The recipe sets:

```bash
MOONCAKE_DISABLE_HIP_DMABUF=1
MC_STORE_USE_HUGEPAGE=1
MC_STORE_HUGEPAGE_SIZE=2MB
MC_STORE_MEMCPY=1
MC_ENABLE_PARALLEL_REG_MR=0
MC_MAX_MR_SIZE=34359738368
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

DMABUF support remains baked and version-checked, but is disabled on this ionic
fleet; the validated transport is rank-local RDMA over registered hugepage host
memory.

## CI-style local run

Use a disposable allocation: the harness removes all containers on the node.

```bash
export SLURM_REUSE_JOBID=<job-id>
export GITHUB_WORKSPACE=<InferenceX checkout>
export MODEL=moonshotai/Kimi-K3
export TP=8 DCP_SIZE=8 EP_SIZE=1 CONC=16 DURATION=3600
export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=mooncake
export TOTAL_CPU_DRAM_GB=1500
export RESULT_DIR=/it-share/$USER/k3-dcp-mooncake-c16
bash benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh
```

For the complete CI chain, invoke the generated arm through
`runners/launch_mi355x-amds.sh`; it reaches the same recipe and performs host
setup before entering the container.

## Acceptance

- server reaches healthy with real weights;
- no HSA `0x1008`, Mooncake `TRANSFER_FAIL`, or RDMA registration failure;
- deterministic GPU-reset preflight reports external hits;
- AgentX conc16 reports non-zero external hits and GPU hits near the trace's
  theoretical reuse (local dummy reference: 73.3% GPU, 36.6% external);
- AIPerf exits zero.
