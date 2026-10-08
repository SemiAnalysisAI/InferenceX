# Native Kimi-K3 PD on MI355X

**English** | [中文](k3-pd-native_zh.md)

Kimi-K3 prefill/decode disaggregation uses the shared Python launcher and native srt-slurm lifecycle. Recipes own serving settings; the master config owns matrix identity.

## Serving configuration

All workers use TP8, MXFP4 target weights, FP8 KV, GMU 0.90 and a 1,048,576-token context. DSpark uses the shipped checkpoint through the pinned framework's default loading path. AgentX throughput uses automatic golden acceptance; evaluations use real block rejection.

Concurrency is global client concurrency. D limits below are per worker.

| Topology | Concurrency | GPUs | P/D DCP | Draft tokens | P CPU offload | P/D max sequences | P/D token budget |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 1P1D | 1 | 16 | 1/1 | 7 | None | 2/2 | 16384/512 |
| 1P1D | 48 | 16 | 8/8 | 4 | 1799 GB | 96/96 | 8192/512 |
| 1P2D | 24 | 24 | 8/8 | 4 | 1799 GB | 48/24 | 8192/512 |
| 1P2D | 48 | 24 | 8/8 | 4 | 1799 GB | 96/48 | 8192/512 |
| 1P3D | 24 | 32 | 8/8 | 4 | 1799 GB | 48/16 | 8192/512 |

P uses PIECEWISE graphs only at c1; other P variants disable graphs. D uses FULL_DECODE_ONLY throughout. The c48 prefill variants retain explicit scratch reclaim. Offload uses SimpleCPUOffloadConnector on P only; each D uses MoRIIO READ. The 1P3D c24 variant inherits the 1P2D c24 policy, adding one D and adjusting the per-D sequence limit.

## Configuration ownership

- `configs/amd-master.yaml` registers `kimik3-fp4-mi355x-vllm-disagg-agentic` and matching topology metadata.
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` owns worker/router images, connectors, graphs and serving arguments. NIC selection belongs to P/D role environment, following the existing AMD disaggregation recipes; `/dev/infiniband` is a recipe-local mount.
- `configs/runners.yaml` declares the target checkpoint location and named draft/provider volumes. The K3 lane selects read-only mounts for the draft checkpoint and node-local Ionic provider.
- `infx/launch/` handles submission, cancellation and artifact collection. The shared backend stages the worker image. The recipe supplies the router image as a native Pyxis/Enroot digest reference, imported by the frontend's job step.

RDMA registration and CPU offload use the cluster's inherited locked-memory limits. When deploying on another cluster, verify the effective worker limits, a routed request and CPU offload reads/writes. Keep site-specific settings scoped to the workload.

## Official image

Master and recipe pin the same immutable worker identity:

`vllm/vllm-openai-rocm:nightly-81198e97ba7eee2a22540caaa756b7fdddcb4d93@sha256:a401e4f46872dcde77e07ae3d7fa2d3a71b7f2773714316116532e2b39deb179`

The worker runs the vLLM installation provided by this image.

## Parameter ownership

Each master point selects one `override_<topology>_c<concurrency>` from the recipe's shared `base`. Engine flags belong in `roles.prefill.args` or `roles.decode.args`; worker environment belongs in the corresponding `env`. YAML anchors share settings, and native override expansion supplies only the differences for a point. JSON-valued engine arguments remain JSON strings because the pinned srt-slurm CLI renderer stringifies ordinary mapping values rather than JSON-encoding them.

The custom benchmark uses the runtime-discovered `SRT_FRONTEND_HOST` and `SRT_FRONTEND_PORT`. Its `benchmark.env` supplies client settings, not engine flags. `KV_OFFLOADING` and `TOTAL_CPU_DRAM_GB` describe the connector configuration and match the master metadata; they do not configure server memory. `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.01` controls live abort only. The shared collector owns the completed-profile threshold.

SSM state remains explicitly `float32`; SiTUv2 activation selection and request-ID randomization follow the pinned image's upstream defaults. State layout, DCP collective selection, nonblocking collectives and c48 scratch reclaim remain explicit compatibility settings.

## Discovery dependency

[srt-slurm #508](https://github.com/NVIDIA/srt-slurm/pull/508) binds allocated discovery endpoints into the explicit MoRIIO/MultiConnector template while preserving the CPU-offload sibling. The [patch inventory](../runners/srt-slurm/patches/README.md) records its upstream revision. The existing `runners/srt-slurm/patches/` mechanism applies the backport to each job's disposable checkout. Retire the patch and its README entry when the shared srt-slurm pin includes this functionality.

The PR generator selects evaluations using the repository's default policy, sample counts and scoring thresholds. The final stack follows the repository's full-sweep and evaluation requirements before merge. Primary sweep labels enable GPU validation.
