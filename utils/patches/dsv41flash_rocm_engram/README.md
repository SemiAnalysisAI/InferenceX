# DeepSeek V4.1 ROCm Engram CPU-offload experiment

`rocm-engram-cpu.patch` is the runtime part of upstream
[vllm-project/vllm#57491](https://github.com/vllm-project/vllm/pull/57491)
("[ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm"), applied to
the pinned image `vllm/vllm-openai-rocm:nightly-af1c01499b289be555c475669ba50a88e96d846e`
(vLLM `af1c01499b289be555c475669ba50a88e96d846e`). The upstream PR's
`tests/kernels/test_engram.py` hunk is not included because the image does not
ship the test tree; `rocm_preflight.py` covers the same ground on the GPU.

The patch adds no AMD-specific code. It widens the two `is_cuda()` gates in
`vllm/config/engram.py` and `vllm/config/vllm.py` to `is_cuda_alike()`, and
`amd/model.py` imports the shared `Engram` from `nvidia/engram.py`, whose
pinned-host tables already read through the ROCm-capable
`get_accelerator_view_from_cpu_tensor` helper and prefetch on a side stream.
Hashing, lookup math, TP head ownership, graph staging and model math are the
upstream code paths. Upstream's `cpu_offload` default is on, so a patched ROCm
server offloads the Engram tables even when `--engram-config` is omitted; the
TP2 recipe passes `--engram-config '{"cpu_offload":true}'` explicitly. DP head
sharding and `dp_shared_memory` arrive with the import but stay inert at
`data_parallel_size=1`, which is all this recipe runs.

`apply.sh` checks the image pin and applies the patch with `git apply --check`.
It then runs `rocm_preflight.py` before model loading. The preflight checks the
widened config gate, the shared `Engram` import in `amd/model.py`, real TP2 shard
outputs against an independent dequantized reference from both HBM and pinned
host memory, masked IDs, noncontiguous hash inputs, graph replay with changed
IDs, the model-level `Engram` constructor with and without offload, and storage
replacement. The two shard owners are exercised on one GPU; real two-GPU
collectives are qualified by the serving sweep and evaluation jobs.

GPU preflight success alone does not qualify this serving recipe. Require the
full AgentX sweep plus the c32 real-rejection evaluation result. AgentX
throughput retains golden AL 3.51; evaluation uses real block rejection. This
experiment is AgentX only. Do not run fixed-sequence or 8k1k benchmarks.

Remove this directory, the `apply.sh` call in the TP2 recipe and
`docs/waiver/3217.md` once a pinned ROCm image ships #57491.

<details>
<summary>中文</summary>

`rocm-engram-cpu.patch` 是上游 [vllm-project/vllm#57491](https://github.com/vllm-project/vllm/pull/57491)
的运行时部分，应用于固定镜像 `vllm/vllm-openai-rocm:nightly-af1c01499b289be555c475669ba50a88e96d846e`
（vLLM `af1c0149`）。上游 PR 中 `tests/kernels/test_engram.py` 的 hunk 不包含在内，
因为镜像不带测试目录；`rocm_preflight.py` 在 GPU 上覆盖同样的内容。

补丁不新增 AMD 专用代码：将 `vllm/config/engram.py` 与 `vllm/config/vllm.py` 中的两个
`is_cuda()` 门控放宽为 `is_cuda_alike()`，并让 `amd/model.py` 从 `nvidia/engram.py`
导入共享的 `Engram`，其 pinned-host 表本就通过支持 ROCm 的设备视图 helper 读取并在
侧流上预取。哈希、查找计算、TP 分片、图暂存和模型计算均为上游代码路径。上游默认开启
`cpu_offload`，因此打补丁后的 ROCm 服务即使不传 `--engram-config` 也会卸载 Engram；
TP2 配方仍显式传入。DP 分片与 `dp_shared_memory` 随导入进入但在
`data_parallel_size=1` 下不生效。

`apply.sh` 校验镜像并以 `git apply --check` 应用补丁，随后在模型加载前运行
`rocm_preflight.py`：验证放宽后的门控、`amd/model.py` 的共享 `Engram` 导入、
HBM/pinned-host 两种存储下的 TP2 分片查找、掩码 ID、非连续输入、变更 ID 后的图回放、
模型级 `Engram` 构造以及存储替换。单卡上分别验证两个 TP 分片，真实双卡通信由服务扫描
和评估验证。仅通过预检不代表配方通过验证；仅运行 AgentX，必须完成全扫描及 c32 真实
拒绝采样评估，不运行固定序列或 8k1k。固定的 ROCm 镜像包含 #57491 后，移除本目录、
TP2 配方中的 `apply.sh` 调用及 `docs/waiver/3217.md`。

</details>
