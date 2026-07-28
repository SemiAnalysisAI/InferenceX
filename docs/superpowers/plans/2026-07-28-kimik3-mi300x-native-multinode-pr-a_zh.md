# Kimi K3 MI300X 原生多节点 PR A 实施计划

<div align="center">

[English](./2026-07-28-kimik3-mi300x-native-multinode-pr-a.md) | **中文**

</div>

> **面向 agentic worker：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。所有步骤都使用 checkbox（`- [ ]`）跟踪状态。

**目标：** 为聚合式 plain-vLLM Kimi K3 AgentX 新增一个显式启用、失败即停的 MI300X 双节点启动路径，固定 TP8 × PP2、EP1，以及并发 1/2/4/8。

**架构：** 在现有 MI300X 单节点启动器前只增加一个早期分流条件，其余旧路径保持不变。新启动器负责双节点 Slurm 分配、在每个节点上调用独立的宿主机预检、每节点启动一个 vLLM rank、在 rank 0 运行现有 AgentX client，并在成功、失败或收到信号时清理 server step 与 allocation。模型和镜像都保留在节点本地；只有受限且由宿主用户持有的结果与日志产物进入 GitHub workspace。

**技术栈：** Bash、Slurm（`salloc`、`srun`、`squeue`、`scancel`）、Pyxis/Enroot、ROCm/gfx942、vLLM、AgentX/AIPerf、Python 3.12、pytest、Pydantic、PyYAML。

---

## 执行边界

本计划只实现并进行 CPU 测试的 PR A。

- 不 push 分支，也不开 pull request。
- 不触发 GitHub Actions sweep 或 K3 端到端 canary。
- 不下载或预置 `moonshotai/Kimi-K3` 权重。
- 不加入 DSpark 或 `Inferact/Kimi-K3-DSpark` draft。
- 不加入 MI325X、Kubernetes、P/D 分离、srt-slurm 支持。
- 不写死 `AITER_SITUV2_A8W4`。调用方设置时只接受 `0` 或 `1` 并原样传递；未设置时保留镜像默认行为。
- 每个节点都使用 `/raid/hf-hub-cache/inferencex/squash` 独立导入镜像。不得使用 `/home`、`/nvme_home` 或 `/raid/squash`。

后续集群阶段必须等本地分支完成 review 后才开始。它的第一个动作是在选定节点上预置 target checkpoint；该阶段不属于本次实施会话。

## 文件职责

| 路径 | 职责 |
|---|---|
| `runners/launch_mi300x-amds.sh` | 保留现有单节点路径，仅在 `NATIVE_MULTINODE=1` 时分流。 |
| `runners/launch_mi300x-amds-native-multinode.sh` | 校验窄范围合同；分配、启动、监控、收集和清理双节点任务。 |
| `runners/mi300x_native_node_preflight.sh` | 在一个已分配节点上校验 8 张 gfx942 GPU、完整的固定版本模型快照，并原子导入/校验节点本地 squash。 |
| `benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh` | 把与调度器无关的 rank 合同转换为 AMD K3 TP8 × PP2 vLLM 命令。 |
| `benchmarks/multi_node/agentic_srt.sh` | 明确现有 client 可用于所有由外部管理的多节点 server，而不只用于 srt-slurm。 |
| `configs/amd-master.yaml` | 精确定义并发 1、2、4、8 四个聚合式 AgentX job。 |
| `utils/matrix_logic/test_kimik3_mi300x_native.py` | 使用外部命令边界的 CPU fake，测试真实配置、shell 语法、server 命令、节点预检、分流、Slurm 生命周期、产物交接与清理。 |
| `docs/kimik3-mi300x-native-multinode.md` 与 `_zh.md` | 记录运维输入、预置合同、本地验证和后续真实集群门槛。 |
| `docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design.md` 与 `_zh.md` | 用已观测的 live-cluster 事实替换过期的 Slurm 和存储假设。 |
| `perf-changelog.yaml` | 在不改变历史字节的前提下追加新配置 key。 |

不修改 workflow 文件，也不修改 matrix generator 的生产代码。现有多节点 AgentX generator 已经做到每个 job 只生成一个并发点，并且已经携带 `pp`。

### 任务 1：冻结 live contract 与矩阵

**文件：**
- 新建：`utils/matrix_logic/test_kimik3_mi300x_native.py`
- 修改：`configs/amd-master.yaml`
- 修改：`docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design.md`
- 修改：`docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design_zh.md`

- [ ] **步骤 1：先写读取真实配置的失败矩阵测试**

新测试模块定义 repo root，并读取真实 AMD master config，不复制一份合成 YAML fixture：

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "utils" / "matrix_logic"))

from generate_sweep_configs import generate_test_config_sweep  # noqa: E402
from validation import load_config_files, load_runner_file  # noqa: E402

CONFIG_KEY = "kimik3-fp4-mi300x-vllm-agentic"


def generate_kimik3_matrix() -> list[dict]:
    configs = load_config_files([str(REPO_ROOT / "configs" / "amd-master.yaml")])
    runners = load_runner_file(str(REPO_ROOT / "configs" / "runners.yaml"))
    args = argparse.Namespace(
        config_keys=[CONFIG_KEY],
        seq_lens=None,
        conc=None,
        scenario_type=["agentic-coding"],
        runner_node_filter=None,
    )
    return generate_test_config_sweep(args, configs, runners)


def test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs() -> None:
    rows = generate_kimik3_matrix()

    assert [row["conc"] for row in rows] == [[1], [2], [4], [8]]
    assert {row["runner"] for row in rows} == {"cluster:mi300x-amds"}
    assert {row["framework"] for row in rows} == {"vllm"}
    assert {row["disagg"] for row in rows} == {False}
    assert {
        (
            row["prefill"]["num-worker"],
            row["prefill"]["tp"],
            row["prefill"]["pp"],
            row["prefill"]["ep"],
            row["prefill"]["dp-attn"],
            row["decode"]["num-worker"],
            row["decode"]["tp"],
            row["decode"]["pp"],
            row["decode"]["ep"],
            row["decode"]["dp-attn"],
        )
        for row in rows
    } == {(1, 8, 2, 1, False, 0, 8, 2, 1, False)}
    settings = rows[0]["prefill"]["additional-settings"]
    assert settings == ["NATIVE_MULTINODE=1"]
    assert all("AITER_SITUV2_A8W4" not in setting for setting in settings)
```

- [ ] **步骤 2：运行测试并确认因配置缺失而失败**

运行：

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py::test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs \
  -q
```

预期：`FAIL`，因为 `configs/amd-master.yaml` 尚无
`kimik3-fp4-mi300x-vllm-agentic`。

- [ ] **步骤 3：新增窄范围 master config**

把以下 entry 追加到 `configs/amd-master.yaml` 的 AgentX 区域：

```yaml
kimik3-fp4-mi300x-vllm-agentic:
  image: vllm/vllm-openai-rocm:kimi-k3
  model: moonshotai/Kimi-K3
  model-prefix: kimik3
  runner: cluster:mi300x-amds
  precision: fp4
  framework: vllm
  multinode: true
  disagg: false
  scenarios:
    agentic-coding:
    - search-space:
      - spec-decoding: none
        kv-offloading: none
        conc-list: [1, 2, 4, 8]
        prefill:
          num-worker: 1
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
          additional-settings:
          - "NATIVE_MULTINODE=1"
        decode:
          num-worker: 0
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
```

本任务不得把 `AITER_SITUV2_A8W4` 写入 matrix。

- [ ] **步骤 4：同步修正两份已提交 design 文档**

英文与中文 specification 必须做等价修改：

- Slurm credential creation、双节点 allocation 和跨节点 `srun` 已验证工作；
- `/home` 实际解析到接近满载的 `/nvme_home` NFS，禁止用于 30.8 GB 镜像；
- `gharunner` 无法创建 `/raid/squash`；
- 每个已分配节点都在 `/raid/hf-hub-cache/inferencex/squash` 下独立导入并校验 squash；
- target 权重尚未存在，预置属于后续显式 gate；
- 在父任务完成 gfx942 exact-shape 测试并选择模式前，`AITER_SITUV2_A8W4` 保持 unset 或 `0|1` 的运行时输入。

- [ ] **步骤 5：运行聚焦测试与现有 matrix gates**

运行：

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py::test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs \
  -q
python3 -m pytest utils/matrix_logic/ -q
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/amd-master.yaml \
  --config-keys kimik3-fp4-mi300x-vllm-agentic \
  --scenario-type agentic-coding \
  --no-evals
```

预期：聚焦测试通过；现有 matrix suite 没有新增失败；生成 JSON 包含四行，`conc` 依次为 `[1]`、`[2]`、`[4]` 和 `[8]`。

- [ ] **步骤 6：提交绿色切片**

```bash
git add \
  configs/amd-master.yaml \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design.md \
  docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design_zh.md
git commit -s \
  -m "feat(config): define Kimi K3 MI300X multi-node AgentX" \
  -m "Add the exact TP8 x PP2 aggregate matrix and update the design with verified Slurm and node-local storage facts.

中文：新增精确的 TP8 x PP2 聚合式矩阵，并用已验证的 Slurm 与节点本地存储事实更新设计。"
```

### 任务 2：新增 AMD rank 启动入口

**文件：**
- 新建：`benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh`
- 修改：`utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **步骤 1：新增失败的 rank 命令与校验测试**

增加一个 subprocess helper，用 `KIMIK3_VLLM_DRY_RUN=1` 运行真实 shell
entrypoint。传入如下完整基线环境：

```python
def server_env(rank: int = 0) -> dict[str, str]:
    return {
        **os.environ,
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PATH": "/models/Kimi-K3",
        "PORT": "8888",
        "CONC_LIST": "4",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "PREFILL_PP_SIZE": "2",
        "PREFILL_EP": "1",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "0",
        "MULTINODE_NODE_COUNT": "2",
        "MULTINODE_GPUS_PER_NODE": "8",
        "MULTINODE_NODE_RANK": str(rank),
        "MULTINODE_MASTER_ADDR": "node-a",
        "KIMIK3_VLLM_DRY_RUN": "1",
    }
```

增加以下行为断言：

```python
def test_rank_zero_serves_tp8_pp2_without_headless() -> None:
    result = run_server(server_env(0))
    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--pipeline-parallel-size 2" in result.stdout
    assert "--nnodes 2" in result.stdout
    assert "--node-rank 0" in result.stdout
    assert "--master-addr node-a" in result.stdout
    assert "--headless" not in result.stdout
    assert "FLASHMLA" not in result.stdout
    assert "FLASHINFER" not in result.stdout


def test_rank_one_is_headless() -> None:
    result = run_server(server_env(1))
    assert result.returncode == 0, result.stderr
    assert "--node-rank 1" in result.stdout
    assert "--headless" in result.stdout


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PREFILL_TP", "16", "TP8 x PP2"),
        ("PREFILL_PP_SIZE", "1", "TP8 x PP2"),
        ("PREFILL_EP", "8", "EP1"),
        ("DECODE_NUM_WORKERS", "1", "aggregated"),
        ("CONC_LIST", "4 8", "one concurrency"),
        ("CONC_LIST", "16", "1, 2, 4, or 8"),
        ("AITER_SITUV2_A8W4", "auto", "0 or 1"),
    ],
)
def test_server_rejects_out_of_contract_values(
    name: str, value: str, message: str
) -> None:
    env = server_env()
    env[name] = value
    result = run_server(env)
    assert result.returncode != 0
    assert message in result.stderr


def test_aiter_mode_is_not_defaulted_and_accepts_both_modes() -> None:
    unset_result = run_server(server_env())
    assert "AITER_SITUV2_A8W4=unset" in unset_result.stdout
    for value in ("0", "1"):
        env = server_env()
        env["AITER_SITUV2_A8W4"] = value
        result = run_server(env)
        assert result.returncode == 0
        assert f"AITER_SITUV2_A8W4={value}" in result.stdout
```

- [ ] **步骤 2：运行测试并确认 entrypoint 缺失**

运行：

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "rank_zero or rank_one or server_rejects or aiter_mode" \
  -q
```

预期：`FAIL`，因为 AMD 多节点 entrypoint 尚不存在。

- [ ] **步骤 3：实现与调度器无关的 server contract**

脚本必须：

1. 使用 `set -euo pipefail`；
2. source `benchmarks/benchmark_lib.sh`；
3. 要求测试使用的全部 topology 变量；
4. 只接受双节点、每节点 8 GPU、一个聚合 worker、TP8 × PP2、EP1、
   不启用 DP attention、零 decode worker，以及 `1 2 4 8` 中的单个并发值；
5. 校验 `MULTINODE_NODE_RANK` 只能为 `0` 或 `1`；
6. 仅在设置了 `AITER_SITUV2_A8W4` 时校验该值，且不指定默认值；
7. export 以下 AMD K3 环境：

```bash
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export PYTHONNOUSERSITE=1
```

用数组构建命令：

```bash
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 8
    --pipeline-parallel-size 2
    --nnodes 2
    --node-rank "$MULTINODE_NODE_RANK"
    --master-addr "$MULTINODE_MASTER_ADDR"
    --trust-remote-code
    --load-format auto
    --moe-backend auto
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
    --max-model-len 1048576
    --max-num-seqs "$CONC_LIST"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
    --mm-encoder-tp-mode data
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
)
if [[ "$MULTINODE_NODE_RANK" == "1" ]]; then
    VLLM_CMD+=(--headless)
fi
```

输出 AITER 模式和经过 shell escaping 的命令。只有
`KIMIK3_VLLM_DRY_RUN=1` 时才在 `exec` 前退出；这个模式也是有效的集群诊断能力，不是只供测试使用的替代实现。

- [ ] **步骤 4：运行聚焦测试与 shell 语法检查**

运行：

```bash
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "rank_zero or rank_one or server_rejects or aiter_mode" \
  -q
```

预期：全部选中测试通过。

- [ ] **步骤 5：提交绿色切片**

```bash
git add \
  benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh \
  utils/matrix_logic/test_kimik3_mi300x_native.py
git commit -s \
  -m "feat(benchmarks): add MI300X Kimi K3 rank entrypoint" \
  -m "Serve the narrow two-rank TP8 x PP2 AMD topology while leaving the gfx942 AITER a8w4 switch caller-configurable.

中文：新增窄范围的双 rank TP8 x PP2 AMD 启动入口，并保留 gfx942 AITER a8w4 开关由调用方配置。"
```

### 任务 3：新增逐节点预置与镜像预检

**文件：**
- 新建：`runners/mi300x_native_node_preflight.sh`
- 修改：`utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **步骤 1：新增真实临时 cache fixture 与失败测试**

创建如下临时 Hugging Face cache：

```text
models--moonshotai--Kimi-K3/
├── refs/main
└── snapshots/0123456789abcdef0123456789abcdef01234567/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-00001-of-00001.safetensors
```

index 内容必须为：

```json
{
  "metadata": {},
  "weight_map": {
    "model.layers.0.weight": "model-00001-of-00001.safetensors"
  }
}
```

在临时 `PATH` 中放入行为确定的 `rocminfo`、`unsquashfs` 和 `enroot`
可执行文件。fake `rocminfo` 输出 8 个 gfx942 agent；fake `unsquashfs`
只对非空 squash 返回成功；fake `enroot import` 创建指定输出并记录调用。

新增以下精确案例：

| 测试 | 输入 | 必须结果 |
|---|---|---|
| `test_preflight_imports_and_validates_image_in_node_local_tree` | 完整 snapshot、8 个 gfx942 agent、squash 不存在 | 退出 0；一条带 revision、`gpu_count=8`、`gpu_arch=gfx942` 的前缀记录；squash 存在；import log 包含 `docker://vllm/vllm-openai-rocm:kimi-k3` |
| `test_preflight_reuses_a_valid_squash_without_import` | 完整 snapshot 与已有有效 squash | 退出 0；import log 不包含 `enroot import` |
| `test_preflight_rejects_seven_gpus` | 7 个 gfx942 agent | 非零；stderr 包含 `exactly 8 gfx942` |
| `test_preflight_rejects_wrong_architecture` | 8 个 gfx950 agent | 非零；stderr 包含 `exactly 8 gfx942` |
| `test_preflight_rejects_missing_main_ref` | 无 `refs/main` | 非零；stderr 指出 `refs/main` |
| `test_preflight_rejects_missing_weight_index` | 无 index JSON | 非零；stderr 指出 `model.safetensors.index.json` |
| `test_preflight_rejects_missing_indexed_shard` | index 指向不存在的 shard | 非零；stderr 包含 `missing weight shard` |

同时断言脚本和 fake command log 都不包含 `hf download`、
`huggingface-cli download`、`wget` 或用于模型预置的 `curl` 命令。

- [ ] **步骤 2：运行预检测试并确认脚本缺失**

运行：

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k preflight \
  -q
```

预期：`FAIL`，因为 `runners/mi300x_native_node_preflight.sh` 尚不存在。

- [ ] **步骤 3：实现节点本地校验与原子导入**

脚本默认值必须是：

```bash
KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_IMAGE="${KIMIK3_IMAGE:-${IMAGE:?IMAGE must be set}}"
```

必须严格按以下顺序执行：

1. 检查 `rocminfo`，要求恰好 8 个 `gfx942` GPU agent；
2. 读取并校验 `refs/main` 中 40 位十六进制 revision；
3. 要求 snapshot 目录和 `config.json` 存在；
4. 用 Python 解析 `model.safetensors.index.json`，要求 `weight_map` 中每个不同文件都存在且非空；
5. 只在 `KIMIK3_SQUASH_DIR` 下创建路径；
6. 把 `ENROOT_CACHE_PATH` 和 `ENROOT_TEMP_PATH` 设置在同一节点本地目录树下，绝不使用 `$HOME`；
7. 使用 `${KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS:-3600}` 获取节点本地 flock；
8. 用 `unsquashfs -s` 校验最终镜像；
9. 镜像无效或不存在时，导入到同目录临时文件，校验后再用 `mv` 原子替换；
10. 输出一行机器可解析记录：

```text
INFERENCEX_KIMIK3_PREFLIGHT hostname=<host> revision=<sha> gpu_count=8 gpu_arch=gfx942 squash_size_bytes=<n>
```

安装 `EXIT`、`INT`、`TERM` trap，只删除本次临时导入文件，绝不删除此前已验证的最终 squash。

- [ ] **步骤 4：运行聚焦测试与语法检查**

运行：

```bash
bash -n runners/mi300x_native_node_preflight.sh
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k preflight \
  -q
```

预期：所有预检测试通过，包括复用已有导入以及全部失败即停场景。

- [ ] **步骤 5：提交绿色切片**

```bash
git add \
  runners/mi300x_native_node_preflight.sh \
  utils/matrix_logic/test_kimik3_mi300x_native.py
git commit -s \
  -m "feat(runners): validate Kimi K3 state on every MI300X node" \
  -m "Verify gfx942 topology and complete pinned weights, then atomically import and validate the K3 image in the writable node-local squash tree.

中文：逐节点验证 gfx942 拓扑与完整的固定版本权重，并在可写的节点本地 squash 目录中原子导入和校验 K3 镜像。"
```

### 任务 4：新增 allocation、生命周期、清理与产物交接

**文件：**
- 修改：`runners/launch_mi300x-amds.sh`
- 新建：`runners/launch_mi300x-amds-native-multinode.sh`
- 修改：`benchmarks/multi_node/agentic_srt.sh`
- 修改：`utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **步骤 1：新增 CPU-only fake-Slurm 生命周期测试**

fake binaries 只替代外部命令边界；断言必须针对 launcher 结果，而不是 fake
本身。fake `salloc` 返回 `4242`；fake rank discovery 返回 `node-a`；fake
preflight 返回 `node-a` 与 `node-b` 两条记录；fake server `srun` 保持存活直到被终止；fake client `srun` 写出受限 handoff archive。

新增以下精确案例：

| 测试 | 必须观测 |
|---|---|
| `test_default_launcher_keeps_existing_single_node_path` | 退出 0；command log 不含 `--nodes=2` 或 native preflight |
| `test_native_launcher_uses_two_full_nodes_and_all_node_preflight` | 退出 0；allocation 包含 `--nodes=2` 与 `--gres=gpu:8`；preflight 包含 `--ntasks=2`；server 包含 `--kill-on-bad-exit=1`；client 包含 `--overlap` 与 `--nodelist=node-a`；cleanup 记录 `scancel 4242` |
| `test_native_launcher_rejects_topology_before_salloc` | 设置 `PREFILL_PP_SIZE=1`；非零并包含 `TP8 x PP2`；command log 无 `salloc` |
| `test_native_launcher_rejects_one_preflight_record` | 只有一条前缀记录；在 server launch 前非零退出 |
| `test_native_launcher_rejects_mismatched_revisions` | 两条记录 revision 不同；在 server launch 前非零退出 |
| `test_server_failure_preserves_failure_and_cancels_allocation` | fake server 在 health 前以 23 退出；launcher 非零、报告提前退出并记录 `scancel 4242` |
| `test_sigterm_returns_143_and_reaps_server_and_allocation` | server 启动后 terminate；十秒内以 143 退出；日志包含 server termination 和 `scancel 4242` |
| `test_success_extracts_only_host_owned_bounded_artifacts` | aggregate、raw artifact 与 server-log archive 存在且 owner 为 `os.getuid()`；handoff 文件已删除 |

- [ ] **步骤 2：运行生命周期测试并确认失败**

运行：

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "launcher or server_failure or sigterm or bounded_artifacts" \
  -q
```

预期：`FAIL`，因为原生多节点分流与 launcher 尚不存在。

- [ ] **步骤 3：新增隔离式分流**

在 `runners/launch_mi300x-amds.sh` 当前 `set -eo pipefail` 后立即加入：

```bash
if [[ "${NATIVE_MULTINODE:-0}" == "1" ]]; then
    exec bash runners/launch_mi300x-amds-native-multinode.sh
fi
```

不得修改该 guard 以下的任何现有行。

- [ ] **步骤 4：实现严格的 allocation 前校验**

新 launcher 使用 `set -euo pipefail`，且除非以下条件全部满足，否则必须在 `salloc` 前拒绝：

```text
IS_MULTINODE=true
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
FRAMEWORK=vllm
MODEL_PREFIX=kimik3
PRECISION=fp4
SPEC_DECODING=none
DISAGG=false
PREFILL_NUM_WORKERS=1
PREFILL_TP=8
PREFILL_PP_SIZE=2
PREFILL_EP=1
PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=0
DECODE_TP=8
DECODE_PP_SIZE=2
DECODE_EP=1
DECODE_DP_ATTN=false
CONC_LIST 必须是 1、2、4、8 之一且只能有一个值
```

如果设置了 `AITER_SITUV2_A8W4`，校验 `0|1`，但不对它赋值。

- [ ] **步骤 5：分配并验证两个节点**

使用以下 allocation：

```bash
salloc \
  --parsable \
  --partition=compute \
  --exclude=chi-mi300x-049,chi-mi300x-121 \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gres=gpu:8 \
  --cpus-per-task=256 \
  --exclusive \
  --time="${KIMIK3_SLURM_TIME_MINUTES:-480}" \
  --no-shell \
  --job-name="$RUNNER_NAME"
```

只从 parsable stdout 中解析数字 job ID，得到 job ID 后立即安装 cleanup traps。

必须从 task rank 解析 head node，而不是依赖 `scontrol` 顺序：

```bash
head_node=$(
  srun --jobid="$job_id" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash -c 'if [[ "$SLURM_PROCID" == "0" ]]; then hostname; fi'
)
```

随后通过一个双 task `srun` 执行
`runners/mi300x_native_node_preflight.sh`。只解析
`INFERENCEX_KIMIK3_PREFLIGHT` 记录，并要求：

- 恰好两条记录；
- 恰好两个不同 hostname；
- 只有一个共同 revision；
- 两个节点均为 `gpu_count=8` 和 `gpu_arch=gfx942`；
- 两个节点的 squash size 都是正数。

这些检查通过前，不得启动任何 server process。

- [ ] **步骤 6：启动 rank 0/rank 1 并监控 readiness**

使用以下公共节点本地路径：

```bash
image_path="/raid/hf-hub-cache/inferencex/squash/<sanitized-image>.sqsh"
model_snapshot="/raid/hf-hub-cache/models--moonshotai--Kimi-K3/snapshots/$revision"
model_container_path="/models/Kimi-K3"
server_script="benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh"
client_script="benchmarks/multi_node/agentic_srt.sh"
```

使用如下参数启动恰好两个 server task：

```text
--nodes=2
--ntasks=2
--ntasks-per-node=1
--kill-on-bad-exit=1
--container-image=<每个节点独立存在的相同路径>
--container-remap-root
--no-container-mount-home
--no-container-entrypoint
```

把 repo mount 到 `/workspace`，把固定 revision snapshot 以只读方式 mount
到 `/models/Kimi-K3`，并 mount `/dev/kfd` 与 `/dev/dri`。在 worker-side
shell 中把 `SLURM_PROCID` 转换为 `MULTINODE_NODE_RANK`。

合并 server 输出写入 workspace 外由宿主用户拥有的 `mktemp -d`。最多等待
`${KIMIK3_STARTUP_TIMEOUT_SECONDS:-7200}`，轮询
`http://$head_node:8888/health`；每次轮询都确认后台 server step 仍存活。
若提前退出或超时，输出最后 200 行日志。

- [ ] **步骤 7：在 rank 0 运行 AgentX 并安全交回产物**

在 rank 0 的以下位置创建 AgentX scratch：

```text
/raid/hf-hub-cache/inferencex/squash/.runs/<job-id>-<run-key>
```

用单 task overlapping `srun` 在 `head_node` 上运行 client，并设置：

```text
INFMAX_CONTAINER_WORKSPACE=/workspace
RESULT_DIR=/results/agentic
AGENTIC_OUTPUT_DIR=/results/output
PORT=8888
AIPERF_SERVER_METRICS_URLS=http://<head-node>:8888/metrics
```

由宿主用户预创建一个 workspace handoff 文件。client 返回后，同一个 rank-0
container 写入一个 gzip tar，内容只能是 `output/*.json` 和 `agentic/**`。
宿主 launcher 必须：

1. 保留 client exit code；
2. 拒绝 archive 中的绝对路径或 `..` 组件；
3. 解压到由宿主用户持有的临时目录；
4. 要求精确存在预期的 `${RESULT_FILENAME}_conc${CONC_LIST}.json`；
5. 把 aggregate 复制到 workspace root；
6. 把 raw artifacts 复制到 `LOGS/agentic/conc_${CONC_LIST}`；
7. 删除 handoff 文件；
8. 用宿主用户持有的 server log 创建 `multinode_server_logs.tar.gz`。

- [ ] **步骤 8：实现幂等的成功/失败/信号清理**

安装 `EXIT`、`INT`、`TERM` 和 `HUP` 处理。保留原始退出状态；把 `INT`
映射为 130、`TERM` 映射为 143、`HUP` 映射为 129。清理顺序：

```text
停止并 reap 仍在运行的 client
→ 停止并 reap server srun
→ 打包可用 server logs
→ 有界且 best-effort 地删除各节点 .runs scratch
→ scancel allocation
→ 轮询 squeue，直到 job 消失或 cleanup timeout
→ 删除本地临时文件
```

清理必须幂等。使用 `${KIMIK3_CLEANUP_TIMEOUT_SECONDS:-120}` 与
`${KIMIK3_CLEANUP_POLL_SECONDS:-2}`，方便 CPU 测试使用短间隔。不得使用
`sudo`，不得在 workspace 中写镜像，也不得用后续成功的 cleanup command
掩盖 client/server failure。

- [ ] **步骤 9：只泛化 AgentX client 注释**

把 `benchmarks/multi_node/agentic_srt.sh` 开头的 “srt-slurm multinode
jobs” 改成 “externally managed multi-node jobs”。不改变 replay 或
aggregation 行为。

- [ ] **步骤 10：运行聚焦生命周期和语法测试**

运行：

```bash
bash -n runners/launch_mi300x-amds.sh
bash -n runners/launch_mi300x-amds-native-multinode.sh
bash -n runners/mi300x_native_node_preflight.sh
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
bash -n benchmarks/multi_node/agentic_srt.sh
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -q
```

预期：不需要 Slurm、GPU、容器导入或网络访问，全部测试通过。

- [ ] **步骤 11：提交绿色切片**

```bash
git add \
  runners/launch_mi300x-amds.sh \
  runners/launch_mi300x-amds-native-multinode.sh \
  benchmarks/multi_node/agentic_srt.sh \
  utils/matrix_logic/test_kimik3_mi300x_native.py
git commit -s \
  -m "feat(runners): orchestrate two-node MI300X Kimi K3" \
  -m "Add fail-fast two-node allocation, per-rank lifecycle, bounded artifact handoff, and cleanup for success, failure, and signals without changing the default launcher path.

中文：新增失败即停的 MI300X 双节点分配、分 rank 生命周期、受限产物交接，以及成功、失败和信号路径清理，同时保持默认启动路径不变。"
```

### 任务 5：记录运维合同并关闭本地 gates

**文件：**
- 新建：`docs/kimik3-mi300x-native-multinode.md`
- 新建：`docs/kimik3-mi300x-native-multinode_zh.md`
- 修改：`perf-changelog.yaml`

- [ ] **步骤 1：编写成对 runbook**

两份文档必须包含相同章节：

1. 支持范围：聚合式 AgentX、plain vLLM、2 × 8 MI300X、TP8 × PP2、
   EP1、并发 1/2/4/8；
2. 必要环境变量与 matrix key；
3. 节点本地模型布局与完整 index 校验；
4. 节点本地 squash 路径与逐节点导入行为；
5. `AITER_SITUV2_A8W4` 语义：unset、`0` 或 `1`，当前不选择默认值；
6. CPU 验证命令；
7. 后续集群 gates 的顺序：exact-shape AITER 测试、target 预置、直接
   vLLM request、AgentX 并发 1 canary、最后运行 1/2/4/8 sweep；
8. 缺少 snapshot、revision 不同、squash 无效、rank failure、
   readiness timeout 与残留 allocation 的排障方式；
9. 明确警告 launcher 永远不会下载约 1.5 TB checkpoint。

文档可以展示后续 staging 命令，但本实施会话不得执行它们。

- [ ] **步骤 2：逐字节追加 changelog entry**

不修改任何历史行，追加：

```yaml
- config-keys:
    - kimik3-fp4-mi300x-vllm-agentic
  description:
    - "Add an opt-in native Slurm path for aggregated Kimi K3 AgentX on two 8xMI300X nodes with TP8 x PP2, EP1, and concurrency 1/2/4/8"
    - "Fail closed on the exact aggregate topology, complete and revision-matched node-local model snapshots, eight gfx942 GPUs per node, and independently validated node-local Enroot squash images"
    - "Keep AITER_SITUV2_A8W4 caller-configurable pending exact-shape gfx942 validation, and never download the approximately 1.5 TB target checkpoint inside a benchmark job"
    - "Preserve the existing single-node MI300X launcher unless NATIVE_MULTINODE=1, with bounded host-owned artifact handoff and signal/failure cleanup"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

`XXX` 是 repo 在 PR 建立前要求的固定字面量，只在获得 PR number 后替换；它不是尚未决定的实现内容。

- [ ] **步骤 3：运行完整本地验证**

不安装项目 package，也不下载权重。只使用现有测试所需 CPU dependencies：

```bash
python3 -m pytest utils/matrix_logic/ -q
python3 -m pytest utils/changelog_gate_tests/test_validate_perf_changelog.py -q
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/amd-master.yaml \
  --config-keys kimik3-fp4-mi300x-vllm-agentic \
  --scenario-type agentic-coding \
  --no-evals
bash -n runners/launch_mi300x-amds.sh
bash -n runners/launch_mi300x-amds-native-multinode.sh
bash -n runners/mi300x_native_node_preflight.sh
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
bash -n benchmarks/multi_node/agentic_srt.sh
git diff --check
```

提交 changelog 后还要运行：

```bash
python3 utils/validate_perf_changelog.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --changelog-file perf-changelog.yaml
```

预期：

- 所有 pytest 命令零失败；
- matrix generation 生成四条精确结果；
- 所有 shell script 通过 `bash -n`；
- changelog validator 报告 matrix 合法；
- `git diff --check` 无输出。

- [ ] **步骤 4：按窄范围检查最终 diff**

运行：

```bash
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git diff origin/main...HEAD -- runners/launch_mi300x-amds.sh
git status --short
```

确认：

- 没有 workflow、srt-slurm、DSpark、MI325X、Kubernetes 或 inference engine
  文件变化；
- 旧 launcher 只有早期 delegation guard 的差异；
- 没有 model、image、archive、生成结果或 cache 文件进入 Git；
- 每个实现 commit 都有 Wenyao sign-off。

- [ ] **步骤 5：提交 docs 与 changelog**

```bash
git add \
  docs/kimik3-mi300x-native-multinode.md \
  docs/kimik3-mi300x-native-multinode_zh.md \
  perf-changelog.yaml
git commit -s \
  -m "docs: add MI300X Kimi K3 operator contract" \
  -m "Document the node-local staging, verification, cleanup, and deferred canary gates, and append the benchmark changelog entry.

中文：记录节点本地预置、验证、清理与后续 canary 门槛，并追加基准测试变更日志条目。"
```

- [ ] **步骤 6：重新生成提交后的证据并保持 worktree 干净**

重复步骤 3 的完整验证，然后运行：

```bash
git log --format='%h %s%n%b' origin/main..HEAD
git status --short --branch
```

预期：全部 gates 仍为绿色；每个新增实现 commit 都包含
`Signed-off-by: Wenyao Gao`；`git status --short` 没有文件条目。

## 后续真实集群验收

以下动作明确不由本计划执行：

1. 完成父任务的 exact-shape gfx942 比较，选择 unset、
   `AITER_SITUV2_A8W4=0` 或 `AITER_SITUV2_A8W4=1`；
2. 不使用 `/home`，在选定节点对上预置 target snapshot；
3. 在两个节点上验证 30.8 GB squash；
4. rank 0 healthy 后发送一个直接 request；
5. 运行一个 AgentX 并发 1 canary；
6. 检查结果有效性、双节点 GPU 使用、cleanup 和残留 job；
7. 只有这些都通过后才运行并发 1/2/4/8，并为 DSpark 开 PR B。
