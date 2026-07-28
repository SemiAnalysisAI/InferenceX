# Kimi K3 MI300X Native Multi-Node PR A Implementation Plan

<div align="center">

**English** | [中文](./2026-07-28-kimik3-mi300x-native-multinode-pr-a_zh.md)

</div>

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, fail-closed, two-node MI300X launcher for aggregated plain-vLLM Kimi K3 AgentX with TP8 × PP2, EP1, and concurrency 1/2/4/8.

**Architecture:** Keep the existing MI300X single-node launcher unchanged below one early delegation guard. The delegated launcher owns the two-node Slurm allocation, invokes a focused host-side preflight on every allocated node, starts one vLLM rank per node, runs the existing AgentX client on rank 0, and cleans up the server step and allocation on success, failure, or signal. Model and image state remain node-local; only bounded, host-owned result and log artifacts cross into the GitHub workspace.

**Tech Stack:** Bash, Slurm (`salloc`, `srun`, `squeue`, `scancel`), Pyxis/Enroot, ROCm/gfx942, vLLM, AgentX/AIPerf, Python 3.12, pytest, Pydantic, PyYAML.

---

## Execution boundary

This plan implements and CPU-tests PR A only.

- Do not push any branch or open a pull request.
- Do not dispatch a GitHub Actions sweep or a K3 end-to-end canary.
- Do not download or stage `moonshotai/Kimi-K3` weights.
- Do not add DSpark or the `Inferact/Kimi-K3-DSpark` draft.
- Do not add MI325X, Kubernetes, P/D disaggregation, or srt-slurm support.
- Do not hard-code `AITER_SITUV2_A8W4`. If the caller sets it, accept only
  `0` or `1` and pass it through. If it is unset, preserve the image default.
- Use `/raid/hf-hub-cache/inferencex/squash` for the independently imported
  image on every allocated node. Do not use `/home`, `/nvme_home`, or
  `/raid/squash`.

The later cluster phase begins only after this local branch is reviewed. Its
first action is staging the target checkpoint on the selected nodes; that phase
is not part of this implementation session.

## File map

| Path | Responsibility |
|---|---|
| `runners/launch_mi300x-amds.sh` | Preserve the current single-node path and delegate only when `NATIVE_MULTINODE=1`. |
| `runners/launch_mi300x-amds-native-multinode.sh` | Validate the narrow contract; allocate, launch, monitor, collect, and clean up the two-node job. |
| `runners/mi300x_native_node_preflight.sh` | On one allocated node, verify eight gfx942 GPUs, verify the complete pinned model snapshot, and atomically import/validate the node-local squash. |
| `benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh` | Translate the scheduler-independent rank contract into the AMD K3 TP8 × PP2 vLLM command. |
| `benchmarks/multi_node/agentic_srt.sh` | Clarify that the existing client is valid for any externally managed multi-node server, not only srt-slurm. |
| `configs/amd-master.yaml` | Define exactly four aggregate AgentX jobs at concurrency 1, 2, 4, and 8. |
| `utils/matrix_logic/test_kimik3_mi300x_native.py` | Exercise the real config, shell syntax, server command, node preflight, delegation, Slurm lifecycle, artifact handoff, and cleanup with CPU-only fakes at external command boundaries. |
| `docs/kimik3-mi300x-native-multinode.md` and `_zh.md` | Document operator inputs, staging contract, local verification, and deferred real-cluster gates. |
| `docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design.md` and `_zh.md` | Replace stale Slurm and storage assumptions with the observed live-cluster facts. |
| `perf-changelog.yaml` | Append the new config key without changing any historical byte. |

No workflow file or matrix-generator production code should change. The
existing multi-node AgentX generator already emits one concurrency per job and
already carries `pp`.

### Task 1: Freeze the live contract and matrix

**Files:**
- Create: `utils/matrix_logic/test_kimik3_mi300x_native.py`
- Modify: `configs/amd-master.yaml`
- Modify: `docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design.md`
- Modify: `docs/superpowers/specs/2026-07-28-kimik3-mi300x-native-multinode-design_zh.md`

- [ ] **Step 1: Write the failing real-config matrix test**

Create the test module with a repository-root constant and load the actual AMD
master config instead of duplicating a synthetic YAML fixture:

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

- [ ] **Step 2: Run the test and observe the missing-config failure**

Run:

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py::test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs \
  -q
```

Expected: `FAIL` because `kimik3-fp4-mi300x-vllm-agentic` is not present in
`configs/amd-master.yaml`.

- [ ] **Step 3: Add the narrow master-config entry**

Append this entry to the AgentX section of `configs/amd-master.yaml`:

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

Do not add `AITER_SITUV2_A8W4` to the matrix in this task.

- [ ] **Step 4: Correct both committed design documents**

Make the same semantic edits in the English and Chinese specifications:

- Slurm credential creation, a two-node allocation, and cross-node `srun` are
  now verified working.
- `/home` resolves to nearly-full `/nvme_home` NFS and is forbidden for the
  30.8 GB image.
- `gharunner` cannot create `/raid/squash`.
- every allocated node imports and validates its own squash below
  `/raid/hf-hub-cache/inferencex/squash`;
- target weights are absent, so staging is a later explicit gate;
- `AITER_SITUV2_A8W4` remains an unset-or-`0|1` runtime input until the parent
  gfx942 exact-shape test selects a mode.

- [ ] **Step 5: Run the focused and existing matrix gates**

Run:

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

Expected: the focused test passes; the existing matrix suite has zero new
failures; the generated JSON contains four rows with `conc` equal to `[1]`,
`[2]`, `[4]`, and `[8]`.

- [ ] **Step 6: Commit the green slice**

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

### Task 2: Add the AMD rank entrypoint

**Files:**
- Create: `benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh`
- Modify: `utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **Step 1: Add failing rank-command and validation tests**

Add a subprocess helper that runs the real shell entrypoint with
`KIMIK3_VLLM_DRY_RUN=1`. Supply this complete baseline environment:

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

Add these behavioral assertions:

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

- [ ] **Step 2: Run the tests and observe the missing-entrypoint failure**

Run:

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "rank_zero or rank_one or server_rejects or aiter_mode" \
  -q
```

Expected: `FAIL` because the AMD multi-node entrypoint does not exist.

- [ ] **Step 3: Implement the scheduler-independent server contract**

The script must:

1. use `set -euo pipefail`;
2. source `benchmarks/benchmark_lib.sh`;
3. require every topology variable used by the tests;
4. reject anything except two nodes, eight GPUs per node, one aggregate worker,
   TP8 × PP2, EP1, no DP attention, zero decode workers, and one concurrency in
   `1 2 4 8`;
5. validate `MULTINODE_NODE_RANK` as `0` or `1`;
6. validate `AITER_SITUV2_A8W4` only when set, without assigning a default;
7. export the AMD K3 environment:

```bash
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export PYTHONNOUSERSITE=1
```

Build the command as an array:

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

Print the AITER mode and shell-escaped command. Exit before `exec` only when
`KIMIK3_VLLM_DRY_RUN=1`; this is also a useful cluster diagnostic and is not a
test-only alternate implementation.

- [ ] **Step 4: Run focused tests and shell syntax**

Run:

```bash
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "rank_zero or rank_one or server_rejects or aiter_mode" \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the green slice**

```bash
git add \
  benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh \
  utils/matrix_logic/test_kimik3_mi300x_native.py
git commit -s \
  -m "feat(benchmarks): add MI300X Kimi K3 rank entrypoint" \
  -m "Serve the narrow two-rank TP8 x PP2 AMD topology while leaving the gfx942 AITER a8w4 switch caller-configurable.

中文：新增窄范围的双 rank TP8 x PP2 AMD 启动入口，并保留 gfx942 AITER a8w4 开关由调用方配置。"
```

### Task 3: Add the per-node staging and image preflight

**Files:**
- Create: `runners/mi300x_native_node_preflight.sh`
- Modify: `utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **Step 1: Add a real temporary-cache fixture and failing tests**

Create a temporary Hugging Face cache with:

```text
models--moonshotai--Kimi-K3/
├── refs/main
└── snapshots/0123456789abcdef0123456789abcdef01234567/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-00001-of-00001.safetensors
```

The index must contain:

```json
{
  "metadata": {},
  "weight_map": {
    "model.layers.0.weight": "model-00001-of-00001.safetensors"
  }
}
```

Place deterministic `rocminfo`, `unsquashfs`, and `enroot` executables in a
temporary `PATH`. The fake `rocminfo` prints eight gfx942 agents. The fake
`unsquashfs` succeeds only for a non-empty squash. The fake `enroot import`
creates the requested output and records its invocation.

Add these exact cases:

| Test | Input | Required result |
|---|---|---|
| `test_preflight_imports_and_validates_image_in_node_local_tree` | complete snapshot, eight gfx942 agents, absent squash | exit 0; one prefixed record with the revision, `gpu_count=8`, `gpu_arch=gfx942`; squash exists; import log contains `docker://vllm/vllm-openai-rocm:kimi-k3` |
| `test_preflight_reuses_a_valid_squash_without_import` | complete snapshot and already-valid squash | exit 0; import log contains no `enroot import` |
| `test_preflight_rejects_seven_gpus` | seven gfx942 agents | nonzero; stderr contains `exactly 8 gfx942` |
| `test_preflight_rejects_wrong_architecture` | eight gfx950 agents | nonzero; stderr contains `exactly 8 gfx942` |
| `test_preflight_rejects_missing_main_ref` | no `refs/main` | nonzero; stderr names `refs/main` |
| `test_preflight_rejects_missing_weight_index` | no index JSON | nonzero; stderr names `model.safetensors.index.json` |
| `test_preflight_rejects_missing_indexed_shard` | index names an absent shard | nonzero; stderr contains `missing weight shard` |

Also assert that neither the script nor the fake command log contains an
`hf download`, `huggingface-cli download`, `wget`, or `curl` model-staging
command.

- [ ] **Step 2: Run the preflight tests and observe the missing-script failure**

Run:

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k preflight \
  -q
```

Expected: `FAIL` because `runners/mi300x_native_node_preflight.sh` does not
exist.

- [ ] **Step 3: Implement node-local verification and atomic import**

The script defaults must be:

```bash
KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_IMAGE="${KIMIK3_IMAGE:-${IMAGE:?IMAGE must be set}}"
```

It must perform this exact order:

1. inspect `rocminfo` and require exactly eight `gfx942` GPU agents;
2. read and validate the 40-hex revision in `refs/main`;
3. require the snapshot directory and `config.json`;
4. parse `model.safetensors.index.json` with Python and require every distinct
   file in `weight_map` to exist and be non-empty;
5. create only paths below `KIMIK3_SQUASH_DIR`;
6. set `ENROOT_CACHE_PATH` and `ENROOT_TEMP_PATH` below that same node-local
   tree, never below `$HOME`;
7. acquire a node-local flock with
   `${KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS:-3600}`;
8. validate the final image with `unsquashfs -s`;
9. when invalid or absent, import into a same-directory temporary file, validate
   it, and atomically `mv` it into place;
10. print one machine-readable line:

```text
INFERENCEX_KIMIK3_PREFLIGHT hostname=<host> revision=<sha> gpu_count=8 gpu_arch=gfx942 squash_size_bytes=<n>
```

Install an `EXIT`, `INT`, and `TERM` trap that removes only the temporary import
file. It must never remove a previously validated final squash.

- [ ] **Step 4: Run focused tests and syntax**

Run:

```bash
bash -n runners/mi300x_native_node_preflight.sh
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k preflight \
  -q
```

Expected: all preflight tests pass, including import reuse and every fail-closed
case.

- [ ] **Step 5: Commit the green slice**

```bash
git add \
  runners/mi300x_native_node_preflight.sh \
  utils/matrix_logic/test_kimik3_mi300x_native.py
git commit -s \
  -m "feat(runners): validate Kimi K3 state on every MI300X node" \
  -m "Verify gfx942 topology and complete pinned weights, then atomically import and validate the K3 image in the writable node-local squash tree.

中文：逐节点验证 gfx942 拓扑与完整的固定版本权重，并在可写的节点本地 squash 目录中原子导入和校验 K3 镜像。"
```

### Task 4: Add allocation, lifecycle, cleanup, and artifact handoff

**Files:**
- Modify: `runners/launch_mi300x-amds.sh`
- Create: `runners/launch_mi300x-amds-native-multinode.sh`
- Modify: `benchmarks/multi_node/agentic_srt.sh`
- Modify: `utils/matrix_logic/test_kimik3_mi300x_native.py`

- [ ] **Step 1: Add CPU-only fake-Slurm lifecycle tests**

The fake binaries are external-boundary substitutes only; assertions must
target launcher outcomes rather than the fakes themselves. The fake `salloc`
returns `4242`; fake rank discovery returns `node-a`; fake preflight returns
two records for `node-a` and `node-b`; fake server `srun` remains alive until
terminated; fake client `srun` writes the bounded handoff archive.

Add these exact cases:

| Test | Required observation |
|---|---|
| `test_default_launcher_keeps_existing_single_node_path` | exit 0; command log contains neither `--nodes=2` nor the native preflight |
| `test_native_launcher_uses_two_full_nodes_and_all_node_preflight` | exit 0; allocation has `--nodes=2` and `--gres=gpu:8`; preflight has `--ntasks=2`; server has `--kill-on-bad-exit=1`; client has `--overlap` and `--nodelist=node-a`; cleanup logs `scancel 4242` |
| `test_native_launcher_rejects_topology_before_salloc` | set `PREFILL_PP_SIZE=1`; nonzero with `TP8 x PP2`; no `salloc` in command log |
| `test_native_launcher_rejects_one_preflight_record` | one prefixed record; nonzero before server launch |
| `test_native_launcher_rejects_mismatched_revisions` | two records with different revisions; nonzero before server launch |
| `test_server_failure_preserves_failure_and_cancels_allocation` | fake server exits 23 before health; launcher is nonzero, reports early exit, and logs `scancel 4242` |
| `test_sigterm_returns_143_and_reaps_server_and_allocation` | terminate after server start; process exits 143 within ten seconds; log contains server termination and `scancel 4242` |
| `test_success_extracts_only_host_owned_bounded_artifacts` | aggregate, raw artifact, and server-log archive exist and are owned by `os.getuid()`; handoff file is gone |

- [ ] **Step 2: Run the lifecycle tests and observe failure**

Run:

```bash
python3 -m pytest \
  utils/matrix_logic/test_kimik3_mi300x_native.py \
  -k "launcher or server_failure or sigterm or bounded_artifacts" \
  -q
```

Expected: `FAIL` because native delegation and the native launcher do not exist.

- [ ] **Step 3: Add the isolated delegation**

Immediately after the current `set -eo pipefail` in
`runners/launch_mi300x-amds.sh`, add:

```bash
if [[ "${NATIVE_MULTINODE:-0}" == "1" ]]; then
    exec bash runners/launch_mi300x-amds-native-multinode.sh
fi
```

Do not alter any existing line below this guard.

- [ ] **Step 4: Implement strict pre-allocation validation**

The new launcher uses `set -euo pipefail` and rejects before `salloc` unless:

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
CONC_LIST is exactly one of 1, 2, 4, 8
```

If `AITER_SITUV2_A8W4` is set, validate `0|1` but do not assign it.

- [ ] **Step 5: Allocate and verify both nodes**

Allocate with:

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

Parse only a numeric job ID from the parsable stdout. Install cleanup traps
immediately after obtaining it.

Resolve the head node from task rank, not `scontrol` ordering:

```bash
head_node=$(
  srun --jobid="$job_id" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash -c 'if [[ "$SLURM_PROCID" == "0" ]]; then hostname; fi'
)
```

Then run `runners/mi300x_native_node_preflight.sh` through one two-task `srun`.
Parse only `INFERENCEX_KIMIK3_PREFLIGHT` records and require:

- exactly two records;
- exactly two unique hostnames;
- one shared revision;
- `gpu_count=8` and `gpu_arch=gfx942` on both;
- a positive squash size on both.

No server process may start before these checks pass.

- [ ] **Step 6: Start rank 0/rank 1 and monitor readiness**

Use the common node-local paths:

```bash
image_path="/raid/hf-hub-cache/inferencex/squash/<sanitized-image>.sqsh"
model_snapshot="/raid/hf-hub-cache/models--moonshotai--Kimi-K3/snapshots/$revision"
model_container_path="/models/Kimi-K3"
server_script="benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh"
client_script="benchmarks/multi_node/agentic_srt.sh"
```

Launch exactly two server tasks with:

```text
--nodes=2
--ntasks=2
--ntasks-per-node=1
--kill-on-bad-exit=1
--container-image=<same path, independently present on each node>
--container-remap-root
--no-container-mount-home
--no-container-entrypoint
```

Mount the repository at `/workspace`, the pinned snapshot read-only at
`/models/Kimi-K3`, and `/dev/kfd` plus `/dev/dri`. Translate
`SLURM_PROCID` to `MULTINODE_NODE_RANK` in the worker-side shell.

Save combined server output in a host-owned `mktemp -d` outside the workspace.
Poll `http://$head_node:8888/health` for at most
`${KIMIK3_STARTUP_TIMEOUT_SECONDS:-7200}` and, on every poll, verify that the
background server step is still alive. Surface the last 200 log lines on early
exit or timeout.

- [ ] **Step 7: Run AgentX on rank 0 and hand artifacts back safely**

Create the AgentX scratch on rank 0 below:

```text
/raid/hf-hub-cache/inferencex/squash/.runs/<job-id>-<run-key>
```

Run the client as a one-task overlapping `srun` pinned to `head_node`. Set:

```text
INFMAX_CONTAINER_WORKSPACE=/workspace
RESULT_DIR=/results/agentic
AGENTIC_OUTPUT_DIR=/results/output
PORT=8888
AIPERF_SERVER_METRICS_URLS=http://<head-node>:8888/metrics
```

Pre-create one workspace handoff file as the host user. After the client
returns, the same rank-0 container writes a gzip tar containing only
`output/*.json` and `agentic/**` into that file. The host launcher must:

1. preserve the client exit code;
2. reject absolute paths or `..` components in the archive;
3. extract it into host-owned temporary storage;
4. require exactly the expected
   `${RESULT_FILENAME}_conc${CONC_LIST}.json`;
5. copy that aggregate to the workspace root;
6. copy raw artifacts to `LOGS/agentic/conc_${CONC_LIST}`;
7. remove the handoff file;
8. create `multinode_server_logs.tar.gz` from the host-owned server log.

- [ ] **Step 8: Implement idempotent success/failure/signal cleanup**

Install `EXIT`, `INT`, `TERM`, and `HUP` handling. Preserve the original exit
status; map `INT` to 130, `TERM` to 143, and `HUP` to 129. Cleanup order:

```text
stop and reap client if still running
→ stop and reap server srun
→ package available server logs
→ bounded best-effort removal of per-node .runs scratch
→ scancel allocation
→ poll squeue until the job disappears or cleanup timeout expires
→ remove local temporary files
```

Make cleanup idempotent. Use `${KIMIK3_CLEANUP_TIMEOUT_SECONDS:-120}` and
`${KIMIK3_CLEANUP_POLL_SECONDS:-2}` so CPU tests can set short intervals.
Do not use `sudo`, do not write an image under the workspace, and do not mask a
client/server failure with a later successful cleanup command.

- [ ] **Step 9: Generalize only the AgentX client comment**

Change the opening comment in `benchmarks/multi_node/agentic_srt.sh` from
“srt-slurm multinode jobs” to “externally managed multi-node jobs.” Do not
change its replay or aggregation behavior.

- [ ] **Step 10: Run focused lifecycle and syntax tests**

Run:

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

Expected: all tests pass without Slurm, GPUs, a container import, or network
access.

- [ ] **Step 11: Commit the green slice**

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

### Task 5: Document the operator contract and close local gates

**Files:**
- Create: `docs/kimik3-mi300x-native-multinode.md`
- Create: `docs/kimik3-mi300x-native-multinode_zh.md`
- Modify: `perf-changelog.yaml`

- [ ] **Step 1: Write the paired runbook**

Both documents must contain the same sections:

1. supported scope: aggregate AgentX, plain vLLM, 2 × 8 MI300X,
   TP8 × PP2, EP1, concurrency 1/2/4/8;
2. required env and matrix key;
3. node-local model layout and complete-index verification;
4. node-local squash path and per-node import behavior;
5. `AITER_SITUV2_A8W4` semantics: unset, `0`, or `1`, with no selected default;
6. CPU verification commands;
7. deferred cluster gates in order:
   exact-shape AITER test, target staging, direct vLLM request, AgentX
   concurrency-1 canary, then 1/2/4/8 sweep;
8. troubleshooting for a missing snapshot, mismatched revisions, invalid
   squash, rank failure, readiness timeout, and residual allocation;
9. explicit warning that the launcher never downloads the 1.5 TB checkpoint.

The documents may show later staging commands, but this implementation session
must not execute them.

- [ ] **Step 2: Append the changelog entry byte-for-byte**

Append, without editing any historical line:

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

`XXX` is the repository-required pre-PR literal and is replaced only after the
PR number exists; it is not an unresolved implementation decision.

- [ ] **Step 3: Run the complete local verification**

Install no project package and download no weights. Use only the CPU
dependencies required by the existing tests:

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

After committing the changelog, also run:

```bash
python3 utils/validate_perf_changelog.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --changelog-file perf-changelog.yaml
```

Expected:

- every pytest command has zero failures;
- matrix generation emits four exact rows;
- every shell script passes `bash -n`;
- changelog validation reports a valid generated matrix;
- `git diff --check` prints nothing.

- [ ] **Step 4: Inspect the final diff against the narrow scope**

Run:

```bash
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git diff origin/main...HEAD -- runners/launch_mi300x-amds.sh
git status --short
```

Confirm:

- no workflow, srt-slurm, DSpark, MI325X, Kubernetes, or inference-engine file
  changed;
- the old launcher differs only by the early delegation guard;
- no model, image, archive, generated result, or cache file is tracked;
- every implementation commit has Wenyao's sign-off.

- [ ] **Step 5: Commit docs and changelog**

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

- [ ] **Step 6: Re-run post-commit evidence and leave a clean worktree**

Repeat the complete verification from Step 3, then run:

```bash
git log --format='%h %s%n%b' origin/main..HEAD
git status --short --branch
```

Expected: all gates remain green, every new implementation commit contains
`Signed-off-by: Wenyao Gao`, and `git status --short` has no file entries.

## Deferred real-cluster acceptance

These are explicitly not executed by this plan:

1. finish the parent task's exact-shape gfx942 comparison and select unset,
   `AITER_SITUV2_A8W4=0`, or `AITER_SITUV2_A8W4=1`;
2. stage the target snapshot on the selected pair without using `/home`;
3. validate the 30.8 GB squash on both nodes;
4. run one direct request after rank 0 becomes healthy;
5. run one AgentX concurrency-1 canary;
6. inspect result validity, two-node GPU use, cleanup, and residual jobs;
7. only then run concurrency 1/2/4/8 and open PR B for DSpark.
