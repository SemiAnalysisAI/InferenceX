[English](2026-08-24-h200-fp4-fp8-power-ab.md) | [中文](2026-08-24-h200-fp4-fp8-power-ab_zh.md)

# H200 FP4 与 FP8 功耗 A/B 实施计划

> **Agent 执行要求：** 必须使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，并按 checkbox 逐项执行。

**目标：** 补齐 H200 NVFP4 Marlin 推理路径，并 dispatch 精确配对的 8K/1K c4 FP4/FP8 canary。

**架构：** 保留现有 H200 FP8 路径作为 control；新增匹配的 H200 FP4 master config 和硬件专用 benchmark script，dense 与 MoE 都使用 Marlin W4A16。通过 matrix generation test 锁定配对关系；canary 使用 `test-config` 精确选择两个配置，避免现有 MTP 配置混入。

**技术栈：** YAML master config、Bash SGLang launcher、Python/Pydantic matrix generator、pytest、GitHub Actions `e2e-tests.yml`。

---

### Task 1：增加配对矩阵回归测试

**文件：**
- 修改：`utils/matrix_logic/test_generate_sweep_configs.py:2611`

- [ ] **Step 1：先写失败测试**

在 `TestAgentXPowerExperimentConfigs` 中增加：

```python
def test_qwen_h200_fp4_fp8_fixed_8k1k_matrix_is_balanced(self):
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config_files([str(repo_root / "configs/nvidia-master.yaml")])
    runners = load_runner_file(str(repo_root / "configs/runners.yaml"))
    args = argparse.Namespace(
        config_keys=[
            "qwen3.5-fp8-h200-sglang",
            "qwen3.5-fp4-h200-sglang",
        ],
        seq_lens=["8k1k"],
        conc=[1, 2, 4, 8, 16, 32, 64],
        scenario_type=["fixed-seq-len"],
        runner_node_filter=None,
    )

    result = generate_test_config_sweep(args, config, runners)

    assert len(result) == 14
    assert {(row["precision"], row["conc"]) for row in result} == {
        (precision, conc)
        for precision in ("fp4", "fp8")
        for conc in (1, 2, 4, 8, 16, 32, 64)
    }
    assert all(row["runner"] == "h200" for row in result)
    assert all(row["tp"] == 8 and row["ep"] == 8 for row in result)
    assert all(row["isl"] == 8192 and row["osl"] == 1024 for row in result)
    assert all(row["spec-decoding"] == "none" for row in result)
```

- [ ] **Step 2：运行测试确认 RED**

```bash
uv run --with pytest --with pydantic --with pyyaml python -m pytest \
  utils/matrix_logic/test_generate_sweep_configs.py::TestAgentXPowerExperimentConfigs::test_qwen_h200_fp4_fp8_fixed_8k1k_matrix_is_balanced -v
```

预期：因为缺少 `qwen3.5-fp4-h200-sglang` 而失败。

### Task 2：增加 H200 FP4 config 与 launcher

**文件：**
- 修改：`configs/nvidia-master.yaml:1796`
- 新增：`benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh`

- [ ] **Step 1：扩展 FP8 control 并新增配对 FP4 配置**

```yaml
qwen3.5-fp8-h200-sglang:
  image: lmsysorg/sglang:v0.5.14-cu130
  model: Qwen/Qwen3.5-397B-A17B-FP8
  model-prefix: qwen3.5
  runner: h200
  precision: fp8
  framework: sglang
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 8192
      osl: 1024
      search-space:
      - { tp: 8, ep: 8, conc-start: 1, conc-end: 64 }

qwen3.5-fp4-h200-sglang:
  image: lmsysorg/sglang:v0.5.14-cu130
  model: nvidia/Qwen3.5-397B-A17B-NVFP4-V2
  model-prefix: qwen3.5
  runner: h200
  precision: fp4
  framework: sglang
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 8192
      osl: 1024
      search-space:
      - { tp: 8, ep: 8, conc-start: 1, conc-end: 64 }
```

- [ ] **Step 2：新增 Hopper FP4 launcher**

沿用 `qwen3.5_fp8_h200.sh` 的生命周期和 benchmark client，SGLang command 必须包含：

```bash
  --kv-cache-dtype fp8_e4m3 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --moe-runner-backend marlin \
  --attention-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --disable-radix-cache \
```

保留 `start_gpu_monitor`、`stop_gpu_monitor`、`wait_for_server_ready`、固定输入输出长度和标准结果文件名；不得复制 B200 的 `trtllm_mha` 或 `flashinfer_trtllm`。

- [ ] **Step 3：重跑 Task 1 测试确认 GREEN**

预期：`1 passed`。

- [ ] **Step 4：提交实现**

```bash
git add configs/nvidia-master.yaml \
  benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh \
  utils/matrix_logic/test_generate_sweep_configs.py
git commit -m "feat: add H200 NVFP4 power benchmark" \
  -m "中文：增加 H200 NVFP4 Marlin 功耗基准测试。"
```

### Task 3：验证实现与精确 canary 矩阵

**文件：**
- 验证：`benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh`
- 验证：`configs/nvidia-master.yaml`
- 验证：`utils/matrix_logic/test_generate_sweep_configs.py`

- [ ] **Step 1：检查 Bash 语法并排除 Blackwell backend**

```bash
bash -n benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh
! rg -n 'trtllm_mha|flashinfer_trtllm' \
  benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh
```

- [ ] **Step 2：运行完整 matrix test suite**

```bash
uv run --with pytest --with pydantic --with pyyaml python -m pytest \
  utils/matrix_logic/ -v
```

- [ ] **Step 3：只生成两个 c4 canary**

```bash
uv run --with pydantic --with pyyaml python \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys qwen3.5-fp4-h200-sglang qwen3.5-fp8-h200-sglang \
  --conc 4 --seq-lens 8k1k --scenario-type fixed-seq-len --no-evals
```

预期：只生成 TP8/EP8、STP、c4 两行，FP4 与 FP8 各一行。

- [ ] **Step 4：检查 diff 和仓库状态**

```bash
git diff origin/main...HEAD --check
git status --short --branch
```

### Task 4：push 并 dispatch 配对 c4 canary

**文件：**
- 远程分支：`codex/h200-fp4-power-ab`
- Workflow：`.github/workflows/e2e-tests.yml`

- [ ] **Step 1：push 并确认 remote SHA**

```bash
git push -u origin codex/h200-fp4-power-ab
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/codex/h200-fp4-power-ab | cut -f1)"
```

- [ ] **Step 2：dispatch 只包含两个 job 的 workflow**

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=codex/h200-fp4-power-ab' \
  -f 'inputs[test-name]=Qwen3.5 H200 FP4 vs FP8 8k1k c4 power canary' \
  -f 'inputs[generate-cli-command]=test-config --config-files configs/nvidia-master.yaml --config-keys qwen3.5-fp4-h200-sglang qwen3.5-fp8-h200-sglang --conc 4 --seq-lens 8k1k --scenario-type fixed-seq-len --no-evals' \
  -f 'inputs[duration-override]='
```

- [ ] **Step 3：定位 run ID 并检查 setup 输出**

使用 `gh run list` 按创建时间和显示名定位新 run。预期：一个 workflow run，只包含两个 fixed-sequence benchmark rows。

### Task 5：监控并执行 canary gate

**文件：**
- 只读：GitHub Actions logs 和下载的 benchmark artifacts

- [ ] **Step 1：不修改共享 workload 地监控**

从最新且匹配的 `workflow_dispatch` run 解析 `RUN_ID`，再以有界间隔读取 `gh run view "$RUN_ID" --json status,conclusion,jobs,url`。除非 run 已确定无效且获得相应授权，否则不 cancel 或 restart。

- [ ] **Step 2：下载并验证 artifacts**

```bash
CANARY_DIR=$(mktemp -d /tmp/h200-fp4-canary.XXXXXX)
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  -n results_bmk -D "$CANARY_DIR"
```

从 `agg_bmk.json` 只提取四舍五入后的行；确认 FP4/FP8 都完成、workload metrics 存在、功耗字段有效，并在 FP4 server log 中确认 dense 与 MoE 均选择 Marlin、没有 progress stall。

- [ ] **Step 3：决定是否扩展**

两组都通过后，才对 c1–c64 的 TP8/EP8 扫描重复三次；任一组失败则停止扩展，先定位启动、显存、kernel 或 progress failure。
