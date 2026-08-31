# 结果、摄取与恢复流程

<div align="center">

[English](./recovery-results-procedures.md) | **中文**

</div>

当吞吐量或评测任务开始产出结果，或 sweep、runner、集群、交接或数据库摄取失败时，请使用本页。本页将证据收集与修复分开，避免把基础设施问题误判为基准性能回归，也避免因摄取问题重新触发昂贵的 GPU sweep。

## 安全关卡

1. **在做任何更改前，先确定准确的 run、attempt、job、event、branch、head SHA 和 runner。** `trigger-ingest` 任务绿色并不能证明有效基准结果行已经进入数据库。
2. **先检查，后变更。** 读取 GitHub 日志、`sinfo`、`squeue`、sysctl、文件所有者和制品是安全的。删除共享文件、更改 sysctl、drain 节点、重启服务或修改集群状态，都需要运维者明确批准。
3. **绝不要仅为修复正式摄取而重跑失败的 push-to-`main` 目标。** 应使用经过验证的制品复用恢复路径；它既能避免 GPU 工作，也能保留源 run 的来源信息。
4. **只重跑已经确诊的偶发故障。** 重试不会修复错误的镜像、recipe、模型、launcher、配置、缺失或过期的制品，也不会修复执行语义变化。
5. **把 `source-run-id` 和 `merge-run-id` 当作一对。** source run 拥有基准/评测制品，merge run 拥有发布/changelog 元数据。在信任摄取前，必须在下游日志中核对两者。
6. **不要仅按时间新旧选择 InferenceX-app 摄取 run。** 过时或伪造的摄取可能比恢复摄取更新。
7. **清理前保留证据。** 记录 run/job URL、run attempt、runner 名称、source/merge ID、制品名称与数量、首个因果错误，以及所有节点或 Slurm job ID。

来源：[sweep 调试安全规则](../.agents/skills/debug-runs/SKILL.md#L78-L152)、[失败摄取安全规则](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L22-L54)。

## 结果流水线：先明确应当存在什么

### 吞吐量结果

对于普通单节点吞吐量任务：

1. launcher 必须在工作区留下 `${RESULT_FILENAME}.json`。工作流会短暂等待；如果该文件一直没有出现，任务失败。
2. `utils/process_result.py` 读取原始 JSON 以及拓扑/运行时环境变量，规范化元数据和每 GPU 吞吐量，把毫秒字段转换成秒，推导 interactivity，并写出 `agg_${RESULT_FILENAME}.json`。
3. 任务将聚合结果上传为制品 `bmk_${RESULT_FILENAME}`。
4. `collect-results.yml` 下载 `bmk_*`，运行 `python3 utils/collect_results.py results/ bmk`，再上传 `results_bmk`；其负载为 `agg_bmk.json`。

对于多节点吞吐量，每个 `${RESULT_FILENAME}_*.json` 都会单独处理。工作流从各文件名推导 GPU 总数、prefill GPU 数和 decode GPU 数，并调用：

```bash
RESULT_FILENAME=${result_file%.json} \
IS_MULTINODE=true \
PREFILL_GPUS="$prefill_gpus" \
DECODE_GPUS="$decode_gpus" \
python3 utils/process_result.py
```

上传的 `bmk_${RESULT_FILENAME}` 制品包含 `agg_${RESULT_FILENAME}_*.json`。缺少源文件属于基准/launcher 故障；缺少 `agg_` 文件属于结果处理故障；缺少 `results_bmk` 属于收集故障。不要把这些问题归类为数据库故障。

来源：[单节点处理/上传](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-tmpl.yml#L289-L323)、[多节点处理/上传](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-multinode-tmpl.yml#L345-L387)、[`process_result.py` 契约](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/utils/process_result.py#L43-L75)、[吞吐量收集器](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/collect-results.yml#L25-L38)。

### 评测结果

评测任务上传以 `eval_${EXP_NAME}_${RESULT_FILENAME}` 命名的逐配置制品。制品包含该评测器实际生成的文件，例如 `meta_env.json`、`results*.json`、`sample*.jsonl`；对于受支持的 agentic 评测器，还可能包含 predictions、reports 或 trajectories。工作流的以下行为都是有意设计的：

- eval-only 任务没有任何评测文件时会报错；
- 评测文件在 `always()` 条件下上传，以保留失败任务的部分证据；
- `utils/evals/validate_scores.py` 在上传后验证 eval-only 分数覆盖范围；
- `collect-evals.yml` 下载 `eval_*`，运行 `collect_eval_results.py`，打印摘要，并上传 `eval_results_all/agg_eval_all.json`。

应用既能摄取聚合行，也能摄取逐配置评测目录。两种输入最终落到相同的自然键，而 sample 文件把详细数据附加到已经解析的评测行。因此，仅有聚合文件只能证明完成了收集，不能证明 sample 完整；当逐 sample 输出很重要时，必须核对逐配置制品。

来源：[单节点评测上传/验证](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-tmpl.yml#L362-L385)、[多节点评测上传/验证](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-multinode-tmpl.yml#L416-L434)、[评测收集器](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/collect-evals.yml#L24-L46)、[应用侧评测制品处理](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/ingest-ci-run.ts#L738-L792)。

### 在不改变 run 的情况下检查结果制品

```bash
RUN_ID=<InferenceX-run-id>

# Run identity and jobs
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --json event,headBranch,headSha,status,conclusion,attempt,jobs

# Exact unexpired artifact inventory
gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --paginate --jq '.artifacts[] | select(.expired == false) | [.name,.created_at,.size_in_bytes] | @tsv'

# Download only the two collection products when they exist
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name results_bmk --dir /tmp/infx-results-$RUN_ID
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name eval_results_all --dir /tmp/infx-evals-$RUN_ID

jq 'length' /tmp/infx-results-$RUN_ID/agg_bmk.json
jq 'length' /tmp/infx-evals-$RUN_ID/agg_eval_all.json
```

**关卡：** 零行、缺失聚合制品或源制品已过期，都要求先完成故障分类再重试。制品存在并不能证明其中有有效行：当两个请求计数字段都存在且 `num_requests_successful == 0` 时，应用会有意跳过该基准结果行。

来源：[失败结果行关卡](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/benchmark-mapper.ts#L203-L216)、[恢复流程使用的制品清单检查](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L149-L174)。

## InferenceX-app 交接与摄取验证

push 到 `main` 时，`run-sweep.yml` 只会在 setup 和适用的收集路径完成后 dispatch `ingest-results`。负载为：

```json
{
  "event_type": "ingest-results",
  "client_payload": {
    "source-run-id": "<artifact-owning-run>",
    "merge-run-id": "<publication-run>"
  }
}
```

普通 run 的两个 ID 都指向当前 run。复用时，`source-run-id` 指向经过验证的 PR sweep，而 `merge-run-id` 仍指向新的 push-to-main 恢复 run。InferenceX-app 会从 source run 中为每个准确制品名称选择最新、未过期的上传；复用时，则用 merge run 的 `changelog-metadata` 替换 source 的 changelog 元数据。如果 source 没有未过期制品，或 merge run 没有 changelog 制品，准备步骤会失败。

之后，应用工作流依次运行：制品准备、迁移、数据库摄取、run overrides、数据库验证、缓存失效和 unmapped entity 检查。数据库写入具有幂等性（`ON CONFLICT DO UPDATE` 或 `DO NOTHING`），因此指向正确目标的摄取在部分失败后可以安全恢复。

来源：[dispatch 负载与关卡](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/run-sweep.yml#L978-L1021)、[制品选择](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/ci-artifact-preparation.ts#L10-L48)、[应用摄取阶段](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/.github/workflows/ingest-results.yml#L50-L124)、[幂等性设计理由](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/docs/data-pipeline.md#L26-L34)。

### 验证准确的下游摄取

```bash
gh run list --repo SemiAnalysisAI/InferenceX-app \
  --workflow "Ingest Benchmark Results" \
  --event repository_dispatch --limit 10 \
  --json databaseId,status,conclusion,createdAt

INGEST_RUN_ID=<candidate-run-id>
SOURCE_RUN_ID=<expected-artifact-run-id>
MERGE_RUN_ID=<expected-publication-run-id>

gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Source run: $SOURCE_RUN_ID"
gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Merge run:  $MERGE_RUN_ID"

gh run watch "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app --exit-status
```

不能只要求结论为绿色；还要满足以下全部条件：

- 两条日志匹配都指向预期的 source/merge run 及其 attempt；
- 制品准备报告了预期的制品名称和数量；
- 复用摄取明确说明 changelog 元数据来自 merge run；
- 数据库摄取报告合理的 benchmark/eval/changelog 数量，并解释所有跳过的行；
- run overrides 和数据库验证成功；
- 已尝试缓存失效；
- 没有 unmapped model、hardware 或 precision，或这些项已经明确完成分类处理。

准备脚本输出的 `Source run:` 和 `Merge run:` 是权威匹配行。复用摄取的 source run 已经完成，所以不会等待五分钟。

来源：[权威准备日志](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/prepare-ci-artifacts.ts#L99-L152)、[正式验证顺序](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L391-L429)。

## 失败摄取恢复

首先把事件分成两种情况。

### 情况 A：正确的应用摄取已存在，但中途失败

如果其准备日志匹配预期的 source/merge 对，且制品仍未过期，那么重跑该 **InferenceX-app 摄取 run** 在数据层面是安全的，因为 ETL 具有幂等性：

```bash
gh run rerun "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app --failed
```

如果 GitHub 拒绝只重跑失败任务，完整重跑同一个应用 run 在数据层面仍然是幂等的：

```bash
gh run rerun "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app
```

当准备日志显示 ID 错误、制品已过期/缺失、changelog 元数据缺失，或候选 source SHA 之后执行语义发生变化时，**不要重跑**。重试只会忠实重复错误操作。

### 情况 B：正式摄取缺失、被跳过、无有效数据或指向错误 source

使用仓库的恢复 PR 流程。不要重跑失败目标，不要新增一次性工作流，也不要手工把行复制进数据库。

#### 1. 检查并证明目标

在干净的 InferenceX checkout 中运行，并确保 `gh`、`git`、`jq` 已认证且 Python 依赖可用：

```bash
python3 utils/recover_failed_ingest.py inspect-target \
  "$FAILED_RUN_OR_JOB_URL" \
  --output /tmp/infx-recovery-target.json

TARGET_RUN_ID=$(jq -r .run_id /tmp/infx-recovery-target.json)
TARGET_JOB_ID=$(jq -r .job_id /tmp/infx-recovery-target.json)
ORIGINAL_PR=$(jq -r .pr_number /tmp/infx-recovery-target.json)
ORIGINAL_MERGE_SHA=$(jq -r .merge_sha /tmp/infx-recovery-target.json)

gh run view "$TARGET_RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --job "$TARGET_JOB_ID" --log > "/tmp/infx-target-$TARGET_RUN_ID.log"

python3 utils/recover_failed_ingest.py audit-changelog \
  --ref "$ORIGINAL_MERGE_SHA"
```

**目标关卡：** 必须是 `main` 上 `.github/workflows/run-sweep.yml` 的已完成 push 事件。目标即使结论为 `success` 或 `cancelled`，仍可能包含无有效数据的摄取：忘记启用复用时，GPU 任务可能被取消，但 `trigger-ingest` 仍针对目标自己的 run ID 成功。不要删除该行；正确恢复后的发布会使用新的 run ID。

#### 2. 验证 source、祖先关系和制品

```bash
SOURCE_RUN_ID=<candidate-pr-sweep-run-id>
SOURCE_JSON=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID")
SOURCE_HEAD_SHA=$(jq -r .head_sha <<<"$SOURCE_JSON")
SOURCE_RUN_ATTEMPT=$(jq -r .run_attempt <<<"$SOURCE_JSON")

jq -e '
  .event == "pull_request" and
  .status == "completed" and
  .path == ".github/workflows/run-sweep.yml"
' <<<"$SOURCE_JSON" >/dev/null

gh api "repos/SemiAnalysisAI/InferenceX/pulls/$ORIGINAL_PR/commits" \
  --paginate --jq '.[].sha' | grep -Fx "$SOURCE_HEAD_SHA"

gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" \
  --paginate --jq '.artifacts[] | select(.expired == false) | .name' \
  | grep -Eq '^(results_bmk|eval_results_all|bmk_agentic_)'
```

未显式固定的 source 只能使用成功 run。明确固定的失败 run 只能恢复其中已完成的点，因为失败的基准结果行会被跳过。将 source SHA 与原始 PR 的最终 head 比较；如果后续编辑改变了待恢复项的镜像、模型、recipe、runner、launcher、基准参数或配置值，必须**停止**。

#### 3. 在更改 changelog 前授权复用

从当前 `main` 创建一个空恢复 PR，只添加一个 full-sweep 标签，并在推送恢复 changelog commit 前固定 source：

```bash
gh pr edit "$RECOVERY_PR" --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast
gh pr comment "$RECOVERY_PR" --repo SemiAnalysisAI/InferenceX \
  --body "/reuse-sweep-run $SOURCE_RUN_ID"
```

把恢复条目追加到 `perf-changelog.yaml` 末尾；绝不要修改历史字节。保留原始 `config-keys`、`description`、`evals-only` 和 `scenario-type`，但使用恢复 PR URL。验证 changelog 和生成的范围：

```bash
python3 utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT"
python3 utils/process_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT" \
  > /tmp/recovery-full-config.json
```

#### 4. 在不改变文件树的情况下保留 source 祖先关系

恢复分支最终 head 必须以恢复 commit 为第一父提交、source SHA 为第二父提交，并且文件树保持不变：

```bash
TARGET_PARENT=$(git rev-parse HEAD)
TARGET_TREE=$(git rev-parse "${TARGET_PARENT}^{tree}")
ATTACH_SHA=$(
  printf 'chore: attach reusable sweep run %s\n' "$SOURCE_RUN_ID" |
    git commit-tree "$TARGET_TREE" \
      -p "$TARGET_PARENT" \
      -p "$SOURCE_HEAD_SHA"
)
git reset --hard "$ATTACH_SHA"

test "$(git rev-parse HEAD^1)" = "$TARGET_PARENT"
test "$(git rev-parse HEAD^2)" = "$SOURCE_HEAD_SHA"
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse HEAD^1^{tree})"
test "$(git diff --name-only origin/main...HEAD)" = "perf-changelog.yaml"
git diff --check origin/main...HEAD
```

推送后，绝不要对这个 carrier commit 执行 rebase、本地 squash、amend 或 force-push。必须确认 source SHA 出现在 PR commit 列表中、Files 中只有 `perf-changelog.yaml`、`check-changelog` 与 `reuse-sweep-gate` 通过，并且 PR GPU 任务被跳过。只有获得明确授权后才能合并，且绝不能绕过失败或等待中的检查。随后使用上文交接流程验证新的 push run 和下游应用 run。

权威来源：[完整失败摄取恢复命令](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md)。

## AMD root-owned 工作区的预防与恢复

### 防止复发

容器可能以 root 身份运行，同时 GitHub 工作区被 bind mount。共享基准库通过以下设置防止工作区中出现 root-owned Python 缓存目录：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
```

不要把这些路径重新覆盖到工作区。MI355X launcher 还会在启动前删除旧基准日志，并安装 EXIT trap：复制 Slurm stdout/stderr 证据、打印错误尾部，然后执行有范围限制的 `sudo rm -rf "$BENCHMARK_LOGS_DIR"`。`KEEP_LOGS=1` 只应在刻意进行本地调试时使用；它会禁用清理 trap。取消任务仍可能绕过 teardown，因此在出现 `EACCES` 清理错误后，应执行下述恢复扫描。

来源：[Python 缓存预防](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/benchmarks/benchmark_lib.sh#L5-L10)、[MI355X 清理 trap](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/runners/launch_mi355x-amds.sh#L49-L76)。

### 恢复 MI355X TW runner 工作区

典型特征：

```text
Deleting the contents of '.../actions-runner/_work/InferenceX/InferenceX'
Error: File was unable to be removed Error: EACCES: permission denied, rmdir '.../benchmark_logs/logs/slurm_job-<id>'
```

jumpbox 没有 sudo；需要使用 agent forwarding 连接到在 `/it-share` 上拥有免密码 sudo 的 hop host。

1. **先进行只读扫描：**

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root 2>/dev/null'"
   ```

2. 检查每一个结果。每条路径都必须位于 `actions-runner/_work/` 下，通常在 `InferenceX/InferenceX/benchmark_logs/` 中。如果任何路径位于 `_work` 外，必须**停止**。
3. 获得明确批准后，只删除已经验证的匹配项：

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root -print0 2>/dev/null \
      | xargs -0 -r sudo rm -rf'"
   ```

4. 再次运行只读扫描，并要求结果为零。
5. 只有清理完成后，才能重跑已确诊的失败 sweep。可使用 `sacct -j <id>` 关联 `slurm_job-<id>`；`CANCELLED` 状态支持“跳过了 teardown”的诊断。

绝不要对 `/it-share` 运行没有范围限制的 `rm -rf`。

权威来源：[MI355X root-owned 文件恢复](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/clean-amd-mi355-runner-root-files.md)。

## MI300X 集群调试：enroot/pyxis 用户命名空间故障

`mi300x-amd_*` / `chi-mi300x-*` 上的典型特征：

```text
error: pyxis:     enroot-nsenter: failed to create user namespace: Permission denied
error: pyxis: couldn't start container
error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
srun: error: chi-mi300x-0XX: task 0: Exited with exit code 1
```

2026 年 7 月已知原因：Ubuntu 24.04 provisioning drift 使部分节点保留 `kernel.apparmor_restrict_unprivileged_userns=1`，阻止实际的 enroot 路径。`unshare -U` 不是有效判据，因为它自己的 AppArmor profile 仍可能允许该操作。

1. 在 GitHub 日志中确认准确特征并记录失败节点：

   ```bash
   gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
   ```

2. 从 root controller 通过 Slurm 访问计算节点；计算节点不接受直接 root SSH：

   ```bash
   ssh amd-vultr-mi300 \
     'srun -w chi-mi300x-043 -N1 --immediate=30 bash -c "<read-only-command>"'
   ```

3. 在不做更改的情况下调查所有可见节点：

   ```bash
   ssh amd-vultr-mi300 'for n in $(sinfo -N -h -o "%N" | sort -u); do
     v=$(srun -w $n -N1 --immediate=20 sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>&1 | tail -1)
     echo "$n: $v"
   done'
   ```

   失败节点为 `1`、工作节点为 `0` 的分裂结果可以确认 drift。如果所有节点都是 `0`，停止把它当成这个已知问题；应与工作节点比较 enroot 版本、pyxis plugin 状态，以及 `/usr/local/bin/enroot-nsenter` 的 AppArmor 覆盖范围。

4. **只有获得明确批准后**，才能把 drift 节点改成工作基线并持久化：

   ```bash
   ssh amd-vultr-mi300 'for n in <drifted-nodes>; do
     srun -w $n -N1 --immediate=30 bash -c \
       "sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 && \
        echo kernel.apparmor_restrict_unprivileged_userns=0 > /etc/sysctl.d/99-enroot-userns.conf"
   done'
   ```

   此操作会禁用一项内核安全缓解措施。验证每个节点的实时值为 `0`，且持久化文件存在。必须把长期修复升级到节点 provisioning image；否则重新 provision 的节点还会复发。

5. 集群基线恢复后，只重跑受影响的偶发失败任务。

权威来源：[MI300X enroot/pyxis 恢复](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/debug-mi300-enroot-pyxis.md)。

## 安全地重跑工作流

先列出失败任务并保存仅失败日志：

```bash
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.conclusion=="failure") | "\(.databaseId)\t\(.name)"'
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed \
  > /tmp/sweep_failed.txt
```

| 诊断 | 安全操作 | 禁止操作 |
| --- | --- | --- |
| PR sweep 上的临时 runner pickup、网络问题或已确认基础设施偶发故障 | `gh run rerun "$RUN_ID" --repo SemiAnalysisAI/InferenceX --failed` | dispatch 新 sweep 并丢失原始 run 上下文 |
| MI355X 工作区清理或 MI300X 修复后的 cancelled run | 先尝试仅重跑失败任务；如果 GitHub 因 run 被取消而拒绝，运行 `gh run rerun "$RUN_ID" --repo SemiAnalysisAI/InferenceX` | 在修复共享状态前完整重跑；它可能以相同方式失败并消耗 GPU 时间 |
| 可复现 OOM、HIP/CUDA/RCCL/NCCL 错误、无效结果、错误分数、镜像/recipe/配置缺陷 | 先修复或验证根因，再重跑最窄的受影响路径 | 仅因为某次重试通过就把事件标记为偶发故障 |
| 正确的 InferenceX-app 摄取发生部分 DB/迁移/验证失败 | 重跑该应用 run 的失败任务；如有需要，完整应用重跑在数据层面具有幂等性 | 重跑生成 GPU 结果的 InferenceX 目标 |
| source 制品错误/缺失/过期，或 source/merge 对错误 | 修复制品选择，或使用恢复 PR 流程 | 反复重跑同一个错误摄取 |
| 正式 push-to-main 摄取缺失、被跳过、无有效数据或失败 | 创建经过验证的恢复 PR，并复用原始 PR 制品 | 重跑失败目标工作流/任务，或创建一次性摄取工作流 |

如果已取消的 InferenceX run 拒绝部分重跑，完整重跑会重复所有符合条件的工作，成本可能很高。移除再重新添加 PR sweep 标签是最后手段，因为它会创建新 sweep，而不是保留失败 run。

来源：[偶发故障重跑规则](../.agents/skills/debug-runs/SKILL.md#L146-L153)、[cancelled run 回退方案](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/clean-amd-mi355-runner-root-files.md#L52-L57)、[应用摄取幂等性](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/docs/data-pipeline.md#L26-L34)。

## 故障分类

按最早出现故障的边界分类，而不是按最后一个红色任务分类。

| 类别 | 证据 | 负责人/操作 |
| --- | --- | --- |
| **基准/运行时** | server 一直未就绪；OOM；HIP/CUDA/RCCL/NCCL；请求失败；原始结果缺失；成功请求数为零 | 复现准确配置，与工作节点/SKU 比较，修复 recipe/镜像/运行时；不要改摄取 |
| **评测** | eval-only 没有 `results*.json`；分数验证失败；`meta_env.json` 覆盖范围/并发不匹配；sample 文件不完整 | 检查评测器输出和请求并发；保留已上传的部分制品；修复评测器/配置后再重跑 |
| **处理/收集** | 原始 JSON 存在但缺少 `agg_*.json`；收集器无法解析；缺少 `results_bmk` 或 `eval_results_all` | 检查 `process_result.py`/收集器日志和制品布局；理解格式问题后才重跑失败工作流任务 |
| **Runner 工作区** | checkout 清理在 `_work/.../benchmark_logs/logs/slurm_job-*` 下报 `EACCES` | 只读所有者扫描、批准后的有范围删除、零结果验证，然后重跑 |
| **MI300X provisioning** | pyxis/enroot 命名空间特征；失败节点 sysctl 为 `1`，工作节点为 `0` | 获批后修复节点并升级 provisioning image，然后重跑受影响任务 |
| **Dispatch/交接** | `trigger-ingest` 后没有应用 run；curl/auth 失败；应用准备日志显示错误 ID 或制品缺失/过期 | 修复 dispatch 凭据/选择，或使用恢复流程；不要重跑 GPU 工作 |
| **ETL/数据库** | 正确 source/merge 对和制品已准备，随后迁移/摄取/验证失败 | 诊断数据库/服务原因后重跑同一应用摄取；依靠幂等性，而非手动删除 |
| **映射/数据质量** | 应用报告跳过失败行、unmapped model/hardware/precision、评测行缺失或数量不合理 | 新增/修复实体映射或源元数据；存在 unmapped 数据时，工作流绿色也不代表完成验证 |
| **发布/来源** | changelog、source-run 来源、merge-run 身份错误，或无有效基准数据的伪造 run | 使用 append-only 恢复 PR；保留 source 祖先关系，并验证准确的下游摄取 |

可执行的事件报告应记录：

```text
InferenceX run / attempt / event / SHA:
Failed job / runner / node / Slurm job:
Failure class and first causal signature:
Raw, per-config, and aggregate artifact names/counts:
Expected source run / attempt / SHA:
Expected merge run:
InferenceX-app ingest run:
Repair performed and approval obtained:
Rerun command and resulting run:
DB verification, skipped rows, and unmapped entities:
Remaining durable fix:
```

这些证据就是完成关卡。如果没有制品身份、source/merge 身份和摄取数量，仅仅“工作流绿色”并不代表结果恢复已经验证。
