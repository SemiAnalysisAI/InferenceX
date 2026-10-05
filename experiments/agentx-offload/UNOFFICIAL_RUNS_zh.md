# AgentX 卸载 unofficial-run 清单

<div align="center">

[English](./UNOFFICIAL_RUNS.md) | **中文**

</div>

本生成清单用于长期保存测试分支的工作流 ID 和 unofficial 图表 URL。
图表参数使用的是 **GitHub workflow run ID**，不是单个 job ID。所有单独的 job ID 均保存在
[`unofficial-runs.json`](./unofficial-runs.json) 中。

在仓库根目录运行以下命令可刷新三个生成文件：

```bash
python3 experiments/agentx-offload/update_unofficial_runs.py --write
```

刷新脚本会查询 `experiment/agentx-b200-offload` 上的全部工作流，记录其 job 和产物，
并调用公开的 unofficial-run API 判断当前是否返回 benchmark 记录。
`renderable` 只表示 API 至少返回一条 benchmark 记录，不代表该运行在科学上有效。
有效性和研究结论应以 [`runs.json`](./runs.json) 为准。

生成时间：`2026-10-05T19:47:17.794309Z`

## URL

全部可由 API 渲染的 benchmark 点
（28 个运行）：

<https://inferencex.semianalysis.com/inference/minimax-m3?i_seq=agentic-traces&i_prec=fp4&i_pctl=p90&i_metric=y_tpPerGpu&unofficialruns=35444664736,35449533638,35454656748,35456989669,35469801850,35476053573,35497846918,35524019140,35524083518,35542179539,35542182560,35542185178,35542190411,35542193122,35542196012,35565505126,35565538871,35635160852,35646317491,35727274307,35766002905,35802396548,35832083322,35882091021,35934408414,35972536894,36796500128,37027095833>

纯 NVMe 且可由 API 渲染的点
（8 个运行）：

<https://inferencex.semianalysis.com/inference/minimax-m3?i_seq=agentic-traces&i_prec=fp4&i_pctl=p90&i_metric=y_tpPerGpu&unofficialruns=35444664736,35524019140,35565538871,35646317491,35727274307,35832083322,35934408414,36796500128>

## 完整分支工作流列表

分支工作流：**68**  
GitHub 单独 job：**816**

| Workflow run ID | 结论 | 图表 benchmark 数 | GitHub job 数 | 工作流 |
| ---: | --- | ---: | ---: | --- |
| [35150085918](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35150085918) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-20260916T2102Z` |
| [35150281066](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35150281066) | cancelled | 0 | 12 | `e2e Test - offload-v1-none-c16-r1-20260916T2104Z` |
| [35150286749](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35150286749) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c16-r1-20260916T2104Z` |
| [35154018672](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35154018672) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-checkoutfix-20260916T214417Z` |
| [35344315025](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35344315025) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-imagefix-20260918T1222Z` |
| [35387609866](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35387609866) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-imagefix2-20260918T194421Z` |
| [35401459251](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35401459251) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-sysfsfix-20260918T222510Z` |
| [35442679581](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35442679581) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16-r1-fa3fix-20260919T1223Z` |
| [35444664736](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35444664736) | success | 1 | 12 | `e2e Test - offload-v1-nvme-c16-r1-fa2fix-20260919T1259Z` |
| [35449533638](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35449533638) | success | 1 | 12 | `e2e Test - offload-v1-none-c16-r1-20260919T144156Z` |
| [35454656748](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35454656748) | success | 1 | 12 | `e2e Test - offload-v1-dram-c16-r1-20260919T162046Z` |
| [35456989669](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35456989669) | failure | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c16-r1-20260919T1705Z` |
| [35469801850](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35469801850) | success | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c16-r2-20260919T2114Z` |
| [35476043213](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476043213) | failure | 0 | 12 | `e2e Test - offload-v1-none-c1024-r1-20260919T2324Z` |
| [35476047627](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476047627) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c1024-r1-20260919T2324Z` |
| [35476050409](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476050409) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c1024-r1-20260919T2324Z` |
| [35476053573](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476053573) | failure | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c1024-r1-20260919T2324Z` |
| [35476061809](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476061809) | failure | 0 | 12 | `e2e Test - offload-v1-none-c4096-r1-20260919T2324Z` |
| [35476067005](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476067005) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c4096-r1-20260919T2324Z` |
| [35476072143](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476072143) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c4096-r1-20260919T2324Z` |
| [35476075885](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476075885) | failure | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c4096-r1-20260919T2324Z` |
| [35476332029](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476332029) | failure | 0 | 12 | `e2e Test - offload-v1-none-c16384-r1-20260919T2330Z` |
| [35476334779](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476334779) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c16384-r1-20260919T2330Z` |
| [35476337411](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476337411) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c16384-r1-20260919T2330Z` |
| [35476340310](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35476340310) | failure | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c16384-r1-20260919T2330Z` |
| [35478297080](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35478297080) | failure | 0 | 12 | `e2e Test - offload-v1-none-c8192-r1-20260920T0015Z` |
| [35478298744](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35478298744) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c8192-r1-20260920T0015Z` |
| [35478300305](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35478300305) | failure | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c8192-r1-20260920T0015Z` |
| [35492330557](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35492330557) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c8192-r1-20260920T0541Z` |
| [35497846918](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35497846918) | success | 1 | 12 | `e2e Test - offload-v1-none-c256-r1-20260920T074810Z` |
| [35524019140](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35524019140) | success | 1 | 12 | `e2e Test - offload-v1-nvme-c256-r1-20260920T1651Z` |
| [35524083518](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35524083518) | success | 1 | 12 | `e2e Test - offload-v1-dram-c256-r1-20260920T1652Z` |
| [35524297747](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35524297747) | failure | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c256-r1-20260920T1656Z` |
| [35542179539](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542179539) | failure | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c256-r2-20260920T2236Z` |
| [35542182560](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542182560) | success | 1 | 12 | `e2e Test - offload-v1-none-c64-r1-20260920T2236Z` |
| [35542185178](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542185178) | failure | 1 | 12 | `e2e Test - offload-v1-dram-c64-r1-20260920T2236Z` |
| [35542187788](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542187788) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c64-r1-20260920T2236Z` |
| [35542190411](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542190411) | failure | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c64-r1-20260920T2236Z` |
| [35542193122](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542193122) | success | 1 | 12 | `e2e Test - offload-v1-none-c128-r1-20260920T2236Z` |
| [35542196012](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542196012) | success | 1 | 12 | `e2e Test - offload-v1-dram-c128-r1-20260920T2236Z` |
| [35542198481](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542198481) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c128-r1-20260920T2236Z` |
| [35542200935](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542200935) | cancelled | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c128-r1-20260920T2236Z` |
| [35542366055](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542366055) | failure | 0 | 12 | `e2e Test - offload-v1-nvme-c64-r2-20260920T2240Z` |
| [35542367394](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35542367394) | cancelled | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c128-r2-20260920T2240Z` |
| [35565505126](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35565505126) | success | 1 | 12 | `e2e Test - offload-v1-none-c1-r1-cleanup-c002-20260921` |
| [35565517309](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35565517309) | cancelled | 0 | 12 | `e2e Test - offload-v1-none-c4-r1-cleanup-c008-20260921` |
| [35565538871](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35565538871) | success | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c256-r1-20260921` |
| [35580481104](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35580481104) | failure | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c512-r1-20260921` |
| [35583529913](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35583529913) | failure | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c512-r2-20260921` |
| [35635160852](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35635160852) | success | 1 | 12 | `e2e Test - offload-v1-none-c1-r2-cleanup-c001-20260921` |
| [35646317491](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35646317491) | success | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c384-r1-20260921` |
| [35727274307](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35727274307) | success | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c320-r1-20260922` |
| [35766002905](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35766002905) | success | 1 | 12 | `e2e Test - offload-v1-none-c320-r1-20260922` |
| [35802396548](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35802396548) | success | 1 | 12 | `e2e Test - offload-v1-none-c384-r1-20260923` |
| [35832083322](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35832083322) | success | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c448-r1-20260923` |
| [35882091021](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35882091021) | success | 1 | 12 | `e2e Test - offload-v1-none-c448-r1-20260923` |
| [35934408414](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35934408414) | success | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c480-r1-20260923` |
| [35972536894](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35972536894) | success | 1 | 12 | `e2e Test - offload-v1-none-c480-r1-20260924` |
| [36739077310](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36739077310) | cancelled | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c496-r1-branchwf-20260930` |
| [36796500128](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36796500128) | failure | 1 | 12 | `e2e Test - offload-v1-nvme4tib-c488-r1-branchwf-20261001` |
| [36879140816](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36879140816) | failure | 0 | 12 | `e2e Test - offload-v1-none-c488-r1-branchwf-20261001` |
| [36943237877](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36943237877) | failure | 0 | 12 | `e2e Test - offload-v1-dram-c488-r1-branchwf-20261001` |
| [37027095833](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37027095833) | failure | 1 | 12 | `e2e Test - offload-v1-dram-nvme-c488-r1-branchwf-20261002` |
| [37064957726](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37064957726) | failure | 0 | 12 | `e2e Test - offload-v1-dram-nvme-c488-r2-guardfix-20261002` |
| [37070896182](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37070896182) | failure | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c492-r1-branchwf-20261002` |
| [37140167126](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37140167126) | failure | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c490-r1-branchwf-20261003` |
| [37236791824](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37236791824) | failure | 0 | 12 | `e2e Test - offload-v1-nvme4tib-c489-r1-branchwf-20261004` |
| [37269968186](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37269968186) | failure | 0 | 12 | `e2e Test - offload-v1-none-c489-r1-branchwf-20261005` |
