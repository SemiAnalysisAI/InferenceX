# B200 AgentX offload crossover study

**English** | [中文](README_zh.md)

This is a fresh study. Do not import previous MXFP8 results, run IDs, diagnostics,
charts or acceptance decisions. The run ledger starts empty. Work on the pushed
`experiment/agentx-b200-offload` branch; no PR or merge is needed.

## Question and comparisons

At a matched AgentX concurrency, when does adding host DRAM, local NVMe, or both
improve the throughput/latency tradeoff over HBM alone? Also compare the combined
tier against each single offload tier. Report a crossover bracket, then repeat and
refine it; do not manufacture a smooth curve or call an isolated noisy win a threshold.

| Arm / config suffix | GPU KV | Host KV | Local NVMe KV | Backend |
| --- | --- | --- | --- | --- |
| `none` | 80 GiB/GPU | 0 | 0 | HBM prefix cache |
| `dram` | 80 GiB/GPU | 739 GB/node | 0 | SimpleCPU, lazy |
| `nvme` | 80 GiB/GPU | No resident KV cache; transfer buffers remain | 4 TiB/node | SimpleCPU disk, lazy, direct I/O |
| `dram-nvme` | 80 GiB/GPU | 739 GB/node | 2 TiB stop guard | Native tiered CPU + filesystem |

The host budget is the actual fresh-main generator output at `dram-utilization:
0.683` and TP4 for B200, interpreted as decimal GB. The study checks it before
starting. HBM is pinned to 80 GiB per GPU (320 GiB total) so different connector
allocation layouts cannot silently change the common budget. All thresholds are
conditional on these budgets; this is not a universal concurrency threshold.

The NVMe-only arm now uses a 4 TiB bounded capacity so a single run can exercise
more of the local array. Runs produced with the earlier 1 TiB budget remain
separately identifiable in the ledger and are not treated as capacity-matched
repeats of the 4 TiB arm. The connector preallocates its configured disk files,
so requested concurrency does not determine disk footprint: admission requires
4,398,046,511,104 bytes plus the 128 GiB reserve. Start the expanded-capacity
series at c256, the highest NVMe concurrency already proven to finish the full
canonical run. The c512 probe completed 5,676 of 5,677 canonical warmup requests
with zero request errors, but one request exceeded the 1,800-second drain limit;
the phase failed before profiling and the allocation then reached its time limit.
Treat c512 as a feasibility bound. Probe c384 next, then fill the remaining
c256-c512 interval before trying a higher concurrency.

The combined tier uses a different connector and storage policy. The pinned FS
tier has no bounded LRU capacity setting: the 2 TiB value is an abort guard, not an
eviction quota. It was raised independently of the NVMe-only capacity after
run `35456989669` completed canonical profiling with zero request errors but reached
1,413,881,142,831 logical filesystem bytes and was correctly invalidated. Only runs
remaining below the guard can support a comparison. A guard hit is a failed
measurement. Confirm backend activity and actual cache
sizes from logs/metrics, and validate a suspected combined-tier win against a
native DRAM control before attributing it solely to storage hardware.

## Full canonical runs

Use fresh main's `nvidia/MiniMax-M3-NVFP4` TP4 recipe with the immutable vLLM
`nightly-dee37d89115db4c94a820a79a78a7828e141c910`, EAGLE3-GQA with the main recipe's
synthetic acceptance length 2.78. All four arms reuse that exact recipe and its
pinned AIPerf submodule. Keep ten extra warmups per lane, all mandatory snapshot
primers, seed 42, recorded assistant replay, the same corpus, idle-gap policy and
**3,600-second profiling window**. Do not use `agentx-fast`, duration overrides,
unsafe mode, synthetic workloads, custom clients, or direct Slurm submissions.
Inspect recorded corpus identity, full commands and actual allocated KV capacity
before declaring a pair matched.

Start with a matched four-arm probe at concurrency 16. Then build full curves for
all four arms. Initial curve points are 1, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1,024,
4,096, 8,192 and 16,384. Add intermediate positive integers near observed changes, and
repeat both sides of a candidate crossover on different nodes.
Keep the maximum at 16,384. Failure or insufficient completed samples is a
feasibility result, never a zero-throughput point. Canonical warmup that exceeds
workflow execution limits needs an explicit new execution budget, not truncation.

The original main image returned registry 404. All four arms therefore use the
same published replacement above; [image-provenance.json](image-provenance.json)
records its registry and AMD64 digests, exact engine commit, and compatibility
checks. The canonical heterogeneous-layout patch applies cleanly and is idempotent
against this source. No performance measurement completed on the unavailable image.
The replacement image defaults the EAGLE3-GQA draft head to FA4 on Blackwell;
runs `35401459251` and `35442679581` reached engine warmup but FA4 rejected a
broadcast descale tensor (`strides[1] == 0`). The latter run confirmed the pinned
engine explicitly redirects requested FA3 back to FA4 on Blackwell. The study
therefore selects FA2 and an unquantized draft KV cache for that FLASH_ATTN draft
head while retaining FlashInfer and FP8 KV for the main model. All four arms use
the same setting. This is an engine-compatibility delta from fresh main, not an
offload optimization.

## InferenceX infrastructure only

Preview each exact single-point dispatch with the real generator:

```bash
uv run --no-project --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate test-config \
  --config-keys agentx-offload-none --conc 16 --no-evals \
  --config-files experiments/agentx-offload/configs.yaml
```

After checking live capacity through InferenceX's dashboard/priority scheduler,
dispatch one arm and one concurrency per run, giving every arm its own run ID:

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref experiment/agentx-b200-offload \
  -f generate-cli-command='test-config --config-keys agentx-offload-none --conc 16 --no-evals --config-files experiments/agentx-offload/configs.yaml' \
  -f test-name='offload-v1-none-c16-r1'
```

Use suffixes `none`, `dram`, `nvme`, `dram-nvme`. Record the exact head SHA,
workflow ID, arm, concurrency, repeat number, node, status and artifact links in
[runs.json](runs.json). A run is not a result until its workflow and result
validation finish. Preserve failures and retries with their own attempt IDs.

Before each dispatch batch, query current B200 allocations and every pending
B200 job, including this study, legacy jobs and other users' work. Use idle
eligible nodes as necessary, but leave at most one B200 job queued across the
cluster after the batch. If one or more B200 jobs are already pending, launch
nothing until the pending count falls below one. Continue to defer to explicit
priority reservations and workflow or scheduler safety limits. Use workflow
logs, artifacts and InferenceX status APIs; no operator SSH, `salloc`, `sbatch`,
`srun` or `scancel`. The existing InferenceX runner itself may use Slurm
internally. The standard job requests `nodes:1`.

Run at most one NVMe-bearing workflow (`nvme` or `dram-nvme`) at a time. The
previous NVMe-bearing workflow must be terminal and its exact task-owned scratch
must have verified `deleted=true` before the next one is dispatched. Require the
declared 4 TiB budget plus the 128 GiB reserve at preflight. Do not dispatch while
an earlier task-owned scratch remains unresolved, and do not repeatedly target a
known-full node unchanged. This single-run rule is independent of the cluster-wide
one-pending-job ceiling above. The workflow currently pins NVMe-bearing allocations
to the node allowlist in `study.json`; it contains the recent c256 node with
17,763,847,192,576 bytes free at preflight and verified cleanup. Expand that list
only from fresh per-node storage evidence.

If a workflow time limit bypasses the normal `finish` trap, register the exact
run identity and scratch name in `cleanup_stale.py`. The c1 and c4 HBM maintenance
probes are pinned to nodes with registered scratch. Before their normal
experiment setup, they verify `owner.json`, delete only that exact scratch child,
and retain a cleanup receipt in the new run artifacts. The cleanup helper cannot
accept an arbitrary path, and the resulting HBM measurements still follow the
full canonical protocol.

Fresh main's `$/` self-repository workflow references are retained. They require
Actions runner 2.336.0 or newer; actionlint 1.7.12 does not recognize this syntax.
Offload setup is selected only by
`experiment: agentx-offload`; ordinary recipes retain their existing behavior.

The experiment's self-hosted checkout uses pinned `actions/checkout@v5.0.1` with
`persist-credentials: false`: the initial NVMe and DRAM workflow jobs failed in
v7.0.1's conditional credential-file setup before reaching GPU allocation.
This is a scoped compatibility probe for the documented [upstream path-matching
issue](https://github.com/actions/checkout/issues/2393), not offload performance
evidence. Ordinary workflow checkouts retain v7.0.1. Remove the fallback once the
runner paths or an upstream release are verified compatible.

After the first checkout failures, retry only the NVMe arm first. Confirm that
checkout and the experiment launch step succeed before dispatching the other
matched arms. Require cluster and scheduler observations no older than 90 seconds;
an available API response with an old scheduler snapshot does not permit launch.

## Evidence and visualization

Retain standard `bmk_agentic_*`, `agentic_*`, server and GPU metrics artifacts.
`results/offload_config.json`, `offload-telemetry.jsonl`, and
`offload_cleanup.json` record exact capacities, direct-I/O/storage proof, observed
file bytes, node memory/disk counters and owned-cache cleanup. Storage guard
failures remain explicit. Node disk counters are not exclusively this process's
I/O. Do not accept fallback buffered FS operation as the declared NVMe arm.
The storage proof resolves the mounted device and its backing leaves through
`/sys/class/block`; the container intentionally does not need the host `/dev/md0`
device node merely to establish that the local filesystem rests on non-rotating
NVMe devices.

After each matched set, compare successful output throughput per GPU, P90 latency,
request failure/cancellation counts and cache-source/I/O evidence. For the original
E2E-normalized X metric, recompute `1 / P90(request E2E seconds / output tokens)`
from a consistently defined completed profiling cohort; never substitute inverse
TPOT under that label. Repeat wins and quantify spread before reporting a threshold.
For a latency SLO, compare each arm's best throughput satisfying that same SLO;
when throughput and latency trade off, report both rather than invent one winner.

The app supports comma-separated workflow IDs:

- Chart: `https://inferencex.semianalysis.com/inference/minimax-m3?i_seq=agentic-traces&i_prec=fp4&i_pctl=p90&i_metric=y_tpPerGpu&unofficialruns=ID1,ID2,ID3,ID4`
- API: `https://inferencex.semianalysis.com/api/unofficial-run?runId=ID1,ID2,ID3,ID4`

Keep every workflow ID in `runs.json`, but append only successful completed runs
to the same open comparison tab and save that URL in the ledger. Refresh after a
new successful result appears, wait for the unofficial overlay to load, disable
`Optimal Only`, and verify that the points are visible. These must be actual new
GitHub workflow IDs, not Slurm IDs. Keep all four runs
individually identifiable even though the app currently reduces offload metadata
to on/off. Its current E2E-normalized chart explicitly suppresses unofficial
overlays because they lack persisted per-request traces. Standard AgentX overlays
are useful immediately after artifacts upload; the E2E-normalized comparison
requires app support or separate analysis of the retained new request traces.
Do not claim this API already supplies that view.

Disposable KV blobs are removed from the exact unique workflow-owned scratch
child after the server stops. Never delete shared model/image/corpus caches.
Temporary downloads should be deleted after analysis; keep only valuable new raw
evidence and its provenance. No legacy dataset is part of this experiment.
