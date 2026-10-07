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
| `nvme` | 80 GiB/GPU | No resident KV cache; transfer buffers remain | 2 TiB/node (current capacity probe) | SimpleCPU disk, lazy, direct I/O |
| `dram-nvme` | 80 GiB/GPU | 739 GB/node | 2 TiB stop guard | Native tiered CPU + filesystem |

The host budget is the actual fresh-main generator output at `dram-utilization:
0.683` and TP4 for B200, interpreted as decimal GB. The study checks it before
starting. HBM is pinned to 80 GiB per GPU (320 GiB total) so different connector
allocation layouts cannot silently change the common budget. All thresholds are
conditional on these budgets; this is not a universal concurrency threshold.

The completed expanded-capacity NVMe-only cohort used a 4 TiB bounded cache.
The next adaptive control halves only this bounded SimpleCPU disk capacity to
2 TiB at c488, where the 4 TiB NVMe and HBM-only arms both completed canonical
profiling. Earlier 1 TiB, 4 TiB, and new 2 TiB runs remain distinct in the
ledger; a cross-capacity difference is local evidence, not a matched repeat.
The connector preallocates its configured disk files, so concurrency does not
determine disk footprint: admission now requires 2,199,023,255,552 bytes plus
the 128 GiB reserve (2,336,462,209,024 bytes total). The completed 4 TiB series started at c256, which completed the full canonical run with 61.680% external
cache hits. The c384 midpoint also completed, but external hits fell to 5.299%,
total throughput per GPU fell 46.716%, and P90 end-to-end latency rose 162.274%
relative to c256. The c512 probe completed 5,676 of 5,677 canonical warmup
requests with zero request errors, but one request exceeded the 1,800-second
drain limit; the phase failed before profiling and the allocation then reached
its time limit. The c320 midpoint completed between c256 and c384 with 24.540%
external hits, 13,193.777 total tokens/s/GPU, P90 interactivity 11.998
tokens/s/user, and P90 TTFT 1,331.284 seconds. This confirms a steep, ordered
cache-reuse and performance cliff across c256, c320, and c384. Its matched
HBM-only c320 control completed 1,010 profiling responses versus NVMe's 1,697.
At the same 3,600-second profiling window, NVMe delivered 13,193.777 versus
7,997.601 total tokens/s/GPU (+64.972%), improved P90 interactivity from 8.047
to 11.998 tokens/s/user (+49.097%), and reduced P90 TTFT from 1,990.296 to
1,331.284 seconds (-33.111%). NVMe also used 4.633% less average power and
43.241% less energy per successful query. This single matched run establishes
that 4 TiB NVMe helps at c320 under this protocol, but a repeat is still needed
before treating the size of the win as stable. Run the matched HBM-only c384
control next to determine whether NVMe still helps after the observed cache cliff.
That control completed 1,048 profiling responses versus NVMe's 1,157. NVMe still
won at c384, but by much less than at c320: 9,198.026 versus 8,277.940 total
tokens/s/GPU (+11.115%), 8.721 versus 8.149 P90 interactivity (+7.028%), and
2,154.172 versus 2,333.924 seconds P90 TTFT (-7.702%). Average power was 2.710%
lower with NVMe. Matched c448 also completed: NVMe delivered 8,839.499 versus
8,251.805 total tokens/s/GPU (+7.122%), improved P90 interactivity from 8.212
to 8.973 tokens/s/user (+9.273%), and completed 1,204 versus 1,120 responses.
P90 TTFT improved only 0.427%, while P90 E2E-normalized interactivity was 0.376%
lower; average power fell 0.416% and energy per successful query fell 7.364%.
The benefit remains positive but continues to narrow. Matched c480 then showed
another mixed NVMe win: 8,642.782 versus 8,104.248 total tokens/s/GPU (+6.645%),
87.219 versus 74.494 output tokens/s/GPU (+17.083%), 8.547 versus 8.119 P90
interactivity (+5.274%), and 2,815.145 versus 2,947.946 seconds P90 TTFT
(-4.505%). NVMe completed 1,186 versus 1,097 responses and used 6.940% less
energy per successful query, although P90 E2E-normalized interactivity was
2.010% lower and average power was 0.610% higher. The c496 probe then remained
in canonical warmup until the reusable job's 500-minute execution limit. Its
last progress sample had returned 5,091 of 5,500 responses, sent 5,459, retained
368 in flight, and reported zero request errors after 22,834.1 seconds; all
5,093 retained request records are warmup and profiling never started. Treat
this as local canonical-warmup infeasibility under the current execution budget,
not as a zero-throughput result. The c488 midpoint then completed all 5,401
canonical warmup requests without request errors after 24,559.47 seconds and
finished the full 3,600-second profiling window. It retained 1,168 successful
responses and delivered 8,541.085 total tokens/s/GPU, 81.795 output
tokens/s/GPU, 8.904 P90 interactivity, 2,904.031 seconds P90 TTFT, and
11,170.326 joules per successful query. External cache served 1.443% of prompt
tokens. The workflow concluded failure only after exports completed because the
launch step exited 143; all result, raw AgentX, server-log, GPU, and power
artifacts uploaded, so c488 is a valid local performance point. The matched
HBM-only c488 control also completed canonical warmup and the full profile before
the same post-export time-limit termination. NVMe delivered 8,541.085 versus
8,052.723 total tokens/s/GPU (+6.065%), 81.795 versus 71.868 output
tokens/s/GPU (+13.813%), 8.904 versus 8.153 P90 interactivity (+9.206%), and
2,904.031 versus 2,998.848 seconds P90 TTFT (-3.162%). It completed 1,168
versus 1,071 responses, used 8.685% less energy per successful query, and
computed 0.847% fewer prompt tokens while serving 1.443% externally. Average
power was 0.415% lower, but P90 E2E-normalized interactivity was 3.729% lower.
Treat c488 as another mixed local NVMe win. The native DRAM c488 control then
sent all 5,401 canonical warmup requests and returned 5,395 with zero request
errors, but six wire requests remained after 25,174.7 seconds when the fixed
1,800-second accelerated drain timeout expired. All 5,395 retained records are
warmup, including eight InvalidInferenceResultError classifications; profiling
never started, so this is a local warmup-feasibility failure rather than a
performance result. The exact task-owned scratch was deleted and independently
confirmed absent. The HBM+DRAM+NVMe c488 control then completed all 5,401
warmup requests without request errors and the full 3,600-second profile. It
retained 1,904 successful responses and measured 13,026.732 total tokens/s/GPU,
144.643 output tokens/s/GPU, 12.898 P90 interactivity, 1,806.766 seconds P90
TTFT, and 5,738.145 joules per successful query. Compared with HBM-only, these
are gains of 61.768%, 101.263%, and 58.193% in total throughput, output
throughput, and P90 interactivity, with 39.751% lower P90 TTFT, 77.778% more
completed responses, and 53.092% lower energy per successful query. However,
the native filesystem tier crossed its 2 TiB stop guard at 2,234,874,994,223
logical bytes and continued growing until cleanup observed 13,255,648,681,519
logical bytes across 1,117,488 files. Late-profile writes then reported ENOSPC.
The full performance export remains useful local evidence of native tiering and
the larger completed cohort, but it is not a clean 2 TiB capacity-matched
control and cannot isolate the storage-tier contribution. The exact scratch was
deleted and independently confirmed absent. Do not dispatch another combined
tier run until the filesystem tier has an enforced capacity bound or a monitor
that stops the workload promptly at the declared guard.

The monitor now resolves the shell's live process tree when a guard is crossed,
signals workload descendants deepest-first, and then signals the shell so its
owned cleanup trap can run. The guard receipt records the intended termination
targets. Corrected c488 requalification run `37064957726` then triggered the
guard during canonical warmup at 2,295,157,759,535 logical bytes. Cleanup
observed 2,299,783,945,775 logical bytes, only 4.626 GB beyond the guard sample,
deleted the exact task-owned scratch, and an independent check confirmed it
absent. The run retained 153 warmup records with zero cancellations; profiling
never started. This verifies prompt deterministic termination and cleanup, but
does not turn the native filesystem tier into a bounded LRU cache. Combined-tier
metrics remain ineligible for a capacity-matched comparison.

The c492 NVMe-only midpoint then failed during canonical warmup before profiling.
Its last progress sample returned 3,599 of 5,446 responses, sent 4,063, retained
464 in flight, and reported zero request errors after 15,692.9 seconds. One root
request then raised ClientOSError while writing its HTTP request body, which
caused the warmup failure and prevented profiling. All 3,605 retained records
are warmup; eight are InvalidInferenceResultError classifications and one is the
terminal ClientOSError. The server log contains no EngineDead, CUDA OOM,
traceback, or error signature. Treat c492 as a local warmup-feasibility failure,
not a performance result. The c490 NVMe-only midpoint completed all 5,423 warmup
requests after 24,686.31 seconds and retained 1,178 successful profiling responses,
but the fixed Slurm allocation expired seven seconds before the nominal profiling
send deadline. AIPerf finalized its raw request records and passed 100% latency
coverage, while aggregate export was interrupted and strict power replay failed
with `sampling_gap_exceeded`. Treat c490 as qualified near-complete request evidence
and a local canonical-protocol feasibility failure under the current execution
budget, not a complete performance result. The c489 NVMe-only point completed all
5,412 warmup requests after 24,710.93 seconds and the full 3,600-second profiling
send window, retaining 1,151 successful responses with 100% latency coverage.
The fixed Slurm allocation then interrupted aggregate export. Strict four-GPU power
replay passed, but the reconstructed request and energy metrics remain qualified
near-complete evidence rather than a complete performance result. The matched
HBM-only c489 arm likewise completed all 5,412 warmup requests after 24,750.81
seconds and the full profiling send window, retained 1,076 successful responses,
and passed 100% latency coverage before the fixed allocation interrupted aggregate
export. Its strict power replay also passed. On the reconstructed matched cohorts,
NVMe improved total throughput/GPU by 4.538%, output throughput/GPU by 9.031%, P90
interactivity by 2.512%, P90 TTFT by 3.392%, completed responses by 6.970%, and
energy/successful query by 7.135%; it also improved P90 E2E-normalized
interactivity by 6.453% and reduced average power by 0.662%. These deltas are
qualified local evidence only because neither c489 arm produced a complete
aggregate result. Both are local canonical-protocol feasibility failures under the
fixed workflow budget, so the integer boundary remains completed c488 versus
infeasible c489 and is not specific to the NVMe tier.

The follow-on c489 requalification changes only the single-node execution envelope.
Canonical AgentX jobs receive a 510-minute Slurm allocation and a 530-minute outer
GitHub job limit instead of 480 and 500 minutes. Both original c489 arms finalized
their raw profiling records roughly 482–483 minutes after job start, so the extra
30 Slurm minutes provide bounded headroom for aggregate export and exact cleanup;
the outer limit remains 20 minutes longer than Slurm. Fixed-sequence single-node
jobs retain their existing 480/500-minute limits. The model, precision, TP4 layout,
corpus, warmup, cache configuration, connector, and 3,600-second profiling window
remain unchanged. Results from this phase are a separate execution-budget
qualification and must not be silently combined with the fixed-budget boundary.

The extended-budget c489 pair completed canonical warmup, the full profiling send
window, aggregate export, and 100% latency coverage. NVMe run `37381547856`
retained 1,171 successful responses. The first HBM run `37428997901` retained
1,088, but a 10.791-second GPU telemetry gap invalidated its energy metric. The
HBM repeat `37490705589` retained 1,111 responses and passed strict four-GPU
UTC power validation (maximum sample gap 1.005 seconds). Relative to this repeat,
the 4 TiB NVMe arm improved total throughput/GPU 4.021%, output throughput/GPU
16.967%, P90 interactivity 8.403%, P90 TTFT 1.453%, completed responses
5.401%, and energy/successful query 0.638%. Average power rose 4.728%, while
P90 E2E-normalized interactivity rose 0.501%. The energy advantage is narrow
local evidence, not a replicated win. The NVMe workflow was cancelled after
export, and the HBM repeat workflow failed in downstream collectors because
their tooling-ref checkout lacked `infx`; both benchmark results and power
validations were complete before those workflow conclusions. Exact host scratch
was independently absent for both runs.

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

Use fresh main's `nvidia/MiniMax-M3-NVFP4` TP4 recipe with the pinned vLLM
manifest digest in [image-provenance.json](image-provenance.json), EAGLE3-GQA with the main recipe's
synthetic acceptance length 2.78. All four arms reuse that exact recipe and its
pinned AIPerf submodule. Keep ten extra warmups per lane, all mandatory snapshot
primers, seed 42, recorded assistant replay, the same corpus, idle-gap policy and
**3,600-second profiling window**. Do not use `agentx-fast`, duration overrides,
unsafe mode, synthetic workloads, custom clients, or direct Slurm submissions.
Inspect recorded corpus identity, full commands and actual allocated KV capacity
before declaring a pair matched.

Start with a matched four-arm probe at concurrency 16. Then build full curves for
all four arms. Initial curve points are 1, 4, 8, 16, 32, 64, 128, 256, 320, 384, 448, 480, 488, 489, 490, 492, 496, 512, 1,024,
4,096, 8,192 and 16,384. Add intermediate positive integers near observed changes, and
repeat both sides of a candidate crossover on different nodes.
Keep the maximum at 16,384. Failure or insufficient completed samples is a
feasibility result, never a zero-throughput point. Canonical warmup that exceeds
workflow execution limits needs an explicit new execution budget, not truncation.

The original main image returned registry 404. All four arms use the same
replacement image; [image-provenance.json](image-provenance.json) records its
registry and AMD64 digests, exact engine commit, and compatibility checks. Its
2026-09-18 nightly tag also disappeared: run `37366720497` reached Slurm allocation
but image import returned registry 404 before the benchmark or scratch creation.
Both recorded digests still resolve, so the study now references the original
multi-platform manifest by digest. The engine image is byte-identical to the
previous tag reference. The canonical heterogeneous-layout patch applies cleanly and is idempotent
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
declared 2 TiB budget plus the 128 GiB reserve at preflight for the capacity probe. Do not dispatch while
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

The complete refreshable branch inventory and canonical chart URLs are maintained in
[`UNOFFICIAL_RUNS.md`](./UNOFFICIAL_RUNS.md).

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
