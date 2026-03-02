# GB300 NVL72 Testing Ideas

Proposed new benchmark configurations and experimental tests for the GB300 NVL72 rack.
These ideas target gaps in the current test coverage and explore workload patterns
that are increasingly relevant for production inference deployments.

## Current GB300 NVL72 Coverage

Today the GB300 NVL72 rack benchmarks are limited to **DeepSeek-R1** (FP4 and FP8)
using two multi-node frameworks:

| Config Key | Model | Precision | Framework | ISL/OSL |
|---|---|---|---|---|
| dsr1-fp8-gb300-dynamo-sglang | DeepSeek-R1 | FP8 | Dynamo-SGLang | 1k1k, 8k1k |
| dsr1-fp4-gb300-dynamo-sglang | DeepSeek-R1 | FP4 | Dynamo-SGLang | 1k1k, 8k1k |
| dsr1-fp8-gb300-dynamo-trt | DeepSeek-R1 | FP8 | Dynamo-TRT | 1k1k, 1k8k, 8k1k |
| dsr1-fp4-gb300-dynamo-trt | DeepSeek-R1 | FP4 | Dynamo-TRT | 1k1k, 1k8k, 8k1k |

---

## Idea 1: GPT-OSS-120B on GB300 NVL72 (Dynamo-TRT)

**Motivation:** GPT-OSS-120B is already benchmarked on GB200 NVL72 via Dynamo-TRT
(`gptoss_fp4_gb200_dynamo-trt`), but there is no GB300 configuration. The GB300's
improved per-GPU memory bandwidth and compute should show meaningful gains over GB200
for this dense (non-MoE) model. This is one of the lowest-hanging fruit additions.

**What to test:**
- FP4 precision, Dynamo-TRT framework
- Disaggregated prefill/decode with varying worker ratios
- ISL/OSL: 1k1k, 1k8k, 8k1k
- Compare against GB200 results to quantify generational improvement

**Expected config key:** `gptoss-fp4-gb300-dynamo-trt`

---

## Idea 2: Long-Context Benchmarks (32K/128K ISL)

**Motivation:** All current NVL72 benchmarks use ISL <= 8192. Production workloads
increasingly involve long-context inputs (RAG with large documents, code repositories,
multi-document summarization). The NVL72's aggregate HBM capacity (72 GPUs x 192GB
on GB300) makes it uniquely suited for long-context inference that cannot fit on
single nodes. Testing long-context performance exposes NVLink domain bandwidth
bottlenecks during prefill and KV cache transfer patterns that short-context
benchmarks miss entirely.

**What to test:**
- DeepSeek-R1 FP8 and FP4 on Dynamo-TRT and Dynamo-SGLang
- New ISL/OSL combinations:
  - `32k1k` (32768 input, 1024 output) - long document Q&A
  - `128k1k` (131072 input, 1024 output) - full context window utilization
  - `32k8k` (32768 input, 8192 output) - long document summarization
- Focus on TTFT scaling: how does time-to-first-token degrade as ISL grows?
- Prefill-heavy worker ratios (e.g., 5 prefill + 2 decode) since long prefills
  dominate latency

**Key metrics to watch:**
- TTFT at low concurrency (latency-sensitive)
- Prefill throughput (tokens/sec) as a function of ISL
- Memory utilization and KV cache pressure

---

## Idea 3: Multi-Turn Conversational Workloads

**Motivation:** The existing `experimental/multiturn/` directory identifies this as a
gap. All current benchmarks use single-turn request patterns, but real chat
deployments involve multi-turn conversations where the KV cache from prior turns can
be reused (prefix caching / radix attention). On NVL72 with disaggregated inference,
multi-turn workloads stress the KV cache transfer between prefill and decode nodes
in a qualitatively different way: subsequent turns need to either re-prefill or
retrieve cached KV state.

**What to test:**
- Synthetic multi-turn conversations (3, 5, 10 turns)
- Per-turn context growth: each turn adds ~1K tokens of history
- Measure with and without prefix caching / HiCache enabled
- DeepSeek-R1 FP8 on Dynamo-SGLang (SGLang has radix attention support)
- Compare effective throughput vs single-turn at equivalent total token counts

**Proposed benchmark script:** `experimental/nvl72_testing_ideas/multiturn_gb300.sh`

**Key metrics to watch:**
- TTFT for turn N vs turn 1 (prefix cache hit rate)
- Total conversation completion time
- KV cache memory utilization over conversation lifetime

---

## Idea 4: Decode-Only Performance Isolation

**Motivation:** The existing `experimental/single_node_decodeonly/` shows that
isolating decode performance via vLLM's `DecodeBenchConnector` reveals dramatically
different characteristics (15x TTFT, 5.3x TPOT improvement in their H200 test).
On NVL72 disaggregated setups, the decode nodes are already isolated, but we don't
independently benchmark their raw decode throughput at various batch sizes. This
would help identify whether the decode nodes or the prefill nodes are the bottleneck
at different concurrency levels.

**What to test:**
- Decode-only throughput on GB300 decode workers at batch sizes: 1, 8, 32, 128, 512
- Pre-filled KV cache via `DecodeBenchConnector` or equivalent
- Measure raw decode tokens/sec per GPU without prefill contention
- Compare decode throughput: TP4 vs TP8 vs DEP32 (full rack decode)
- Profile NVLink utilization during decode at scale

**Key metrics to watch:**
- Peak decode tokens/sec/GPU at saturation
- Decode latency (TPOT) vs batch size curve
- NVLink bandwidth utilization percentage

---

## Idea 5: MoE Expert Load Balancing (EPLB) Sensitivity Analysis

**Motivation:** Several GB300 Dynamo-TRT configs reference `eplb` (Expert Load
Balancing) values like `eplb0`, `eplb256`, `eplb288`. MoE models like DeepSeek-R1
have 256 experts; how experts are distributed across GPUs in an NVL72 rack
significantly impacts all-to-all communication volume and tail latency. A systematic
sensitivity analysis would reveal the optimal EPLB configuration for each
concurrency regime.

**What to test:**
- DeepSeek-R1 FP4 on Dynamo-TRT, 1k1k sequence length
- Sweep EPLB slot counts: 0, 64, 128, 256, 288, 512
- Fixed concurrency points: low (4), mid (128), high (1024+)
- Measure the throughput delta between EPLB=0 (no rebalancing) and optimal EPLB
- Fixed prefill/decode worker ratios to isolate the EPLB effect

**Key metrics to watch:**
- Throughput improvement from EPLB at high concurrency
- Tail latency (P99 TPOT) sensitivity to EPLB
- Expert activation distribution skew

---

## Idea 6: Prefill/Decode Worker Ratio Sweep

**Motivation:** Current GB300 configs test a few hand-picked worker ratios (e.g.,
1 prefill + 4 decode for low-latency, 2 prefill + 1 decode for mid, 1+1 for max
throughput). But the NVL72 has 18 nodes (72 GPUs / 4 GPUs per node), and the
optimal split depends heavily on the workload's ISL/OSL ratio and concurrency.
A systematic sweep would map the full Pareto frontier.

**What to test:**
- DeepSeek-R1 FP8 on Dynamo-SGLang
- ISL/OSL: 1k1k (decode-heavy) and 8k1k (prefill-heavy)
- Worker ratio sweep (prefill:decode):
  - 1:17, 2:16, 3:15, 4:14, 6:12, 9:9, 12:6, 14:4, 16:2, 17:1
- Fixed concurrency per ratio to find the Pareto-optimal split
- Plot throughput vs TTFT for each ratio

**Key metrics to watch:**
- Throughput/GPU at each ratio
- TTFT as a function of prefill worker count
- Where the crossover point is between prefill-bound and decode-bound

---

## Idea 7: Qwen 3.5-397B MoE on GB300 NVL72

**Motivation:** Qwen 3.5-397B-A17B is a large MoE model already benchmarked on
single-node B200 and AMD MI325X/MI355X. It has a very different expert
architecture than DeepSeek-R1 (17B active out of 397B total, vs R1's architecture).
Running it on NVL72 would test how the disaggregated inference stack handles a
different MoE topology and whether the EP/TP configurations that work well for
DeepSeek-R1 transfer to other MoE models.

**What to test:**
- BF16 and FP8 precision
- Dynamo-SGLang and Dynamo-TRT
- ISL/OSL: 1k1k, 1k8k
- Varying EP sizes: EP4, EP8, EP16

**Expected config keys:**
- `qwen3.5-bf16-gb300-dynamo-sglang`
- `qwen3.5-fp8-gb300-dynamo-trt`

---

## Idea 8: NVLink Domain Scaling Efficiency

**Motivation:** The NVL72 connects 72 GPUs in a single NVLink domain. Current
configs either use all 72 GPUs or small subsets (e.g., TP4 per node). Understanding
how throughput scales as we use more of the rack (from 1 node to 18 nodes) would
reveal NVLink scaling efficiency and help cloud operators decide how to slice the
rack for multi-tenant deployments.

**What to test:**
- DeepSeek-R1 FP4 on Dynamo-TRT, fixed ISL/OSL (1k1k)
- Scale from 1 node (4 GPUs) to 18 nodes (72 GPUs):
  - 1 node (TP4), 2 nodes (TP8), 4 nodes (TP16), 9 nodes (TP36), 18 nodes (TP72)
- For each scale point, find the concurrency that saturates throughput
- Plot tokens/sec/GPU vs total GPU count to show scaling efficiency
- Also test with disaggregated mode: how does adding more decode workers scale?

**Key metrics to watch:**
- Throughput per GPU at each scale point
- Linear scaling coefficient (ideal = 1.0)
- NVLink all-reduce overhead as node count grows

---

## Idea 9: DeepSeek Sparse Attention (DSA) on NVL72

**Motivation:** The `experimental/dsv32/` directory already models the theoretical
FLOPs and bytes savings from DeepSeek Sparse Attention. Once DSA is available in
vLLM/SGLang for multi-node deployments, NVL72 is the ideal platform to validate
those theoretical gains. DSA reduces decode-time KV cache reads from O(L) to O(k)
for context length L, which should be especially impactful for the long-context
benchmarks proposed in Idea 2.

**What to test:**
- DeepSeek-R1 with sparse attention enabled (when framework support lands)
- Compare dense vs sparse attention at ISL = 32K, 64K, 128K
- Sweep Top-k values: 1024, 2048, 4096, 8192
- Measure accuracy degradation (via eval tasks: gsm8k, gpqa_diamond) alongside
  throughput gains
- Profile memory bandwidth savings on decode nodes

**Key metrics to watch:**
- Decode TPOT improvement: dense vs sparse
- Accuracy impact at various Top-k values
- Memory bandwidth utilization reduction

---

## Idea 10: Mixed-Precision A/B Comparison (FP4 vs FP8 vs BF16)

**Motivation:** The GB300 tests FP4 and FP8 for DeepSeek-R1, but the two precisions
use different model checkpoints and framework configurations, making direct comparison
hard. A controlled A/B test with identical workload parameters, worker ratios, and
concurrency levels would provide a definitive throughput-vs-accuracy tradeoff analysis
for each precision on the same hardware.

**What to test:**
- DeepSeek-R1 on GB300 Dynamo-TRT with identical configs across:
  - FP4 (`deepseek-r1-0528-fp4-v2`)
  - FP8 (`deepseek-r1-0528`)
- Same prefill/decode worker ratios: 2 prefill + 3 decode (a balanced config)
- Same concurrency sweep: 4, 32, 256, 1024, 4096
- Same ISL/OSL: 1k1k
- Run evals (gsm8k, gpqa_diamond) at each precision to measure accuracy delta

**Key metrics to watch:**
- Throughput/GPU ratio: FP4 vs FP8
- Accuracy delta on gsm8k and gpqa_diamond
- Effective tokens/watt (if power monitoring is available)

---

## Priority Ranking

| Priority | Idea | Effort | Impact |
|---|---|---|---|
| P0 | 1. GPT-OSS-120B on GB300 | Low | High - direct gap fill |
| P0 | 6. Prefill/Decode ratio sweep | Medium | High - foundational for all other tests |
| P1 | 2. Long-context (32K/128K) | Medium | High - production relevance |
| P1 | 5. EPLB sensitivity analysis | Low | Medium - tuning insight |
| P1 | 10. FP4 vs FP8 controlled A/B | Low | Medium - precision tradeoff clarity |
| P2 | 7. Qwen 3.5 on GB300 | Medium | Medium - model diversity |
| P2 | 8. NVLink scaling efficiency | High | High - architectural insight |
| P2 | 3. Multi-turn conversations | Medium | Medium - production relevance |
| P3 | 4. Decode-only isolation | Low | Medium - debugging/profiling |
| P3 | 9. DeepSeek Sparse Attention | High | High - blocked on framework support |
