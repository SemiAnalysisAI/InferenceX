# Op-attributed vLLM profiles for AgentX points

Profiles a single-node vLLM srt-slurm AgentX point (for example
`dsv4-fp4-b200-vllm-agentic-mtp`) during its real replay and attributes every
device kernel to the CPU-side call stack that launched it: the module stack,
the torch op, and for kernels launched straight from Python, the launcher and
its vLLM callers. Kernels replayed from CUDA graphs are attributed through the
graph's capture.

## Running

Dispatch `e2e-tests.yml` on a branch carrying this directory with a `profile`
input (a JSON object; `{}` takes every default):

```sh
gh workflow run e2e-tests.yml -R SemiAnalysisAI/InferenceX --ref <branch> --json <<'EOF'
{
  "ref": "<branch>",
  "agentx-fast": "true",
  "test-name": "profile test",
  "profile": "{}",
  "generate-cli-command": "test-config --config-files configs/nvidia-master.yaml --scenario-type agentic-coding --no-evals --conc 32 --config-keys dsv4-fp4-b200-vllm-agentic-mtp"
}
EOF
```

One concurrency selects one recipe variant (for DSV4 on B200: `tp8_c*` or
`dep8_c*`). Settings (`infx/srt_slurm/single_node.py`, `PROFILE_DEFAULTS`):

| Setting | Default | Meaning |
|---|---|---|
| `windows` | `[["warmup", 0, 32], ["decode", 0, 32]]` | `[anchor, delay_seconds, iterations]` per torch profiler window. `warmup` and `profiling` anchor on aiperf logging that phase's start; AgentX warmup is the lanes' long first turns, all sent at its start (prefill-heavy; at TP8 c4 they have drained by 60 s in). `decode` anchors on steady decode: from the warmup start, the first 20 s in which at least 90% of the engines' logged steps replayed FULL CUDA graphs (falling back to 150 s before the measured replay's cap). Each measured-phase turn re-prefills first, so under data parallelism, where one prefilling rank keeps every rank eager, steady decode arrives late and at a time that depends on concurrency |
| `capture_ranks` | `"dp0_tp0"` | Workers whose CUDA graph capture is profiled (`"all"` for every rank) |
| `host_headroom_gib` | `128` | Host memory taken from a CPU KV-offload pool for the profiler's buffers |
| `duration` | last `profiling` delay + 600 s | Cap on the measured replay. Once its last window closes (and the measured phase has run 60 s), the window client SIGINTs aiperf, which exports what it measured and exits zero, so the run does not replay past its windows |

A profiled point holds its node for about 30 minutes (engine start-up and
graph capture are most of it). Its throughput is not a result: windows pause
the engine while they export, and profiled runs tolerate failed requests.

Deviations from the recipe, all recorded in the run's config: vLLM's torch
profiler config, the patch's environment, a raised `VLLM_RPC_TIMEOUT`, and on
CPU-offload recipes an offload pool smaller by `host_headroom_gib`.

## Artifacts

- `profile_<result>`: `infx-profile.tar`, the full record.
  - `torch/`: vLLM's per-rank traces, one per window.
  - `capture/`: the CUDA graph capture trace (full Python stacks).
  - `steps/<rank>.jsonl`: every step's batch composition on that rank.
  - `copies/<rank>.jsonl`: every CPU KV-offload block copy the rank queued
    (direction, blocks, bytes, step, vLLM callers).
  - `graphs/`, `env/`, `windows_conc<c>.jsonl`: graph ordinals, versions and
    config per rank, and the window log.
- `profile_steps_<result>`: the same profile broken down by step
  (`extract.py`).
  - `index.json`: every step file with its kernel count, device span,
    tokens, request count and cudagraph mode.
  - `report.json`: per-trace attribution coverage, graph joins, and the
    kernel-name check.
  - `window<w>/<rank>/step<k>.json.gz` (`dummy<k>` for an idle DP rank's
    forward, `unstepped` for kernels outside any step):

    ```
    {"rank", "window", "kind", "step",
     "summary": {"kernels", "busy_us", "t0_us", "t1_us", "span_us", "sources"},
     "batch":   {"total_tokens", "reqs": [{"req", "scheduled", "computed", "spec",
                                           "new", "prompt_len"}], "dispatch": [...]},
     "kernels": [{"kernel", "cat", "stream", "device", "ts_us", "dur_us",
                  "source": "eager" | "graph" | "offload_copy" | "unattributed",
                  "module_stack": [[qualified name, [[shape, dtype], ...]], ...],
                  "op", "op_chain", "input_dims", "input_types", "concrete_inputs",
                  "launcher", "launcher_callers", "annotations",
                  "graph", "node_pos", "graph_node_id", "grid", "block",
                  "copy": {"store", "blocks", "bytes", "issue_lag_us"}, ...}]}
    ```

## How attribution works

`sitecustomize.py` loads through `PYTHONPATH` and does nothing unless
`INFX_PROF_DIR` is set. Its markers are `record_function` ranges emitted only
while a profiler runs:

- `infx_step#k` / `infx_dummy#k`: a scheduled step and an idle DP rank's
  dummy forward. Each step's batch composition is logged under the same `k`.
- `infx_mod#<qualified name>#<dims>:<dtype>;...` (e.g. `7x7168:bfloat16`): every
  module call and its tensor inputs, through
  global module hooks registered only while a window is open, and only when
  the engine is not torch.compiled.
- `infx_py#<launcher>#<vLLM frame>|...`: the Python entry points that launch
  kernels without a torch op (Triton, TileLang, CuTe DSL, vLLM's DeepGEMM and
  FlashInfer wrappers).
- `infx_graph_capture#n` / `infx_graph_replay#n`: CUDA graph ordinals.

Marker names carry no quotes or backslashes: Kineto writes event names into the
trace JSON unescaped, and the extractor fails on a trace it cannot decode.

An eager kernel joins its launch through the CUDA correlation id. A kernel
replayed from graph `n` carries a `graph node id`: ordered by node id, graph
`n`'s nodes pair with the launches recorded while graph `n` was captured,
which carry that launch's full context. `report.json` counts a graph whose
node count differs from its captured launches as `count_mismatch`, and lists
graph kernels whose op never launched that kernel eagerly.

CPU KV offload (SimpleCPUOffloadConnector) copies blocks with
`cuMemcpyBatchAsync` from the connector's copy thread. Kineto records neither
that driver call nor `record_function` ranges on that thread, so those memcpys
have no CPU launch. Each queued copy is one memcpy on its direction's stream,
run in queue order, and `copies/` logs each one as it is queued. Per direction,
the extractor pairs the memcpys in order with the logged copies issued before
them (with equal bytes), taking the pairing with the least total issue lag.
The step markers put the log's wall clock on the trace clock.
