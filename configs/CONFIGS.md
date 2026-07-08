# Configs

The config files in this directory are meant to be a "source of truth" for what benchmark configurations can/should be run. As such, they must follow a precise format which is described below.

## Master Configs (AMD, NVIDIA, etc.)

```yaml
entry-name:
  image: string
  model: string
  model-prefix: string
  runner: string
  precision: string
  framework: string
  scenarios:
    fixed-seq-len:
    - isl: int
      osl: int
      search-space:
      - { tp: int, conc-start: int, conc-end: int }
      # Optionally, specify expert/data-parallel attention and context-parallel sizes
      - { tp: int, ep: int, dp-attn: bool, dcp-size: int, pcp-size: int, conc-start: int, conc-end: int }
      - ...
    - ...
    agentic-coding:  # optional
    - trace-source: string
      search-space:
      - { tp: int, conc-start: int, conc-end: int }
      - ...
```

Heterogeneous disaggregated search-space entries declare hardware on each
worker pool. Omit both `hardware` fields for homogeneous hardware:

```yaml
multinode: true
disagg: true
scenarios:
  fixed-seq-len:
  - isl: 1024
    osl: 1024
    search-space:
    - conc-list: [64]
      prefill:
        hardware: b200
        num-worker: 1
        tp: 8
        ep: 8
        dp-attn: false
      decode:
        hardware: h100
        num-worker: 2
        tp: 8
        ep: 8
        dp-attn: false
```

Note: while not required, `entry-name` typically takes the format `<INFMAX_MODEL_PREFIX>-<PRECISION>-<GPU>-<FRAMEWORK>`.

The below list describes what each field is:

- `image`: The image used to serve the benchmark, e.g., `vllm/vllm-openai:v0.10.2`
- `model`: The model to server, e.g., `openai/gpt-oss-120b`
- `model-prefix`: The canonical InferenceMAX model prefix reference, i.e., `dsr1` for Deepseek, `gptoss` for gptoss-120b, etc. This value is used to decipher which script in `benchmarks/` should be used in order to launch the benchmark.
- `runner`: This is the runner label on which to run the benchmark. This must be a valid key under `labels` in `runners.yaml`.
  Agentic configs must use an exact `cluster:<name>` runner label, not a broad
  SKU or capacity label, so every search-space point runs on the same hardware
  fleet.
- `precision`: The precision to run the benchmark. Again, this is used to find which script to run in `benchmarks/`.
- `framework`: The framework (serving runtime) to serve the benchmark, e.g., `vllm`, `sglang`, `trt`.
- `disagg`: Enables disaggregated serving and may only be `true` when
  `multinode` is also `true`.
- `hardware`: Optional metadata within each `prefill` and `decode` worker block
  for heterogeneous disaggregated deployments. If one worker declares a GPU
  SKU, the other must also declare one. Omit both fields for homogeneous
  hardware. These values flow into aggregate results but do not affect runner
  scheduling.
- `scenarios`: A dictionary of benchmark scenario types. At least one must be specified. Currently supported:
  - `fixed-seq-len`: Fixed input/output sequence length benchmarks. Each entry must have:
    - `isl`: An integer representing the input sequence length, e.g., `1024`
    - `osl`: An integer representing the output sequence length, e.g., `8192`
    - `search-space`: A list of configurations to run with respective `isl` and `osl`, each entry must be a dict with the following fields:
      - `tp`: An integer representing the tensor parallelism level that the configuration will be served at.
      - `conc-start`: An integer representing the starting level of concurrency e.g., `4`
      - `conc-end`: An integer representing the ending level of concurrency (inclusive) e.g., `128`
      - Note: the step factor between `conc-start` and `conc-end` is 2, so if `conc-start` is 4 and `conc-end` is 128, all concurrencies `4, 8, 16, 32, ..., 128` will be run.
      - (Optional) `ep`: An integer representing the expert parallelism level that the configuration will be served at. Default is 1 (no expert parallelism) when not specified.
      - (Optional) `dp-attn`: A boolean representing whether or not to activate data parallel attention for the configuration. Default is false when not specified.
      - (Optional) `dcp-size`: Decode context-parallel size. Default is 1. It must be a positive divisor of `tp`; DCP reuses the TP GPUs.
      - (Optional) `pcp-size`: Prefill context-parallel size. Default is 1. A single-node job allocates `tp * pcp-size` GPUs.
      - `dcp-size` and `pcp-size` are single-node fields. They are not accepted inside multinode `prefill` or `decode` worker blocks.
  - `agentic-coding`: Agentic trace replay benchmarks using real conversation traces. Each entry must have:
    - `trace-source`: Identifier for the trace dataset to use.
    - `search-space`: Same structure as `fixed-seq-len` search-space entries.

Agentic duration is not a master YAML field. Matrix generation defaults agentic
jobs to 3600 seconds; reusable workflow callers may override the `duration`
input.

Notes:
- No extra fields besides the ones listed may be specified, or else the benchmarks will fail to run.
- Setting the fields above only guarantees that their values are passed as environment variables to benchmark scripts (`ep` as `EP_SIZE`, `dp-attn` as `DP_ATTENTION`, `dcp-size` as `DCP_SIZE`, and `pcp-size` as `PCP_SIZE`). Actually using those variables is an implementation detail of the benchmark Bash script.

## Runners

The `runners.yaml` config represents available runner labels and reusable
hardware facts in the repository. It has two top-level sections:

```yaml
labels:
  cluster:b300-nv:
    - b300-nv_01
    - b300-nv_02

hardware:
  cluster:b300-nv:
    available-cpu-dram-mib: 2964436
    gpus-per-node: 8
```

`labels` maps a schedulable runner label to the concrete runner node names that
can satisfy that label. `hardware` maps hardware or fleet keys to host resource
facts. Matrix generation reads the `hardware` entry whose key matches the
master config's `runner` label when a benchmark needs derived hardware facts.
Use `cluster:<name>` labels for hardware metadata that depends on an exact
cluster/fleet rather than a broad SKU label. Agentic master configs must use a
`cluster:<name>` runner label.
`available-cpu-dram-mib` is the host CPU DRAM available to benchmark jobs, in
MiB. Agentic DRAM KV-offload matrices combine it with `gpus-per-node` and the
master config's `dram-utilization` to emit `total-cpu-dram-gb` for benchmark
templates.

### Model path registry (`model-paths`)

`runners.yaml` also carries the per-cluster registry of staged model
checkpoints:

```yaml
model-paths:
  cluster:gb300-nv:
    - model-prefix: dsv4
      precision: fp4
      framework: dynamo-trt   # optional; omit to match any framework
      path: /scratch/models/DeepSeek-V4-Pro
      alias: deepseek-ai/DeepSeek-V4-Pro   # srt-slurm model_paths key
      served-model-name: deepseek-v4-pro   # optional
    - model-prefix: dsv4
      precision: fp4
      path-candidates:                      # probe in order, first existing
        - /data/models/dsv4-pro             # dir wins; else first entry
        - /data/models/DeepSeek-V4-Pro
      alias: deepseek-v4-pro
```

Runner launch scripts resolve `MODEL_PATH` / `SRT_SLURM_MODEL_PREFIX` /
`SERVED_MODEL_NAME` from this section via
`runners/lib/multinode.sh:infx_resolve_model_paths` — model staging paths
must live here, not in if/elif ladders inside runner bash. Entries are
matched first-to-last on `model-prefix` + `precision` (+ `framework` when
present), so framework-specific rows go above generic ones. Staging a new
checkpoint on a cluster is a one-line registry change. See
`benchmarks/multi_node/srt_slurm/README.md` for how this feeds the
multinode contract.
