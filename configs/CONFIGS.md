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
  # Optional defaults for every search-space entry in this config.
  router: { name: string, version: string }
  kv-p2p-transfer: string
  scenarios:
    fixed-seq-len:
    - isl: int
      osl: int
      search-space:
      - { tp: int, conc-start: int, conc-end: int }
      # Optionally, specify pipeline/expert/data-attention/context parallelism.
      - { tp: int, pp: int, ep: int, dp-attn: bool, dcp-size: int, pcp-size: int, conc-start: int, conc-end: int }
      # Optionally, declare router metadata and the P2P KV transfer engine.
      - tp: int
        router: { name: string, version: string }
        kv-p2p-transfer: string
        conc-start: int
        conc-end: int
      - ...
    - ...
    agentic-coding:  # optional
    - trace-source: string
      search-space:
      - tp: int
        kv-offloading: dram
        kv-offload-backend: { name: string, version: string } # version optional
        conc-start: int
        conc-end: int
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
        pp: 1
        dcp-size: 1
        pcp-size: 2
        ep: 8
        dp-attn: false
      decode:
        hardware: h100
        num-worker: 2
        tp: 8
        pp: 1
        dcp-size: 2
        pcp-size: 1
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
      - (Optional) `router`: Router metadata containing exactly non-empty `name` and `version` strings.
      - (Optional) `kv-p2p-transfer`: Non-empty name of the engine used to move KV state between workers. It is valid only for `multinode: true` configs and does not carry version metadata.
      - (Optional) `pp`: Pipeline parallelism level. Default is 1. It must be a positive integer.
      - (Optional) `dcp-size`: Decode context-parallel size. Default is 1. It must be a positive divisor of `tp`; DCP reuses the TP GPUs.
      - (Optional) `pcp-size`: Prefill context-parallel size. Default is 1. A topology consumes `tp * pp * pcp-size` GPUs per worker; DCP does not add GPUs.
      - For single-node entries, set `pp`, `dcp-size`, and `pcp-size` directly in the search-space entry.
      - For multinode entries, set them independently inside each `prefill` and `decode` worker block. A worker pool allocates `num-worker * tp * pp * pcp-size` GPUs.
  - `agentic-coding`: Agentic trace replay benchmarks using real conversation traces. Each entry must have:
    - `trace-source`: Identifier for the trace dataset to use.
    - `search-space`: Same structure as `fixed-seq-len` search-space entries.

`router` and `kv-p2p-transfer` may be omitted independently. Router metadata
requires non-empty `name` and `version` strings. Its `version` must be the
component's exact release, package or wheel version, or source commit. Do not
copy a container image name or image tag into `version`; container image
references are rejected. `kv-p2p-transfer` is intentionally
name-only and is reserved for `multinode: true` configurations. It may describe
an aggregated multinode topology, but every `disagg: true` config must declare
it either at the top level or in every search-space entry.

Top-level declarations apply to every scenario and search-space entry in that
master config. For `router` and `kv-p2p-transfer`, choose exactly one scope:
declaring a field both at the top level and in any search-space entry is
rejected. Search-space values may differ between entries.

`kv-offload-backend` is separate from peer-to-peer transfer. It requires a
non-empty `name`; `version` is optional because framework-native implementations
such as vLLM's built-in offload backends and SGLang HiCache do not have an
independent component version. Supply `version` for independently versioned
backends such as LMCache or Mooncake. Additional keys and image references in
`version` are rejected.

Agentic duration is not a master YAML field. Matrix generation defaults agentic
jobs to 3600 seconds; reusable workflow callers may override the `duration`
input.

Notes:
- No extra fields besides the ones listed may be specified, or else the benchmarks will fail to run.
- Setting the fields above only guarantees that their values are passed as environment variables to benchmark scripts. Single-node jobs receive `PP_SIZE`, `DCP_SIZE`, and `PCP_SIZE`; multinode jobs receive `PREFILL_PP_SIZE`, `PREFILL_DCP_SIZE`, `PREFILL_PCP_SIZE`, `DECODE_PP_SIZE`, `DECODE_DCP_SIZE`, and `DECODE_PCP_SIZE`. Actually using those variables is an implementation detail of the benchmark Bash script.

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
