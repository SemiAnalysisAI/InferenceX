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
  multinode: bool
  disagg: bool  # optional, multinode only
  # Optional defaults for every search-space entry in this config.
  router: { name: string, version: string }
  kv-p2p-transfer: string  # multinode only
  # Recipe directory under benchmarks/<single|multi>_node/srt-slurm-recipes/.
  srt-recipe-dir: string
  scenarios:
    fixed-seq-len:
    - isl: int
      osl: int
      search-space:
      - { tp: int, conc-start: int, conc-end: int, srt-recipe: string }
      # Optionally, specify pipeline/expert/data-attention/context parallelism.
      - { tp: int, pp: int, ep: int, dp-attn: bool, dcp-size: int, pcp-size: int, conc-start: int, conc-end: int }
      # Optionally, declare router metadata.
      - tp: int
        router: { name: string, version: string }
        conc-start: int
        conc-end: int
      - ...
    - ...
    agentic-coding:  # optional
    - dram-utilization: float  # required by kv-offloading: dram
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
- `model`: The model to serve, e.g., `deepseek-ai/DeepSeek-R1-0528`
- `model-prefix`: The canonical InferenceX model prefix reference, i.e., `dsr1` for DeepSeek-R1 or `qwen3.5` for Qwen3.5. Consult `docs/MODELS.md` for supported model/scenario combinations. The launcher's workload policy rows in [`infx/launch/policy.py`](../infx/launch/policy.py) match on this value.
- `runner`: This is the runner label on which to run the benchmark. This must be a valid key under `labels` in `runners.yaml`.
  Agentic configs must use an exact `cluster:<name>` runner label, not a broad
  SKU or capacity label, so every search-space point runs on the same hardware
  fleet.
- `precision`: The precision to run the benchmark. Launcher workload policy rows may also match on it.
- `framework`: The framework (serving runtime) to serve the benchmark, e.g., `vllm`, `sglang`, `trt`.
- `multinode`: Required. `true` selects the multinode schema with `prefill`/`decode` or `worker` blocks, and `false` the single-node schema with `tp` entries.
- `disagg`: Enables disaggregated serving and may only be `true` when
  `multinode` is also `true`.
- `hardware`: Optional metadata within each `prefill` and `decode` worker block
  for heterogeneous disaggregated deployments. If one worker declares a GPU
  SKU, the other must also declare one. Omit both fields for homogeneous
  hardware. These values flow into aggregate results but do not affect runner
  scheduling.
- `scenarios`: A dictionary of benchmark scenario types. At least one must be specified. Currently supported:
  - `fixed-seq-len`: Fixed input/output sequence length benchmarks. Each entry must have:
    - `isl`: An integer representing the input sequence length, e.g., `8192`
    - `osl`: An integer representing the output sequence length, e.g., `1024`
    - PR sweeps run only `isl`/`osl` pairs 1024/1024 and 8192/1024; the changelog planner rejects any other pair.
    - `search-space`: A list of configurations to run with respective `isl` and `osl`, each entry must be a dict with the following fields:
      - `tp`: A positive integer representing the tensor parallelism level that the configuration will be served at.
      - `conc-start`: An integer representing the starting level of concurrency e.g., `4`
      - `conc-end`: An integer representing the ending level of concurrency (inclusive) e.g., `128`
      - Note: the step factor between `conc-start` and `conc-end` is 2, so if `conc-start` is 4 and `conc-end` is 128, all concurrencies `4, 8, 16, 32, ..., 128` will be run.
      - (Optional) `ep`: An integer representing the expert parallelism level that the configuration will be served at. Default is 1 (no expert parallelism) when not specified.
      - (Optional) `dp-attn`: A boolean representing whether or not to activate data parallel attention for the configuration. Default is false when not specified.
      - (Optional) `router`: Router metadata containing exactly non-empty `name` and `version` strings.
      - (Optional) `kv-p2p-transfer`: Non-empty name of the engine used to move KV state between workers. It is valid only for `multinode: true` configs and does not carry version metadata.
      - (Optional) `pp`: Pipeline parallelism level. Default is 1. It must be a positive integer.
      - (Optional) `dcp-size`: Decode context-parallel size. Default is 1. It must be a positive divisor of `tp`. DCP reuses the TP GPUs.
      - (Optional) `pcp-size`: Prefill context-parallel size. Default is 1. A topology consumes `tp * pp * pcp-size` GPUs per worker. DCP does not add GPUs.
      - For single-node entries, set `pp`, `dcp-size`, and `pcp-size` directly in the search-space entry.
      - For multinode entries, set them independently inside each `prefill` and `decode` worker block. A worker pool allocates `num-worker * tp * pp * pcp-size` GPUs.
  - `agentic-coding`: Agentic trace replay benchmarks using real conversation traces. Each entry has:
    - (Optional) `dram-utilization`: The fraction of each node's available CPU DRAM that DRAM KV offloading may use, greater than 0 and at most 1. It is required when any search-space entry sets `kv-offloading: dram`.
    - `search-space`: Same structure as `fixed-seq-len` search-space entries, plus `kv-offloading` (`none` or `dram`, required for single-node entries) and the optional `kv-offload-backend`.

`router` and `kv-p2p-transfer` may be omitted independently. Router metadata
requires non-empty `name` and `version` strings. Its `version` must be the
component's exact release, package or wheel version, or source commit. Do not
copy a container image name or image tag into `version`. Container image
references are rejected. `kv-p2p-transfer` is intentionally
name-only and is reserved for `multinode: true` configurations. It may describe
an aggregated multinode topology, but every `disagg: true` config must declare
it either at the top level or in every search-space entry.

Top-level declarations apply to every scenario and search-space entry in that
master config. For `router` and `kv-p2p-transfer`, choose exactly one scope:
declaring a field both at the top level and in any search-space entry is
rejected. Search-space values may differ between entries.

`kv-offload-backend` is separate from peer-to-peer transfer. It requires a
non-empty `name`. Its `version` is optional because framework-native implementations
such as vLLM's built-in offload backends and SGLang HiCache do not have an
independent component version. Supply `version` for independently versioned
backends such as LMCache or Mooncake. Additional keys and image references in
`version` are rejected.

Most `router` and `kv-offload-backend` versions label what the image ships. When a
repository setup script installs the component instead, `version` is required and is
the installed version; the launcher binds or checks it
([recipe fragments](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md#recipe-fragments)).

Agentic duration is not a master YAML field. Matrix generation defaults agentic
jobs to 3600 seconds. Reusable workflow callers may override the `duration`
input.

Notes:
- The fields above are the common ones, not the full schema. The Pydantic models in [`infx/matrix/validation.py`](../infx/matrix/validation.py) are the authoritative contract; they also accept fields such as `spec-decoding`, `num-nodes`, `require-power` and `power`, and they reject any field they do not define, which fails matrix generation.
- A multinode search-space entry with `power: true` measures DCGM power: its rows carry `power: true`, the launcher merges the shared telemetry block into the recipe and rejects lanes without a power rule.
- srt-slurm entries require `srt-recipe-dir`. Every row requires `srt-recipe: <file>[:<selector>]` under it (selector: `base`, `override_<name>` or `zip_override_<name>[<i>]`). Multinode rows may add `eval-srt-recipe`. Validation rejects missing files or variants, and `CONFIG_FILE`/`EVAL_CONFIG_FILE` settings. Only `llmd-vllm` keeps `CONFIG_FILE`. Generated rows carry the path from `inferencex-e2e/`.
- Setting the fields above only guarantees that their values are passed as environment variables to benchmark scripts. Single-node jobs receive `PP_SIZE`, `DCP_SIZE`, and `PCP_SIZE`. Multinode jobs receive `PREFILL_PP_SIZE`, `PREFILL_DCP_SIZE`, `PREFILL_PCP_SIZE`, `DECODE_PP_SIZE`, `DECODE_DCP_SIZE`, and `DECODE_PCP_SIZE`. Actually using those variables is an implementation detail of the benchmark Bash script.

### DWDP prefill groups

Multi-node prefill workers may set `dwdp-size` to the number of ranks in a
distributed weight-data parallel group. It defaults to 1 (disabled). For DWDP,
`tp`, `ep`, `pp`, `dcp-size`, and `pcp-size` must all be 1 and `dp-attn` must be
false. `num-worker` counts groups, not ranks: two workers with `dwdp-size: 8`
allocate 16 prefill GPUs. Decode workers cannot enable DWDP.

The native TensorRT-LLM recipe must enable the corresponding DWDP mode and match
the declared group layout. This field records topology and sizes allocations;
it does not inject backend launch flags or enable a benchmark recipe. Results
carry `prefill_dwdp_size` when greater than 1; TP remains 1. Existing result
identities are unchanged when DWDP is disabled.

See NVIDIA's [DWDP implementation notes](https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.md).

## Runners

`runners.yaml` holds the schedulable runner labels and the static facts of each
physical cluster:

```yaml
labels:
  cluster:b300-dsxe:
    - b300-dsxe_00
    - b300-dsxe_01

clusters:
  b300-dsxe:
    gpus-per-node: 8
    available-cpu-dram-mib: 3977095
    arch: x86_64
    models:
      entries:
        Kimi-K3: {root: scratch, dir: Kimi-K3}
    scheduler: slurm
    slurm:
      partition: batch_1
      account: benchmark
      exclusive: true
      volumes:
        scratch: {path: /scratch/models, visibility: node-local}
      squash: {dir: /data/home/sa-gha-runner/squash, visibility: shared, import: compute}
```

The Pydantic models in [`infx/clusters/`](../infx/clusters) are the authoritative
schema; unknown keys fail.

- `labels` maps each schedulable label to concrete runner names. Every runner is in
  exactly one `cluster:<id>` label, and every such label has a `clusters.<id>` record;
  `python -m infx.launch` resolves its cluster from `RUNNER_NAME` this way.
- Matrix generation reads `gpus-per-node` and `available-cpu-dram-mib` from the cluster a
  master config's `cluster:<id>` label names; broad SKU labels use the GPU family when its
  clusters agree. Agentic master configs must use a `cluster:<id>` label; their DRAM
  KV-offload matrices combine both fields with `dram-utilization` into per-node caps:
  `total-cpu-dram-gb` for each node a single-node, aggregate, or prefill server occupies, and
  `decode-cpu-dram-gb` (omitted when 0) for each node a decode server occupies. Each is the
  node's share held by that server's GPUs on it; a server spanning nodes gets a full node's
  share on each. Servers must tile their nodes, and per-worker `hardware` is not supported.
  Results record them as `allocated_cpu_dram_gb` and `decode_allocated_cpu_dram_gb`.
- `available-cpu-dram-mib` is the host DRAM a benchmark job can use on a node: the lower of
  Slurm's `RealMemory - MemSpecLimit` and `MemTotal` less 16 GiB for the OS, over the
  cluster's GPU nodes; omit it where unmeasured. With `slurm.srt-slurm.single-node-exclusive`
  (the default), a single-node srt-slurm job requests `--mem=0`, all the memory Slurm has,
  so its host KV pools must fit the node. Otherwise the job and its srun step get the serving
  GPUs' share, `available-cpu-dram-mib / gpus-per-node * GPU_COUNT` MiB, which the pools and
  server processes must fit; Slurm rejects a share above what it can allocate.
- `env` is the workload environment of every launch on the cluster. Like
  `slurm.srt-slurm.env`, it overrides the runner's own environment but never a name the
  point's additional-settings set. Values cannot contain commas: Slurm hands them to jobs
  in an `srun --export` list.
- `models.entries` keys pre-staged checkpoints by directory name, each with its `root`
  volume and `dir`. The optional `models.download-root` names the shared volume that
  receives missing checkpoints as `<root>/<HF basename>`.
- `scheduler` names a scheduler registered in `infx.clusters.SCHEDULERS`; the
  sub-record of that name holds its settings, and records for other schedulers are
  rejected. Its `volumes` name the checkpoint roots and caches (`hf-home`,
  `hf-hub-cache`, `shared-hf-hub-cache`, `aiperf-cache`, `dynamo-wheels`),
  each `shared` (the default) or `node-local`.
- `slurm:` ([`infx/clusters/slurm.py`](../infx/clusters/slurm.py)) holds the partition,
  account, exclusivity, GRES, excluded nodes and extra `srun`/`salloc` options; its
  volumes are host `path`s that jobs see at the same place.
- Optional `slurm.partitions` lists independent allocation partitions inside one
  cluster. When nonempty, every runner in that cluster must have exactly one
  `partition:<name>` inventory label naming an allowed partition, and the default
  `slurm.partition` must be in the allowlist. The launcher resolves its actual
  partition from the selected anchor's `RUNNER_NAME`, without changing the cluster
  identity, model paths, or workload policies. Inventory labels must match GitHub.
  The dashboard controller must support partition-aware leases before enabling
  these runners; it must never combine capacity across partition labels.
- `slurm.squash` is the Pyxis squash cache: `dir`, `visibility`, `lock-timeout-s`,
  `key-style` (`underscore`, `plus` or `plus-strip-nvcr`) and `import`: `submit-host`
  (on the launching host), `compute` (once on one compute node), `all-nodes` (on every
  node of the job), `pre-staged` (only validate what operators staged) or `unchecked`
  (use the squash path untouched). Imports lock `<squash>.lock`; `lock-file: locks-dir`
  locks `<dir>/.locks/<key>.lock`. Jobs srtctl submits allocate themselves, so nothing is imported inside them:
  they reuse a valid squash or let Pyxis import the registry image, unless
  `single-node-import: true` or `multi-node-import` (the default) imports first.
  `framework-dirs.<framework>` (and its `model-prefixes.<prefix>`) and
  `helper-dirs.nginx`/`helper-dirs.dcgm-exporter` give multi-node images another `dir`,
  `key-style` or `import`; unset fields are the cache's, and a model prefix's location
  replaces its framework's. `import-step-args` adds options to a standalone `compute`
  import step. Without `slurm.squash`, every job starts from the registry image, which
  Pyxis imports inside the job.
- `slurm.srt-slurm` is the cluster's srt-slurm profile: `volume-mounts` maps
  volumes every job mounts to container paths, `mounts` does the same for host
  paths outside the volumes (devices), `agentic-volume-mounts` adds world-writable
  volumes for multi-node AgentX jobs, `power-exporter-port` is the DCGM exporter port
  of power points, `env` is the environment of the srt-slurm
  launch, `container-aliases` and `nginx-aliases` name the recipe containers that
  resolve to the main image and to the staged nginx, and `outputs`,
  `shared-run-root` and `uv-cache-root` are the directories the srt-slurm launcher
  itself uses.
- `slurm.srt-slurm.fabric` holds the hosts' network facts that recipes take as
  `'@fabric.<name>'` values: the lists `rdma-devices` (the RDMA HCAs, as NCCL, Mooncake,
  SGLang `disaggregation-ib-device` and `IBDEVICES` read them), `rdma-device-ports` (the
  HCAs as `device:port`, as UCX reads them) and `socket-ifname` (NCCL and Gloo), rendered
  comma-joined, and the numbers `rdma-gid-index` and `rdma-tc` (the RDMA GID index and
  traffic class). Fields name cluster facts, not the software that reads them; a recipe
  maps one onto its engine's setting, as in `MC_GID_INDEX: '@fabric.rdma-gid-index'`.
  The cluster's own `env` and `srt-slurm.host-setup.env` values may take a field the
  same way; they resolve when the record loads, and `env` still rejects the commas of a
  list.
- `slurm.srt-slurm.extra.default_gpu_exporter` is the cluster's native power exporter.
  Native single-node throughput and AgentX jobs inherit it when the bound recipe
  enables telemetry without selecting its own exporter; eval-only jobs collect no power.
  A cluster without the block stops preparation before the benchmark starts. An invalid
  measurement fails the job under `REQUIRE_POWER=1` and otherwise records an invalid
  verdict without energy metrics. NVIDIA clusters run `dcgm-exporter`; AMD clusters use
  srt-slurm's
  [`kind: custom` schema](https://github.com/NVIDIA/srt-slurm/blob/ae3d6877c46f3147e3a5dafa46ee4cd89322e52f/docs/power-telemetry.md#gpu-exporter-labels-and-metrics),
  which requires `gpu_labels` and `gpu_metrics` and rejects unknown keys such as the
  former `power_profile`. The AMD exporter reads
  [`runners/srt-slurm/exporters/amd-power.json`](../runners/srt-slurm/exporters/amd-power.json);
  [`runners/srt-slurm/patches/README.md`](../runners/srt-slurm/patches/README.md) lists
  the srt-slurm patches native power collection relies on.
- `slurm.srt-slurm.workspace-mounts` maps files relative to `inferencex-e2e/` into
  job containers. The renderer resolves each source against the job checkout;
  host-absolute paths belong in `mounts`, and volume aliases in `volume-mounts`.
