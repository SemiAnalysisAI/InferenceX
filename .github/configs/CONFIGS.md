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
  seq-len-configs:
  - isl: int
    osl: int
    search-space:
    - { tp: int, conc-start: int, conc-end: int }
    # Optionally, specify 'ep' (expert-parallelism) and 'dp-attn' (data parallel attention)
    - { tp: int, ep: int, dp-attn: bool, conc-start: int, conc-end: int }
    - ...
  - ...
```
Note: while not required, `entry-name` typically takes the format `<INFMAX_MODEL_PREFIX>-<PRECISION>-<GPU>-<FRAMEWORK>`.

The below list describes what each field is:

- `image`: The image used to serve the benchmark, e.g., `vllm/vllm-openai:v0.10.2`
- `model`: The model to server, e.g., `openai/gpt-oss-120b`
- `model-prefix`: The canonical InferenceMAX model prefix reference, i.e., `dsr1` for Deepseek, `gptoss` for gptoss-120b, etc. This value is used to decipher which script in `benchmarks/` should be used in order to launch the benchmark.
- `runner`: This is the runner on which to run the benchmark. This must be a valid runner (key or value) from `runners.yaml`.
- `precision`: The precision to run the benchmark. Again, this is used to find which script to run in `benchmarks/`.
- `framework`: The framework (serving runtime) to serve the benchmark, e.g., `vllm`, `sglang`, `trt`.
- `seq-len-configs`: A list of possible sequence lengths to benchmark. Each entry must have the following fields:
  - `isl`: An integer representing the input sequence length, e.g., `1024`
  - `osl`: An integer representing the output sequence length, e.g., `8192`
  - `search-space`: A list of configurations to run with respective `isl` and `osl`, each entry must be a dict with the following fields:
    - `tp`: An integer representing the tensor parallelism level that the configuration will be served at.
    - `conc-start`: An integer representing the starting level of concurrency e.g., `4`
    - `conc-end`: An integer representing the ending level of concurrency (inclusive) e.g., `128`
    - Note: the step factor between `conc-start` and `conc-end` is 2, so if `conc-start` is 4 and `conc-end` is 128, all concurrencies `4, 8, 16, 32, ..., 128` will be run.
    - (Optional) `ep`: An integer representing the expert parallelism level that the configuration will be served at. Default is 1 (no expert parallelism) when not specified.
    - (Optional) `dp-attn`: A boolean representing whether or not to activate data parallel attention for the configuration. Default is false when not specified.

Notes:
- No extra fields besides the ones listed may be specified, or else the benchmarks will fail to run.
- Setting the fields above, particularly `ep` and `dp-attn`, only guarantee that the respective values will be passed as environment variables to the benchmark scripts! Actually using those environment variables is an implementation detail at the level of the benchmark Bash script.

## Multi-node srt-slurm recipes

Multi-node configs that dispatch via `srt-slurm` (i.e. `srtctl apply -f …`) reference their recipe as a first-class field on the search-space entry:

```yaml
search-space:
- spec-decoding: "mtp"
  conc-list: [1214]
  recipe: "trtllm/b200-fp4/1k1k/mtp/ctx1_gen2_dep8_batch64_eplb0_mtp2.yaml"
  prefill:
    num-worker: 1
    tp: 4
    ep: 4
    dp-attn: true
  decode:
    num-worker: 2
    tp: 8
    ep: 8
    dp-attn: true
```

- `recipe` is a path **relative to `benchmarks/multi_node/srt-slurm-recipes/`** in this repo. The schema validator rejects entries whose recipe file does not exist on disk, so adding a new entry requires upstreaming the recipe yaml here first.
- The path may carry an `:override[N]` / `:override_<name>` suffix to select a named override section inside an sglang-style recipe yaml (e.g. `"dsr1/sglang/b200-fp4/1k1k/disagg/1k1k.yaml:zip_override_mtp_lowlat[0]"`). The launcher strips this suffix before reading the file but passes the full string to `srtctl`.
- `recipe` is optional: multi-node entries that do *not* go through srt-slurm (e.g. dynamo-sglang aggregated topologies that drive their own bash) leave it unset.
- Recipes live under `benchmarks/multi_node/srt-slurm-recipes/` organized as `<model>/<framework>/<hw>-<precision>/<isl><osl>/<agg|disagg>/<stp|mtp>/<recipe-name>.yaml` — e.g. `dsr1/trtllm/b200-fp4/1k1k/disagg/mtp/ctx1_gen2_dep8_batch64_eplb0_mtp2.yaml`. A handful of sglang-style files that carry override sections spanning both stp and mtp are parked one level shallower (the trailing `<stp|mtp>/` segment is omitted). The benchmark template resolves `recipe` to an absolute path and passes it to the launcher as `CONFIG_FILE`, so launchers do not see the relative form.

### Custom-script benchmarking

Recipes are migrating from srt-slurm's bundled `benchmark.type: sa-bench` to `benchmark.type: custom` so the benchmark client lives in this repo (`utils/bench_serving/benchmark_serving.py`) instead of being maintained twice. New shape:

```yaml
container_mounts:
  "$INFMAX_WORKSPACE": "/infmax-workspace"

benchmark:
  type: "custom"
  command: "bash /infmax-workspace/benchmarks/multi_node/srt_bench.sh"
  env:
    PREFILL_GPUS: "4"               # per prefill worker  (filename component)
    DECODE_GPUS: "8"                # per decode worker   (filename component)
    TOTAL_GPUS: "20"                # sum across workers  (filename component)
    # MODEL_NAME: "..."             # only when server's served-model-name
                                    # differs from master-yaml's `model:`
    # USE_CHAT_TEMPLATE: "false"    # only when overriding default (true)
```

`MODEL`, `ISL`, `OSL`, `CONC_LIST`, `DISAGG`, `RANDOM_RANGE_RATIO` are exported by `benchmark-multinode-tmpl.yml` at the workflow step and propagate through the launcher → `srtctl` → `srun` (default `--export=ALL`) → pyxis into the benchmark container, so they don't need to be re-declared in `benchmark.env`. The recipe only carries per-recipe topology knobs (`PREFILL_GPUS`/`DECODE_GPUS`/`TOTAL_GPUS`, used in the result filename) plus the rare overrides (`MODEL_NAME` when the server's served-model-name diverges from `model:`, `USE_CHAT_TEMPLATE: false` for tokenizers that have no chat template, etc.).

`benchmarks/multi_node/srt_bench.sh` is a thin wrapper around `run_benchmark_serving()` in `benchmarks/benchmark_lib.sh` (the same shim every single-node bench script uses). It loops once per concurrency in `$CONC_LIST` and writes results to `/logs/sa-bench_isl_<ISL>_osl_<OSL>/results_concurrency_<N>_gpus_<TOT>_ctx_<P>_gen_<D>.json` so existing launcher result-harvesters pick them up unchanged. Tokenizer is loaded from `/model` — `srtctl`'s `RuntimeContext.create` auto-mounts the model dir at that path in every container, so we don't need any HF Hub egress.

The `container_mounts` block bind-mounts the host-side `$INFMAX_WORKSPACE` (set by the launcher to `$GITHUB_WORKSPACE`) at `/infmax-workspace` inside srt-slurm's benchmark container, so the wrapper and bench client are reachable at known paths. `srtctl` resolves `$INFMAX_WORKSPACE` via `os.path.expandvars` at submission time.

## Runners

The `runners.yaml` config represents the available runners in the repository. The keys are the runner *types* (i.e., the GPUs as well as some specific combinations like `b200-trt`) whereas the value is a list of *runner nodes*. This config is used to verify the master configs.
