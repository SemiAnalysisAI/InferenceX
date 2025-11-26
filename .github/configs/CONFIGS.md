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

## Runners

The `runners.yaml` config represents the available runners in the repository. The keys are the runner *types* (i.e., the GPUs as well as some specific combinations like `b200-trt`) whereas the value is a list of *runner nodes*. This config is used to verify the master configs.
