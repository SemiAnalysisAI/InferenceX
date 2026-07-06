# Registering Recipes from srtslurm

For disaggregated multi-node configurations (`dynamo-sglang`, `dynamo-trt`), recipes are stored in the external [srtslurm](https://github.com/NVIDIA/srt-slurm) repository. This doc covers staging those recipes in InferenceX.

## 1. Locate source recipes in srtslurm

```bash
# Example: H200 sglang disagg recipes
ls /path/to/srtslurm/recipes/h200/
# 1k1k/  8k1k/
```

## 2. Recipe structure

Each recipe YAML contains:
- `name`: Recipe identifier
- `model`: Model path/container info
- `resources`: GPU type, prefill/decode node/worker counts
- `backend.sglang_config`: Prefill and decode configuration (tp-size, dp-size, ep-size, dp-attention, etc.)
- `benchmark`: ISL/OSL and concurrency settings

## 3. Add config to nvidia-master.yaml

```yaml
dsr1-fp8-h200-dynamo-sglang:
  image: lmsysorg/sglang:v0.5.8-cu130-runtime
  model: deepseek-ai/DeepSeek-R1-0528
  model-prefix: dsr1
  runner: h200-multinode-slurm
  precision: fp8
  framework: dynamo-sglang
  multinode: true
  disagg: true
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - conc-list: [1, 4, 16, 32, 64, 128, 256, 512]
        prefill:
        num-worker: 1
        tp: 8
        ep: 1
        dp-attn: false
        additional-settings:
        - "CONFIG_FILE=recipes/h200/1k1k/bs128-agg-tp.yaml"
      decode:
        num-worker: 0
        tp: 8
        ep: 1
        dp-attn: false
```

## 4. Field mapping (srtslurm → nvidia-master.yaml)

| srtslurm field | nvidia-master.yaml field |
|----------------|-------------------------|
| `resources.prefill_workers` | `prefill.num-worker` |
| `resources.decode_workers` | `decode.num-worker` |
| `sglang_config.prefill.tp-size` | `prefill.tp` |
| `sglang_config.prefill.ep-size` | `prefill.ep` |
| `sglang_config.prefill.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` (parsed) | `conc-list` |
| Recipe file path | `additional-settings: CONFIG_FILE=...` |

## 5. Common patterns

- **Aggregated (AGG)**: Single node, `num-worker: 1` for prefill, `num-worker: 0` for decode
- **TEP (Tensor-Expert Parallel)**: `dp-attn: false`, `ep: 1`
- **DEP (Data-Expert Parallel)**: `dp-attn: true`, `ep: 8` (typically)
- **Low latency**: More decode workers (e.g., 9), lower concurrencies
- **High throughput**: Fewer decode workers, higher concurrencies

## 6. Add perf-changelog entry

```yaml
- config-keys:
    - dsr1-fp8-h200-dynamo-sglang
  description:
    - "Add DSR1 FP8 H200 Dynamo SGLang disaggregated multinode configuration"
    - "Image: lmsysorg/sglang:v0.5.8-cu130-runtime"
    - "Recipes sourced from srtslurm repo (recipes/h200/)"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

## 7. Validate

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang
```
