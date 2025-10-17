# Search Space Configuration

This file defines which benchmark configurations to run for each model, GPU, and precision combination.

## Quick Start

Add a new configuration by following this pattern:
```yaml
model_name:
  precision:
    gpu_type:
      sequence_length:
        - { tp: [1, 2, 4], conc: { start: 4, end: 64 } }
```

## Field Definitions

### Required Fields

- **`tp`**: Tensor Parallelism (number of GPUs)
  - Single value: `tp: 4`
  - Multiple values: `tp: [2, 4, 8]`

- **`conc`**: Concurrency (number of simultaneous requests)
  - `start`: First value to test
  - `end`: Last value to test
  - `step`: Multiplier (default: 2)
  - Example: `{start: 4, end: 64}` → tests [4, 8, 16, 32, 64]

### Optional Fields

- **`ep`**: Expert Parallelism for MoE models (default: 1)

- **`dp_attention`**: Data Parallel Attention (default: `"false"`)

## Examples

### Basic configuration
```yaml
gptoss:
  fp4:
    h100:
      1k1k:  # 1024 input, 1024 output
        - { tp: [2, 4, 8], conc: { start: 4, end: 64 } }
```
This tests 15 combinations: 3 TP values × 5 concurrency values

### Configuration with optional fields
```yaml
dsr1:
  fp4:
    b200-trt:
      1k1k:
        - { tp: 4, ep: 4, dp_attention: "true", conc: { start: 256, end: 256 } }
```

### Custom step factor
```yaml
llama:
  fp8:
    b200:
      1k8k:
        - { tp: 2, conc: { start: 4, end: 64, step: 4 } }
```
This tests [4, 16, 64] (multiplies by 4 instead of default 2)

## Key Points

1. **Models**: `gptoss`, `llama`, `dsr1`
2. **Precisions**: `fp4`, `fp8`
3. **Sequence lengths**: `1k1k`, `1k8k`, `8k1k` (input×output)
4. Each entry expands to test all combinations of TP and concurrency values
5. Use `-trt` suffix for TensorRT-optimized hardware configs

## Testing Your Changes

Run the flattening script to validate:
```bash
python utils/flatten_matrix.py
```