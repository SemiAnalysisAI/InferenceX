# Practical Guide: Adapting Workflows for Matrix Batching

## When Do You Need This?

You need batching when a single model-prefix generates more than 256 configurations. This can happen when:
- Adding many new model variants
- Expanding the concurrency search space
- Adding more sequence length configurations
- Testing across many runner types

## Quick Start: Check If You Need Batching

Before modifying a workflow, check how many configs will be generated:

```bash
python3 utils/matrix-logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
                .github/configs/amd-master.yaml \
  --seq-lens 1k1k \
  --model-prefix dsr1 | jq 'length'
```

If the output is > 256, you need batching.

## Solution 1: Simple Batching (Recommended for Most Cases)

If your model-prefix generates between 256-512 configs, split it into 2 batches:

### Before (Original):
```yaml
jobs:
  get-model-configs:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - id: get-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
            --seq-lens 1k1k \
            --model-prefix mymodel)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark:
    needs: get-model-configs
    uses: ./.github/workflows/benchmark-tmpl.yml
    strategy:
      matrix:
        config: ${{ fromJson(needs.get-model-configs.outputs.search-space-config) }}
```

### After (With Batching):
```yaml
jobs:
  # Batch 0
  get-model-configs-batch-0:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - id: get-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
            --seq-lens 1k1k \
            --model-prefix mymodel \
            --batch-index 0)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark-batch-0:
    needs: get-model-configs-batch-0
    uses: ./.github/workflows/benchmark-tmpl.yml
    name: mymodel 1k1k batch-0 /
    strategy:
      matrix:
        config: ${{ fromJson(needs.get-model-configs-batch-0.outputs.search-space-config) }}
    # ... rest of with: parameters

  # Batch 1
  get-model-configs-batch-1:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - id: get-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
            --seq-lens 1k1k \
            --model-prefix mymodel \
            --batch-index 1)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark-batch-1:
    needs: get-model-configs-batch-1
    uses: ./.github/workflows/benchmark-tmpl.yml
    name: mymodel 1k1k batch-1 /
    strategy:
      matrix:
        config: ${{ fromJson(needs.get-model-configs-batch-1.outputs.search-space-config) }}
    # ... rest of with: parameters

  # Collect results from both batches
  collect-results:
    needs: [benchmark-batch-0, benchmark-batch-1]
    if: ${{ always() }}
    uses: ./.github/workflows/collect-results.yml
    with:
      exp-name: "mymodel_1k1k"
```

## Solution 2: Split by Additional Criteria

Instead of batching, you can split configurations by other dimensions:

### Split by Framework
```yaml
jobs:
  get-vllm-configs:
    # ... get configs with --framework vllm
  
  get-trt-configs:
    # ... get configs with --framework trt
  
  benchmark-vllm:
    # ... benchmark vllm configs
  
  benchmark-trt:
    # ... benchmark trt configs
```

### Split by Precision
```yaml
jobs:
  get-fp8-configs:
    # ... get configs with --precision fp8
  
  get-fp4-configs:
    # ... get configs with --precision fp4
```

## Solution 3: Reduce Configuration Space

If you're hitting the 256 limit, consider:

1. **Use test-mode**: Test with highest TP and lowest concurrency only
   ```bash
   --test-mode
   ```

2. **Filter by specific sequence lengths**:
   ```bash
   --seq-lens 1k1k  # Only test one sequence length
   ```

3. **Use larger step-size**: Increase concurrency step size
   ```bash
   --step-size 4  # Instead of default 2
   ```

4. **Filter by runner-type**: Test on specific hardware only
   ```bash
   --runner-type h200
   ```

## Determining Number of Batches Needed

Use `--get-batch-count` to programmatically determine how many batches you need:

```bash
BATCH_COUNT=$(python3 utils/matrix-logic/generate_sweep_configs.py \
  full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --seq-lens 1k1k \
  --model-prefix mymodel \
  --get-batch-count)
echo "Need $BATCH_COUNT batches"
```

## Real Example: Adapting full-sweep-1k1k-scheduler.yml

If dsr1 configs grow beyond 256, modify the workflow like this:

```yaml
jobs:
  # Check if batching is needed (optional diagnostic step)
  check-dsr1-count:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - run: |
          pip install pydantic
          COUNT=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
                          ${GITHUB_WORKSPACE}/.github/configs/amd-master.yaml \
            --seq-lens 1k1k \
            --model-prefix dsr1 | jq 'length')
          echo "Total dsr1 configs: $COUNT"
          if [ $COUNT -gt 256 ]; then
            echo "⚠️  Warning: dsr1 has $COUNT configs (>256). Using batching."
          fi

  # Batch 0: First 256 configs
  get-dsr1-configs-batch-0:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-dsr1-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - id: get-dsr1-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
                          ${GITHUB_WORKSPACE}/.github/configs/amd-master.yaml \
            --seq-lens 1k1k \
            --model-prefix dsr1 \
            --batch-index 0)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark-dsr1-batch-0:
    needs: get-dsr1-configs-batch-0
    uses: ./.github/workflows/benchmark-tmpl.yml
    name: dsr1 1k1k batch-0 /
    strategy:
      fail-fast: false
      matrix:
        config: ${{ fromJson(needs.get-dsr1-configs-batch-0.outputs.search-space-config) }}
    secrets: inherit
    with:
      exp-name: "dsr1_1k1k"
      # ... rest of parameters

  # Batch 1: Next 256 configs (if needed)
  get-dsr1-configs-batch-1:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-dsr1-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - id: get-dsr1-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 ${GITHUB_WORKSPACE}/utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files ${GITHUB_WORKSPACE}/.github/configs/nvidia-master.yaml \
                          ${GITHUB_WORKSPACE}/.github/configs/amd-master.yaml \
            --seq-lens 1k1k \
            --model-prefix dsr1 \
            --batch-index 1)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark-dsr1-batch-1:
    needs: get-dsr1-configs-batch-1
    uses: ./.github/workflows/benchmark-tmpl.yml
    name: dsr1 1k1k batch-1 /
    strategy:
      fail-fast: false
      matrix:
        config: ${{ fromJson(needs.get-dsr1-configs-batch-1.outputs.search-space-config) }}
    secrets: inherit
    with:
      exp-name: "dsr1_1k1k"
      # ... rest of parameters

  # Update collect-results to wait for all batches
  collect-dsr1-results:
    needs: [benchmark-dsr1-batch-0, benchmark-dsr1-batch-1, benchmark-gb200]
    if: ${{ always() }}
    uses: ./.github/workflows/collect-results.yml
    secrets: inherit
    with:
      exp-name: "dsr1_1k1k"
```

## Best Practices

1. **Monitor config counts**: Regularly check how many configs are generated
   ```bash
   python3 utils/matrix-logic/generate_sweep_configs.py full-sweep \
     --config-files ... --model-prefix ... | jq 'length'
   ```

2. **Use batch names**: Add "batch-0", "batch-1" to job names for clarity

3. **Keep batches under 256**: Default batch size is 256, which is the GitHub limit

4. **Test locally first**: Verify batching works before pushing to GitHub
   ```bash
   # Get total count
   python3 generate_sweep_configs.py ... --get-batch-count
   
   # Get each batch
   python3 generate_sweep_configs.py ... --batch-index 0
   python3 generate_sweep_configs.py ... --batch-index 1
   ```

5. **Document splits**: Add comments explaining why batching is needed

## Troubleshooting

### Error: "Invalid batch-index X. Valid range is 0 to Y"
- The batch index is out of range
- Run with `--get-batch-count` to see valid range

### All configs go to batch 0
- Total configs is under 256, no batching needed
- This is normal and expected behavior

### Missing configs between batches
- Verify each batch size: `| jq 'length'`
- Sum should equal total: batch-0 + batch-1 + ... = total

### GitHub Actions workflow fails with matrix error
- Ensure each batch is under 256 configs
- Check that the output is valid JSON
