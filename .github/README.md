# How to Test Workflows

In order to test configurations described in `.github/configs`, the primary workflow file used is `.github/workflows/e2e-tests.yml`. As input, this workflow takes in the CLI arguments for the `utils/matrix-logic/generate_sweep_configs.py` script. The usage for this script is shown below:

```
usage: generate_sweep_configs.py [-h] {full-sweep,test-config,runner-model-sweep,runner-sweep,custom} ...

Generate benchmark configurations from YAML config files

positional arguments:
  {full-sweep,test-config,runner-model-sweep,runner-sweep,custom}
                        Available commands
    full-sweep          Generate full sweep configurations with optional filtering by model, precision, framework, runner type, and sequence lengths
    test-config         Given a config key, run that configuration as specified. Optionally specify --test-mode to only run one parallelism-concurrency pair for the config.
    runner-model-sweep  Given a runner type, find all configurations matching the type, and     run that configuration on all individual runner nodes for the specified runner type. This is meant to validate
                        that all runner nodes work on all configurations for a runner type. For instance, to validate that all configs that specify an h200 runner successfully run across all h200 runner
                        nodes.
    runner-sweep        Given a model (and optionally a precision and framework), find all configurations matching the inputs, and run those configurations across all compatible runner nodes. This is
                        meant to validate all runner nodes that should run a particular model can. For instance, this should be used to validate that all runners nodes that should run gptoss-120b
                        actually do so successfully.
    custom              Enter custom values

options:
  -h, --help            show this help message and exit
```

Instead of explaining each command at a high level, let's just walk through some common testing scenarios and describe how to run them.

**Scenario 1**: I want to change increase the concurrency from 128 to 256 in the 1k1k scenario for the `dsr1-fp4-b200-sglang` config (from `.github/configs/nvidia-master.yaml`) and then test it.

Go to the GitHub Actions UI, click on the `End-to-End Tests` workflow, and enter the text following command as the text input:
```
test-config --key dsr1-fp4-b200-sglang --seq-len 1k1k --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

Workflow Run Example: https://github.com/InferenceMAX/InferenceMAX/actions/runs/18986046399

If we wanted to also test 1k8k or 8k1k scenarios, we would simply append `1k8k` or `8k1k` to `--seq-len`, respectively.

Further, if we wanted to run that config on *one specific* runner node, we could specify that by appending `--runner-node` to the argument list. Note that if the specified runner node is not compatible with the specified config key (as dictated by `.github/configs/runners.yaml`), then the workflow will error:

```
test-config --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml --key dsr1-fp4-b200-sglang --seq-len 1k1k --runner-node mi300x-amd_0

ValueError: Runner node 'mi300x-amd_0' is not compatible with config 'dsr1-fp4-b200-sglang' which runs on runner type 'b200'. Available runner nodes for this config are 'b200-nb_0, b200-nb_1, b200-nvd_0, b200-nvd_1, b200-nvd_2, b200-nvd_3, b200-tg_0'.
```

Workflow Run Example: https://github.com/InferenceMAX/InferenceMAX/actions/runs/18986053019/job/54229839736

**Scenario 2**: I just made a change to the `benchmarks/dsr1_fp8_b200_docker.sh` and I need to verify that these changes work across all B200 runners.

Go to the GitHub Actions UI, click on the `End-to-End Tests` workflow, and enter the text following command as the text input:
```
runner-sweep --runner-type b200 --model-prefix dsr1 --precision fp8 --config-files .github/configs/amd-master.yaml .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

Workflow Run Example: https://github.com/InferenceMAX/InferenceMAX/actions/runs/18986283169

This will run a test (just the highest available parallelism and lowest available concurrency) for each B200 runner node for each Deepseek config that runs on B200 with fp8 precision. I.e., this can be used to "sweep" across runners for a particular model to test that all runners still work with changes that have been made.

**Scenario 3**: I just upgraded the CUDA drivers on all H200 runners and need to verify that all models that use H200 still work correctly across all H200 nodes.

Go to the GitHub Actions UI, click on the `End-to-End Tests` workflow, and enter the following command as the text input:
```
runner-model-sweep --runner-type h200 --config-files .github/configs/amd-master.yaml .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

Workflow Run Example: https://github.com/InferenceMAX/InferenceMAX/actions/runs/18986292917

This will run a test (just the highest available parallelism and lowest available concurrency) for each configuration that specifies the `h200` runner type, across all H200 runner nodes defined in `.github/configs/runners.yaml`.

For example, if you have configs `dsr1-fp8-h200-sglang`, `dsr1-fp8-h200-trt`, and `gptoss-fp4-h200-vllm` that all use `runner: h200`, and you have 8 H200 nodes (`h200-cw_0`, `h200-cw_1`, etc.), this will run all 3 configs on all 8 nodes (24 total test runs).

This is particularly useful when:
- You've made infrastructure changes to a specific runner type (driver updates, system configuration, Docker setup)
- You've added new runner nodes and want to validate they work with all existing model configurations
- You want to verify that all models remain compatible with a specific GPU type after system updates

**Key difference from Scenario 2**:
- `runner-sweep`: Fix a **model**, sweep across runners → "Does this model work on all its runners?"
- `runner-model-sweep`: Fix a **runner type**, sweep across models → "Do all models work on this runner type?"

## Additional Use Cases with `full-sweep`

The `full-sweep` command supports multiple filters that can be combined for targeted testing:

**Test all gptoss configurations on B200 with 1k1k sequence lengths:**
```
full-sweep --model-prefix gptoss --runner-type b200 --seq-lens 1k1k --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

**Test all fp8 precision configs across all runners for 1k8k workloads:**
```
full-sweep --precision fp8 --seq-lens 1k8k --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml --runner-config .github/configs/runners.yaml
```

**Test all TRT configs on H200 runners:**
```
full-sweep --framework trt --runner-type h200 b200-trt --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

**Quick smoke test of all configs (highest TP, lowest concurrency only):**
```
full-sweep --test-mode --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml --runner-config .github/configs/runners.yaml
```

**Test specific model on specific hardware with specific sequence lengths:**
```
full-sweep --model-prefix dsr1 --runner-type b200 --precision fp4 --framework sglang --seq-lens 1k1k 8k1k --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

## Custom One-off Tests

**Scenario 4**: I want to run a quick test with a custom image, model, or configuration that isn't in the config files yet.

Use the `custom` command to specify all parameters manually:
```
custom --runner-label b200-nb_0 --image vllm/vllm-openai:v0.11.0 --model meta-llama/Llama-3.1-70B --framework vllm --precision fp8 --exp-name llama70b_test --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml
```

This runs a single 1k1k test job with your custom parameters on the specified runner node. Useful for:
- Testing new images before adding them to config files
- Quick validation of new models
- Experimenting with different frameworks or precisions
