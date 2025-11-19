# Getting Started

## Overview
Currently we have 2 B300 nodes set up as self-hosted runners: `b200-nvd_0` and `b200-nvd_1`. These runners are named with the `b200` prefix to maintain compatibility with existing B200 scripts and allow them to run with minimal changes. This naming convention can be updated as B300-specific scripts are added.

## Running Benchmarks

Instructions to trigger workflows are listed in [.github/workflows/README.md](./.github/workflows/README.md).

For B300 benchmarks, please trigger the workflows on either `b200-nvd_0` or `b200-nvd_1` runners. You can specify the runner node using the `--runner-node` flag as explained in the [workflows README](./.github/workflows/README.md).

**Example:** To run the DSR1 FP4 TRT config on the B300 runner `b200-nvd_0`:
```
test-config --config-files .github/configs/nvidia-master.yaml --runner-config .github/configs/runners.yaml --key dsr1-fp4-b200-trt --runner-node b200-nvd_0
```

You can replace `b200-nvd_0` with `b200-nvd_1` to use the other B300 runner.

## Sample TRT Runs on B300 Runners

- **DSR1 FP4 1k1k**: https://github.com/NVIDIA/InferenceMAX/actions/runs/19487563472
- **DSR1 FP4 8k1k**: https://github.com/NVIDIA/InferenceMAX/actions/runs/19490188545
- **DSR1 FP4 1k8k**: https://github.com/NVIDIA/InferenceMAX/actions/runs/19490171737
- **DSR1 FP8**: https://github.com/NVIDIA/InferenceMAX/actions/runs/19493354530
- **GPTOSS FP4**: https://github.com/NVIDIA/InferenceMAX/actions/runs/19509231645

## Workflow Guidelines

**Important:** Please make changes in feature/side branches and trigger workflows from there.

**Do not merge changes into `main`** - This will cause merge conflicts during upstream synchronization.