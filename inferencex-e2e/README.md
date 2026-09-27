# InferenceX end-to-end benchmarks

**English** | [中文](README_zh.md)

Model-serving benchmarks, configurations, launchers, result tooling, and the
performance changelog live here. Start with [the documentation index](docs/index.md).

Run end-to-end commands from this directory:

```bash
cd inferencex-e2e
uv run --locked python -m infx.matrix.generate test-config \
  --config-files configs/nvidia-master.yaml --config-keys <key>
```

The Python manifest and lockfile stay at the repository root. `uv` discovers them
from this directory. GitHub workflows, repository policy, and CODEOWNERS also stay
at the repository root. Workflow dispatch generator arguments use paths relative
to this directory. Historical checkouts retain their original execution root.
