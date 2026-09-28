# InferenceX end-to-end benchmarks

<div align="center">

**English** | [中文](README_zh.md)

</div>

This project contains model-serving benchmarks, configurations, launchers, Python
tooling, documentation, and performance history. Start with the
[documentation index](docs/index.md).

The [Python manifest](pyproject.toml) and [lockfile](uv.lock) belong to this
project. From the repository root, install and run the local tooling with:

```bash
cd inferencex-e2e
uv sync --locked
uv run --locked python -m infx.matrix.generate test-config \
  --config-files configs/nvidia-master.yaml --config-keys <key>
```

Replace `<key>` with the selected config key. Workflow dispatch generator arguments
also use paths relative to this directory. See the [testing guide](docs/testing.md)
for dependency groups and checks.

The project-local [Python version](.python-version) selects its interpreter.
Repository policy and GitHub workflows remain at the repository root. The local [license](LICENSE) keeps attribution
available when containers mount only this project directory.
