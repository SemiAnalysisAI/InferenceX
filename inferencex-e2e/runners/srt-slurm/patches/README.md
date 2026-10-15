# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch backports an upstream fix that the shared submodule pin does not include. Once the pin includes the fix, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `573-participating-gpus.patch` | [NVIDIA/srt-slurm#573](https://github.com/NVIDIA/srt-slurm/pull/573) | Record only the GPUs workers occupy on a worker node; an exporter such as AMD's reports every GPU, which the collector otherwise flags as `unexpected_device`. |
| `602-vllm-sidecar-reasoning-parser.patch` | [NVIDIA/srt-slurm#602](https://github.com/NVIDIA/srt-slurm/pull/602) | Drop the `none` reasoning-parser sentinel for Python multi-node vLLM sidecar launches; keep it for `vllm-rs` and reject unsupported `auto`. Backports commit `fe957cc8baf1f3be4e0c792b1e1a66e489e85247`, including its command-generation regression tests. |
