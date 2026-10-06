# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch is a temporary fix for an open upstream PR. When the PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `amd-native-power-profile.patch` | [AMD producer source](https://github.com/edwingao28/srt-slurm/commit/fa9a497cd1c1ad0253dac161926376ae973dd022) (upstream PR pending) | Select AMD device-metrics-exporter metrics and retain only participating GPUs. |

The pin includes [NVIDIA/srt-slurm#548](https://github.com/NVIDIA/srt-slurm/pull/548), so its former patch is removed. The AMD patch contains the production changes from the linked source, rebased onto upstream `848f72d4`. Its upstream PR remains required before merge.
