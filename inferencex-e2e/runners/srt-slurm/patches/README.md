# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch is a temporary local delta. Delete it when the submodule pin includes the same behavior.

| Patch | Base API | Local delta |
|-------|-------------|-----|
| `572-participating-gpus.patch` | [NVIDIA/srt-slurm#572](https://github.com/NVIDIA/srt-slurm/pull/572) | Persist only worker-assigned GPUs from a node-wide exporter scrape. |

The pin includes the generic exporter mapping in the open upstream #572 head. The participant filter remains local until the pinned upstream commit includes it.
