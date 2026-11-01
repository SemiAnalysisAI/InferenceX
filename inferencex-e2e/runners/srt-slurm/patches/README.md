# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch backports an upstream fix that the shared submodule pin does not include. Once the pin includes the fix, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `573-participating-gpus.patch` | [NVIDIA/srt-slurm#573](https://github.com/NVIDIA/srt-slurm/pull/573) | Record only the GPUs workers occupy on a worker node; an exclusive node's exporter (DCGM or AMD) reports every GPU, which the collector otherwise flags as `unexpected_device`. The pin includes #573, but upstream merged a different design: it keeps every row and excuses idle worker-node GPUs only after a DCGM worker GPU UUID probe. [`infx/results/power/multinode.py`](../../../infx/results/power/multinode.py) does not apply that excusal yet, so keep this filter until it does. |
| `603-rapl-cpu-power-source.patch` | [NVIDIA/srt-slurm#603](https://github.com/NVIDIA/srt-slurm/pull/603) | Add `--source rapl` to the CPU power exporter and the `package` rail kind, so x86 hosts (AMD EPYC) report per-socket package power from the RAPL energy counters, and keep the cluster GPU exporter when a recipe adds that CPU leg. |
| `603-rapl-exporter-asset.patch` | (interim, not upstream) | `make setup` on x86_64 installs the `cpu-power-exporter` built from #603 commit `64d5b0ac` from a pinned fork release and checks it against a pinned SHA-256; a missing asset fails the setup. The NVIDIA release binary has no `rapl` source. Delete with the row above. |
