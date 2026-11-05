# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch backports an upstream fix that the shared submodule pin does not include. Once the pin includes the fix, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `573-participating-gpus.patch` | [NVIDIA/srt-slurm#573](https://github.com/NVIDIA/srt-slurm/pull/573) | Record only the GPUs workers occupy on a worker node; an exclusive node's exporter (DCGM or AMD) reports every GPU, which the collector otherwise flags as `unexpected_device`. The pin includes #573, but upstream merged a different design: it keeps every row and excuses idle worker-node GPUs only after a DCGM worker GPU UUID probe. [`infx/results/power/multinode.py`](../../../infx/results/power/multinode.py) does not apply that excusal yet, so keep this filter until it does. |
| `644-nic-traffic.patch` | [NVIDIA/srt-slurm#644](https://github.com/NVIDIA/srt-slurm/pull/644), with [#649](https://github.com/NVIDIA/srt-slurm/pull/649) | Sample every RDMA NIC's byte counters once per second on each node of a multi-node job and write per-window mean and peak Gbit/s to `LOGS/nic_traffic/`. Pollara (`ionic`) RDMA traffic is visible only in its `hw_counters`, which node-exporter cannot read. It also carries #649, which #644 builds on: a node outside every het component now skips the CPU power host leg instead of failing the sweep. Source files only; the upstream PRs carry the tests and docs. |
