# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch backports an upstream fix that the shared submodule pin does not include. Once the pin includes the fix, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `640-tokenspeed-smg-zmq.patch` | [NVIDIA/srt-slurm#640](https://github.com/NVIDIA/srt-slurm/pull/640) at [`07e090a957067fec5557e2f3aeb6997ec7e8d688`](https://github.com/NVIDIA/srt-slurm/commit/07e090a957067fec5557e2f3aeb6997ec7e8d688) | Launch native headless TokenSpeed workers through SMG's ZMQ transport, wait for healthy worker registrations, resolve gateway model paths, and discover each worker's separate HTTP metrics listener. Carries the complete upstream PR diff, including tests and documentation. |
| `573-participating-gpus.patch` | [NVIDIA/srt-slurm#573](https://github.com/NVIDIA/srt-slurm/pull/573) | Record only the GPUs workers occupy on a worker node; an exclusive node's exporter (DCGM or AMD) reports every GPU, which the collector otherwise flags as `unexpected_device`. The pin includes #573, but upstream merged a different design: it keeps every row and excuses idle worker-node GPUs only after a DCGM worker GPU UUID probe. [`infx/results/power/multinode.py`](../../../infx/results/power/multinode.py) does not apply that excusal yet, so keep this filter until it does. |
