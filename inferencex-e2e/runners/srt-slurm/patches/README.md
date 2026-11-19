# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch backports an upstream fix that the shared submodule pin does not include. Once the pin includes the fix, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `573-participating-gpus.patch` | [NVIDIA/srt-slurm#573](https://github.com/NVIDIA/srt-slurm/pull/573) | Record only the GPUs workers occupy on a worker node; an exclusive node's exporter (DCGM or AMD) reports every GPU, which the collector otherwise flags as `unexpected_device`. The pin includes #573, but upstream merged a different design: it keeps every row and excuses idle worker-node GPUs only after a DCGM worker GPU UUID probe. [`infx/results/power/multinode.py`](../../../infx/results/power/multinode.py) does not apply that excusal yet, so keep this filter until it does. |
| `900-pool-expected-devices.patch` | pending | Plan every service pool node's `gpus_per_node` GPUs as expected power-telemetry devices under the service's name (`worker_role=<service>`, `worker_index=<node ordinal in the pool>`), so a services-only recipe (`frontend.type: none`, one `services[].nodes` pool, `benchmark.type: custom`) reaches collector readiness and publishes instead of failing with `exporter_startup_timeout` + `expected_device_missing` + `unexpected_device`. A services-only job records `worker_identity_proof.error = "no engine workers to probe"` instead of a probe traceback, the identity gate only scopes engine-role devices, and the offline validator accepts pool roles. Engine-only manifests are unchanged. With `573-participating-gpus.patch` an exclusive pool node's unplanned exporter rows are dropped at scrape time, so a recipe whose `gpus_per_node` is below the node's physical GPU count still validates; upstream alone reports them as `unexpected_device`. |
