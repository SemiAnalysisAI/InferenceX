# srt-slurm patches

`setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../slurm_utils.sh) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule. TileRT jobs use the fork checkout and skip these patches.

Patches are temporary fixes pending upstream integration. When the corresponding change merges and the submodule pin includes it, delete the patch and its row. A local candidate without an upstream PR is identified explicitly below.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `504-post-eval-srun-options.patch` | [NVIDIA/srt-slurm#504](https://github.com/NVIDIA/srt-slurm/pull/504) | Forward recipe `srun_options` (e.g. `container-writable`) to post-eval steps |
| `local-version-compute-setup.patch` | Local candidate; upstream PR pending | Give Hatch the distribution name for its scoped version override, reuse the locally derived package version on compute nodes, and stop if compute environment synchronization fails |
