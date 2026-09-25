# srt-slurm patches

`setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../slurm_utils.sh) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule. TileRT jobs use the fork checkout and skip these patches.

Each patch is a temporary fix for an open upstream PR. When the PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `031-amd-power-profiles.patch` | [SemiAnalysisAI/srt-slurm#31](https://github.com/SemiAnalysisAI/srt-slurm/pull/31) at `4f95eee1b9f50fc0dadbc7163f93800680bca0a7` | AMD exporter profile validation on MI300X, MI325X, and MI355X |
| `504-post-eval-srun-options.patch` | [NVIDIA/srt-slurm#504](https://github.com/NVIDIA/srt-slurm/pull/504) | Forward recipe `srun_options` (e.g. `container-writable`) to post-eval steps |
