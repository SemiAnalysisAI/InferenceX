# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on NVIDIA for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

`setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../slurm_utils.sh) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule. TileRT jobs use the fork checkout and skip these patches.

Each patch is a temporary fix for an open upstream PR. When the PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| _(none)_ | | No patches are currently carried; the pinned submodule (v2.30.0) includes everything the runners need. |
