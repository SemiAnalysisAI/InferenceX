# srt-slurm patches

`setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../slurm_utils.sh) applies every `*.patch` here, in glob order, to the job's srt-slurm clone after checking out the pinned submodule.

Each patch is temporary. When its upstream PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `504-post-eval-srun-options.patch` | [NVIDIA/srt-slurm#504](https://github.com/NVIDIA/srt-slurm/pull/504) | Forward recipe `srun_options` (e.g. `container-writable`) to post-eval steps |
| `tilert-backend-router.patch` | none yet (ported from [SemiAnalysisAI/srt-slurm#13](https://github.com/SemiAnalysisAI/srt-slurm/pull/13)) | Add the `tilert` engine (vLLM prefill image, TileRT decode, one-time weight conversion step) and the `tilert-router` frontend, plus `configs/tilert_setup.sh` |
