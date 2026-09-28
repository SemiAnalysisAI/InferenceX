# srt-slurm patches

`setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../slurm_utils.sh) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule. TileRT jobs use the fork checkout and skip these patches.

Each patch is a temporary fix for an open upstream PR. When the PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `507-lmcache-server-atom-sglang.patch` | [SemiAnalysisAI/srt-slurm#32](https://github.com/SemiAnalysisAI/srt-slurm/pull/32) (includes [NVIDIA/srt-slurm#507](https://github.com/NVIDIA/srt-slurm/pull/507)) | LMCache for vLLM, SGLang and ATOM: the `lmcache-server` service, and ATOM `extra-kv-connectors` wrapped with Mooncake in `multi` |
