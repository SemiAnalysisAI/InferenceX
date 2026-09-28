# OperatorX GitHub Actions

**English** | [中文](CI_zh.md)

[OperatorX Sweep](../.github/workflows/operatorx-sweep.yml) runs on `workflow_dispatch` only (pull requests run just the hosted planner).

| GPU | Runner label | GPUs / node | Image platform | Result cluster |
| --- | --- | ---: | --- | --- |
| H100 | `cluster:h100-dgxc` (default) | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `cluster:h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `cluster:b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `cluster:b300-dsxe` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `cluster:gb200-nv` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `cluster:gb300-nv` | 4 | `linux/arm64` | `gb300_nvl72_4x` |
| MI300X | `cluster:mi300x-amd` | 8 | `linux/amd64` | `mi300x_amds_8x` |
| MI325X | `cluster:mi325x-amds` | 8 | `linux/amd64` | `mi325x_amds_8x` |
| MI355X | `cluster:mi355x-amds` | 8 | `linux/amd64` | `mi355x_8x` |

## Dispatch

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f runner=cluster:h100-dgxc -f backends=vllm \
  -f testlists=gemm -f world_sizes=1 -f chunk_size=500 -f ingest=false
```

- Inputs: `runner`, `backends` (`vllm`; AMD also `torch`), `testlists`, `mode` (`timing` | `counters`), `world_sizes` (1, 2, 4, 8), `chunk_size` (1-500), `recovery_run_id`, `ingest` (default true; `false` for test runs).
- `mode=counters`: each op once under Nsight Compute (NVIDIA) or rocprofv3 (AMD, 12 passes: use a smaller `chunk_size`); raw files in `results/counters/`; latencies perturbed.
- Smoke check: `testlists=gemm_perf`, `chunk_size=50`. Dispatch `attn_*` testlists apart from gemm / moe ones.

## Execution contract

- Planning: shards per backend image, `parallel` split and InferenceX recipe (`recipes.py`: image, launch env, `vllm serve` args), chunked by `chunk_size`; at most 256 shards; a case with no recipe runs on the backend's image from `containers.toml`.
- One exclusive Slurm node per shard (GB200/GB300: one 4-GPU tray, no world size 8); 45 min allocation, 70 min job.
- Image: digest resolved at planning, imported with enroot on the allocated node, cached by image + digest + arch; a tag that moved since planning fails the shard.
- Ranks: one process per GPU of the split's world size, env:// rendezvous; runner settings from CollectiveX's platform registry overlaid by `platforms.json`.
- Results: rows checkpointed after each op; unsupported cases stay visible as rows, errors or zero successful rows fail the shard; artifacts `operatorx-manifest-<run_id>` and `operatorx-shard-<run_id>-<attempt>-<shard>`; ingested into the OperatorX database unless `ingest=false`.
- Cleanup: cancellation and the `always()` step cancel the allocation and keep partial results; after a failed cleanup, rerun one shard with `recovery_run_id`.

## Local tests

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml --with torch --with numpy \
  python -m pytest operatorx/tests/ -q
```
