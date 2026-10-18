# Serving MiniMax-H3 on vLLM-Omni

Configuration observed for one MiniMax-H3 request on 2x H100 80GB on 2026-10-17.

## Pins

| Component | Value |
| --- | --- |
| vLLM-Omni | release v0.30.0, commit `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b` |
| Image | `vllm/vllm-omni:v0.30.0`, manifest digest `sha256:fc77cfaac7b1b43c30cbf32eaea214864434c6e0f303e6e9eb9f64104241a994`, linux/amd64 `sha256:b09c9eab7d2e30817208de92ccb482756d13d4339377d7cc4c4ce9f47bf04174` |
| In-image versions | vllm 0.30.0, vllm-omni 0.30.0, torch 2.13.0+cu130 |
| Model | `MiniMaxAI/MiniMax-H3` revision `42ed227ee7df40d41602854ae760620d6eb651fe` (same as `specs/h3.yaml` and PR #2916) |

Only the `FL2VA/` partition plus root `model_index.json`, `LICENSE`, `README.md` and `Ref2VA/model_index.json` need to be on disk (129 GB); `--task-type fl2va` loads nothing else. Leave the root `modular_model_index.json` out of a local copy: with it present, vllm-omni misdetects the directory as FastH3 (upstream issue [#8796](https://github.com/vllm-project/vllm-omni/issues/8796)).

## Server command

```
vllm serve <snapshot dir> --served-model-name MiniMaxAI/MiniMax-H3 --omni --host 0.0.0.0 --port 30283 \
  --trust-remote-code --task-type fl2va --num-gpus 2 --tensor-parallel-size 2 --usp 1 --ring 1 \
  --text-encoder-tp-size 2 --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --enforce-eager
```

`<snapshot dir>` is the Hugging Face cache snapshot directory for the revision above. Serving the Hub ID with `--revision` while `HF_HUB_OFFLINE=1` fails at startup with `huggingface_hub.errors.IncompleteSnapshotError` because `snapshot_download` requires every file of the repository and resolves `refs/main`; the directory path avoids that. `--served-model-name` keeps the model name the evaluator sends.

Environment:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_CACHE=<cache root>
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
```

## Parallelism

Tensor parallel 2 across both GPUs with Ulysses and ring degree 1, text encoder TP 2, VAE patch-parallel 2 in tile mode with tiling. This is the upstream recipe's only two-GPU resident layout; pure Ulysses replicates the whole DiT on every rank, so SGLang's `ulysses_degree 2 / tp_size 1` layout from #2916 does not fit 80 GB here. The observed request used `--enforce-eager`.

## Attention backend

Resolved by the server at startup, no flag passed: `FLASH_ATTN` for `role='minimax_h3.token_refiner'` and for `role='self'` (log line from `selector.py:84/93`).

## Observed on 2x H100 (informational, not a benchmark)

- Model load: 61.3 GiB and 282 s per TP worker; `/health` 200 about 5 min 20 s after launch.
- One request from the `frozen-c124` set with `repetitions=1, warmup_runs=0`: 50 denoise steps at 2.35 s/it, 125.4 s submit to validated clip, both GPUs at the 700 W cap with 70.2 GB used.
- The request carries width, height, fps and num_frames plus `extra_params` (`task`, `target`, `audio_flow_shift`); the server logged `Applied extra_params` and produced 1344x768, 107 frames, 24 fps, 32 kHz stereo, matching the spec's clip checks.
