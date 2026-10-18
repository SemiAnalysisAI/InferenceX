# Serving Wan2.2-T2V-A14B on vLLM-Omni

Configuration observed for one Wan2.2-T2V-A14B request on 8x H100 80GB on 2026-10-17 (dsxe, Slurm job 445).

## Pins

| Component | Value |
| --- | --- |
| vLLM-Omni | release v0.30.0, commit `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b` (contains the upstream Wan2.2-T2V recipe's pinned `4c7a98c2`) |
| Image | `vllm/vllm-omni:v0.30.0`, manifest digest `sha256:fc77cfaac7b1b43c30cbf32eaea214864434c6e0f303e6e9eb9f64104241a994`, linux/amd64 `sha256:b09c9eab7d2e30817208de92ccb482756d13d4339377d7cc4c4ce9f47bf04174` |
| In-image versions | vllm 0.30.0, vllm-omni 0.30.0, torch 2.13.0+cu130, av 18.1.0, numpy 2.2.6 |
| Model | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7` (126 GB, 49 files; same as `specs/wan.yaml` and PR #3733) |

## Server command

```
vllm serve <snapshot dir> --omni --served-model-name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --host 127.0.0.1 --port 8091 \
  --use-hsdp --cfg-parallel-size 2 --usp 4 --vae-patch-parallel-size 8 --vae-use-tiling \
  --init-timeout 3600 --stage-init-timeout 3600
```

`<snapshot dir>` is the Hugging Face cache snapshot directory for the revision above (serve the directory, not the Hub ID, under `HF_HUB_OFFLINE=1`). `--served-model-name` keeps the model name the evaluator sends.

Environment:

```
HF_HUB_OFFLINE=1
HF_HUB_CACHE=<cache root>
VLLM_OMNI_SERVER_STORAGE__PATH=<job dir>/server-storage
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## Parallelism

The upstream recipe's "official open-source model (with CFG)" layout: CFG parallel 2 x Ulysses sequence parallel 4 = 8 GPUs, HSDP weight sharding, VAE patch parallel 8 with tiling. The server resolved the `FLASH_ATTN` attention backend for both `self` and `cross` roles at startup. Both init timeouts must be raised: the 600 s default aborted the first cold start (job 443) while the eight workers were still reading the checkpoint; the warm-cache restart was ready after 574 s.

## Request mapping

`POST /v1/videos` is a multipart form at this pin; the field names come from `vllm_omni/entrypoints/openai/video/generation/helpers.py::_parse_video_form` and `vllm_omni/entrypoints/openai/protocol/videos.py::VideoGenerationRequest`.

| spec field | form field |
| --- | --- |
| `model.id` | `model` |
| case prompt, case seed | `prompt`, `seed` |
| `generation.width`, `generation.height` | `width`, `height` |
| `generation.frame_count` | `num_frames` |
| `generation.fps` | `fps` |
| `generation.num_inference_steps` | `num_inference_steps` |
| `generation.guidance_scale`, `generation.guidance_scale_2` | `guidance_scale`, `guidance_scale_2` |
| `generation.flow_shift` | `flow_shift` (per-request scheduler shift; the server default is 5.0) |
| `generation.negative_prompt` | `negative_prompt` (FastAPI drops the empty field; `pipeline_wan2_2.py` encodes `""` for it) |

`boundary_ratio` is not sent; both engines use the checkpoint's 0.875. The runner polls `GET /v1/videos/{id}` until `completed` and downloads `GET /v1/videos/{id}/content`. At this pin a failed job answers the poll with a non-2xx status, which the ported runner records as a transport error rather than `provider_failed` (the same holds for the H3 vLLM-Omni path in #3733).

Runner call (the SGLang path is unchanged; only `runtime` differs):

```python
from specs import h3_spec
from evaluator import mvp_runner

spec = h3_spec.load_spec("specs/wan.yaml")
mvp_runner.run_plan(
    h3_spec.plan_for(spec, "shared-v1"),
    "<out>",
    endpoint="http://127.0.0.1:8091",
    runtime="vllm-omni",
    runtime_revision="a8576ccb725c4e21cd13c3eb5f9a546b21149d2b",
    hardware_label="<hardware>",
    model_revision=spec["model"]["revision"]["value"],
)
```

## Observed on 8x H100 (informational, not a benchmark)

- `/health` 200 after 574 s with a warm page cache; eight `DiffusionWorker_SP{0-3}_CFG{0,1}` processes, `peak_memory_mb` 25566 in the server's job record.
- One request of case `drum-taps-s11` (a one-case copy of `shared-v1` with `warmup_runs` 0): 40 denoise steps at 1.23 it/s, server `inference_time_s` 37.43, 37.97 s submit to validated clip.
- Clip: 832x480, 81 frames, 16 fps, 5.0625 s, h264, no audio stream, 2,225,797 bytes; every frame has pixel std 69 to 71 on the 0 to 255 scale, and frames 1, 41 and 81 show the prompted drummer.
- Power (nvidia-smi 1 Hz): the sum of per-GPU board means was 4518 W during generation, using 42 of 60 samples per GPU, against 543 W idle before launch. This is a single-request observation, not a PowerX curve or a matched power benchmark.
- Evidence: `tests/fixtures/wan-vllm-omni-a8576ccb/` holds the runner request, the server's job record and the ffprobe of the clip.
