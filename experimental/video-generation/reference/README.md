# MiniMax-H3 Diffusers reference

This command produces reference media for the selected frozen H3 workload. It
does not benchmark serving latency, publish results, or establish cross-engine
quality. A selected subset is recorded as a subset. The original float32 video
and audio are retained alongside the MP4, before lossy encoding.

## Pinned environment

Use Linux x86_64, glibc >= 2.28 and CPython 3.12.14. `requirements-linux.txt`
locks every runtime dependency and one compatible wheel hash per package.
PyPI's torch 2.10.0 wheel supplies CUDA 12.8; torchvision is paired at 0.25.0.
The environment also pins Transformers 5.22.1, Accelerate 1.12.0, Hugging Face
Hub 1.32.0, safetensors 0.8.0, PyAV 16.1.0 and NumPy 2.3.5.

Diffusers is the archive of commit
[`d8c1b13`](https://github.com/huggingface/diffusers/tree/d8c1b13c756df0059bcdb6e477a1eafcd502098d),
version 0.42.0.dev0. This is newer than the integration commit linked by the
[MiniMax recipe](https://github.com/MiniMax-AI/MiniMax-H3/blob/199b7d0e1fd95e7b3eafa8c15eca692534142427/requirements.txt).
It is a selected source version, not a claim that MiniMax tested this dependency
combination. Real imports and CPU scheduler checks can pass without validating
model loading, GPU kernels or generated quality.

From `experimental/video-generation/`, prepare the environment on persistent
storage before allocating GPUs. Reuse an existing environment only if these
versions and the archive identity match. The first install supplies the pinned
build tools; the second installs the hash-locked runtime without isolated,
unversioned build dependencies.

```sh
REFERENCE_ENV=/path/to/persistent/h3-diffusers-reference
uv venv --python 3.12.14 "$REFERENCE_ENV"
uv pip install --python "$REFERENCE_ENV/bin/python" setuptools==80.9.0 wheel==0.45.1
uv pip sync --python "$REFERENCE_ENV/bin/python" --require-hashes --no-build-isolation \
  reference/requirements-linux.txt
uv pip check --python "$REFERENCE_ENV/bin/python"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$REFERENCE_ENV/bin/python" \
  -m evaluator.reference --spec specs/h3-v0.1.yaml --set release-v1 --case all --check
```

The check imports the actual ModularPipeline and Qwen3-VL classes, validates the
installed direct pins/archive, selects 20 measured cases plus the separate warmup,
and evaluates both pinned schedulers on CPU. It loads no model weights. Diffusers'
50 sigma points produce 49 updates; both sigma arrays and their hashes are printed.
The serving engines use 49 in their request field under this same workload.

## Generate on an authorized H100 allocation

Use the complete, verified snapshot at
`models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe`.
It must contain `modular_model_index.json`, the text encoder, tokenizer, processor,
base transformer, both VAEs and both schedulers. An SGLang FL2VA view is not this
snapshot. Retain the existing model staging/checksum receipt: this helper records
local metadata hashes and the requested revision but does **not** attest weight
contents. Offline mode and the explicit local loading override prevent downloading
missing weights. Missing components fail the attempt.

The supported reference route uses one 80 GB H100, BF16, native PyTorch SDPA,
and component CPU offload with a 12 GB GPU reserve. Plan for at least 256 GiB host
RAM and sufficient local output storage. Reference speed is not an engine baseline.
It uses no quantization, LoRA, external attention-kernel download or prompt rewrite.

```sh
H3_SNAPSHOT=/path/to/hf-hub-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
REFERENCE_OUTPUT=/path/to/persistent/campaign/reference-pilot

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$REFERENCE_ENV/bin/python" \
  -m evaluator.reference --spec specs/h3-v0.1.yaml --set release-v1 \
  --case ceramic-wheel --snapshot "$H3_SNAPSHOT" --output "$REFERENCE_OUTPUT" \
  --device cuda:0 --dtype bfloat16 --attention-backend native --offload-reserve 12GB \
  --encoder-threads 1 --audio-bitrate 128000
```

The first case is a bounded loading/generation pilot. After its acceptance and
within a separately approved population budget, use `--case all` and a new output
directory for the complete 20+1 reference set. Multiple `--case` arguments select
explicit cases; existing output directories are never overwritten.

Generation, prompt/seed, schedule and MP4 codec/preset/CRF settings come from the
selected YAML. The command explicitly supplies execution and encoder thread/AAC
bitrate settings. Each case resets one CPU generator; the pinned T2VA source draws
float32 video noise and then audio noise from it. The receipt records generator
states before/after; it does not claim to capture initial noise or an iteration
trace. Equal integer seeds across engines do not establish equal noise.

Each case retains `video.npy` (frames, channels, height, width), `audio.npy`
(channels, samples), and `reference.mp4`. Encoding truncates float32 RGB×255 to
uint8, then uses the selected libx264/preset/CRF/yuv420p and AAC settings. A 124-frame
float32 video occupies about 1.43 GiB; reserve at least 35 GiB for the complete set.
`reference.json` retains exact selected cases, environment, schedules, metadata,
encoder options and file hashes. `SHA256SUMS` seals completed and failed attempts.
`selection_complete` means only that the selected files were written. Complete
decode, model/runtime evidence and independent quality evaluation are still
required; `quality_status` remains `unjudged` and `release_qualified` remains false.

To deliberately refresh dependencies, edit `requirements.in`, resolve for the same
Python/platform with `uv pip compile --generate-hashes`, and inspect the chosen
wheels and hashes before replacing the lock. Revalidate imports and schedules;
do not silently substitute an upstream branch or latest package at execution time.
