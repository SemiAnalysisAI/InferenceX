## Summary

* Add SGLang disaggregated prefill–decode (PD) benchmark recipe for **zai-org/GLM-5-FP8** on MI355X (MoRI backend, NSA attention)
* New CI config `glm5-fp8-mi355x-sglang-disagg` in `amd-master.yaml`: 1P+1D TP8/EP1 smoke sweep for ISL 1K1K and 8K1K, concurrency 8–512, image `lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260517` (same as aggregated `glm5-fp8-mi355x-sglang`)
* New `GLM-5-FP8` entry in `benchmarks/multi_node/amd_utils/models.yaml` (MoRI transfer, glm tool/reasoning parsers, multithread load)
* New multinode entry script `benchmarks/multi_node/glm5_fp8_mi355x_sglang-disagg.sh`
* **`setup_deps.sh`**: idempotent in-container patches required for GLM-5 on ROCm images
  * aiter gluon `pa_mqa_logits` 3D `instr_shape` fix (NSA / Triton ≥ 3.5)
  * install transformers with `glm_moe_dsa` when `MODEL_NAME=GLM-5-FP8`
* **`env.sh`**: GLM-5-specific exports (`SGLANG_ROCM_FUSED_DECODE_MLA=0`, quick reduce, safetensors fast GPU)
* **`server.sh`**: `source setup_deps.sh` before `env.sh` (patches run once per container start)

## Test plan

* [ ] Verify CI passes: `amd-master.yaml` validates `glm5-fp8-mi355x-sglang-disagg` (`scenarios.fixed-seq-len`)
* [ ] Confirm `setup_deps.sh` is a no-op or fast on non–GLM-5 disagg jobs (transformers install gated on `MODEL_NAME`)
* [ ] Smoke on mia1: `glm5-fp8-mi355x-sglang-disagg` 1P+1D, ISL/OSL 1024/1024, conc=64
* [ ] Verify model weights resolve at `/it-share/data/GLM-5-FP8` (runner sets `MODEL_NAME` from `zai-org/GLM-5-FP8`)

## Dependencies

* Builds on shared MI355X disagg plumbing from `main` (and optionally after PR #1570 for Qwen3.5 disagg)
* Does **not** include Qwen3.5 config, `benchmark_serving` tokenizer fallback, or MoRI `conn.py` overlay (follow-up PRs)

---

**Note**

**Risk**  
Medium: `server.sh` now sources `setup_deps.sh` for **all** SGLang disagg runs; gluon patch is global but idempotent; transformers install only runs for `GLM-5-FP8`.

**Overview**  
Enables GLM-5 PD disaggregation in InferenceX CI. Image patches address gaps not yet in upstream aiter/transformers/docker images; runtime env tweaks reflect GLM-5 NSA architecture vs MLA-based models like DeepSeek-R1.
