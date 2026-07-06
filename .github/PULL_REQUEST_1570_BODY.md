## Summary

* Add SGLang disaggregated prefill–decode (PD) benchmark recipe for **Qwen/Qwen3.5-397B-A17B-FP8** on MI355X (MoRI backend), mirroring the existing DeepSeek-R1 disagg structure with model-specific tuning (no MTP, no DeepEP dispatch)
* New CI config `qwen3.5-fp8-mi355x-sglang-disagg` in `amd-master.yaml`: 1P+1D TP8/EP1 smoke sweep for ISL 1K1K and 8K1K, concurrency 8–512, image `lmsysorg/sglang-rocm:v0.5.11-rocm700-mi35x-20260511`
* New `Qwen3.5-397B-A17B-FP8` entry in `benchmarks/multi_node/amd_utils/models.yaml` (MoRI transfer, FP8 KV cache, aiter attention, DP / no-DP / EP-only prefill and decode profiles)
* New multinode entry script `benchmarks/multi_node/qwen3.5_fp8_mi355x_sglang-disagg.sh` (maps CI sweep params → shared `amd_utils/submit.sh`; optional `NODE_LIST` for local smoke on fixed nodes)
* `runners/launch_mi355x-amds.sh`: `KEEP_LOGS=1` skips EXIT trap log cleanup for local debugging
* No changes to shared `job.slurm` / `submit.sh` / `server.sh` (uses existing `main` disagg plumbing)

## Test plan

* [ ] Verify CI passes: `amd-master.yaml` loads and validates the new `qwen3.5-fp8-mi355x-sglang-disagg` entry (`scenarios.fixed-seq-len` schema)
* [ ] Confirm existing **SGLang** disagg benchmarks are unaffected (this PR does not modify shared `amd_utils` scripts)
* [ ] Smoke on mia1: `qwen3.5-fp8-mi355x-sglang-disagg` 1P+1D, ISL/OSL 1024/1024, conc=64 (target: requests complete, results JSON produced)
* [ ] Optional local: re-run with `KEEP_LOGS=1` and confirm Slurm logs are retained under `benchmark_logs/`

## Out of scope (follow-up PRs)

* `utils/bench_serving/benchmark_serving.py` Qwen3.5 tokenizer fallback
* GLM-5 disagg config, `setup_deps.sh` image patches, v0.5.12 + MoRI `conn.py` overlay
* Upstream sglang fix for Qwen3.5 + MoRI + `enable-dp-attention` (`is_deepep_class_backend` / shared-expert slot accounting)

---

**Note**

**Risk**  
Low: only adds new config, model block, and launch script; reuses unmodified `amd_utils` from `main`.

**Overview**  
First InferenceX recipe for **Qwen3.5-397B-A17B-FP8** PD disaggregation on MI355X using the existing multinode SGLang + MoRI stack. Scope is intentionally narrow (PR-1): CI config, model flags, launch script, and `KEEP_LOGS` runner tweak—no shared `amd_utils` edits, no in-container patches, no `benchmark_serving` changes.
