# Kimi-K3 TP8×PP2 Runtime Patches

English | [中文](README_zh.md)

This directory contains the runtime patches used to bring up and optimize
Kimi-K3 FP4 with TP8×PP2 on the pinned vLLM ROCm image at commit
`7c5dc571cbd1064ecc8a9b1045637ff647aa22cb`.

## Pipeline transport

[`pp_async_activation/`](pp_async_activation/) contains the two-slot PP
activation ring, PP1 receive preposting, the scheduler batch cap, the upstream
send-buffer lifetime fix, and the 16-rank transport regression.

The v5 prepost patch applies v2 first. Mount this directory as `/ppasync` and
set:

```bash
K3_PP_ASYNC_ACTIVATION=1
K3_PP_PREPOST_RECV=1
python3 /ppasync/apply_pp_async_activation_v5.py
```

## DSpark under pipeline parallelism

[`apply_vllm_50514_pp_spec.sh`](apply_vllm_50514_pp_spec.sh) and the vendored
`vllm_pr50514_*` files place the drafter on the last PP stage and transport the
five ordered Kimi-K3 auxiliary hidden states across PP.

[`dspark_cache_reuse/`](dspark_cache_reuse/) contains compatibility fixes for:

- duplicate capture of the PP boundary layer;
- preservation of the global draft KV-cache group through PP projection;
- Mamba state retention at the EAGLE replay boundary; and
- the optional EAGLE trailing-block no-drop configuration.

Apply the cache fixes after the PP speculative-decoding patch. The order is:

```bash
python3 patch_dspark_aux_contract_v028.py
python3 patch_dspark_pp_scheduler_eagle_groups.py
python3 apply_dspark_replay_boundary_fix.py
python3 patch_disable_eagle_block_drop_v028.py  # only for no-drop experiments
```

## Validation evidence

The v5 transport regression passed 100 TP8×PP2 steps with changing tensor
shapes and sender-side gather policies.

For DSpark K=4, a controlled dummy-weight test sent the same 18,089-token
prompt twice:

- replay-boundary retention: 15,360 cached tokens on the second request;
- replay-boundary retention plus no-drop: 16,896 cached tokens (93.41%).

Real-weight AgentX screening raised the measured prefix-cache hit rate from
6.53% with draft-group annotation alone to 33.95% with replay-boundary
retention and 37.46% with no-drop. These short screening runs are implementation
evidence, not publishable benchmark results.
