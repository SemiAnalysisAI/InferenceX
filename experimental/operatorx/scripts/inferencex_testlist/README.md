# inferencex_testlist

Derive `operatorx/testlists/*.json` from the InferenceX matrix configs.

## Approach

Pure static enumeration — no GPU time, no tracing.

For each `(model, framework, precision, runner, TP, EP, dp-attn, ISL, OSL,
conc, spec-decoding)` tuple in the InferenceX matrix, we:

1. Look up the model's HuggingFace `config.json` (locally cached).
2. Resolve the architecture (DeepSeek-MLA / GLM-MLA / Kimi-MLA / MiniMax-GQA /
   GPT-OSS-GQA / Qwen3.5-MoE).
3. Emit canonical OperatorX op shapes that the workload would touch:
   - **GEMM**: attention projections, MLP gate/up/down (for dense layers),
     sharded by TP.
   - **Attention**: MHA or MLA, parameterised by TP head-shard and DP-attn
     batch-shard. Prefill at full ISL; decode at `mtp_factor` query tokens
     and `ISL + OSL/2` KV length.
   - **MoE**: `moe_forward` + `topk_routing` at prefill (`conc*ISL` tokens)
     and decode (`conc*mtp_factor` tokens) regimes.
   - **Collectives**: AllReduce (plain TP), AllGather + ReduceScatter
     (DP-attn), dispatch/combine (EP > 1).
4. Aggregate across rows, dedupe by canonical `(op_type, args)`, and write
   testlist JSONs. Existing entries are unioned with new ones unless
   `--no-merge-existing` is passed.

Matrix expansion is delegated to InferenceX's own
`utils/matrix_logic/generate_sweep_configs.py full-sweep` so the enumerator
stays in sync with InferenceX semantics automatically.

## Known limitations

Static enumeration captures canonical shapes, not engine-specific transformations:

- Fused QKV / fused gate-up GEMMs are emitted as separate canonical projections.
- MLA "absorbed" decode form (one big GEMM instead of separate Q/KV ups) is not
  modeled — we emit the naive MLA decomposition.
- Speculative-decoding draft-model GEMMs (EAGLE-style drafters) are not
  enumerated separately; MTP is modeled by inflating the decode token count.
- Qwen3.5's hybrid linear/full attention layers are modeled as full-attention
  only (linear-attention layers TODO).
- Engine-version drift is not detected.

If you need engine-faithful shapes, plug in the empirical-trace fingerprint
path on top (separately, optional).

## Run

```bash
# From a clean venv with pydantic + pyyaml installed:
pip install pydantic pyyaml

# Enumerate against both NVIDIA and AMD matrices:
python -m scripts.inferencex_testlist enumerate \
    --inferencex /path/to/InferenceX \
    --config .github/configs/nvidia-master.yaml \
    --config .github/configs/amd-master.yaml \
    --models-dir /models \
    --models-dir /scratch/fsw/models \
    --out testlists/
```

`--models-dir` is repeatable; each is searched for a `<model-short-name>/config.json`.
HuggingFace cache (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/`)
is searched automatically.

Use `--no-merge-existing` to overwrite the testlists rather than union.

## Adding a new architecture

When InferenceX adopts a new model family, add a `_build_<family>(cfg, name)`
function in `models.py` and wire it into `_BUILDERS`. The function should
populate `AttentionArch` (mha vs mla, sharded dims) and `MoeArch`
(num_experts, top_k, intermediate, n_shared_experts, optional dense MLP).
The rest of the enumerator picks up from there.

## Adding a new parameterisation

Nothing to do — new `(TP, EP, ISL, OSL, conc, spec-decoding)` tuples added to
`.github/configs/*.yaml` are picked up automatically on next run.
