# DSv4 EPLB image layer (MI355X / ROCm)

English | [中文](README_zh.md)

Build context for `jiahcao/vllm-dsv4:optimal-fse-aiter4269-sparse4382-megamoe-eplb`,
the image the DSv4 MI355X EPLB arms run on.

EPLB needs two DSv4 model files the base image does not ship. They were
previously bind-mounted over the image's editable vLLM install at container
start, which meant a clean checkout could not reproduce an EPLB arm. This
directory is that patch, as a build layer.

| File | Purpose |
|---|---|
| `Dockerfile` | Copies both patched files over `/src/vllm/vllm/models/deepseek_v4/amd/`, then runs the verifier. `/src/vllm` is an editable install, so the real file must be replaced — PYTHONPATH shadowing does not work. |
| `deepseek_v4/amd/model.py` | `DeepseekV4MixtureOfExperts` mixin so the runner registers the model, plus physical-slot expert sizing for both MoE backends. |
| `deepseek_v4/amd/mega_moe_experts.py` | EPLB on the FlyDSL mega-MoE path: multi-slot weight loader, `get_expert_weights()` over the shuffled kernel layout, the logical→physical remap in `forward()`, and mori symmetric-heap sizing from the `MegaMoEShape`. |
| `verify_eplb_patch.py` | Build-time proof the patch is really installed. Fails the build rather than the benchmark. |

## Build

```sh
docker build -t jiahcao/vllm-dsv4:optimal-fse-aiter4269-sparse4382-megamoe-eplb .
```

`BASE` defaults to the `-megamoe` tag; override it with `--build-arg BASE=…`
when the base image is bumped. The layer adds ~97 kB.

## Why the model needs patching at all

With the stock `amd/model.py` the runner never registers the model as a
`MixtureOfExperts`, so `--enable-eplb` is a **silent no-op** rather than an
error: an arm can "run with EPLB" while measuring nothing. That is what the
verifier exists to catch. Two of its checks are worth knowing about:

- The redundancy read in `_init_mega_moe_experts` must be unconditional, not
  gated on `self.enable_eplb`. MTP reuses `DeepseekV4DecoderLayer` with EPLB
  off, so gating it would size MTP layers 384/48 while target layers sat at
  392/49 — two `MegaMoEShape` keys, hence two `MegaMoEV2` instances and two
  ~7.7 GiB sets of mori symmetric buffers instead of one shared runtime.
- The verifier parses with `ast` instead of importing. Importing pulls in
  `vllm.platforms.rocm`, whose module-level `_get_gcn_arch()` calls
  `torch.cuda` — there is no GPU in a docker build.

It is also a `COPY`'d file rather than a `RUN python - <<HEREDOC`. The legacy
(non-BuildKit) builder does not support heredocs: that form feeds python an
empty stdin and exits 0, i.e. a check that passes on an unpatched image.
Confirmed by building against the stock files.

## Running an EPLB arm

`ENABLE_EPLB=1` on `dsv4_agentic_g05.sh` or `dsv4_quick_ab_g05.sh`, with
`EP_SIZE > 1`. `NUM_REDUNDANT_EXPERTS` must keep
`(n_routed_experts + N) % ep_size == 0` — a multiple of 8 for this 384-expert
checkpoint on 8 ranks. EPLB stays off by default.
