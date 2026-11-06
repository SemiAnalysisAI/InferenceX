# NCCL EP

## Which rows follow an engine

vLLM and SGLang expose no NCCL EP selector. TensorRT-LLM does: its `NcclEP` MoE communication
strategy runs NCCL EP's low-latency algorithm with the rank-major layout, and nothing else. It is
auto-selected after TensorRT-LLM's NVLink one- and two-sided strategies, or forced with
`TRTLLM_FORCE_COMM_METHOD=NCCL_EP`. Claims below were checked against TensorRT-LLM main
`a6142c920` (`tensorrt_llm/_torch/moe/fused_moe/`) and nccl-extensions `1ee7136`, the last commit
before the 2026-10-05 nightly. They have not been rechecked against the pinned 2026-10-21 nightly.

| Row | Configuration | `maturity` |
|---|---|---|
| `low-latency`, rank-major, BF16, inside one scale-up domain (zero-copy, the default; graphed or eager) | TensorRT-LLM's `NcclEP` (NVIDIA/TensorRT-LLM#20631) | `production` |
| `low-latency`, rank-major, BF16, scale-out (staged, graphed or eager) | TensorRT-LLM's `NcclEP` | `production` |
| `low-latency`, rank-major, BF16, inside one scale-up domain, `COLLX_NCCL_LL_ZERO_COPY=0` (staged) | library | `candidate` |
| `low-latency`, rank-major, DS-FP8 | library | `candidate` |
| `low-latency`, expert-major (`COLLX_NCCL_LL_LAYOUT=expert-major`) | library | `candidate` |
| `normal` (HT) | library | `candidate` |

`maturity` is set per row, so `implementation.maturity` separates engine rows from library rows.
The registry's per-backend `backend_maturity` still says `candidate`.

Scale-out means the EP group is larger than the case's scale-up domain (`world_size >
scale_up_domain`, `_internode` in `ep_nccl.py`; the case's `scope` is `scale-out`), so dispatch
crosses the GIN network path. Node count does not decide it: GB200/GB300 EP8 and EP16 span two and
four nodes inside one 72-GPU MNNVL domain and zero-copy. H100, H200, B200 and B300 EP16 are
scale-out.

## Low-latency: TensorRT-LLM's configuration

- **Group.** `LOW_LATENCY`, the routed expert count, `max_dispatch_tokens_per_rank` = capacity,
  `max_recv_tokens_per_rank` = EP size x capacity, `max_token_bytes` = 2 x hidden. SM budgets are
  left to the library, which gives low-latency every SM (`nccl_ep_utils.py:251-257`).
- **Capacity.** TensorRT-LLM sizes its one group to `min(max_num_tokens, moe_max_num_tokens)`
  (`nccl_ep.py:98-103`, default 8192) and rejects a step above it (`nccl_ep.py:124-129`). Here the
  capacity is exactly the ladder's top, 512 for decode and 8192 for prefill: the smallest TensorRT-LLM configuration that
  admits every rung, so low-latency runs the 512 rung that the other backends' fixed 256-token
  receive drops. Capacity changes what LL measures: each rank-major dispatch resets every routing
  slot of the receive plane, `EP x capacity x top-k` entries (`ll_dispatch_lsa.cuh:652-659`).
- **Zero-copy wherever the library supports it.** Low-latency dispatch writes a registered NCCL
  window directly, graphed or eager. The window path exists only within one NVLink/MNNVL scale-up
  domain, for rank-major dispatch (`nccl_ep.cc:4589-4600` at the 2026-10-05 source), so scale-out
  and expert-major rows stage in both regimes. FP8 registers its scale plane alongside the token
  window and follows the same policy. TensorRT-LLM matches this with NVIDIA/TensorRT-LLM#20631:
  its dispatch output is a registered window reserved for the context's lifetime, eager and under
  CUDA graph capture alike, and it hands combine a registered expert-output window. Before #20631
  it dispatched into a plain tensor under capture and the library copied every received token out
  of its staging buffer. So in-domain zero-copy rows and scale-out rows carry `engine_contract`
  `tensorrt-llm` in both regimes. A zero-copy row creates its group with `ZeroCopyMode.ON`, which
  fails any dispatch that lacks the window, so the `-zc` label cannot name a path that did not
  run. A staged row uses `OFF`. Here the low-latency combine reads the registered receive window
  that dispatch wrote, which is the expert output in this harness.
- **`COLLX_NCCL_LL_ZERO_COPY=0`.** Turns low-latency zero-copy off (default `1`; any other value is
  rejected), so in-domain rows run the staged dispatch TensorRT-LLM used under capture before
  #20631. Those rows are library rows. HT ignores it. The row records it twice: the family loses
  `-zc`, and `implementation.library_config.ll_zero_copy_opt_out` is `true`. Like
  `COLLX_NCCL_LL_LAYOUT`, it is read from the launching shell's environment, which every rank
  inherits (`srun --export=ALL`), and the `case_id` does not change, so a run with it replaces the
  case's committed default result.
- **Handle.** One handle, rebound to each ladder point's routing in untimed code. TensorRT-LLM
  also creates one handle and updates it every step. A low-latency update does no device work
  (`nccl_ep.cc:4014-4020`).
- **Remaining differences.** TensorRT-LLM passes int32 top-k ids and asks for global receive ids.
  This adapter passes int64 ids and rebases local ids in the untimed oracle. Neither changes the
  bytes on the wire, whose routing header carries 16-bit expert ids (`include/common.hpp:193`).

## High-throughput: a library row

No engine runs HT. Its timed dispatch still carries `ncclEpUpdateHandle`, because the API requires
it for every routing step. Its HT work is sized by the handle's capacity, not by the token count:
capacity-wide memsets, a routing-map conversion, an `ncclAllGather` of capacity rows, and a
metadata scan over them (`nccl_ep.cc:4057-4137`, `4281`). So every rung carries a term set by the
ladder's top, 512 for decode and 8192 for prefill. That capacity is part of `kernel_generation`,
so a ladder change never pools with older rows. NVIDIA's `ep_bench` times the update separately.
HT runs 16 SMs for dispatch and combine. `-static` marks the combine input: FLAT combine takes
the dispatch output's full static `[num_recv_slots, hidden]` receive plane, because a count-sized
slice needs an `NCCL_EP_AUTO` group (`ep_nccl.py` `_bind_ht_recv_count`). HT also supports caller-quantized FP8 with forwarded
scales (`QUANT_FWD`), which this suite does not measure.

## FP8

FP8 uses NCCL EP's internal `DS_FP8E3M4` recipe, for low-latency prefill and decode. TensorRT-LLM never
selects it: `NcclEP` sets `uses_internal_fp8_dispatch = False` and transports DeepSeek-FP8
activations in BF16 (`nccl_ep.py:78-79`), so FP8 rows are library rows. They run in the default
matrix, like every other backend's FP8.

## Row identity

`kernel_generation` is a family, `nccl-ep-v02-ht-routed-zc-static` (HT, plus `-gum1` when graphed),
`nccl-ep-v02-ll-rm` (plus `-zc` on the window path) or `nccl-ep-v02-ll-em`, then
`-fp8-ds-e3m4` on FP8 rows, `-cap<N>-<wheel version>`, and the harness's regime suffix. HT's `-zc`
is part of its family: HT always takes the window. Examples:

- `nccl-ep-v02-ll-rm-zc-cap512-0.2.0.dev20261021-cudagraph-spin-ar`: graphed low-latency, zero-copy.
- `nccl-ep-v02-ll-rm-cap512-0.2.0.dev20261021-cudagraph-spin-ar`: graphed low-latency, staged
  (scale-out, or `COLLX_NCCL_LL_ZERO_COPY=0`; `library_config.ll_zero_copy_opt_out` tells them apart).
- `nccl-ep-v02-ht-routed-zc-static-gum1-cap512-0.2.0.dev20261021-cudagraph-spin-ar`: graphed
  HT decode.
- `nccl-ep-v02-ht-routed-zc-static-cap8192-0.2.0.dev20261021-eager-spin-ar`: HT prefill.

`-gum1` appears only on graphed HT rows, the only capture that holds a host collective on this
communicator (the routing `ncclAllGather`). The communicator runs graph usage mode 1 in every mode,
but the mode only matters to that capture.

`EPBackend.implementation_details()` returns the realized configuration, which the result document
publishes as `implementation.library_config`: `capacity_tokens_per_rank`, `zero_copy`, `num_sms`
(`dispatch` and `combine`), `ll_zero_copy_opt_out` (`COLLX_NCCL_LL_ZERO_COPY=0` on a low-latency
row), and `engine_contract` (`tensorrt-llm`, or null for a library row). The
wheel version stays in `implementation.library_version`. `num_sms` repeats the library's
resolution: 16 for HT and every SM for low-latency, then `NCCL_EP_COMM_SMS`, then
`NCCL_EP_DISPATCH_SMS` and `NCCL_EP_COMBINE_SMS`, each ignored outside 1 to the device's SM count
(`nccl_ep.cc:2219-2276`). The group reads these variables when it is created.

## Nightly version

`runtime/common.sh` pins `nccl-extensions[cu13]==0.2.0.dev20261021` alongside
`nccl4py[cu13]==0.5.0` and `cuda-core==1.2.1`. Installation uses `--pre --extra-index-url https://pypi.nvidia.com`;
the package specification keys the backend cache. Each result records the installed
wheel version verbatim in `implementation.library_version`, including the nightly date, and in
`kernel_generation`. The value comes from distribution metadata, not the requested pin or native
library version.

Prefill runs eager; decode remains graph-capable. FP8 zero-copy on the October 15 wheel
still needs GPU validation. Combine remains BF16 and staged for both LL phases.
