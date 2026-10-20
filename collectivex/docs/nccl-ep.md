# NCCL EP

## Which rows follow an engine

vLLM and SGLang expose no NCCL EP selector. TensorRT-LLM does: its `NcclEP` MoE communication
strategy runs NCCL EP's low-latency algorithm with the rank-major layout, and nothing else. It is
auto-selected after TensorRT-LLM's NVLink one- and two-sided strategies, or forced with
`TRTLLM_FORCE_COMM_METHOD=NCCL_EP`. Claims below were checked against TensorRT-LLM main
`a6142c920` (`tensorrt_llm/_torch/moe/fused_moe/`) and nccl-extensions `1ee7136`, the last commit
before the pinned nightly.

| Row | Configuration | `maturity` |
|---|---|---|
| `low-latency`, rank-major, BF16 | TensorRT-LLM's `NcclEP` | `production` |
| `low-latency`, rank-major, DS-FP8 (`--precisions fp8`) | library | `candidate` |
| `low-latency`, expert-major (`COLLX_NCCL_LL_LAYOUT=expert-major`) | library | `candidate` |
| `normal` (HT) | library | `candidate` |

`maturity` is set per row, so `implementation.maturity` separates engine rows from library rows.
The registry's per-backend `backend_maturity` still says `candidate`.

## Low-latency: TensorRT-LLM's configuration

- **Group.** `LOW_LATENCY`, the routed expert count, `max_dispatch_tokens_per_rank` = capacity,
  `max_recv_tokens_per_rank` = EP size x capacity, `max_token_bytes` = 2 x hidden. SM budgets are
  left to the library, which gives low-latency every SM (`nccl_ep_utils.py:251-257`).
- **Capacity.** TensorRT-LLM sizes its one group to `min(max_num_tokens, moe_max_num_tokens)`
  (`nccl_ep.py:98-103`, default 8192) and rejects a step above it (`nccl_ep.py:124-129`). Here the
  capacity is exactly the ladder's top, 512 for decode: the smallest TensorRT-LLM configuration that
  admits every rung, so low-latency runs the 512 rung that the other backends' fixed 256-token
  receive drops. Capacity changes what LL measures: each rank-major dispatch resets every routing
  slot of the receive plane, `EP x capacity x top-k` entries (`ll_dispatch_lsa.cuh:652-659`).
- **Zero-copy follows the regime.** TensorRT-LLM dispatches into a registered NCCL window only
  when it is not capturing. Under CUDA graph capture it dispatches into a plain tensor, and the
  library copies every received token out of its staging buffer (`nccl_ep.py:305-327`,
  `ll_dispatch_lsa.cuh:484-490`). So graphed rows, the default, run the staged dispatch, and eager
  rows (`COLLX_CUDA_GRAPH=0`) write the window directly. The window path exists only within one
  NVLink domain, for rank-major dispatch without internal quantization (`nccl_ep.cc:4589-4600`).
  Scale-out, expert-major and DS-FP8 rows stage in both regimes. A zero-copy row creates its group
  with `ZeroCopyMode.ON`, which fails any dispatch that lacks the window, so the `-zc` label cannot
  name a path that did not run. A staged row uses `OFF`, which equals TensorRT-LLM's default `AUTO`
  at this pin. Combine always stages in low-latency mode.
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

FP8 uses NCCL EP's internal `DS_FP8E3M4` recipe, for low-latency decode only. TensorRT-LLM never
selects it: `NcclEP` sets `uses_internal_fp8_dispatch = False` and transports DeepSeek-FP8
activations in BF16 (`nccl_ep.py:78-79`). The default matrix therefore omits nccl-ep FP8
(`sweep_matrix.OFF_PATH_PRECISIONS`, which lists no other precision for any backend).
`--precisions fp8` adds it back.

## Row identity

`kernel_generation` is a family, `nccl-ep-v02-ht-routed-zc-static` (HT, plus `-gum1` when graphed),
`nccl-ep-v02-ll-rm` (plus `-zc` on the eager window path) or `nccl-ep-v02-ll-em`, then
`-fp8-ds-e3m4` on FP8 rows, `-cap<N>-<wheel version>`, and the harness's regime suffix. HT's `-zc`
is part of its family: HT always takes the window. Examples:

- `nccl-ep-v02-ll-rm-cap512-0.2.0.dev20261005-cudagraph-spin-ar`: graphed low-latency, staged.
- `nccl-ep-v02-ll-rm-zc-cap512-0.2.0.dev20261005-eager-spin-ar`: eager low-latency, zero-copy.
- `nccl-ep-v02-ht-routed-zc-static-gum1-cap512-0.2.0.dev20261005-cudagraph-spin-ar`: graphed
  HT decode.
- `nccl-ep-v02-ht-routed-zc-static-cap8192-0.2.0.dev20261005-eager-spin-ar`: HT prefill.

`-gum1` appears only on graphed HT rows, the only capture that holds a host collective on this
communicator (the routing `ncclAllGather`). The communicator runs graph usage mode 1 in every mode,
but the mode only matters to that capture.

`EPBackend.implementation_details()` returns the realized configuration, which the result document
publishes as `implementation.library_config`: `capacity_tokens_per_rank`, `zero_copy`, `num_sms`
(`dispatch` and `combine`), and `engine_contract` (`tensorrt-llm`, or null for a library row). The
wheel version stays in `implementation.library_version`. `num_sms` repeats the library's
resolution: 16 for HT and every SM for low-latency, then `NCCL_EP_COMM_SMS`, then
`NCCL_EP_DISPATCH_SMS` and `NCCL_EP_COMBINE_SMS`, each ignored outside 1 to the device's SM count
(`nccl_ep.cc:2219-2276`). The group reads these variables when it is created.

## Nightly version

`runtime/common.sh` pins `nccl-extensions[cu13]==0.2.0.dev20261005` alongside
`nccl4py[cu13]==0.5.0`. Installation uses `--pre --extra-index-url https://pypi.nvidia.com`;
the package specification keys the backend cache. Each result records the installed
wheel version verbatim in `implementation.library_version`, including the nightly date, and in
`kernel_generation`. The value comes from distribution metadata, not the requested pin or native
library version.
