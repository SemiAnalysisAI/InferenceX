# Kimi Disaggregated Experiment Notes

## Separation of Concerns

- CI recipe changes should be reproducible without MIA-local paths.
- Local replay belongs in `benchmarks/multi_node/local_runner/`.
- Persist logs under `/it-share/yichaozhu/...` or `/data/yichaozhu/...`.

## MoRI + LMCacheMP

Current intended pattern for the Kimi MoRI LMCache agentic line:

- PD transfer uses `MoRIIOConnector`.
- LMCacheMP provides prefill-side DRAM/L2 prefix reuse.
- Decode should normally use MoRIIO for transferred KV and should not independently load/store the same paged KV blocks through LMCacheMP unless that path has been proven safe.
- In MultiConnector, ensure only the PD transfer connector owns proxy-visible `kv_transfer_params`; LMCacheMP should not conflict with PD handoff metadata.

Important generated fields:

- `kv-offloading: dram`
- `kv-offload-backend: lmcache`
- `total-cpu-dram-gb`
- `_kvdram-lmcache` suffix in `exp-name`

## Mooncake + LMCache

The old Mooncake TCP path was useful as a host-staged compatibility baseline. When using vLLM router with Mooncake connector:

- Confirm whether the path is host-staged TCP or HIP/RDMA.
- If HIP registered memory descriptors are missing, verify whether the run is accidentally on a GPU-memory/RDMA path instead of the intended host-staged TCP route.
- Do not assume toy proxy behavior maps to vLLM router behavior; compare routing and PD handshake responsibilities explicitly.

## Router vs Toy Proxy

Prefer vLLM router for productizable recipes. Use toy proxy only when isolating protocol handshakes or debugging a feature gap.

When replacing toy proxy:

- Identify how prefill/decode registration and discovery happen.
- Confirm router policy and backend endpoints.
- Confirm PD transfer metadata reaches decode.
- Confirm benchmark client uses the served model name expected by the router.

## Long Context Runs

Agentic 262k requests can spend a long time in decode. Do not declare a hang only because a long request is still running. Check:

- `num_requests_running`
- `num_requests_waiting`
- decode logs advancing
- output token counters
- cache usage
- router timeout or backend error logs

## Result Interpretation

Keep these counters separate:

- `profile_export.jsonl` line count: total records, may include invalid/error/cancelled metadata records.
- `num_requests_successful`: successful completed requests in the aggregate result.
- AIPerf phase `completed`: successful completed requests at phase end.
- AIPerf phase `cancelled`: in-flight credits cancelled when profiling duration expires.

For clean runs these can match, but AgentX/K2.5 artifacts can have `profile_export.jsonl` lines greater than successful requests. Do not use JSONL line count as completed request count without checking errors/cancelled records.

The K2.7 1P2D data collected in this investigation has no discovered 60-minute LMCache conc32 result. Found LMCache conc32 JSONs were short runs (`duration_seconds` around 629s or 930s). Keep short K2.7 conc32 and K2.5 AgentX conc32 separate from the K2.7 60-minute plots.
