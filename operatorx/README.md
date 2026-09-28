# operatorx

Inference operator benchmarks: times one op at a time on NVIDIA and AMD through the serving framework's own layers (vLLM).

| Op type | Schema | What it is |
| --- | --- | --- |
| `gemm` | `ops/gemm.py` | dense GEMM, quantized operands (`a` activation, `b` weight) |
| `moe` | `ops/moe.py` | one MoE layer, router GEMM to combined output |
| `mla`, `mla_dsa`, `dsv4_attn`, `gqa`, `qsa`, `gdn`, `kda` | `ops/attention.py` | whole attention modules, one op type each |

- Split over devices: each case's `parallel` arg (`{"tp", "dp", "ep", "dcp"}`), defined in `core/parallel.py`.
- Testlists: `testlists/*.json`; every entry lists its `sources` (`<org>/<model>/<role>`). Dispatch `attn_*` testlists apart from gemm / moe ones (each builds its own vLLM engine).

## Running

Dispatch the **OperatorX Sweep** workflow ([CI.md](CI.md)). The same planner runs by hand on a login node:

```bash
PYTHONPATH=.:inferencex-e2e python3 -m operatorx.ci plan --platform-config operatorx/platforms.json \
  --runner cluster:h200-dgxc --backends vllm --testlists gemm_parallel --world-sizes 1,2,4,8 \
  --chunk-size 500 --mode timing --run-id local --attempt 1 --source-sha "$(git rev-parse HEAD)" \
  --out manifest.json
```

Results: one JSON per run under `results/<platform>/<cluster>/`; in CI, the `operatorx-shard-*` artifacts, then ingested into the OperatorX database.

## Adding a backend / op

- Backend: `runners/<platform>/backends/<name>.py` exporting `IMPLS = [BackendImpl(...)]`; its image in `containers.toml`.
- Op: `ops/<name>.py` calling `register(OpSpec(type=..., arg_schema=..., parallel_axes=...))`.
- Shapes: `testlists/<name>.json`.
