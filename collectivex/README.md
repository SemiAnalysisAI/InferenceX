# CollectiveX - Experimental (Beta)

CollectiveX is an experimental MoE expert-parallel communication benchmark. It measures dispatch,
combine and paired roundtrip latency across EP libraries and accelerator systems, and writes one
result file per case into `results/`, where it is committed. A second suite, the
[vLLM `swap_blocks` benchmark](docs/swap-blocks.md), measures pinned CPU↔GPU and same-GPU block
copies through the same matrix and pool launchers.

CollectiveX is run by hand on each pool (see [Running By Hand](#running-by-hand)). Committing the
results is the publish step: the dashboard builds from whatever `results/` holds on `main`. It does
not rank, recommend, or select; any downstream display or comparison is the consumer's
responsibility. [docs/methodology-summary.md](docs/methodology-summary.md) states the measurement
rules, including each backend's engine contract, and [docs/methodology.md](docs/methodology.md)
holds the evidence behind them.

## Coverage

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain. `configs/platform_config.json` registers which backend runs where. `normal` mode
runs the decode and prefill ladders; `low-latency` is decode-only.

| Backend | Engine selector (`maturity`) | `normal` | `low-latency` |
|---|---|---|---|
| `deepep-v2` | vLLM `--all2all-backend deepep_v2`, SGLang `--moe-a2a-backend deepep` (`production`) | EP8, EP16: H100, H200, B200, B300, GB200, GB300 | EP8: H200. EP8, EP16: B200, GB200, GB300 |
| `mori` | vLLM `--all2all-backend mori_*`, SGLang `--moe-a2a-backend mori` (`production`) | EP8: MI300X, MI325X, MI355X. EP16: MI355X | EP16: MI355X |
| `uccl-ep` | none; engines load it as a drop-in `deep_ep` under vLLM's DeepEP backends (`candidate`) | EP8: H100, H200, B200, MI300X, MI325X, MI355X. EP16: B200 | EP8: H100, H200, B200 |
| `nccl-ep` | TensorRT-LLM `NcclEP` (`production` for low-latency rank-major BF16, `candidate` otherwise) | EP8, EP16: H100, H200, B200, B300, GB200, GB300 | EP8: H100, H200, B300. EP8, EP16: B200, GB200, GB300 |
| `flashinfer-ep` | vLLM `--all2all-backend flashinfer_nvlink_one_sided` (`production`) | EP8, EP16: GB200, GB300 | none |

Every backend runs a BF16 dispatch control and an FP8 dispatch, except that nccl-ep's FP8 (its
internal DS_FP8E3M4 recipe, low-latency decode only) is off every engine's path and runs only when
`--precisions fp8` names it. Combine is always BF16. Coverage is uniform routing only.

`results/cases.json` is the committed copy of the full matrix: 195 requested cases, of which 163 are
runnable EP cases, 18 are swap-blocks cases (two per pool) and 14 are unsupported EP cells (UCCL-EP
EP16 on H100, H200, MI300X, MI325X and MI355X, and MoRI EP16 on MI300X and MI325X, once per phase).
A cell is listed as unsupported only when its backend runs on that SKU at another EP size, and
low-latency adds only runnable cells, so a missing low-latency cell (DeepEP V2 on B300, for example)
appears nowhere; [docs/methodology.md](docs/methodology.md#backends) gives each reason. The NCCL EP
contract, row identity and wheel pin are in [docs/nccl-ep.md](docs/nccl-ep.md).

## Running By Hand

Run from the pool's Slurm login node, in a checkout of the commit you want to measure:

```bash
collectivex/run.sh h200-dgxc --list-shards          # what will run, one allocation per shard
setsid nohup collectivex/run.sh h200-dgxc > run-h200.log 2>&1 < /dev/null &
collectivex/run.sh h200-dgxc --shard h200-dgxc-nccl-ep-bf16-n2   # rerun one failed shard
```

`run.sh` resolves the pool's cases with `sweep_matrix.py` (`--suite`, `--backend`, `--ep`,
`--modes` and `--precisions` narrow it, and `--swap-profile` picks the swap-blocks grid), gives
each shard (one backend, mode, precision and node count) its own allocation through the pool's
launcher, and copies every result to `results/<sku>/<case_id>.json`, replacing the previous result
for that case. A failed shard does not stop the others; the closing summary lists each shard's exit
code and runs `summarize.py` over the pool's results. Interrupting `run.sh` releases its
allocations. Runtime logs stay under `/tmp/inferencex-collectivex-<uid>/logs/<execution>/` on the
login node.

h200-dgxc, b200-nscale and gb200 run from the registry alone. h100-dgxc, b300, gb300 and the AMD
pools also need an operator document that names a stage directory (see
[Runner Configuration](#runner-configuration)).

These variables change what a row measures. The result records each one (in `kernel_generation`,
`implementation.fp8_consume` or `implementation.rank_alignment`), but the `case_id` does not, so a
run with one replaces the case's committed default-configuration result:

| Variable | Default | Effect |
|---|---|---|
| `COLLX_CUDA_GRAPH=0` | `1` | times graph-capable decode cells eager (`-eager-spin-ar`) |
| `CX_FP8_CONSUME=dequant` | `native` | stages every FP8 pair inside the chain, a verification hatch |
| `COLLX_NCCL_LL_LAYOUT=expert-major` | `rank-major` | runs nccl-ep low-latency on its expert-major layout |
| `COLLX_ALIGN_SPIN_US` | `2000` | the per-sample rank-alignment spin |

`COLLX_TIME` (allocation minutes: 300 per EP shard and 45 per swap-blocks shard unless set),
`COLLX_RUN_TIMEOUT` (the per-case hang guard, 5400 s) and `COLLX_IMAGE_REFRESH=1` (re-import the
image) change only how a shard runs.

Results record the commit they were measured at, so `run.sh` refuses to start while tracked files
under `collectivex/` (outside `results/`) have uncommitted edits. `--allow-dirty` runs anyway and
marks the results `source_dirty`, so they cannot be published. To publish, review
`git diff collectivex/results/<sku>`, check it with
`python3 -m unittest collectivex/tests/test_results.py`, and open a PR with the files. That test
also runs in CI: every committed result must be a runnable case in `results/cases.json`, come from
committed code, stay under 1 MiB, and contain no IP addresses or host-identifying fields.
Regenerate `cases.json` whenever the configs change:

```bash
python3 collectivex/sweep_matrix.py --suites ep,swap-blocks --cases-json collectivex/results/cases.json
```

## Runner Configuration

Each SKU's Slurm, storage, image and network values come from its entry in
`configs/platform_config.json`, which also records architecture and product, fixed placement,
launcher, runnable backend/EP pairs and the scale-out `fabric` identity (NIC and switch, so
same-GPU clusters on different fabrics are distinct entries). An optional operator document at
`COLLECTIVEX_OPERATOR_CONFIG`, or else `${XDG_CONFIG_HOME:-$HOME/.config}/inferencex/collectivex.json`,
overrides the entry's `operator` block field by field:

```json
{"runners": {"b300": {"storage_roots": ["/data/home/<you>"]}}}
```

It accepts `partition`, `account`, `qos`, `squash_dir`, `stage_dir`, `storage_roots`,
`enroot_cache_path`, `exclude_nodes`, `nodelist`, `lock_dir` and the network fields below. An
unknown field, a SKU the registry lacks and non-JSON input fail closed, and no value is ever
evaluated as shell. Duplicate JSON keys are not rejected (`json.load` keeps the last), and entries
for other SKUs are not validated, so a misspelled SKU is silently ignored. The registry's `network`
block is applied after the document and wins for every key it sets.

`stage_dir` must be visible at the same path on the login node and every allocated node. Each shard
copies `collectivex/` into a mode-0700 `job_<execution>` child there and removes it once its
allocation is confirmed released. Without one, the stage goes beside the image under `squash_dir`,
which h200-dgxc and b200-nscale do; h100-dgxc, b300, both GB pools and the AMD pools require one,
and gb200's registry `storage_roots` supplies it.
`storage_roots` derives both `stage_dir` and `squash_dir` (`<root>/collectivex/stage` and
`<root>/collectivex/containers`, from the first root where both can be created) after the document
is merged in, so on gb200, whose registry entry carries `storage_roots`, a lone `stage_dir` in the
document is overridden: give the document its own `storage_roots` there. A run under an account
other than the one that owns the registry's `squash_dir` (a CI account's home on h100-dgxc and b300)
needs its own `squash_dir` or `storage_roots` as well: each shard creates its backend cache under
`squash_dir`, and an image squash another account owns is reused only while it is current.

Every non-MNNVL EP16 placement requires `rdma_devices`, the operator-approved scale-out devices.
The other network fields are `socket_ifname`, `ib_gid_index`, `rdma_service_level`,
`rdma_traffic_class`, `rail_isolated`, `rdma_fabric`, `rdma_relaxed_ordering`, `rdma_gbs` and
`single_node_rdma_devices`. Without `socket_ifname` each node's default-route interface carries the
cross-node sockets (h100-dgxc, h200-dgxc, b300). HCAs are never guessed, and every scale-out node must
show its socket interface up and an active port on each listed device before backend setup; the
methodology's [Execution Isolation](docs/methodology.md#execution-isolation) says what each field
drives.

Enroot imports the image once per pool (once per node where the squash dir is node-local, as on
mi355x) into a squash keyed by image tag and platform; the tag's
manifest digest in its container registry, resolved at launch, decides when it is stale, and
`COLLX_IMAGE_REFRESH=1` forces a re-import. An image owned by another account is reused when
readable and current, but never replaced or deleted; the launcher stops with an error instead.
Backends the image does not ship (DeepEP V2 and UCCL-EP, built at their source pins in
`runtime/common.sh`, the `nccl-extensions` wheels, and aws-ofi-nccl where an EFA host needs it)
are cached in a cluster-local backend cache that reaches the container only as `/cx-cache`.

## Local Checks

```bash
python3 -m unittest discover collectivex/tests -p 'test_*.py'
python3 collectivex/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
for f in collectivex/run.sh collectivex/runtime/*.sh collectivex/launchers/*.sh; do bash -n "$f"; done
collectivex/run.sh h200-dgxc --dry-run   # each shard's environment, without allocating
```

Core paths are `run.sh`, `configs/`, `sweep_matrix.py`, `bench/`, `runtime/`, `launchers/`,
`results/` and `tests/`. `summarize.py` and `bandwidth.py` read results and gate nothing.
