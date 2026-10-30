---
name: serving-bench
description: Benchmark an LLM serving config on a VNG GPU node with the AgentX coding scenario, the tau2 or LiveKit agent traces, the E2/E3 prefix traces, or the synthetic ISL/OSL/cache grid (fixed concurrency or SLO search), starting from a vllm/sglang launch command, a Dynamo frontend + engine pair, a docker-compose file or a k8s manifest. Use when the user says "benchmark agentx với config…", "chạy agentx/tau2/livekit cho config này", "bench sweep", "tìm max CCU", or pastes a serve command and asks for a benchmark. Writes one serving YAML and dispatches it with one command.
---

# serving-bench

You turn the user's serving config into **one YAML file** and run **one command**. Do not write recipes, edit `configs/runners.yaml`, or call `gh workflow run` yourself. How it runs end to end: `inferencex-e2e/docs/serving/ARCHITECTURE.md`.

## 1. Write `inferencex-e2e/configs/serving/<name>.yaml`

```yaml
name: qwen38-bf16-tp1               # [a-z0-9.-], no "_"; prefixes the artifact names
runner: cluster:hcm-hgx02           # see "Runner" below
image: vllm/vllm-openai:v0.29.0     # required; or /abs/path.sqsh on the runner's node
model: Qwen/Qwen3.8-27B             # HF repo id, never a local path
gpus: 1                             # GPUs the command needs
replicas: 1                         # optional: N engine copies, gpus/N each; N > 1 needs router: with {{engine_urls}}
precision: bf16                     # optional (bf16 default), naming only
env:                                # optional, the engine's env vars; quote booleans
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
serve: |                            # the launch command, as given
  vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --max-model-len 262144
router:                             # optional: any router in front of the engine(s)
  image: <router image>
  env: {}
  # router /metrics on its own port: put {{router_metrics_port}} in serve (picked free at run time)
  serve: |                          # {{router_port}} = the port the benchmark calls, {{engine_url(s)}} = the engine(s)
    python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}} --worker-urls={{engine_url}}
storage:                            # optional: a non-GPU sidecar started first (e.g. Mooncake master)
  image: <image>
  serve: mooncake_master --port={{storage_master_port}} --http_metadata_server_port={{storage_meta_port}}
arms:                               # optional: engine variants in one run, e.g. base vs speculative (see ARCHITECTURE.md, Arms)
  base: {}
  mtp6: {append: "--speculative-config '...'"}
scenario: agentx                    # agentx | tau2 | livekit | shared-prefix-e2 | hidden-prefix-e3 | synthetic | greenspec | speedbench
overrides: {duration: 180}          # optional variant, see the table below
conc: [4, 8]                        # one benchmark job per value
```

Working examples, one per pattern: `inferencex-e2e/configs/serving/*.yaml`. They are format references: never dispatch them or copy their values for a user's request.

### Scenarios

| `scenario:` | Workload | Default length |
|---|---|---|
| `agentx` | SemiAnalysis coding traces (weka 062126), needs 262144 context | 3600 s |
| `tau2` | tau2 customer-service agent traces | 1200 s |
| `livekit` | LiveKit voice-agent traces, V1.1 warm (dial warm-ups count in TTFT) | 1200 s |
| `shared-prefix-e2` / `hidden-prefix-e3` | prefix-cache A/B traces (HiCache L2 vs Mooncake L3) | 1800 s |
| `synthetic` | synthetic ISL/OSL with a shared-prefix cache hit, needs `workload:` | 300 s per point |
| `greenspec` | speculative-decoding bench from production chat traffic (private; result has `spec_decode.acceptance_length`) | the whole selection, no duration |
| `speedbench` | SPEED-Bench acceptance-length prompts (private copy of public data; result has `spec_decode.acceptance_length`) | the whole selection, no duration |

Variants (`overrides:`, only these keys):

| You want | Write |
|---|---|
| a smoke run | `overrides: {duration: 180}` |
| the 256k AgentX corpus | `overrides: {dataset: semianalysis_cc_traces_weka_062126_256k}` |
| AgentX without the 262144 cap | `overrides: {max_context_length: null}` |
| sampling instead of greedy | `overrides: {generation: {top_p: 0.95, temperature: 1.0}}` |
| LiveKit V1 / V2 trace | `overrides: {dataset: livekit-111868-v1}` (or `-v2`) |
| a different trace idle-gap cap | `overrides: {trace_idle_gap_cap: 1}` |
| GreenSpec subset, lite, bucket, cap | `scenario: greenspec` + `overrides: {subset: agentic_tools, lite: true, max_tokens: 4096, requests: 200}` (only these keys; no `duration`) |
| SPEED-Bench config, category, cap | `scenario: speedbench` + `overrides: {config: qualitative, category: coding, requests: 40}` (config: `qualitative` or `throughput_{1k,2k,8k,16k,32k}`; no `duration`) |

Synthetic grid, one job per point, all in one run:

```yaml
scenario: synthetic
workload: {isl: [8k, 32k, 64k, 128k], osl: [512, 1k], cache: [0, 50, 90], search: true}
slo: "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"   # optional; this is the default
conc: 1                             # search: true finds the concurrency itself
```

`search: false` runs each point at every `conc` value instead (then only one point, or one `conc`).

Old scenario names (`agentx-cc-062126-smoke`, `custom-agentic-tau2`, `sweep-isl8k-osl1k-cache50`, …) map as in the table of `docs/serving/ARCHITECTURE.md` § Scenarios. Results keep the old names.

### Remote endpoint (no engine)

To benchmark an endpoint already running (GreenNode MaaS, a k8s service, OpenAI/Anthropic/Gemini
via an OpenAI-compatible API), write `endpoint:` instead of `image`/`model`/`gpus`/`serve`
(example: `inferencex-e2e/configs/serving/remote-openai-agentx-smoke.yaml`):

- `url`: base URL; a trailing `/v1` is fine. `model`: the served name as the API expects it.
- `api-key-secret`: the GitHub secret holding the key (`gh secret set <NAME> --repo vngcloud/InferenceX`).
  Never put the key in the YAML. Omit for an open endpoint.
- `tokenizer`: `builtin` (default; exact for OpenAI GPT-4o/GPT-5, approximate for other closed
  models), `cl100k_base`/`o200k_harmony`, or the HF repo id of an open model.
- `runner`: any VNG runner; the job takes 16 CPUs and 64 GB, no GPU.
- No server metrics or GPU telemetry in the result. ISL/OSL come from the API's `usage`; if it
  streams none, the job counts client-side and the aggregate says `token_count_source: client`.
- A probe request runs first: 401/404/400 fail the job in seconds with the fix. A 400 naming
  `ignore_eos` means the API rejects the field the AgentX scenario always sends; tell the user,
  there is no workaround yet.

### Translation rules

- **Model path → HF id.** `--model-path /models/<org>/<model>` becomes `--model-path <org>/<model>` and `model: <org>/<model>`. If you cannot tell the HF id from the path, ask.
- **Host/port:** leave them or drop them; the benchmark picks its own free ports.
- **Router:** the router part goes under `router:` with its own `image`; the engine stays in `serve`. Engine env → `env:`, router env → `router.env:`.
  - **Replicas** (e.g. 2× TP1): `replicas: N`, `gpus` = N × per-engine GPUs, the router lists `{{engine_urls}}`. Not with the Dynamo frontend.
  - **Dynamo** (`python3 -m dynamo.frontend` + `python3 -m dynamo.sglang`): leave out `DYN_*`, `ETCD_ENDPOINTS`, NATS and `--http-host/--http-port`; the benchmark wires them.
  - **Any other router:** keep the user's flags except the wiring. Listen flags → `--host=0.0.0.0 --port={{router_port}}`. A worker URL → `{{engine_url}}` (or `{{engine_port}}`). A metrics port flag (`--prometheus-port=18888`) → `--prometheus-port={{router_metrics_port}}`, picked free at run time like every other port. Never a literal port: two jobs on one node would collide. (`router.metrics_port: <port>` still exists for a port the user cannot change.)
  - **Router command missing** (only flags + image): infer it from the flag family and say so. `--worker-urls`, `--policy=cache_aware`, `--prometheus-port` → `python3 -m sglang_router.launch_router`. Unknown family → ask. Slim images without python3 run a binary path.
  - Placeholders exist only in `router.*`, `storage.*` and (storage ports only) `env:`. No `$VAR` anywhere.
- **Env vars** (`NCCL_P2P_LEVEL=NVL`, compose `environment:`, k8s `env:`) go under `env:`. No secrets: the YAML is a run input and an artifact. Dataset access uses the runner's `GREENLAB_HF_TOKEN`.
- **Images:** public ones and `vcr.vngcloud.vn` ones the runner can read need nothing. Another private registry fails at `enroot import`: tell the user it needs infra (credentials for `stackops` or a pre-imported squash).
- **A `.sqsh` file** (an image built with enroot, e.g. `/data/<user>-enroot/images/x.sqsh`): use the absolute path as `image:`; it runs as is, never imported. It exists on one node only, so ask which node holds it and set `runner: cluster:<id>` of that node; a pool label fails validation. The file and its directories must be readable by `stackops`; the driver checks that and `unsquashfs -l` before submitting. An image several nodes need belongs in a registry instead.
- **gpus:** tensor-parallel × pipeline-parallel (× replicas). If unclear, ask.
- **Runner** (each VNG node is one cluster; the job takes only its GPUs, 16 CPUs and a memory share per GPU, the rest of the node stays usable by the team):

  | Node | `runner:` | GPUs |
  |---|---|---|
  | hgx-h200-01 | `cluster:hcm-hgx01` | 4 (until infra declares all 8) |
  | hgx-h200-02 | `cluster:hcm-hgx02` | 8 |
  | either hgx | `h200-hcm` | up to 4 |
  | RTX PRO 6000 node 1 / 2 | `cluster:hcm-rtx01` / `cluster:hcm-rtx02` | 2 |
  | either RTX node | `rtx6000pro-hcm` | up to 2 |
  | HAN H200 node 1 / 2 / 4 | `cluster:han-h200-01` / `-02` / `-04` | 1 (16 CPU, ~110 GB RAM) |
  | either HAN node | `h200-han` | 1 |

  Any value outside this table (e.g. the legacy `hardware-hcm-8x_01`-style labels) is out-of-scheme: dispatch rejects it, and it would pin the whole run to one runner slot.

  Check load first: `ssh slurm-client squeue -p hardware-hcm` (or `-p rtx6000pro-hcm`, `-p test` for HAN). If the node is busy, tell the user the job waits in the Slurm queue. Never cancel another user's job.
- **docker-compose / k8s:** `image` from `image:`; `serve` from `command:` + `args:`; `env` from `environment:` / `env:`.
- **Drop `--numa-node …`.**
- **DeepEP on hgx** (`--moe-a2a-backend deepep`): add `NVSHMEM_REMOTE_TRANSPORT: none` to `env:`.
- **JSON in a flag must be single-quoted:** `--kv-events-config='{"publisher":"zmq"}'`.
- **Context length:** `agentx` needs 262144 (`--max-model-len 262144` / `--context-length 262144`); ask if the config cannot. Trace scenarios are fine with the model default unless it is below 32k.

## 2. Ask the user (do not guess) when

Asking means: stop, write nothing, dispatch nothing, and end your reply with the question.

- No image and no version → ask for the exact tag. Never take it from `configs/serving/*.yaml` or past runs.
- A **version** but no image → resolve it: SGLang → `lmsysorg/sglang:v<ver>`, vLLM → `vllm/vllm-openai:v<ver>`. Verify the tag exists: `curl -s "https://hub.docker.com/v2/repositories/<repo>/tags?name=<tag>&page_size=50"`. Not found → ask. Router images are never resolved this way.
- No scenario → list the six with their lengths and ask.
- `gpus` fits no node in the runner table → ask.
- A Dynamo router with a non-Dynamo engine, or the reverse → the validator rejects it; explain and ask.

## 3. Dispatch

```bash
inferencex-e2e/utils/serving/dispatch.sh inferencex-e2e/configs/serving/<name>.yaml
```

It validates first. On `ERROR <field>: <fix>` lines, fix the YAML and rerun; nothing was dispatched. On success it prints the run URL. The branch must be pushed (the script says so); do not push without the user's OK.

## 4. Results

- Status: `gh run view <run-id> --repo vngcloud/InferenceX`. One job per point (`conc` value or grid point). Expect about 20 min of image import, model load and warmup on top of the scenario length.
- Artifacts: `agentic_<exp-name>_…` (`results/`: `serving.yaml`, `scenario.yaml`, `server_command.txt`, `server.log`, `router.log`, `storage.log`, `benchmark_command.txt`, `benchmark.log`, `aiperf_artifacts/`) and `bmk_agentic_<exp-name>_…` (the aggregate JSON). For a report, use the `result-report` skill.
- A search point's aggregate carries `state` (`ok`, `cliff`, `no_pass_in_range`, `no_failure_in_range`, `max_iterations`, `no_data`), `max_conc` and the metrics at `max_conc`. Every state is a result, not a failure.
- A failed job: `server.log` first, then `benchmark.log`; `router.log` / `storage.log` when it fails before aiperf starts. `gh run view <id> --repo vngcloud/InferenceX --log-failed | tail -100` shows the Slurm/enroot side.
- If the engine crashes, report the error and the config as given. Do not change flags or redispatch a modified config on your own.
- Report: run URL, per-point success/failure, where the artifacts are.
