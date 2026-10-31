# Serving YAML: remote endpoint mode

Date: 2026-10-27. Branch: `refactor/vng-benchmark`.

## Goal

Benchmark an endpoint we do not launch (GreenNode MaaS gateway, a k8s service, OpenAI,
Anthropic or Gemini behind an OpenAI-compatible API) from a serving YAML that names only the
URL, the model, the API key secret and, optionally, the tokenizer. Same scenarios, `conc`
lists, artifacts and `result-report` as the engine path.

Replaces the BYO-endpoint path the refactor dropped (`remote-bench-e2e.yml`,
`remote_bench_preflight`, one recipe per model). That path required a `/metrics` URL, an
image, a hardware key and a recipe file; none of those exist for a closed API.

## YAML

```yaml
name: gpt5-agentx-remote
runner: cluster:hcm-rtx01          # any VNG runner; the job takes no GPU
endpoint:
  url: https://api.openai.com/v1   # a trailing / or /v1 is stripped
  model: gpt-5                     # the served name, sent as "model"; free-form
  api-key-secret: OPENAI_API_KEY   # optional; name of a GitHub Actions secret
  tokenizer: builtin               # optional, default builtin
scenario: agentx
overrides: {duration: 180}
conc: [4, 8]
```

Rules (`infx/serving/schema.py`, every problem reported at once like today):

- `endpoint:` excludes `image`, `serve`, `gpus`, `replicas`, `precision`, `env`, `router`,
  `storage`. Without `endpoint:`, `image`, `serve`, `gpus` stay required.
- `endpoint.url` must start with `http://` or `https://`.
- `endpoint.model` is free-form (`gpt-5`, `z-ai/glm-5.2`); no HF-id check.
- `endpoint.api-key-secret` matches `[A-Z][A-Z0-9_]*` (GitHub secret names). The key itself
  never appears in the YAML, the run inputs or any artifact.
- `endpoint.tokenizer`: `builtin`, a tiktoken encoding (`cl100k_base`, `o200k_base`,
  `o200k_harmony`), or an HF repo id `<org>/<name>`.
- No `metrics-url`: a remote run collects neither server metrics nor GPU telemetry.

## Tokenizer

The harness uses the tokenizer for three things. What a mismatch with the served model costs:

| Use | Effect of a non-matching tokenizer |
|---|---|
| Prompt synthesis (trace `hash_ids` + lengths → token ids → decoded text) | Prompt lengths are approximate in the served model's tokens; equal `hash_ids` still decode to equal text, so shared-prefix structure is kept |
| `maxContextLength` trace filter | Approximate; leave headroom against the target's window |
| ISL/OSL in results | None while the server reports usage (see below) |

Default `builtin` = tiktoken `o200k_base`, no network or HF access. It is the real tokenizer
for OpenAI GPT-4o/GPT-5. For other closed models it is the best available approximation.
For open models set the HF repo id; the harness fetches the tokenizer files only, not the
weights. Gated repos use the job's existing `HF_TOKEN`.

## Token counting

Every scenario sets `useServerTokenCount: true`: the harness adds
`stream_options.include_usage` and reads ISL/OSL from the last `usage` chunk. When no chunk
carries `usage`, harness 1.0.12 logs a warning and records `None`; it does not fall back to
client-side counting, and the CLI has no flag to turn the setting off.

The client therefore probes the endpoint before the run (next section) and picks the source:

- usage present → keep `useServerTokenCount: true`.
- usage absent → set `benchmark.endpoint.useServerTokenCount: false` in the rendered
  `results/scenario.yaml`; the harness counts with the configured tokenizer. A `WARNING`
  line goes to `benchmark.log`.

The aggregate JSON gets `"token_count_source": "server" | "client"` next to the `scenario`
label `label_result` already writes, so a report shows which numbers are approximate.
Client-side counting cannot see hidden reasoning tokens, so OSL undercounts on reasoning
models that stream without usage; the warning says so.

## Pre-flight probe

`python3 -m infx.serving.client prepare` in remote mode sends one streaming request shaped
like the harness's (stdlib `urllib`, 60 s timeout):

```json
{"model": "<model>", "messages": [{"role": "user", "content": "Say OK."}],
 "stream": true, "stream_options": {"include_usage": true},
 "max_completion_tokens": 16, "ignore_eos": true}
```

| Result | Action |
|---|---|
| 200 with a `usage` chunk | write `server` to `results/.token_count` |
| 200 without `usage` | write `client` |
| 401 / 403 | fail: key rejected; name the secret, say whether it was empty |
| 404 | fail: wrong URL or model name |
| 400 | fail and print the response body (typically `ignore_eos` or `max_completion_tokens` rejected; see Limits) |
| other / timeout | fail with status and body |

The probe also replaces the old `remote_bench_preflight` reachability check: a wrong key or
model fails in seconds instead of after warmup. It never prints the key.

## Flow

```
dispatch.sh       validate locally; if api-key-secret is set, check it is in `gh secret list`
gn-benchmark.yml  to_matrix -> points; secret SERVING_API_KEY: ${{ secrets[matrix.api_key_secret] }}
benchmark-tmpl    new optional secret SERVING_API_KEY -> env (upstream file, 3 added lines)
infx.launch       policy -> SERVING driver (unchanged)
drivers/serving   remote: stage the client image only, submit_batch(remote.sh, gpus=0)
remote.sh         srun --container-image=<client image> python3 -m infx.serving.client prepare && run
client.py         probe, pick token source, agentx profile --url <endpoint> --api-key ...
```

### `to_matrix.py`

- Row: `image` = `CLIENT_IMAGE` (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`: python
  and uv, so the AgentX venv bootstrap needs no curl), `model` = `endpoint.model`,
  `precision` = `remote`, `framework` = `remote`, `tp` = 1. `tp` stays 1 because upstream
  derives `GPU_COUNT` from it and divides by it in result processing; the driver, not
  `GPU_COUNT`, decides that the job takes no GPU.
- Point: adds top-level `api_key_secret` (empty when unset) for the workflow's `secrets[...]`
  lookup.
- `SERVING_JSON`: adds `remote_url`, `tokenizer`; `serve_cmd` is empty.

### `drivers/serving.py`

When `Serving.remote_url` is set: no DCGM, router or storage images; the `hf-hub-cache`
volume stays required (remote.sh mounts it so HF tokenizers are cached); `job_env` adds `SERVING_REMOTE_URL`,
`SERVING_TOKENIZER`, and the batch goes to `remote.sh` with `gpus=0`. CPU and memory are
requested through `SBATCH_CPUS_PER_TASK=4` and `SBATCH_MEM_PER_NODE=16G` in the
environment `sbatch` inherits, so `SlurmBackend.submit_batch` stays untouched.

### `benchmarks/single_node/serving/remote.sh` (new, ~15 lines)

Same container flags as `job.sh` (workspace and HF cache mounts), one foreground
`srun --container-image="$SERVING_ENGINE_IMAGE"` running `client prepare` then `client run`.
No port picking, no sidecars, no wait step. No `set -x`.

### `client.py`

- `agentx_argv`: `--url` = `SERVING_REMOTE_URL` when set, else `http://localhost:$PORT`;
  `--tokenizer` = `SERVING_TOKENIZER` when set, else `MODEL`; `--api-key` when
  `SERVING_API_KEY` is set. GPU telemetry and server metrics follow the existing env checks,
  which remote mode never sets.
- `benchmark_command.txt` writes `***` in place of the key.
- `prepare()` remote: run the probe; skip `hf download` (it would pull the weights).
- `run()` remote: apply the token-count source to `scenario.yaml` before writing it; add
  `token_count_source` to the aggregate.

## Limits (documented in the serving-bench skill, not handled)

- `ignore_eos`: the `inferencex-agentx-mvp` scenario always injects `ignore_eos: true`, and
  harness 1.0.12 offers no way to omit it (explicit `false` is still sent). An API that
  rejects unknown body fields fails the probe with a 400. Add a field-stripping local proxy
  when a target we need does this. APIs that ignore the field run, but the model may stop
  before the trace's output length; OSL then follows the model, not the trace.
- `max_completion_tokens`: sent by default. An API that only takes `max_tokens` fails the
  probe with a 400; add `--use-legacy-max-tokens` as an `endpoint:` option when one shows up.
- No TLS skip, custom headers or reset URL. Add each when a target needs it.
- A remote result's per-GPU numbers are meaningless (`tp` = 1 placeholder).

## Upstream files touched

| File | Change |
|---|---|
| `.github/workflows/benchmark-tmpl.yml` | optional secret `SERVING_API_KEY`, env `SERVING_API_KEY` |

Everything else is in VNG `serving` files.

## Tests (`infx/tests/serving/`, per AGENTS.md test rules)

- schema: `endpoint` with `image` or `gpus` → one error per field; `endpoint` without
  `image`/`serve`/`gpus` passes; bad `api-key-secret` and non-http `url` rejected; free-form
  `model` accepted; `url` ending `/v1/` normalised.
- `to_matrix`: remote YAML → `tp` 1, client image, `api_key_secret` on the point,
  `remote_url`/`tokenizer` in `SERVING_JSON`.
- `agentx_argv`: remote env → endpoint URL, configured tokenizer, `--api-key`; engine env
  unchanged (localhost, `MODEL` tokenizer).
- command file: key absent from `benchmark_command.txt`.
- probe: local `http.server` stub returning (a) a stream with usage, (b) without usage,
  (c) 401, (d) 400 with a body → `server`, `client`, failure naming the secret, failure
  printing the body.
- token source applied: `client` → rendered `scenario.yaml` has
  `useServerTokenCount: false`; aggregate carries `token_count_source`.

## Verification before merge

1. `sbatch --test-only` with no `--gres` on `hardware-hcm` and `rtx6000pro-hcm` from
   `slurm-client`: the partitions accept a GPU-less job.
2. One smoke dispatch against an internal OpenAI-compatible endpoint with a key
   (`overrides: {duration: 180}`, `conc: [1]`): job green, `token_count_source: server`.
3. One smoke against a closed API: either green, or the probe fails with a readable 400
   that this spec's Limits section names.
