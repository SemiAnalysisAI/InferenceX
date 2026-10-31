# Serving Remote Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is one ticket: run it in a clean context, read only the task, the Global Constraints and the spec.

**Goal:** A serving YAML with an `endpoint:` block (url, model, api-key secret, tokenizer) benchmarks an endpoint we do not launch, through the same `gn-benchmark` flow, scenarios and artifacts as the engine path.

**Architecture:** `infx/serving/schema.py` accepts `endpoint:` in place of `image`/`serve`/`gpus`. `to_matrix` emits a point whose image is a small uv/python client image and whose `SERVING_JSON` carries `remote_url`/`tokenizer`/`api_key_secret`. The serving driver sees `remote_url`, stages only the client image and `sbatch`es `remote.sh` as a CPU-only job. `remote.sh` runs `infx.serving.client prepare` (probe → token-count source) and `run` (agentx against the endpoint with `--api-key`). The key travels as the GitHub secret named in the YAML, mapped to `SERVING_API_KEY`.

**Tech Stack:** Python 3.12 (pydantic 2, PyYAML, stdlib `urllib`/`http.server`), bash, Slurm + Pyxis, GitHub Actions, `agentx-harness==1.0.12`.

**Spec:** `docs/superpowers/specs/2026-10-27-serving-remote-endpoint-design.md`

## Global Constraints

- Branch `refactor/vng-benchmark`, worktree `/home/thanglq5/greennode-code/InferenceOptimization/InferenceX-refactor`. Paths below are relative to `inferencex-e2e/` unless they start with `.github/` or `.agents/`.
- Test command, from `inferencex-e2e/`: `uv run --no-project --python 3.12 --with pytest --with pydantic --with pyyaml python -m pytest <paths> -q` (written `PYTEST <paths>`).
- Lint, from the repo root, before each commit that touches Python: `uvx ruff check inferencex-e2e/infx && uvx ruff format --check inferencex-e2e/infx`.
- Tests follow AGENTS.md "Test quality": run real code on small inputs, assert outputs; no source-grepping, no pinning config.
- The API key never appears in a YAML, `SERVING_JSON`, `benchmark_command.txt`, or any printed line. `remote.sh` has no `set -x`.
- Upstream file touched: only `.github/workflows/benchmark-tmpl.yml` (optional secret + one env line). `SlurmBackend` is not modified.
- Client image: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`.
- Remote job resources: `--cpus-per-task=4 --mem=16G`, no `--gres`.
- Git: stage files by name, author from repo config (`Ly Quang Thang <thanglq5@greennode.ai>`). **Never push or dispatch** without the user's explicit OK.

## Deviations from the spec (decided while planning, from reading the code)

1. The spec requested CPU/memory through `SBATCH_CPUS_PER_TASK`/`SBATCH_MEM_PER_NODE` and `submit_batch(gpus=0)`. That cannot work: every VNG cluster record puts `--cpus-per-gpu=16 --mem-per-gpu=…` in `salloc-args`, `submit_batch` always appends them, and Slurm rejects `--cpus-per-gpu` without a GPU request (and with `--cpus-per-task`). The driver instead calls the public `cli.sbatch` with `cli.Resources` (no gres) and `salloc-args` minus `--cpus-per-gpu*`/`--mem-per-gpu*`, then follows the job with the backend's `stream_logs`/`state` as today.
2. The remote YAML has no top-level `model:`; `endpoint.model` is the model. Top-level `model` joins the fields `endpoint:` excludes.
3. `test_configs.py` pins the example count (7). Adding the remote example makes it 8; update the number in the same commit.

## File map

| File | Change |
|---|---|
| `infx/serving/schema.py` | `EndpointSpec`; `image`/`serve`/`gpus`/`model` optional; remote rules in `load_serving` |
| `infx/serving/render.py` | `render_remote()` (empty engine/router/storage fields, framework `remote`) |
| `infx/serving/to_matrix.py` | remote row and `SERVING_JSON` fields; point-level `api_key_secret` |
| `infx/serving/client.py` | remote argv, key redaction, `probe`, token-count source, `token_count_source` in the aggregate |
| `infx/launch/drivers/serving.py` | remote branch: client image only, CPU-only `sbatch` of `remote.sh` |
| `benchmarks/single_node/serving/remote.sh` | new: one client step |
| `.github/workflows/gn-benchmark.yml` | `SERVING_API_KEY: secrets[matrix.api_key_secret]` |
| `.github/workflows/benchmark-tmpl.yml` | optional secret `SERVING_API_KEY` + env |
| `utils/serving/dispatch.sh` | check the named secret exists |
| `configs/serving/remote-openai-agentx-smoke.yaml` | new example |
| `.agents/skills/serving-bench/SKILL.md`, `docs/serving/ARCHITECTURE.md` | remote mode section |
| `infx/tests/serving/test_schema.py`, `test_to_matrix.py`, `test_client.py`, `test_driver.py`, `test_configs.py` | tests |

---

### Task 1: Schema accepts `endpoint:`

**Files:**
- Modify: `infx/serving/schema.py`
- Test: `infx/tests/serving/test_schema.py`

**Interfaces:**
- Produces: `EndpointSpec` with `url: str` (normalised: no trailing `/`, no trailing `/v1`), `model: str`, `api_key_secret: str | None`, `tokenizer: str` (default `"builtin"`); `ServingSpec.endpoint: EndpointSpec | None`; `ServingSpec.image/serve/model: str | None`, `ServingSpec.gpus: int | None`; constants `TOKENIZER_ENCODINGS`, `REMOTE_EXCLUDED`.

- [ ] **Step 1: Write the failing tests** (append to `test_schema.py`)

```python
REMOTE = """\
name: gpt5-remote
runner: cluster:hcm-rtx01
endpoint:
  url: https://api.openai.com/v1/
  model: gpt-5
  api-key-secret: OPENAI_API_KEY
scenario: agentx
conc: [4]
"""


def test_remote_endpoint_needs_no_engine_fields():
    spec = load_serving(REMOTE)
    assert spec.endpoint.url == "https://api.openai.com"
    assert spec.endpoint.model == "gpt-5"
    assert spec.endpoint.api_key_secret == "OPENAI_API_KEY"
    assert spec.endpoint.tokenizer == "builtin"
    assert spec.image is None and spec.gpus is None


def test_remote_tokenizer_accepts_encodings_and_hf_ids():
    for tokenizer in ("o200k_harmony", "zai-org/GLM-5.2"):
        text = REMOTE.replace("  api-key-secret", f"  tokenizer: {tokenizer}\n  api-key-secret")
        assert load_serving(text).endpoint.tokenizer == tokenizer


@pytest.mark.parametrize("extra, message", [
    ("image: vllm/vllm-openai:v0.29.0\n", "image: not used with endpoint:"),
    ("gpus: 1\n", "gpus: not used with endpoint:"),
    ("model: Qwen/Qwen3.8-27B\n", "model: not used with endpoint:"),
    ("router: {image: x, serve: y}\n", "router: not used with endpoint:"),
])
def test_remote_rejects_engine_fields(extra, message):
    assert message in errors(REMOTE + extra)


@pytest.mark.parametrize("old, new, message", [
    ("url: https://api.openai.com/v1/", "url: api.openai.com", "endpoint.url: must start with http:// or https://"),
    ("api-key-secret: OPENAI_API_KEY", "api-key-secret: openai-key", "endpoint.api-key-secret: a GitHub secret name"),
    ("model: gpt-5", "model: gpt 5", "endpoint.model: no whitespace or quotes"),
    ("api-key-secret: OPENAI_API_KEY", "tokenizer: gpt5", "endpoint.tokenizer: builtin, cl100k_base"),
    ("api-key-secret: OPENAI_API_KEY", "apikey: x", "endpoint.apikey: unknown key (allowed: url, model, api-key-secret, tokenizer)"),
])
def test_remote_field_errors(old, new, message):
    assert message in errors(REMOTE.replace(old, new))


def test_engine_yaml_without_image_says_what_is_missing():
    text = BASE.replace("image: vllm/vllm-openai:v0.29.0\n", "")
    assert "image: required (or use endpoint: for a remote endpoint)" in errors(text + "scenario: agentx\nconc: 8\n")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST infx/tests/serving/test_schema.py -q`
Expected: the new tests FAIL (`endpoint` is an unknown key); the old ones pass.

- [ ] **Step 3: Implement**

In `schema.py`, after `PLACEHOLDER = ...` add:

```python
TOKENIZER_ENCODINGS = ("builtin", "cl100k_base", "o200k_base", "o200k_harmony")  # tiktoken, no HF access
SECRET_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
# Engine fields a remote endpoint has no use for.
REMOTE_EXCLUDED = ("image", "serve", "gpus", "model", "replicas", "precision", "env", "router", "storage")
```

Add `model_validator` to the pydantic import. Before `class Workload`, add:

```python
class EndpointSpec(BaseModel):
    """An endpoint we do not launch: an OpenAI-compatible URL, its served model and its key."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    url: str
    model: str = Field(min_length=1)
    api_key_secret: str | None = Field(default=None, alias="api-key-secret")
    tokenizer: str = "builtin"

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        if not re.fullmatch(r"https?://[^\s'\"]+", value):
            raise ValueError("must start with http:// or https:// and hold no whitespace or quotes")
        value = value.rstrip("/")
        return value.removesuffix("/v1").rstrip("/")  # the scenario appends /v1/chat/completions

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        if re.search(r"[\s'\"]", value):
            raise ValueError("no whitespace or quotes; use the served model name")
        return value

    @field_validator("api_key_secret")
    @classmethod
    def _secret(cls, value: str | None) -> str | None:
        if value is not None and not SECRET_NAME.fullmatch(value):
            raise ValueError("a GitHub secret name: capitals, digits and _ (e.g. OPENAI_API_KEY)")
        return value

    @field_validator("tokenizer")
    @classmethod
    def _tokenizer(cls, value: str) -> str:
        if value not in TOKENIZER_ENCODINGS and not MODEL_PATTERN.fullmatch(value):
            raise ValueError(f"{', '.join(TOKENIZER_ENCODINGS)}, or an HF repo id <org>/<name>")
        return value
```

In `ServingSpec`, change these fields and add `endpoint`:

```python
    image: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    gpus: int | None = Field(default=None, gt=0)
    serve: str | None = Field(default=None, min_length=1)
    endpoint: EndpointSpec | None = None
```

In `_format_validation_error`:
- add `"endpoint": EndpointSpec` to `models`;
- list aliases in the unknown-key message:
  `message = "unknown key (allowed: " + ", ".join(f.alias or n for n, f in model.model_fields.items()) + ")"`;
- for a field error inside `endpoint`, print the alias: build `location` as
  `".".join(_alias(item["loc"]))` where

```python
def _alias(loc: tuple) -> list[str]:
    """Field path as written in the YAML (api-key-secret, not api_key_secret)."""
    if len(loc) == 2 and loc[0] == "endpoint":
        field = EndpointSpec.model_fields.get(str(loc[1]))
        return ["endpoint", (field.alias if field and field.alias else str(loc[1]))]
    return [str(part) for part in loc]
```

and strip pydantic's `"Value error, "` prefix: `message = item["msg"].removeprefix("Value error, ")`.

In `load_serving`, right after `spec = ServingSpec.model_validate(raw)` succeeds, split the two modes. Replace the body from `problems: list[str] = []` down to (not including) `if len(set(spec.conc)) != len(spec.conc):` with:

```python
    problems: list[str] = []
    if spec.endpoint is not None:
        problems += [f"{field}: not used with endpoint: (remote mode); remove it"
                     for field in REMOTE_EXCLUDED if field in spec.model_fields_set]
    else:
        problems += [f"{field}: required (or use endpoint: for a remote endpoint)"
                     for field in ("image", "model", "gpus", "serve") if getattr(spec, field) is None]
        if problems:
            raise ServingError("\n".join(problems))
        problems += _engine_problems(spec)
```

and move the old engine-only checks into a new function `_engine_problems(spec) -> list[str]` (image pattern, model path/pattern, `_command_problems`, `_placeholder_problems`, the `replicas` checks), unchanged in content. At the end of `load_serving`, the `runner_gpus` block becomes:

```python
    try:
        per_node = runner_gpus(spec.runner, inventory)  # also checks runner is a VNG runner
        if spec.endpoint is None:
            if spec.gpus > per_node:
                problems.append(f"gpus: {spec.gpus} > {per_node} GPUs per node on runner {spec.runner}; pick a bigger node")
            problems += _local_squash_problems(spec, inventory)
    except ServingError as error:
        problems.append(str(error))
```

- [ ] **Step 4: Run tests**

Run: `PYTEST infx/tests/serving -q`
Expected: all pass (the existing 146 plus the new ones).

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check inferencex-e2e/infx && uvx ruff format --check inferencex-e2e/infx
git add inferencex-e2e/infx/serving/schema.py inferencex-e2e/infx/tests/serving/test_schema.py
git commit -m "feat(serving): endpoint: block for a remote endpoint in the serving YAML"
```

---

### Task 2: Matrix points for a remote endpoint

**Files:**
- Modify: `infx/serving/render.py`, `infx/serving/to_matrix.py`
- Test: `infx/tests/serving/test_to_matrix.py`

**Interfaces:**
- Consumes: `ServingSpec.endpoint` (Task 1).
- Produces: `render.CLIENT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"`; `render.render_remote() -> dict[str, str]`; each point is `{"config": row, "serving": json_str, "api_key_secret": str}`; `SERVING_JSON` gains `"remote_url"`, `"tokenizer"`, `"api_key_secret"` (strings; `""` on the engine path).

- [ ] **Step 1: Write the failing tests** (append to `test_to_matrix.py`)

```python
REMOTE = """\
name: gpt5-remote
runner: cluster:hcm-rtx01
endpoint: {url: "https://api.openai.com/v1", model: gpt-5, api-key-secret: OPENAI_API_KEY}
scenario: agentx
overrides: {duration: 180}
conc: [4, 8]
"""


def test_remote_point_runs_the_client_image_against_the_endpoint():
    points = build(REMOTE)["points"]
    assert [p["config"]["exp-name"] for p in points] == ["gpt5-remote_c4", "gpt5-remote_c8"]
    row = points[0]["config"]
    assert set(row) == ROW_KEYS
    assert row["image"] == "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
    assert (row["model"], row["framework"], row["precision"], row["tp"]) == ("gpt-5", "remote", "remote", 1)
    assert points[0]["api_key_secret"] == "OPENAI_API_KEY"
    serving = json.loads(points[0]["serving"])
    assert serving["remote_url"] == "https://api.openai.com"
    assert serving["tokenizer"] == "builtin"
    assert serving["api_key_secret"] == "OPENAI_API_KEY"
    assert serving["serve_cmd"] == "" and serving["router_cmd"] == ""


def test_engine_point_carries_no_remote_fields():
    point = build(ENGINE + "scenario: tau2\nconc: 4\n")["points"][0]
    serving = json.loads(point["serving"])
    assert (serving["remote_url"], serving["tokenizer"], point["api_key_secret"]) == ("", "", "")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST infx/tests/serving/test_to_matrix.py -q`
Expected: the two new tests FAIL (`KeyError` / validation error).

- [ ] **Step 3: Implement**

`render.py`, after `_ROUTER_OWNED_FLAGS`:

```python
CLIENT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"  # python + uv: the AgentX venv bootstrap needs no curl


def render_remote() -> dict[str, str]:
    """A remote endpoint launches nothing: every engine, router and storage field is empty."""
    return {
        "framework": "remote", "serve-cmd": "", "serve-env": "", "served-model-name": "",
        "router-image": "", "router-cmd": "", "router-env": "", "router-profile": "",
        "router-metrics-port": "", "storage-image": "", "storage-cmd": "", "storage-env": "",
    }  # fmt: skip
```

`to_matrix.py`: import `CLIENT_IMAGE, render, render_remote` from `infx.serving.render`. Change `_row`:

```python
def _row(spec: ServingSpec, framework: str, conc: int, duration: int, exp_name: str) -> dict:
    remote = spec.endpoint
    return {
        "image": CLIENT_IMAGE if remote else spec.image, "model": remote.model if remote else spec.model,
        "model-prefix": spec.name, "precision": "remote" if remote else spec.precision,
        "framework": framework, "runner": spec.runner,
        # tp stays 1 for a remote point: upstream derives GPU_COUNT from it and divides by it;
        # the serving driver, not GPU_COUNT, decides that a remote job takes no GPU.
        "tp": 1 if remote else spec.gpus, "pp": 1, "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False,
        "spec-decoding": "none", "conc": conc, "kv-offloading": "none", "total-cpu-dram-gb": 0,
        "duration": duration, "exp-name": exp_name, "scenario-type": "agentic-coding",
    }  # fmt: skip
```

In `_serving`, add to the dict:

```python
        "remote_url": spec.endpoint.url if spec.endpoint else "",
        "tokenizer": spec.endpoint.tokenizer if spec.endpoint else "",
        "api_key_secret": (spec.endpoint.api_key_secret or "") if spec.endpoint else "",
```

In `build`: `rendered = render_remote() if spec.endpoint else render(spec)`, and the point becomes

```python
            secret = (spec.endpoint.api_key_secret or "") if spec.endpoint else ""
            points.append({"config": row, "serving": _serving(spec, rendered, variant, text),
                           "api_key_secret": secret})
```

- [ ] **Step 4: Run tests**

Run: `PYTEST infx/tests/serving -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check inferencex-e2e/infx && uvx ruff format --check inferencex-e2e/infx
git add inferencex-e2e/infx/serving/render.py inferencex-e2e/infx/serving/to_matrix.py inferencex-e2e/infx/tests/serving/test_to_matrix.py
git commit -m "feat(serving): remote endpoint points run the client image with remote_url and tokenizer"
```

---

### Task 3: Client talks to the remote endpoint

**Files:**
- Modify: `infx/serving/client.py`
- Test: `infx/tests/serving/test_client.py`

**Interfaces:**
- Consumes env (set by the driver, Task 4): `SERVING_REMOTE_URL`, `SERVING_TOKENIZER`, `SERVING_API_KEY_SECRET` (name only), `SERVING_API_KEY` (the key, from the workflow), `MODEL`.
- Produces: `class ProbeError(RuntimeError)`; `probe(url: str, model: str, api_key: str, secret: str = "", timeout: float = 60.0) -> str` returning `"server"` or `"client"`; `redacted(argv: list[str]) -> list[str]`; `with_token_source(scenario_yaml: str, source: str) -> str`; `label_result(path, label, token_count_source: str | None = None)`; marker file `results/.token_count`.

- [ ] **Step 1: Write the failing tests** (append to `test_client.py`; add imports at the top)

```python
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from infx.serving.client import ProbeError, probe, redacted, with_token_source

REMOTE_ENV = {
    "MODEL": "gpt-5", "CONC": "4", "RESULT_DIR": "/workspace/results", "SERVING_SEARCH": "0",
    "SERVING_REMOTE_URL": "https://api.example.com", "SERVING_TOKENIZER": "builtin",
    "SERVING_API_KEY": "sk-secret-123",
}


def test_remote_argv_targets_the_endpoint_with_its_key_and_tokenizer():
    argv = agentx_argv("agentx", Path("s.yaml"), REMOTE_ENV)
    assert argv[argv.index("--url") + 1] == "https://api.example.com"
    assert argv[argv.index("--tokenizer") + 1] == "builtin"
    assert argv[argv.index("--api-key") + 1] == "sk-secret-123"
    assert "--no-gpu-telemetry" in argv and "--server-metrics" not in argv


def test_engine_argv_sends_no_key():
    assert "--api-key" not in agentx_argv("agentx", Path("s.yaml"), ENV)


def test_redacted_hides_the_key():
    argv = agentx_argv("agentx", Path("s.yaml"), REMOTE_ENV)
    assert "sk-secret-123" not in " ".join(redacted(argv))
    assert redacted(argv)[argv.index("--api-key") + 1] == "***"


def test_client_token_source_turns_server_counting_off():
    doc = {"benchmark": {"endpoint": {"type": "chat", "useServerTokenCount": True}}}
    text = yaml.safe_dump(doc)
    assert with_token_source(text, "server") == text
    assert yaml.safe_load(with_token_source(text, "client"))["benchmark"]["endpoint"] == {
        "type": "chat", "useServerTokenCount": False}


def test_label_result_records_the_token_count_source(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"conc": 4}))
    label_result(path, "agentx-cc-062126-smoke", token_count_source="client")
    assert json.loads(path.read_text()) == {
        "conc": 4, "scenario": "agentx-cc-062126-smoke", "token_count_source": "client"}


def serve(status: int, body: bytes):
    """A one-route stub endpoint on a free port; returns (base url, received requests)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(length))))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", seen, server


WITH_USAGE = (b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
              b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1}}\n\n'
              b"data: [DONE]\n\n")
WITHOUT_USAGE = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'


def test_probe_reads_server_usage_and_sends_the_harness_request_shape():
    url, seen, server = serve(200, WITH_USAGE)
    try:
        assert probe(url, "gpt-5", "sk-1") == "server"
    finally:
        server.shutdown()
    [(path, headers, body)] = seen
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-1"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["ignore_eos"] is True and body["model"] == "gpt-5"


def test_probe_without_usage_falls_back_to_client_counting():
    url, _, server = serve(200, WITHOUT_USAGE)
    try:
        assert probe(url, "gpt-5", "") == "client"
    finally:
        server.shutdown()


@pytest.mark.parametrize("status, body, secret, message", [
    (401, b'{"error":"bad key"}', "OPENAI_API_KEY", "HTTP 401: key rejected; check GitHub secret OPENAI_API_KEY"),
    (401, b"", "", "HTTP 401: the endpoint wants a key; set endpoint.api-key-secret"),
    (404, b"no such model", "", "HTTP 404: check endpoint.url and endpoint.model"),
    (400, b'{"error":"Unrecognized request argument supplied: ignore_eos"}', "",
     "HTTP 400: Unrecognized request argument supplied: ignore_eos"),
])
def test_probe_failures_name_the_fix(status, body, secret, message):
    url, _, server = serve(status, body)
    try:
        with pytest.raises(ProbeError, match=message.replace("(", r"\(")) as caught:
            probe(url, "gpt-5", "sk-1" if secret else "", secret)
    finally:
        server.shutdown()
    assert "sk-1" not in str(caught.value)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST infx/tests/serving/test_client.py -q`
Expected: ImportError on `ProbeError` (all tests in the file fail to collect).

- [ ] **Step 3: Implement** (`client.py`)

Imports: add `import urllib.error`, `import urllib.request`. Constants after `FAILED_REQUEST_THRESHOLD`:

```python
TOKEN_SOURCE = ".token_count"
PROBE_BODY = {  # the shape agentx sends: streaming, usage requested, ignore_eos injected by the scenario
    "messages": [{"role": "user", "content": "Say OK."}], "stream": True,
    "stream_options": {"include_usage": True}, "max_completion_tokens": 16, "ignore_eos": True,
}
```

Replace `agentx_argv` and add the helpers:

```python
def agentx_argv(agentx: str, scenario: Path, env: Mapping[str, str]) -> list[str]:
    argv = [agentx, "profile", "--config", str(scenario)]
    argv += ["--url", env.get("SERVING_REMOTE_URL") or f"http://localhost:{env['PORT']}"]
    argv += ["--model", env.get("SERVED_MODEL_NAME") or env["MODEL"]]
    argv += ["--tokenizer", env.get("SERVING_TOKENIZER") or env["MODEL"]]
    if key := env.get("SERVING_API_KEY"):
        argv += ["--api-key", key]
    if env.get("SERVING_SEARCH") != "1":
        argv += ["--concurrency", env["CONC"]]
    if telemetry := env.get("AIPERF_GPU_TELEMETRY_URL"):
        argv += ["--gpu-telemetry", telemetry]
    else:
        argv.append("--no-gpu-telemetry")
    if urls := env.get("AIPERF_SERVER_METRICS_URLS"):
        argv += ["--server-metrics", *urls.split(",")]
    argv += ["--output-artifact-dir", f"{env['RESULT_DIR']}/aiperf_artifacts"]
    return argv


def redacted(argv: list[str]) -> list[str]:
    """``argv`` with the --api-key value replaced, for benchmark_command.txt (an uploaded artifact)."""
    return [("***" if index and argv[index - 1] == "--api-key" else arg) for index, arg in enumerate(argv)]


class ProbeError(RuntimeError):
    """The remote endpoint refused the pre-flight request; the message names the fix."""


def _probe_hint(status: int, detail: str, has_key: bool, secret: str) -> str:
    if status in (401, 403):
        if has_key:
            return f"HTTP {status}: key rejected; check GitHub secret {secret or '(unnamed)'}"
        return f"HTTP {status}: the endpoint wants a key; set endpoint.api-key-secret"
    if status == 404:
        return f"HTTP 404: check endpoint.url and endpoint.model ({detail})"
    if status == 400:
        return (f"HTTP 400: {detail} -- the endpoint may reject ignore_eos or max_completion_tokens "
                "(spec 2026-10-27, Limits)")
    return f"HTTP {status}: {detail}"


def probe(url: str, model: str, api_key: str, secret: str = "", timeout: float = 60.0) -> str:
    """One streaming request shaped like agentx's: "server" if a chunk carries usage, else "client"."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps({"model": model, **PROBE_BODY}).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except ValueError:
                    continue
                if isinstance(chunk, dict) and chunk.get("usage"):
                    return "server"
    except urllib.error.HTTPError as error:
        detail = error.read(2000).decode(errors="replace").strip()
        try:  # an OpenAI-style {"error": {"message": ...}} or {"error": "..."} reads better unwrapped
            parsed = json.loads(detail)["error"]
            detail = parsed["message"] if isinstance(parsed, dict) else str(parsed)
        except (ValueError, KeyError, TypeError):
            pass
        raise ProbeError(_probe_hint(error.code, detail, bool(api_key), secret)) from None
    except (urllib.error.URLError, OSError) as error:
        raise ProbeError(f"{url} unreachable: {getattr(error, 'reason', error)}") from None
    return "client"


def with_token_source(scenario_yaml: str, source: str) -> str:
    """The scenario with server-side token counting off when the endpoint reports no usage."""
    if source != "client":
        return scenario_yaml
    import yaml  # in the AgentX venv (agentx-harness depends on PyYAML)

    doc = yaml.safe_load(scenario_yaml)
    doc["benchmark"]["endpoint"]["useServerTokenCount"] = False
    return yaml.safe_dump(doc, sort_keys=False)
```

`label_result` gains the keyword:

```python
def label_result(path: Path, label: str, token_count_source: str | None = None) -> None:
    """File the aggregate under the legacy scenario label result-report reads first."""
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result["scenario"] = label
    if token_count_source:
        result["token_count_source"] = token_count_source
    path.write_text(json.dumps(result, indent=2))
```

`prepare`: after the file-dataset block and before `return proc.call([... "download" ...])`, insert:

```python
    if url := env.get("SERVING_REMOTE_URL"):
        try:
            source = probe(url, env["MODEL"], env.get("SERVING_API_KEY", ""), env.get("SERVING_API_KEY_SECRET", ""))
        except ProbeError as error:
            print(f"ERROR remote endpoint: {error}", file=sys.stderr)
            return 1
        (result_dir / TOKEN_SOURCE).write_text(source)
        if source == "client":
            print("WARNING remote endpoint streams no usage; ISL/OSL are counted client-side with "
                  f"tokenizer {env.get('SERVING_TOKENIZER')} (approximate unless it is the model's own; "
                  "hidden reasoning tokens are not counted)", file=sys.stderr)
        return 0  # no weights to download; agentx fetches an HF tokenizer itself
```

`run`: replace `scenario.write_text(env["SERVING_SCENARIO_YAML"])` with

```python
    marker = result_dir / TOKEN_SOURCE
    source = marker.read_text().strip() if marker.exists() else None
    scenario.write_text(with_token_source(env["SERVING_SCENARIO_YAML"], source or "server"))
```

replace the `benchmark_command.txt` line with `shlex.join(redacted(argv))`, and pass `token_count_source=source` to `label_result`. Rename the existing dataset `marker` variable in `run` to `dataset_marker` to avoid the clash.

- [ ] **Step 4: Run tests**

Run: `PYTEST infx/tests/serving -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check inferencex-e2e/infx && uvx ruff format --check inferencex-e2e/infx
git add inferencex-e2e/infx/serving/client.py inferencex-e2e/infx/tests/serving/test_client.py
git commit -m "feat(serving): client probes a remote endpoint, picks the token-count source, redacts the key"
```

---

### Task 4: Driver runs a remote point as a CPU-only Slurm job

**Files:**
- Create: `benchmarks/single_node/serving/remote.sh`
- Modify: `infx/launch/drivers/serving.py`
- Test: `infx/tests/serving/test_driver.py`

**Interfaces:**
- Consumes: `SERVING_JSON` fields `remote_url`, `tokenizer`, `api_key_secret` (Task 2); client env names (Task 3).
- Produces: `remote_sbatch_args(salloc_args: Sequence[str]) -> list[str]`; job env `SERVING_REMOTE_URL`, `SERVING_TOKENIZER`, `SERVING_API_KEY_SECRET`.

- [ ] **Step 1: Write the failing tests** (append to `test_driver.py`)

```python
def test_remote_point_is_a_cpu_only_job_with_one_client_step(harness):
    shutil.copy(ROOT / "benchmarks/single_node/serving/remote.sh",
                harness.workspace / "benchmarks/single_node/serving/remote.sh")
    env = {**harness.env, "IMAGE": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim", "GPU_COUNT": "1",
           "SERVING_API_KEY": "sk-secret-123",
           "SERVING_JSON": serving_json(framework="remote", serve_cmd="", remote_url="https://api.example.com",
                                        tokenizer="builtin", api_key_secret="OPENAI_API_KEY")}
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [submit] = lines(harness.logs, "sbatch")
    flags = submit.split()
    assert {"--cpus-per-task=4", "--mem=16G", "--nodelist=hgx-h200-02"} <= set(flags)
    assert not [f for f in flags if f.startswith(("--gres", "--cpus-per-gpu", "--mem-per-gpu"))]
    assert submit.endswith("remote.sh")
    [step] = lines(harness.logs, "srun")  # no DCGM sidecar
    assert "--container-image=" in step and "infx.serving.client" in step  # staged reference, not the tag
    assert "sk-secret-123" not in result.stdout + result.stderr


def test_remote_sbatch_args_keep_the_node_pin_only():
    from infx.launch.drivers.serving import remote_sbatch_args

    assert remote_sbatch_args(["--nodelist=n1", "--cpus-per-gpu=16", "--mem-per-gpu=110000M"]) == [
        "--ntasks=1", "--cpus-per-task=4", "--mem=16G", "--nodelist=n1"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST infx/tests/serving/test_driver.py -q`
Expected: FAIL (`remote.sh` missing; `remote_sbatch_args` not defined; `Serving` rejects `remote_url`).

- [ ] **Step 3: Implement**

`benchmarks/single_node/serving/remote.sh` (mode 0755):

```bash
#!/usr/bin/env bash
# One remote-endpoint serving point inside its CPU-only Slurm allocation. infx/launch/drivers/serving.py
# sbatches this file with --chdir=<workspace>; every input is env (see the driver's job_env).
# No engine, sidecars or ports: one client step in the client image probes the endpoint and replays
# the scenario against it. SERVING_API_KEY reaches the step through the environment; never trace it.
set -euo pipefail

: "${SERVING_ENGINE_IMAGE:?}" "${SERVING_HF_CACHE_HOST:?}"
WS="$PWD"
mkdir -p "$WS/results"
HF_TARGET="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"

srun --container-image="$SERVING_ENGINE_IMAGE" \
    --container-mounts="$WS:/workspace,$SERVING_HF_CACHE_HOST:${HF_TARGET%/}" \
    --no-container-mount-home --container-remap-root --container-writable \
    --container-workdir=/workspace --no-container-entrypoint \
    bash -c 'set -eo pipefail
: "${RESULT_DIR:?}" "${MODEL:?}"
export PYTHONPATH=/workspace AGENTIC_OUTPUT_DIR=/workspace
mkdir -p "$RESULT_DIR"
printf "%s" "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"
python3 -m infx.serving.client prepare
python3 -m infx.serving.client run'
```

`drivers/serving.py`:

- imports: `from collections.abc import Sequence`, `from infx.launch.backends.slurm import SlurmBackend, SlurmJob, cli`.
- constants:

```python
REMOTE_JOB_SCRIPT = Path("benchmarks/single_node/serving/remote.sh")
REMOTE_RESOURCES = ("--ntasks=1", "--cpus-per-task=4", "--mem=16G")
GPU_SHARE_FLAGS = ("--cpus-per-gpu", "--mem-per-gpu")  # a GPU-less job cannot ask for a per-GPU share
```

- function:

```python
def remote_sbatch_args(salloc_args: Sequence[str]) -> list[str]:
    """A remote point's sbatch flags: a few CPUs and some memory, the node pin, no per-GPU share."""
    return [*REMOTE_RESOURCES, *(arg for arg in salloc_args if not arg.startswith(GPU_SHARE_FLAGS))]


def submit_remote(backend: SlurmBackend, request: ServingRequest, minutes: int, log: Path) -> SlurmJob:
    """sbatch remote.sh without gres; cancelled with the launch like submit_batch's jobs."""
    settings = backend.settings
    resources = cli.Resources(partition=settings.partition, account=settings.account, time_min=minutes,
                              job_name=backend.request.runner_name, exclude=settings.exclude)
    job = cli.sbatch(request.workspace / REMOTE_JOB_SCRIPT, resources, output=log,
                     chdir=request.workspace, extra=remote_sbatch_args(settings.salloc_args))
    backend.life.callback(cli.cancel, job)
    backend.life.event.set(slurm_job_id=job.id)
    backend.life.event.evidence(log)
    return SlurmJob(job.id, log=log)
```

- `Serving`: add `remote_url: str = ""`, `tokenizer: str = ""`, `api_key_secret: str = ""`.
- `job_env`: `"SERVING_DCGM_IMAGE": images.get("dcgm", "")`, and add `"SERVING_REMOTE_URL": serving.remote_url`, `"SERVING_TOKENIZER": serving.tokenizer`, `"SERVING_API_KEY_SECRET": serving.api_key_secret`.
- `run`: build `wanted` as `{"engine": request.image}` when `serving.remote_url`, else the current dict (engine + dcgm + router + storage). In the submit stage:

```python
    with event.stage("submit"):
        try:
            if serving.remote_url:
                job = submit_remote(backend, request, minutes, log)
            else:
                job = backend.submit_batch(request.workspace / JOB_SCRIPT, gpus=request.gpu_count,
                                           time_min=minutes, log=log)
        except BackendError as error:
            event.fail(type(error).__name__, f"batch allocation unavailable: {error}", retriable=error.retriable)
            return 1
```

  (`cli.SlurmError` subclasses `BackendError`, so the existing `except` covers it.)

- [ ] **Step 4: Run tests**

Run: `PYTEST infx/tests/serving infx/tests/launch -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check inferencex-e2e/infx && uvx ruff format --check inferencex-e2e/infx
chmod 755 inferencex-e2e/benchmarks/single_node/serving/remote.sh
git add inferencex-e2e/benchmarks/single_node/serving/remote.sh inferencex-e2e/infx/launch/drivers/serving.py inferencex-e2e/infx/tests/serving/test_driver.py
git commit -m "feat(serving): remote points run as a CPU-only Slurm job with one client step"
```

---

### Task 5: Key secret through the workflows; dispatch checks it

**Files:**
- Modify: `.github/workflows/gn-benchmark.yml`, `.github/workflows/benchmark-tmpl.yml`, `utils/serving/dispatch.sh`

**Interfaces:**
- Consumes: point-level `api_key_secret` (Task 2); env `SERVING_API_KEY` (Task 3).

No unit test: workflow YAML and a `gh` call. Verified by `actionlint` here and by the smoke dispatch in Task 7.

- [ ] **Step 1: `benchmark-tmpl.yml`** — under `on.workflow_call.secrets`, after `INFERENCEX_OFFICIAL_RO_HF_TOKEN`:

```yaml
      SERVING_API_KEY:
        required: false
```

and in the top-level `env:` block, after `SERVING_JSON: ${{ inputs.serving }}`:

```yaml
  SERVING_API_KEY: ${{ secrets.SERVING_API_KEY }}
```

- [ ] **Step 2: `gn-benchmark.yml`** — in the `bench` job's `secrets:` mapping, after the HF token line:

```yaml
            SERVING_API_KEY: ${{ matrix.api_key_secret && secrets[matrix.api_key_secret] || '' }}
```

- [ ] **Step 3: `dispatch.sh`** — after the `title=$(...) || exit 2` validation, before the ref checks:

```bash
secret=$(cd "$E2E" && uv run -q --no-project --python 3.12 --with pyyaml python -P -c \
    'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])).get("endpoint") or {}).get("api-key-secret") or "")' \
    "$yaml_file")
if [[ -n "$secret" ]]; then
    if names=$(gh secret list --repo "$GH_REPO" --json name --jq '.[].name' 2>/dev/null); then
        if ! grep -qx "$secret" <<<"$names"; then
            echo "ERROR endpoint.api-key-secret: $secret is not a secret of $GH_REPO; add it: gh secret set $secret --repo $GH_REPO" >&2
            exit 2
        fi
    else
        echo "WARNING cannot list secrets of $GH_REPO (needs admin); assuming $secret exists" >&2
    fi
fi
```

- [ ] **Step 4: Lint the workflows**

Run: `uvx --from actionlint-py actionlint .github/workflows/gn-benchmark.yml .github/workflows/benchmark-tmpl.yml`
Expected: no new findings on the edited lines (compare with `git stash; …; git stash pop` if the file already has findings).
Run: `bash -n inferencex-e2e/utils/serving/dispatch.sh` → no output.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/gn-benchmark.yml .github/workflows/benchmark-tmpl.yml inferencex-e2e/utils/serving/dispatch.sh
git commit -m "ci(serving): pass the endpoint's API key secret to the job; dispatch checks it exists"
```

---

### Task 6: Example config and docs

**Files:**
- Create: `configs/serving/remote-openai-agentx-smoke.yaml`
- Modify: `infx/tests/serving/test_configs.py`, `.agents/skills/serving-bench/SKILL.md`, `docs/serving/ARCHITECTURE.md`

- [ ] **Step 1: Example YAML**

```yaml
# Remote endpoint (no engine launched): AgentX smoke against an OpenAI-compatible API.
# The key is the GitHub secret named below (gh secret set OPENAI_API_KEY --repo vngcloud/InferenceX).
# tokenizer: builtin is tiktoken o200k_base, exact for GPT-4o/GPT-5; use an HF repo id for open models.
name: remote-openai-agentx-smoke
runner: cluster:hcm-rtx01
endpoint:
  url: https://api.openai.com/v1
  model: gpt-5
  api-key-secret: OPENAI_API_KEY
  tokenizer: builtin
scenario: agentx
overrides: {duration: 180}
conc: [1]
```

- [ ] **Step 2: `test_configs.py`** — rename `test_seven_examples_one_per_pattern` to `test_eight_examples_one_per_pattern` and assert `len(CONFIGS) == 8`.

- [ ] **Step 3: Run tests**

Run: `PYTEST infx/tests/serving -q`
Expected: all pass, including `test_every_serving_config_validates[remote-openai-agentx-smoke]`.

- [ ] **Step 4: SKILL.md** — add a section after "1. Write ...", before "2. Ask the user":

```markdown
### Remote endpoint (no engine)

To benchmark an endpoint already running (GreenNode MaaS, a k8s service, OpenAI/Anthropic/Gemini
via an OpenAI-compatible API), write `endpoint:` instead of `image`/`model`/`gpus`/`serve`
(example: `configs/serving/remote-openai-agentx-smoke.yaml`):

- `url`: base URL; a trailing `/v1` is fine. `model`: the served name as the API expects it.
- `api-key-secret`: the GitHub secret holding the key (`gh secret set <NAME> --repo vngcloud/InferenceX`).
  Never put the key in the YAML. Omit for an open endpoint.
- `tokenizer`: `builtin` (default; exact for OpenAI GPT-4o/GPT-5, approximate for other closed
  models), `cl100k_base`/`o200k_harmony`, or the HF repo id of an open model.
- `runner`: any VNG runner; the job takes 4 CPUs and 16 GB, no GPU.
- No server metrics or GPU telemetry in the result. ISL/OSL come from the API's `usage`; if it
  streams none, the job counts client-side and the aggregate says `token_count_source: client`.
- A probe request runs first: 401/404/400 fail the job in seconds with the fix. A 400 naming
  `ignore_eos` means the API rejects the field the AgentX scenario always sends; tell the user,
  there is no workaround yet.
```

- [ ] **Step 5: ARCHITECTURE.md** — in the flow section, add one paragraph:

```markdown
**Remote endpoint.** A YAML with `endpoint:` launches nothing. Its point runs the client image
(`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`); the driver `sbatch`es `remote.sh` with 4 CPUs,
16 GB and no GPU, keeping only the node pin from the cluster record. `remote.sh` runs one client
step: `prepare` probes the endpoint (key, model, usage reporting) and `run` replays the scenario
against it with `--api-key`. The key is the GitHub secret the YAML names, passed as `SERVING_API_KEY`.
```

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/configs/serving/remote-openai-agentx-smoke.yaml inferencex-e2e/infx/tests/serving/test_configs.py .agents/skills/serving-bench/SKILL.md inferencex-e2e/docs/serving/ARCHITECTURE.md
git commit -m "docs(serving): remote endpoint mode in the serving-bench skill, ARCHITECTURE and an example"
```

---

### Task 7: Verify on the cluster (needs the user's OK for push and dispatch)

- [ ] **Step 1: Slurm accepts the remote job shape** (read-only, no allocation):

```bash
ssh slurm-client 'sbatch --test-only --partition=rtx6000pro-hcm --account=greennode --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --nodelist=hoanq3-rtx6000x2-slurm-1 --wrap=true'
ssh slurm-client 'sbatch --test-only --partition=hardware-hcm --account=greennode --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --nodelist=hgx-h200-02 --wrap=true'
```

Expected: `sbatch: Job ... to start at ...` for both. An error here means the partition requires a GPU request; stop and report.

- [ ] **Step 2: Full local test run**

Run: `PYTEST infx/tests/serving infx/tests/launch -q` → all pass. Lint commands from Global Constraints → clean.

- [ ] **Step 3: Ask the user** to push `refactor/vng-benchmark` and to name the internal endpoint + secret for the first smoke. Do not push before the answer.

- [ ] **Step 4: Smoke against an internal OpenAI-compatible endpoint** (after OK): write `configs/serving/<name>.yaml` with that endpoint, `overrides: {duration: 180}`, `conc: [1]`, then `utils/serving/dispatch.sh configs/serving/<name>.yaml`. Expected: job green; `results/benchmark_command.txt` shows `--api-key ***`; the aggregate has `token_count_source: server` and non-zero throughput; `grep -c <key prefix>` over the artifacts is 0.

- [ ] **Step 5: Smoke against a closed API** (after OK, with the user's secret): expected either green, or a probe failure whose message is the readable 400 from the spec's Limits section. Report which.
