"""The lm-evaluation-harness framework: pinned install, per-request token budget, and one run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from infx.bench import env, proc, uv
from infx.bench.eval.context import EvalContext, EvalOutcome

REPOSITORY = "https://github.com/EleutherAI/lm-evaluation-harness"
REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
DEFAULT_TASKS = "infx/evals/gsm8k.yaml"
PATCH = "infx/evals/patches/lm_eval_sitecustomize.py"
# The client's complete dependency closure, hash-pinned and installed without resolution: torch (CPU) from the
# PyTorch CPU index, everything else (transformers 5.16.1, huggingface-hub 1.33.0, datasets 5.1.0, ...) from PyPI.
# lm-eval itself is installed at REF with --no-deps.
CLIENT_LOCKS = (("infx/evals/lm_eval_client_torch.lock", "https://download.pytorch.org/whl/cpu"),
                ("infx/evals/lm_eval_client.lock", "https://pypi.org/simple"))
CLIENT_PYTHON = "INFX_EVAL_CLIENT_PYTHON"
CLIENT_PROBE = (
    "import json, importlib.metadata as m; from lm_eval.models.openai_completions import LocalChatCompletion; "
    "u = json.loads(m.distribution('lm-eval').read_text('direct_url.json') or '{}'); "
    "print(json.dumps({'ref': (u.get('vcs_info') or {}).get('commit_id') or u.get('url', ''), "
    "'huggingface_hub': m.version('huggingface-hub'), 'transformers': m.version('transformers')}))"
)
FALLBACK_CONTEXT = 16384
PROMPT_RESERVE = 4096
MAX_OUTPUT_TOKENS = 16384
CONTEXT_FIELDS = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")


def install(environ: dict[str, str]) -> None:
    """Choose the interpreter that runs lm-eval and record it in ``environ[CLIENT_PYTHON]``.

    This interpreter when it already imports lm-eval's chat-completions client at ``REF``; otherwise an isolated
    virtual environment built from ``CLIENT_LOCK``. The serving interpreter's packages are never changed: installing
    lm-eval[api] into a serving image can replace dependencies its own packages pin (huggingface-hub 2.x under a
    transformers that requires <2.0).
    """
    found = _client(sys.executable, environ)
    if found is not None:
        environ[CLIENT_PYTHON] = sys.executable
        print(f"Eval client: this interpreter {sys.executable} {json.dumps(found, sort_keys=True)}")
        return
    try:
        python = _isolated(environ)
    except env.BenchError as error:
        print(f"WARN: isolated eval client unavailable: {error}", file=sys.stderr)
        return
    found = _client(python, environ)
    if found is None:
        print(f"WARN: isolated eval client at {python} does not import lm-eval at {REF}", file=sys.stderr)
        return
    environ[CLIENT_PYTHON] = python
    print(f"Eval client: isolated {python} {json.dumps(found, sort_keys=True)} lock sha256 {_lock_sha256()}")


def _client(python: str, environ: Mapping[str, str]) -> dict | None:
    """lm-eval's REF and key versions as ``python`` sees them, or ``None`` when it cannot run the client at REF."""
    try:
        out = subprocess.run([python, "-c", CLIENT_PROBE], env=dict(environ), capture_output=True, text=True,
                             timeout=300, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        found = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return found if REF in str(found.get("ref", "")) else None


def _lock_sha256() -> str:
    digest = hashlib.sha256()
    for lock, _ in CLIENT_LOCKS:
        digest.update((proc.REPO_ROOT / lock).read_bytes())
    return digest.hexdigest()


def _isolated(environ: Mapping[str, str]) -> str:
    """Build the isolated client environment once per container; return its interpreter."""
    root = Path(tempfile.gettempdir()) / f"infx-lm-eval-client-{REF[:12]}-{_lock_sha256()[:12]}"
    python = root / "bin" / "python"
    uv_bin = uv.find()
    if not python.exists():
        if proc.call([uv_bin, "venv", "-q", "--python", sys.executable, str(root)], environ):
            raise env.BenchError(f"uv venv {root} failed")
        for lock, index in CLIENT_LOCKS:
            if proc.call([uv_bin, "pip", "install", "-q", "--no-cache", "--no-deps", "--require-hashes", "-r",
                          str(proc.REPO_ROOT / lock), "--index-url", index, "--python", str(python)], environ):
                shutil.rmtree(root, ignore_errors=True)
                raise env.BenchError(f"installing {lock} failed")
        pinned = ["pip", "install", "-q", "--no-cache", "--no-deps", "--python", str(python)]
        git = shutil.which("git", path=environ.get("PATH"))
        if not (git and proc.call([uv_bin, *pinned, f"git+{REPOSITORY}.git@{REF}"], environ) == 0) and proc.call(
            [uv_bin, *pinned, f"{REPOSITORY}/archive/{REF}.tar.gz"], environ
        ):
            shutil.rmtree(root, ignore_errors=True)
            raise env.BenchError(f"installing lm-eval at {REF} failed")
    return str(python)


def native_context_length(model: str) -> int:
    """The model config's maximum sequence length, or 0 when it cannot be read."""
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        config = None
    if isinstance(config, dict):
        for name in CONTEXT_FIELDS:
            value = config.get(name)
            if type(value) is int and value > 0:
                return value
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception:  # noqa: BLE001
        return 0
    for name in CONTEXT_FIELDS:
        if hasattr(config, name):
            value = getattr(config, name)
            return value if isinstance(value, int) and value > 0 else 0
    return 0


def context_length(environ: Mapping[str, str]) -> int:
    """``EVAL_MAX_MODEL_LEN``, else ``MAX_MODEL_LEN`` capped at the model's native maximum."""
    explicit = env.optional("EVAL_MAX_MODEL_LEN", env=environ)
    if explicit is not None:
        return env.parse_positive_int("EVAL_MAX_MODEL_LEN", explicit)
    benchmark = env.optional("MAX_MODEL_LEN", env=environ) or "0"
    benchmark_length = 0 if benchmark == "0" else env.parse_positive_int("MAX_MODEL_LEN", benchmark)
    local = env.optional("MODEL_PATH", env=environ)
    model = local if local and Path(local).is_dir() else env.require("MODEL", env=environ)["MODEL"]
    native = native_context_length(model)
    length = min(benchmark_length or native, native) if native else benchmark_length
    if length:
        return length
    print(f"WARN: no context length known for {model}; using {FALLBACK_CONTEXT}", file=sys.stderr)
    return FALLBACK_CONTEXT


def max_output_tokens(context: int) -> int:
    """Leave room for the prompt, and bound the KV cache TRT-LLM reserves per request."""
    budget = context - PROMPT_RESERVE if context > PROMPT_RESERVE else context // 2
    return min(budget, MAX_OUTPUT_TOKENS)


def tasks(environ: Mapping[str, str]) -> str:
    return env.optional("EVAL_TASKS_DIR", env=environ) or DEFAULT_TASKS


def suite(environ: Mapping[str, str]) -> str:
    """The task YAML's stem, or the task name."""
    name = Path(tasks(environ)).name
    for extension in (".yaml", ".yml"):
        name = name.removesuffix(extension)
    return name


def prepare(environ: dict[str, str]) -> int:
    """Validate the inputs, install lm-eval, and return every request's context length."""
    env.require("OPENAI_API_KEY", env=environ)
    length = context_length(environ)
    install(environ)
    return length


def run(ctx: EvalContext) -> EvalOutcome:
    """Evaluate ``ctx.model`` through the server's chat completions route."""
    max_tokens = max_output_tokens(ctx.context_length)
    print(f"Eval budget: eval_context_len={ctx.context_length}, max_output_tokens={max_tokens}")
    model_args = [
        f"model={ctx.model}",
        f"base_url={ctx.base_url}/v1/chat/completions",
        f"api_key={ctx.env['OPENAI_API_KEY']}",
        "eos_string=</s>",
        "max_retries=5",
        f"num_concurrent={ctx.concurrency}",
        "timeout=1800",
        "tokenized_requests=False",
        f"max_length={ctx.context_length}",
    ]
    python = ctx.env.get(CLIENT_PYTHON) or sys.executable
    (ctx.results_dir / "eval_client.json").write_text(
        json.dumps({"python": python, "isolated": python != sys.executable, "lm_eval_ref": REF,
                    "client_locks": [lock for lock, _ in CLIENT_LOCKS], "client_lock_sha256": _lock_sha256()}, indent=1) + "\n"
    )
    argv = [
        python, "-m", "lm_eval", "--model", "local-chat-completions",
        "--apply_chat_template", "--tasks", tasks(ctx.env),
        "--output_path", str(ctx.results_dir), "--log_samples",
        "--model_args", ",".join(model_args),
        "--gen_kwargs", f"max_tokens={max_tokens},temperature=0,top_p=1",
    ]  # fmt: skip
    if limit := env.optional("EVAL_LIMIT", env=ctx.env):
        argv += ["--limit", limit]
    with tempfile.TemporaryDirectory(prefix="lm-eval-patch-") as patch:
        shutil.copyfile(proc.REPO_ROOT / PATCH, Path(patch, "sitecustomize.py"))
        pythonpath = os.pathsep.join([patch, proc.pythonpath(ctx.env)])
        rc = proc.call(argv, {**ctx.env, "PYTHONPATH": pythonpath}, cwd=proc.REPO_ROOT)
    return EvalOutcome(rc, suite(ctx.env))
