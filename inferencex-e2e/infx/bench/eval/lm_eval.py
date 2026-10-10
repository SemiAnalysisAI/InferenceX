"""The lm-evaluation-harness framework: pinned install, per-request token budget, and one run."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from infx.bench import env, proc, uv
from infx.bench.eval.context import EvalContext, EvalOutcome

REPOSITORY = "https://github.com/EleutherAI/lm-evaluation-harness"
REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"  # installed over the lm-eval[api] release
DEFAULT_TASKS = "infx/evals/gsm8k.yaml"
PATCH = "infx/evals/patches/lm_eval_sitecustomize.py"
FALLBACK_CONTEXT = 16384
PROMPT_RESERVE = 4096
MAX_OUTPUT_TOKENS = 16384
CONTEXT_FIELDS = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")
_UNSAFE_CODE = re.compile(r"^unsafe_code:\s*true\s*$", re.MULTILINE)


def install(environ: Mapping[str, str]) -> None:
    """Install lm-eval at ``REF`` into this interpreter; failures only warn, as images may ship it."""
    try:
        uv.find()
    except env.BenchError as error:
        print(f"WARN: {error}", file=sys.stderr)
        return
    # torchvision causes circular imports in ATOM; TRT-LLM/SGLang need it at module level.
    if "atom" in environ.get("IMAGE", ""):
        _pip(environ, "uninstall", "torchvision")
    _pip(environ, "install", "lm-eval[api]")
    pinned = ("install", "--no-deps", "--reinstall")
    git = shutil.which("git", path=environ.get("PATH"))
    if git and _pip(environ, *pinned, f"git+{REPOSITORY}.git@{REF}"):
        return
    _pip(environ, *pinned, f"{REPOSITORY}/archive/{REF}.tar.gz")


def _pip(environ: Mapping[str, str], command: str, *args: str) -> bool:
    argv = uv.pip(command, "-q", "--no-cache", "--break-system-packages", *args)
    rc = proc.call(argv, environ)
    if rc:
        print(f"WARN: uv {' '.join(argv[1:])} failed with exit code {rc}", file=sys.stderr)
    return rc == 0


def native_context_length(model: str) -> int:
    """The model config's maximum sequence length, or 0 when it cannot be read."""
    # A local config.json works even when transformers does not know the model type yet.
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
        from transformers import AutoConfig  # the serving image's copy, loaded only here

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception:  # noqa: BLE001 - any failure leaves the native maximum unknown
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
    benchmark = env.optional("MAX_MODEL_LEN", env=environ) or "0"  # 0: the native maximum
    benchmark_length = 0 if benchmark == "0" else env.parse_positive_int("MAX_MODEL_LEN", benchmark)
    # MODEL can be a served alias that is neither a repo id nor a path (deepseek-r1-fp4 on B300).
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


def task_args(task: str) -> list[str]:
    """``--tasks`` for ``task``, plus what a repo task YAML needs to load and run.

    The pinned lm-eval looks every task up in its index by name, so a YAML path whose
    task it does not bundle raises KeyError unless its directory is an include path.
    A task that executes model output declares ``unsafe_code: true`` and runs only
    with ``--confirm_run_unsafe_code``.
    """
    args = ["--tasks", task]
    path = proc.REPO_ROOT / task
    if path.suffix not in (".yaml", ".yml") or not path.is_file():
        return args
    args = ["--include_path", str(Path(task).parent), *args]
    if _UNSAFE_CODE.search(path.read_text()):
        args.append("--confirm_run_unsafe_code")
    return args


def suite(environ: Mapping[str, str]) -> str:
    """The task YAML's stem, or the task name."""
    name = Path(tasks(environ)).name
    for extension in (".yaml", ".yml"):
        name = name.removesuffix(extension)
    return name


def prepare(environ: Mapping[str, str]) -> int:
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
    argv = [
        sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
        "--apply_chat_template", *task_args(tasks(ctx.env)),
        "--output_path", str(ctx.results_dir), "--log_samples",
        "--model_args", ",".join(model_args),
        "--gen_kwargs", f"max_tokens={max_tokens},temperature=0,top_p=1",
    ]  # fmt: skip
    if limit := env.optional("EVAL_LIMIT", env=ctx.env):
        argv += ["--limit", limit]
    with tempfile.TemporaryDirectory(prefix="lm-eval-patch-") as patch:
        shutil.copyfile(proc.REPO_ROOT / PATCH, Path(patch, "sitecustomize.py"))
        pythonpath = os.pathsep.join([patch, proc.pythonpath(ctx.env)])
        # Relative task paths resolve against the checkout, whatever the image WORKDIR.
        rc = proc.call(argv, {**ctx.env, "PYTHONPATH": pythonpath}, cwd=proc.REPO_ROOT)
    return EvalOutcome(rc, suite(ctx.env))
