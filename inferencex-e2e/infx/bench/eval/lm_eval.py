"""The lm-evaluation-harness framework: pinned install, per-request token budget, and one run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from infx.bench import env, proc
from infx.bench.eval.context import EvalContext, EvalOutcome

REPOSITORY = "https://github.com/EleutherAI/lm-evaluation-harness"
REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
DEFAULT_TASKS = "infx/evals/gsm8k.yaml"
PATCH = "infx/evals/patches/lm_eval_sitecustomize.py"
# The client's complete dependency closure, hash-pinned and installed without resolution: torch (CPU) from the
# PyTorch CPU index, everything else (transformers 5.16.1, huggingface-hub 1.33.0, datasets 5.1.0, ...) from PyPI.
# lm-eval itself is installed at REF with --no-deps.
CLIENT_LOCKS = (
    ("infx/evals/lm_eval_client_torch.lock", "https://download.pytorch.org/whl/cpu"),
    ("infx/evals/lm_eval_client.lock", "https://pypi.org/simple"),
)
CLIENT_PYTHON = "INFX_EVAL_CLIENT_PYTHON"
CLIENT_RECEIPT = "INFX_EVAL_CLIENT_RECEIPT"
RECEIPT = "infx-receipt.json"
# The installer the client locks were checked with; a serving image's own uv and its UV_*/PIP_* settings and uv.toml
# files are not used for the client.
CLIENT_UV = "0.12.5"
# sha256 of the CLIENT_UV release archive for each platform.machine().
CLIENT_UV_SHA256 = {
    "x86_64": "68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2",
    "aarch64": "9bf43b4d1a07665bf64d4c4e710930b382321a785e0eb10aac07f46471f86a31",
}
CLIENT_PROBE = (
    "import json, sys, importlib.metadata as m; "
    "from lm_eval.models.openai_completions import LocalChatCompletion; "
    "u = json.loads(m.distribution('lm-eval').read_text('direct_url.json') or '{}'); "
    "print(json.dumps({'ref': (u.get('vcs_info') or {}).get('commit_id') or u.get('url', ''), "
    "'versions': {n: next((d.version for d in m.distributions(name=n)), None) "
    "for n in sys.argv[1:]}}))"
)
FALLBACK_CONTEXT = 16384
PROMPT_RESERVE = 4096
MAX_OUTPUT_TOKENS = 16384
CONTEXT_FIELDS = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")


def install(environ: dict[str, str]) -> None:
    """Select the interpreter that runs lm-eval; record it and its receipt in ``environ``.

    This interpreter when it runs lm-eval's chat-completions client at exactly ``REF`` over exactly the
    versions ``CLIENT_LOCKS`` pin; otherwise an isolated virtual environment built from them and verified
    the same way, else ``BenchError``. The serving interpreter's packages are never changed: installing
    lm-eval[api] into a serving image can replace dependencies its own packages pin (huggingface-hub 2.x
    under a transformers that requires <2.0).
    """
    environ.pop(CLIENT_PYTHON, None)
    environ.pop(CLIENT_RECEIPT, None)
    found = _client(sys.executable, environ)
    text = _isolated(environ) if found is None else _receipt(sys.executable, found, isolated=False)
    receipt = json.loads(text)
    environ[CLIENT_PYTHON] = receipt["python"]
    environ[CLIENT_RECEIPT] = text
    print(
        f"Eval client: {receipt['python']} runs lm-eval from {receipt['lm_eval_source']} over "
        f"{len(receipt['versions'])} locked versions; lock sha256 {receipt['client_lock_sha256']}"
    )


def _client(python: str, environ: Mapping[str, str]) -> dict | None:
    """``python``'s lm-eval source and locked versions; ``None`` unless exactly REF and the locks."""
    pins = _pins()
    try:
        probe = subprocess.run(
            [python, "-c", CLIENT_PROBE, *pins],
            env=dict(environ),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode:
        return None
    try:
        found = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    # direct_url.json records a git install's commit and an archive install's URL.
    sources = (REF, f"{REPOSITORY}/archive/{REF}.tar.gz")
    if (
        not isinstance(found, dict)
        or found.get("ref") not in sources
        or found.get("versions") != pins
    ):
        return None
    return found


def _pins() -> dict[str, str]:
    """``{name: version}`` of every requirement in ``CLIENT_LOCKS``."""
    pins = {}
    for lock, _ in CLIENT_LOCKS:
        for line in (proc.REPO_ROOT / lock).read_text().splitlines():
            name, pinned, version = line.partition("==")
            if pinned and line[:1].isalnum():
                pins[name] = version.rstrip(" \\")
    return pins


def _receipt(python: str, found: dict, *, isolated: bool) -> str:
    """The receipt of ``python`` verified as ``found``, as ``run`` records it."""
    receipt = {
        "verified": True,
        "isolated": isolated,
        "python": python,
        "lm_eval_ref": REF,
        "lm_eval_source": found["ref"],
        "client_locks": [lock for lock, _ in CLIENT_LOCKS],
        "client_lock_sha256": _lock_sha256(),
        "versions": found["versions"],
        "uv": CLIENT_UV if isolated else None,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return json.dumps(receipt, indent=1, sort_keys=True) + "\n"


def _lock_sha256() -> str:
    digest = hashlib.sha256()
    for lock, _ in CLIENT_LOCKS:
        digest.update((proc.REPO_ROOT / lock).read_bytes())
    return digest.hexdigest()


def _uv_version(
    binary: str | os.PathLike[str], environ: Mapping[str, str], cwd: Path
) -> str | None:
    try:
        probe = subprocess.run(
            [str(binary), "--version"],
            env=dict(environ),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = probe.stdout.split()
    return parts[1] if not probe.returncode and len(parts) > 1 and parts[0] == "uv" else None


def _client_uv(environ: Mapping[str, str], scratch: Path) -> str:
    """``uv`` at CLIENT_UV: the one on PATH when it is that version, else its release archive's."""
    found = shutil.which("uv", path=environ.get("PATH"))
    image_version = _uv_version(found, environ, scratch) if found else None
    print(f"Eval client installer: uv {CLIENT_UV}; image uv {image_version or 'absent'}")
    if found and image_version == CLIENT_UV:
        return found
    arch = platform.machine()
    if arch not in CLIENT_UV_SHA256:
        raise env.BenchError(f"no uv {CLIENT_UV} release archive for {arch!r}")
    name = f"uv-{arch}-unknown-linux-gnu"
    url = f"https://github.com/astral-sh/uv/releases/download/{CLIENT_UV}/{name}.tar.gz"
    archive = scratch / f"{name}.tar.gz"
    download = ["curl", "-fsSL", "--proto", "=https", "--retry", "3", "-o", str(archive), url]
    if proc.call(download, environ, cwd=scratch):
        raise env.BenchError(f"downloading {url} failed")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != CLIENT_UV_SHA256[arch]:
        raise env.BenchError(f"{url} has sha256 {digest}, not {CLIENT_UV_SHA256[arch]}")
    binary = scratch / "uv"
    try:
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember(f"{name}/uv")
            source = tar.extractfile(member) if member.isfile() else None
            if source is not None:
                binary.write_bytes(source.read())
                binary.chmod(0o755)
    except (KeyError, OSError, tarfile.TarError) as error:
        raise env.BenchError(f"{url} has no {name}/uv: {error}") from error
    if _uv_version(binary, environ, scratch) != CLIENT_UV:
        raise env.BenchError(f"{url} does not hold uv {CLIENT_UV}")
    return str(binary)


def _build(build: Path, python: str, environ: Mapping[str, str]) -> str:
    """Install and verify the client in a new environment at ``build``; its receipt names ``python``."""
    clean = {k: v for k, v in environ.items() if not k.startswith(("UV_", "PIP_"))}
    ignored = sorted(k for k in environ if k not in clean)
    print(f"Eval client installer: --no-config, ignoring {ignored}")
    target = str(build / "bin" / "python")
    with tempfile.TemporaryDirectory(prefix="infx-uv-") as scratch:
        work = Path(scratch)
        uv_bin = _client_uv(clean, work)
        flags = ["-q", "--no-config", "--no-cache"]
        # Relocatable: the verified environment is renamed into place.
        venv = [uv_bin, "venv", *flags, "--relocatable", "--python", sys.executable, str(build)]
        if proc.call(venv, clean, cwd=work):
            raise env.BenchError(f"uv venv {build} failed")
        pip = [uv_bin, "pip", "install", *flags, "--no-deps", "--python", target]
        for lock, index in CLIENT_LOCKS:
            locked = ["--require-hashes", "-r", str(proc.REPO_ROOT / lock), "--index-url", index]
            if proc.call([*pip, *locked], clean, cwd=work):
                raise env.BenchError(f"installing {lock} failed")
        git = shutil.which("git", path=clean.get("PATH"))
        sources = [f"git+{REPOSITORY}.git@{REF}"] if git else []
        sources.append(f"{REPOSITORY}/archive/{REF}.tar.gz")
        if all(proc.call([*pip, source], clean, cwd=work) for source in sources):
            raise env.BenchError(f"installing lm-eval at {REF} failed")
    found = _client(target, environ)
    if found is None:
        raise env.BenchError(f"{build} does not run lm-eval at {REF} over the locked versions")
    return _receipt(python, found, isolated=True)


def _reusable(root: Path, lock_sha256: str) -> str | None:
    """``root``'s receipt when it records a verified build of REF and the locks there."""
    try:
        text = (root / RECEIPT).read_text()
        receipt = json.loads(text)
    except (OSError, ValueError):
        return None
    expected = {
        "verified": True,
        "isolated": True,
        "python": str(root / "bin" / "python"),
        "lm_eval_ref": REF,
        "client_lock_sha256": lock_sha256,
        "versions": _pins(),
        "uv": CLIENT_UV,
    }
    if isinstance(receipt, dict) and all(receipt.get(k) == v for k, v in expected.items()):
        return text
    return None


def _isolated(environ: Mapping[str, str]) -> str:
    """The receipt of the isolated client environment, built and verified by one launcher at a time."""
    lock_sha256 = _lock_sha256()
    root = Path(tempfile.gettempdir()) / f"infx-lm-eval-client-{REF[:12]}-{lock_sha256[:12]}"
    python = str(root / "bin" / "python")
    try:
        with open(f"{root}.lock", "a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            text = _reusable(root, lock_sha256)
            if text is not None and _client(python, environ) is not None:
                print(f"Eval client: reusing the verified {root}")
                return text
            shutil.rmtree(root, ignore_errors=True)
            # Builds hold the lock, so a .build-* sibling is a killed build's leftover.
            for stale in root.parent.glob(f"{root.name}.build-*"):
                shutil.rmtree(stale, ignore_errors=True)
            build = Path(tempfile.mkdtemp(prefix=f"{root.name}.build-", dir=root.parent))
            try:
                text = _build(build, python, environ)
                (build / RECEIPT).write_text(text)
                build.rename(root)
            finally:
                shutil.rmtree(build, ignore_errors=True)
    except OSError as error:
        raise env.BenchError(f"isolated eval client {root}: {error}") from error
    print(f"Eval client: built and verified {root}")
    return text


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


def _verified(environ: Mapping[str, str]) -> dict:
    """``install``'s receipt for ``environ[CLIENT_PYTHON]``, with the receipt's path and sha256.

    ``BenchError`` unless it verifies that interpreter at REF over the current locks and the
    interpreter and its receipt are still in place.
    """
    python = environ.get(CLIENT_PYTHON, "")
    text = environ.get(CLIENT_RECEIPT, "")
    try:
        receipt = json.loads(text)
    except ValueError:
        receipt = None
    expected = {
        "verified": True,
        "python": python,
        "lm_eval_ref": REF,
        "client_lock_sha256": _lock_sha256(),
    }
    if not (
        python
        and isinstance(receipt, dict)
        and all(receipt.get(k) == v for k, v in expected.items())
        and os.access(python, os.X_OK)
    ):
        raise env.BenchError(f"no verified lm-eval client for {CLIENT_PYTHON}={python!r}")
    path = Path(python).parent.parent / RECEIPT if receipt.get("isolated") else None
    try:
        recorded = path.read_text() if path else text
    except OSError:
        recorded = None
    if recorded != text:
        raise env.BenchError(f"{path} no longer holds the receipt of {python}")
    sha256 = hashlib.sha256(text.encode()).hexdigest()
    return {**receipt, "receipt": str(path) if path else None, "receipt_sha256": sha256}


def run(ctx: EvalContext) -> EvalOutcome:
    """Evaluate ``ctx.model`` through the server's chat completions route with the verified client."""
    client = _verified(ctx.env)
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
    (ctx.results_dir / "eval_client.json").write_text(
        json.dumps(client, indent=1, sort_keys=True) + "\n"
    )
    argv = [
        client["python"], "-m", "lm_eval", "--model", "local-chat-completions",
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
