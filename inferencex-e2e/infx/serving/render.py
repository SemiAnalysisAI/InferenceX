"""Turn a validated ServingSpec's launch commands into what the recipes run.

serve-cmd is shlex-quoted so `eval "set -- $SERVE_CMD"` in bash restores the
exact argv; --host/--port are removed here and appended by the recipe, which
owns the port. With a router block (V2), router-cmd carries the Dynamo frontend
argv the same way: it loses --http-host/--http-port here and serve_router.sh
appends them. Any other router is the "generic" profile: its command is kept whole, {{router_port}} / {{engine_url}} / {{engine_port}} are expanded at runtime by serve_router.sh.
"""

from __future__ import annotations

import json
import re
import shlex

from infx.serving.schema import ServingError, ServingSpec

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_OWNED_FLAGS = ("--host", "--port")
_ROUTER_OWNED_FLAGS = ("--http-host", "--http-port")


def split_serve(serve: str, field: str = "serve") -> list[str]:
    joined = re.sub(r"\\[ \t]*\r?\n", " ", serve)
    try:
        return shlex.split(joined, comments=True)
    except ValueError as error:
        raise ServingError(
            f"{field}: cannot parse the command ({error}); check the quotes"
        ) from error


def _pop_env_prefix(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    index = 0
    while index < len(argv) and _ENV_ASSIGNMENT.match(argv[index]):
        key, value = argv[index].split("=", 1)
        env[key] = value
        index += 1
    return env, argv[index:]


def _python_module(argv: list[str]) -> str | None:
    if len(argv) >= 3 and argv[0] in ("python", "python3") and argv[1] == "-m":
        return argv[2]
    return None


def infer_framework(argv: list[str]) -> str:
    if argv[:2] == ["vllm", "serve"]:
        return "vllm"
    if (
        argv[:2] == ["sglang", "serve"]
        or "sglang.launch_server" in argv
        or _python_module(argv) == "dynamo.sglang"
    ):
        return "sglang"
    raise ServingError(
        "serve: unknown engine; the command must start with `vllm serve`, "
        "`python3 -m sglang.launch_server`, `sglang serve` or `python3 -m dynamo.sglang`"
    )


def _strip_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg in flags:
            skip_value = True
            continue
        if arg.startswith(tuple(f"{flag}=" for flag in flags)):
            continue
        out.append(arg)
    return out


def _check_json_values(field: str, argv: list[str]) -> None:
    """A flag value that starts with "{" must still be JSON once the shell removed its quotes."""
    for index, arg in enumerate(argv):
        if arg.startswith("--") and "=" in arg:
            flag, value = arg.split("=", 1)
        elif index and argv[index - 1].startswith("--") and "=" not in argv[index - 1]:
            flag, value = argv[index - 1], arg
        else:
            continue
        # "{%" / "{{" is an inline Jinja template (e.g. --chat-template), not JSON
        if not value.startswith("{") or value.startswith(("{%", "{{")):
            continue
        try:
            json.loads(value)
        except ValueError:
            raise ServingError(
                f"{field}: {flag} is not valid JSON after shell quoting; "
                f"wrap it in single quotes: {flag}='{{...}}'"
            ) from None


def _render_router(spec: ServingSpec, engine_argv: list[str]) -> dict[str, str]:
    is_dynamo_engine = _python_module(engine_argv) == "dynamo.sglang"
    if spec.router is None:
        if is_dynamo_engine:
            raise ServingError(
                "router: required for python3 -m dynamo.sglang (a Dynamo engine has no HTTP port); "
                "add a router block running python3 -m dynamo.frontend"
            )
        return {
            "router-image": "", "router-cmd": "", "router-env": "",
            "router-profile": "", "router-metrics-port": "",
        }
    prefix_env, argv = _pop_env_prefix(split_serve(spec.router.serve, "router.serve"))
    if not argv:
        raise ServingError("router.serve: empty command")
    is_dynamo_router = _python_module(argv) == "dynamo.frontend"
    if is_dynamo_router and not is_dynamo_engine:
        raise ServingError("serve: python3 -m dynamo.frontend needs a python3 -m dynamo.sglang engine")
    if is_dynamo_engine and not is_dynamo_router:
        raise ServingError(
            "router.serve: a python3 -m dynamo.sglang engine needs python3 -m dynamo.frontend as its router"
        )
    _check_json_values("router.serve", argv)
    if is_dynamo_router:
        if spec.router.metrics_port is not None:
            raise ServingError(
                "router.metrics_port: not used with python3 -m dynamo.frontend "
                "(metrics come from the frontend port and the Dynamo system port); remove it"
            )
        argv = _strip_flags(argv, _ROUTER_OWNED_FLAGS)
    env = {**prefix_env, **spec.router.env}
    return {
        "router-image": spec.router.image,
        "router-cmd": shlex.join(argv),
        "router-env": "\n".join(f"{key}={value}" for key, value in env.items()),
        "router-profile": "dynamo" if is_dynamo_router else "generic",
        "router-metrics-port": "" if spec.router.metrics_port is None else str(spec.router.metrics_port),
    }


def _render_storage(spec: ServingSpec) -> dict[str, str]:
    if spec.storage is None:
        return {"storage-image": "", "storage-cmd": "", "storage-env": ""}
    prefix_env, argv = _pop_env_prefix(split_serve(spec.storage.serve, "storage.serve"))
    if not argv:
        raise ServingError("storage.serve: empty command")
    _check_json_values("storage.serve", argv)
    env = {**prefix_env, **spec.storage.env}
    return {
        "storage-image": spec.storage.image,
        "storage-cmd": shlex.join(argv),
        "storage-env": "\n".join(f"{key}={value}" for key, value in env.items()),
    }


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def render(spec: ServingSpec) -> dict[str, str]:
    prefix_env, argv = _pop_env_prefix(split_serve(spec.serve))
    if not argv:
        raise ServingError("serve: empty command")
    framework = infer_framework(argv)
    _check_json_values("serve", argv)
    if framework == "vllm" and "--disable-log-stats" in argv:
        raise ServingError("serve: drop --disable-log-stats; the benchmark scrapes vLLM metrics")
    router = _render_router(spec, argv)
    storage = _render_storage(spec)
    argv = _strip_flags(argv, _OWNED_FLAGS)
    # A dynamo.sglang engine must bring its own --enable-metrics (spec 2026-10-13 §1);
    # the V1 append rule stays for the V1 sglang engines.
    if framework == "sglang" and "--enable-metrics" not in argv:
        if _python_module(argv) == "dynamo.sglang":
            raise ServingError(
                "serve: python3 -m dynamo.sglang needs --enable-metrics "
                "(the metrics gate requires sglang:* series)"
            )
        argv.append("--enable-metrics")
    env = {**prefix_env, **spec.env}
    return {
        "framework": framework,
        "serve-cmd": shlex.join(argv),
        "serve-env": "\n".join(f"{key}={value}" for key, value in env.items()),
        "served-model-name": _flag_value(argv, "--served-model-name") or "",
        **router,
        **storage,
    }
