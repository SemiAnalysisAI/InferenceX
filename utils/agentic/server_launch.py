"""Apply the explicit AgentX server launch extension before starting a server.

Only the server command/environment is changed. Replay inputs and recipe
topology stay owned by the launcher. No shell evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

PROTECTED_ARGS = frozenset(
    {
        "--model",
        "--model-path",
        "--max-model-len",
        "--context-length",
        "--served-model-name",
        "--host",
        "--port",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--pp",
        "--pp-size",
        "--pipeline-parallel-size",
        "--ep",
        "--ep-size",
        "--expert-parallel-size",
        "--dp",
        "--dp-size",
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-size-local",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--enable-dp-attention",
        "--enable-expert-parallel",
        "--decode-context-parallel-size",
        "--prefill-context-parallel-size",
        "--dcp-size",
        "--pcp-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
    }
)
PROTECTED_ENV = frozenset(
    {
        "MODEL",
        "MODEL_PATH",
        "MAX_MODEL_LEN",
        "MODEL_NAME",
        "SERVED_MODEL_NAME",
        "MODEL_PREFIX",
        "TP",
        "PP_SIZE",
        "EP_SIZE",
        "DP_ATTENTION",
        "DCP_SIZE",
        "PCP_SIZE",
        "CONC",
        "DURATION",
        "PORT",
        "RUN_EVAL",
        "EVAL_ONLY",
        "IS_AGENTIC",
        "SCENARIO_TYPE",
        "SCENARIO_SUBDIR",
        "RECIPE_FINGERPRINT",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    }
)
PROTECTED_ENV_PREFIXES = ("AGENTX_", "AGENTIC_", "AIPERF_", "SGLANG_SIMULATE_ACC_")
FIELDS = {
    "version",
    "append_args",
    "remove_args",
    "replace_args",
    "env",
    "unset_env",
    "source_files",
    "absent_source_files",
    "executable",
}
RUNTIME_ENV_NAMES = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "SGLANG_SRC_PYTHONPATH",
        "GPU_ARCHS",
    }
)
RUNTIME_ENV_PREFIXES = (
    "SGLANG_",
    "VLLM_",
    "AITER_",
    "TRITON_",
    "TORCH_",
    "PYTORCH_",
    "HIP_",
    "ROCR_",
    "HSA_",
    "NCCL_",
    "RCCL_",
    "OMP_",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def validate_overrides(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("AgentX launch_overrides requires version: 1")
    if unknown := value.keys() - FIELDS:
        raise ValueError(f"Unknown AgentX launch override fields: {sorted(unknown)}")
    result = {
        "version": 1,
        "append_args": [],
        "remove_args": [],
        "replace_args": False,
        "env": {},
        "unset_env": [],
        "source_files": {},
        "absent_source_files": [],
        "executable": None,
        **value,
    }
    for key in ("append_args", "remove_args", "unset_env", "absent_source_files"):
        if not isinstance(result[key], list) or any(
            not isinstance(item, str) or "\0" in item for item in result[key]
        ):
            raise ValueError(f"{key} must be a list of strings without NUL bytes")
    if type(result["replace_args"]) is not bool:
        raise ValueError("replace_args must be a boolean")
    for key in ("env", "source_files"):
        if not isinstance(result[key], dict) or any(
            not isinstance(name, str)
            or not isinstance(item, str)
            or "\0" in name
            or "\0" in item
            for name, item in result[key].items()
        ):
            raise ValueError(f"{key} must map strings to strings without NUL bytes")
    if len(set(result["remove_args"])) != len(result["remove_args"]):
        raise ValueError("remove_args must not contain duplicates")
    for name in result["remove_args"]:
        if not re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"remove_args must contain long option names: {name!r}")
        if name in PROTECTED_ARGS:
            raise ValueError(f"Cannot remove protocol option {name}")
    names = set(result["env"]) | set(result["unset_env"])
    if set(result["env"]) & set(result["unset_env"]):
        raise ValueError("An environment variable cannot be both set and unset")
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if name in PROTECTED_ENV or name.startswith(PROTECTED_ENV_PREFIXES):
            raise ValueError(f"Cannot override protocol environment variable {name}")
    for name, checksum in result["source_files"].items():
        if not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(
                "source_files requires absolute paths and lowercase SHA256 values"
            )
    for name in result["absent_source_files"]:
        if not Path(name).is_absolute() or name in result["source_files"]:
            raise ValueError("absent_source_files requires distinct absolute paths")
    executable = result["executable"]
    if executable is not None and (
        not isinstance(executable, str)
        or "\0" in executable
        or not Path(executable).is_absolute()
    ):
        raise ValueError("executable must be an absolute path or null")
    return result


def option_groups(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    """Separate fixed executable/positionals and unambiguous long options."""
    prefix: list[str] = []
    groups: list[list[str]] = []
    for token in argv:
        if token.startswith("--"):
            name = token.partition("=")[0]
            if not re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Ambiguous server option: {token!r}")
            groups.append([token])
        elif groups:
            # Short options are ambiguous with values and cannot safely be
            # removed as part of the preceding long option. Negative numbers
            # are ordinary values. Equals-style options already own a value.
            if "=" in groups[-1][0] or (
                token.startswith("-")
                and not re.fullmatch(r"-\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token)
            ):
                raise ValueError(f"Ambiguous server option value: {token!r}")
            groups[-1].append(token)
        else:
            prefix.append(token)
    return prefix, groups


def apply_launch_args(
    argv: list[str],
    overrides: dict[str, Any],
    framework: str,
) -> list[str]:
    """Apply the v1 argv contract without reading files or the process environment."""
    overrides = validate_overrides(overrides)
    if framework not in {"sglang", "vllm"} or not argv:
        raise ValueError(
            "AgentX launch extension supports nonempty SGLang/vLLM commands"
        )
    effective = list(argv)
    # Empty requests do not parse/rewrite an existing command. This preserves
    # every canonical launcher token, including options unfamiliar to v1.
    if (
        overrides["append_args"]
        or overrides["remove_args"]
        or overrides["replace_args"]
    ):
        prefix, groups = option_groups(argv)
        extra_prefix, extra = option_groups(overrides["append_args"])
        if extra_prefix:
            raise ValueError(
                "append_args cannot inject an executable or positional arguments"
            )
        if any(group[0].partition("=")[0] in PROTECTED_ARGS for group in extra):
            raise ValueError("append_args cannot change model, topology, host or port")
        for name in overrides["remove_args"]:
            matches = [group for group in groups if group[0].partition("=")[0] == name]
            if len(matches) != 1:
                raise ValueError(
                    f"remove_args must match exactly one existing option: {name}"
                )
            groups.remove(matches[0])
        if overrides["replace_args"]:
            # Keep the executable/model positionals and all protocol options;
            # only the optional serving configuration is replaced.
            groups = [
                group
                for group in groups
                if group[0].partition("=")[0] in PROTECTED_ARGS
            ]
        effective = prefix + [token for group in groups + extra for token in group]
    if overrides["executable"] is not None:
        effective[0] = overrides["executable"]
    return effective


def prepare_launch(
    argv: list[str],
    overrides: dict[str, Any],
    framework: str,
    environ: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    overrides = validate_overrides(overrides)
    environ = dict(os.environ if environ is None else environ)
    effective = apply_launch_args(argv, overrides, framework)
    if overrides["executable"] is not None:
        executable = Path(overrides["executable"])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                f"Requested server executable is not executable: {executable}"
            )
    sources = {}
    for name, expected in overrides["source_files"].items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"AgentX source file changed before server launch: {name}")
        sources[name] = actual
    for name in overrides["absent_source_files"]:
        if os.path.lexists(name):
            raise ValueError(
                f"AgentX deleted source file reappeared before launch: {name}"
            )
    touched = sorted(set(overrides["env"]) | set(overrides["unset_env"]))
    base_env = {name: environ.get(name) for name in touched}
    effective_env = {name: overrides["env"].get(name) for name in touched}
    child_env = dict(environ)
    for name in overrides["unset_env"]:
        child_env.pop(name, None)
    child_env.update(overrides["env"])
    resolved_executable = shutil.which(effective[0], path=child_env.get("PATH"))
    if resolved_executable is None:
        raise ValueError(f"Server executable not found: {effective[0]}")
    evidence = {
        "version": 1,
        "framework": framework,
        "base_argv": argv,
        "effective_argv": effective,
        "base_env": base_env,
        "effective_env": effective_env,
        "overrides_sha256": digest(overrides),
        "source_files": sources,
        "absent_source_files": overrides["absent_source_files"],
        "resolved_executable": str(Path(resolved_executable).resolve()),
        "runtime_environment": {
            name: value
            for name, value in sorted(child_env.items())
            if (name in RUNTIME_ENV_NAMES or name.startswith(RUNTIME_ENV_PREFIXES))
            and not any(
                secret in name.upper()
                for secret in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")
            )
        },
    }
    evidence["evidence_sha256"] = digest(evidence)
    # The env utility execs the original server; overrides never leak to the
    # router or replay subprocesses in the parent launcher.
    command = []
    if touched:
        command.append("env")
        for name in overrides["unset_env"]:
            command.extend(["-u", name])
        command.extend(f"{name}={value}" for name, value in overrides["env"].items())
    return command + effective, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--overrides", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        request = validate_overrides(json.loads(args.overrides.read_text()))
        if digest(request) != args.expected_sha256:
            raise ValueError("AgentX launch override hash does not match the request")
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        command, evidence = prepare_launch(argv, request, args.framework)
        args.output.write_bytes(b"".join(token.encode() + b"\0" for token in command))
        temporary = args.evidence.with_name(args.evidence.name + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.evidence)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"AgentX server launch rejected: {exc}\n")


if __name__ == "__main__":
    main()
