from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.bench import env

JOB_STATUS = "native-job-status.json"
CREDENTIAL_ENV = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH|BEARER|"
    r"API_KEY|ACCESS_KEY|PRIVATE_KEY|SSH_KEY)(?:_|$)",
    re.IGNORECASE,
)


def cell_dir(concurrency: int) -> str:
    return f"gpu/c{concurrency}"


def window_file(concurrency: int) -> str:
    return f"benchmark_concurrency_{concurrency}.json"


def run_complete(run: Mapping[str, Any]) -> bool:
    summary = run.get("summary") or {}
    return (
        run.get("status") == "complete"
        and bool(summary.get("scheduled"))
        and summary.get("valid") == summary.get("scheduled")
    )


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench video")
    commands = parser.add_subparsers(dest="action", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--model-view", required=True, choices=("snapshot", "h3-fl2va"))
    server.add_argument("command", nargs=argparse.REMAINDER)
    client = commands.add_parser("client")
    client.add_argument("--request-seconds", required=True, type=float)
    args = parser.parse_args(argv)
    if args.action == "serve":
        if not args.command or args.command[0] != "--" or len(args.command) < 2:
            raise env.InputError("video serve requires -- followed by the engine command")
        args.command = args.command[1:]
    elif not 0 < args.request_seconds < float("inf"):
        raise env.InputError("--request-seconds must be finite and positive")
    return args


def endpoint() -> str:
    inputs = env.require("VIDEO_SERVER_SERVICE", "VIDEO_SERVER_PORT")
    key = re.sub(r"[^A-Z0-9_]", "_", inputs["VIDEO_SERVER_SERVICE"].upper())
    host = env.require(f"SRT_SERVICE_{key}_IPS")[f"SRT_SERVICE_{key}_IPS"].split(",")[0]
    if not host or any(char.isspace() for char in host) or any(c in host for c in "/?#@"):
        raise env.InputError("the video service must publish a valid host address")
    port = env.positive_int("VIDEO_SERVER_PORT")
    if port > 65535:
        raise env.InputError("VIDEO_SERVER_PORT exceeds 65535")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inputs = env.require("VIDEO_SPEC_PATH", "VIDEO_RESULT_DIR")
    spec_path, output = Path(inputs["VIDEO_SPEC_PATH"]), Path(inputs["VIDEO_RESULT_DIR"])
    try:
        if args.action == "serve":
            env.require("VIDEO_RECIPE_ENV_KEYS")
        selected = env.require("VIDEO_SET")["VIDEO_SET"]
        from evaluator import srt_selected

        native = env.require("VIDEO_PLAN_SHA256", "VIDEO_RUNTIME", "VIDEO_RUNTIME_REVISION")
        plan = srt_selected.selected_plan(spec_path, selected, native["VIDEO_PLAN_SHA256"])
        gpus = env.positive_int("GPU_COUNT")
        if args.action == "serve":
            model = env.require("VIDEO_MODEL_PATH")["VIDEO_MODEL_PATH"]
            srt_selected.serve(
                plan,
                runtime=native["VIDEO_RUNTIME"],
                runtime_revision=native["VIDEO_RUNTIME_REVISION"],
                model_path=Path(model),
                gpus=gpus,
                output=output / "server",
                command=args.command,
                model_view=args.model_view,
            )
            return 0

        concurrency_values = env.require("CONC_LIST")["CONC_LIST"].split()
        concurrencies = [env.parse_positive_int("CONC_LIST", c) for c in concurrency_values]
        if not concurrencies or len(set(concurrencies)) != len(concurrencies):
            raise env.InputError("CONC_LIST must contain distinct positive integers")
        window_dir = Path(env.require("SRT_MEASUREMENT_WINDOW_DIR")["SRT_MEASUREMENT_WINDOW_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        window_dir.mkdir(parents=True, exist_ok=True)
        url = endpoint()
        runs = srt_selected.run_matrix(
            plan,
            runtime=native["VIDEO_RUNTIME"],
            runtime_revision=native["VIDEO_RUNTIME_REVISION"],
            endpoint=url,
            output=output,
            concurrencies=concurrencies,
            gpus=gpus,
            window_dir=window_dir,
            request_seconds=args.request_seconds,
        )
        for concurrency, run in runs.items():
            print(
                json.dumps(
                    {
                        "concurrency": concurrency,
                        "summary": run.get("summary"),
                        "status": run.get("status"),
                    }
                )
            )
        return 0 if all(map(run_complete, runs.values())) else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise env.BenchError(str(error)) from error
