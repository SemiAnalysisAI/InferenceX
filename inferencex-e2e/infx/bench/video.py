from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from infx.bench import env


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
    inputs = env.require("H3_SERVER_SERVICE", "H3_SERVER_PORT")
    key = re.sub(r"[^A-Z0-9_]", "_", inputs["H3_SERVER_SERVICE"].upper())
    host = env.require(f"SRT_SERVICE_{key}_IPS")[f"SRT_SERVICE_{key}_IPS"].split(",")[0]
    if not host or any(char.isspace() for char in host) or any(c in host for c in "/?#@"):
        raise env.InputError("the video service must publish a valid host address")
    port = env.positive_int("H3_SERVER_PORT")
    if port > 65535:
        raise env.InputError("H3_SERVER_PORT exceeds 65535")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inputs = env.require("H3_SPEC_PATH", "H3_RESULT_DIR")
    spec_path, output = Path(inputs["H3_SPEC_PATH"]), Path(inputs["H3_RESULT_DIR"])
    try:
        if args.action == "serve":
            env.require("H3_RECIPE_ENV_KEYS")
        selected = env.require("H3_SET")["H3_SET"]
        from evaluator import srt_selected

        native = env.require("H3_PLAN_SHA256", "H3_RUNTIME", "H3_RUNTIME_REVISION")
        plan = srt_selected.selected_plan(spec_path, selected, native["H3_PLAN_SHA256"])
        gpus = env.positive_int("GPU_COUNT")
        if args.action == "serve":
            model = env.require("H3_MODEL_PATH")["H3_MODEL_PATH"]
            srt_selected.serve(
                plan,
                runtime=native["H3_RUNTIME"],
                runtime_revision=native["H3_RUNTIME_REVISION"],
                model_path=Path(model),
                port=env.positive_int("H3_SERVER_PORT"),
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
            runtime=native["H3_RUNTIME"],
            runtime_revision=native["H3_RUNTIME_REVISION"],
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
        return (
            0
            if all(
                run["summary"]["valid"] == run["summary"]["scheduled"]
                and run["summary"]["scheduled"]
                for run in runs.values()
            )
            else 1
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise env.BenchError(str(error)) from error
