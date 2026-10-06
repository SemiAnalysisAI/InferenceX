"""Fixed-sequence throughput lanes; ``client_argv`` owns the client policy they share."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from infx.bench import env, proc, server

# The client needs numpy and transformers, so it runs in a child, never in this process.
PYTHON = "python3"
# Single-node srt frameworks and the client backend that speaks their completions API.
SINGLE_NODE_BACKENDS = {"sglang": "vllm", "atom": "vllm", "trt": "openai"}


@dataclass(frozen=True)
class Point:
    """One client run: ``num_prompts`` random ISL/OSL requests at concurrency ``conc``."""

    base_url: str
    model: str
    backend: str
    isl: int
    osl: int
    random_range_ratio: str
    conc: int
    num_prompts: int
    result: Path
    tokenizer: str | None = None
    use_chat_template: bool = False
    trust_remote_code: bool = False


def client_argv(point: Point) -> list[str]:
    """An unthrottled burst that ignores EOS, after 2x-concurrency warmups."""
    argv = [
        PYTHON, "-m", "infx.bench_serving.benchmark_serving",
        "--model", point.model,
        "--backend", point.backend,
        "--base-url", point.base_url,
        "--dataset-name", "random",
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--random-range-ratio", point.random_range_ratio,
        "--num-prompts", str(point.num_prompts),
        "--max-concurrency", str(point.conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups", str(2 * point.conc),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--result-dir", str(point.result.parent),
        "--result-filename", point.result.name,
    ]  # fmt: skip
    if point.tokenizer is not None:
        argv += ["--tokenizer", point.tokenizer]
    if point.use_chat_template:
        argv.append("--use-chat-template")
    if point.trust_remote_code:
        argv.append("--trust-remote-code")
    return argv


def _run_client(point: Point) -> int:
    with proc.RelaySignals() as relay:
        return relay.run(client_argv(point))


def served_model(base_url: str) -> str:
    """The first model id the frontend lists."""
    listing = server.http_json(f"{base_url}/v1/models")
    data = listing.get("data") if isinstance(listing, dict) else None
    first = data[0] if isinstance(data, list) and data else None
    model = first.get("id") if isinstance(first, dict) else None
    if not isinstance(model, str) or not model:
        raise env.BenchError(f"{base_url}/v1/models lists no served model")
    return model


def srt_single(args: argparse.Namespace) -> int:
    """One single-node point; srt-slurm owns the server and power sampling."""
    values = env.require(
        "MODEL", "CONC", "ISL", "OSL", "RANDOM_RANGE_RATIO", "RESULT_FILENAME", "RESULT_DIR",
        "SRT_FRONTEND_HOST", "SRT_FRONTEND_PORT", "RUN_EVAL", "EVAL_ONLY",
        "USE_CHAT_TEMPLATE", "FRAMEWORK",
    )  # fmt: skip
    env.flag("RUN_EVAL")
    eval_only = env.flag("EVAL_ONLY")
    backend = SINGLE_NODE_BACKENDS.get(values["FRAMEWORK"])
    if backend is None:
        raise env.InputError(f"unsupported fixed-sequence FRAMEWORK: {values['FRAMEWORK']}")
    conc = env.positive_int("CONC")
    result_dir = Path(values["RESULT_DIR"])
    point = Point(
        base_url=f"http://{values['SRT_FRONTEND_HOST']}:{env.positive_int('SRT_FRONTEND_PORT')}",
        model=values["MODEL"],
        backend=backend,
        isl=env.positive_int("ISL"),
        osl=env.positive_int("OSL"),
        random_range_ratio=values["RANDOM_RANGE_RATIO"],
        conc=conc,
        num_prompts=10 * conc,
        result=result_dir / f"{values['RESULT_FILENAME']}.json",
        use_chat_template=env.flag("USE_CHAT_TEMPLATE"),
        trust_remote_code=args.trust_remote_code,
    )
    if not result_dir.is_dir():
        raise env.InputError("RESULT_DIR must be an existing runtime-provided directory")
    if eval_only:
        print("EVAL_ONLY mode: skipping throughput benchmark", flush=True)
        return 0
    # Removing the local sampler must not silently turn measured points into unmeasured ones.
    env.require("SRT_MEASUREMENT_WINDOW_DIR")
    with proc.RelaySignals() as relay:
        rc = relay.run(client_argv(point))
        if rc:
            return rc
        return relay.run([PYTHON, "-m", "infx.results.power.window", str(point.result), str(conc)])


def srt_sweep(args: argparse.Namespace) -> int:
    """Every ``CONC_LIST`` point of an srt-slurm multi-node job; srt-slurm owns the servers."""
    values = env.require(
        "ISL", "OSL", "RANDOM_RANGE_RATIO", "SRT_FRONTEND_HOST", "SRT_FRONTEND_PORT",
        "CONC_LIST", "PREFILL_NUM_WORKERS", "PREFILL_TP", "DECODE_NUM_WORKERS", "DECODE_TP",
    )  # fmt: skip
    isl = env.positive_int("ISL")
    osl = env.positive_int("OSL")
    base_url = f"http://{values['SRT_FRONTEND_HOST']}:{env.positive_int('SRT_FRONTEND_PORT')}"
    # Aggregated rows export DECODE_NUM_WORKERS=0; the result name still records it.
    ctx = env.non_negative_int("PREFILL_NUM_WORKERS") * env.positive_int("PREFILL_TP")
    gen = env.non_negative_int("DECODE_NUM_WORKERS") * env.positive_int("DECODE_TP")
    concs = [env.parse_positive_int("CONC_LIST", word) for word in values["CONC_LIST"].split()]
    windows = env.optional("SRT_MEASUREMENT_WINDOW_DIR")
    # The name the workers registered; the workflow's MODEL is the HF id, which can differ.
    model = served_model(base_url)
    tokenizer = env.optional("TOKENIZER") or model
    result_dir = args.logs_dir / f"sa-bench_isl_{isl}_osl_{osl}"
    result_dir.mkdir(parents=True, exist_ok=True)
    with proc.RelaySignals() as relay:
        for conc in concs:
            name = f"results_concurrency_{conc}_gpus_{ctx + gen}_ctx_{ctx}_gen_{gen}.json"
            point = Point(
                base_url=base_url,
                model=model,
                backend="openai",
                tokenizer=tokenizer,
                isl=isl,
                osl=osl,
                random_range_ratio=values["RANDOM_RANGE_RATIO"],
                conc=conc,
                num_prompts=10 * conc,
                result=result_dir / name,
                use_chat_template=True,
                trust_remote_code=True,
            )
            rc = relay.run(client_argv(point))
            # Power lanes: tell srt-slurm which interval this concurrency's result measured.
            if rc == 0 and windows:
                window = [PYTHON, "-m", "infx.results.power.window", str(point.result), str(conc)]
                rc = relay.run(window)
            if rc:
                return rc
    return 0


def explicit_point(args: argparse.Namespace) -> int:
    """One point described entirely by flags."""
    return _run_client(
        Point(
            base_url=args.base_url,
            model=args.model,
            backend=args.backend,
            tokenizer=args.tokenizer,
            isl=env.parse_positive_int("--isl", args.isl),
            osl=env.parse_positive_int("--osl", args.osl),
            random_range_ratio=args.random_range_ratio,
            conc=env.parse_positive_int("--conc", args.conc),
            num_prompts=env.parse_positive_int("--num-prompts", args.num_prompts),
            result=args.result,
        )
    )


def main(argv: list[str]) -> int:
    """Run the ``fixed-seq`` command."""
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench fixed-seq")
    modes = parser.add_subparsers(dest="mode", required=True)

    single = modes.add_parser("srt-single", help="one srt-slurm single-node point (env)")
    single.add_argument("--trust-remote-code", action="store_true")
    single.set_defaults(run=srt_single)

    sweep = modes.add_parser("srt-sweep", help="every CONC_LIST point of a multi-node job (env)")
    sweep.add_argument("--logs-dir", type=Path, required=True, help="srt-slurm's log mount")
    sweep.set_defaults(run=srt_sweep)

    point = modes.add_parser("point", help="one point from flags")
    point.add_argument("--base-url", required=True)
    point.add_argument("--model", required=True, help="served model name")
    point.add_argument("--backend", required=True, help="benchmark_serving backend")
    point.add_argument("--tokenizer", required=True, help="tokenizer name or path")
    point.add_argument("--isl", required=True, help="input sequence length")
    point.add_argument("--osl", required=True, help="output sequence length")
    point.add_argument("--random-range-ratio", required=True)
    point.add_argument("--conc", required=True, help="max concurrency")
    point.add_argument("--num-prompts", required=True)
    point.add_argument("--result", type=Path, required=True, help="result JSON path")
    point.set_defaults(run=explicit_point)

    args = parser.parse_args(argv)
    return args.run(args)
