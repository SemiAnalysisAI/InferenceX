"""The ``eval`` command and its srt-slurm shims against a fake OpenAI server, uv, and lm-eval."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.bench import uv
from infx.bench.env import BenchError, InputError
from infx.bench.eval import FRAMEWORKS, evaluate, lm_eval
from infx.bench.eval.context import EvalOutcome
from infx.evals.validate_scores import validate_batch_manifest
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]

STUBS = {
    "lm_eval/__init__.py": "",
    "lm_eval/models/__init__.py": "",
    "lm_eval/models/openai_completions.py": "class LocalChatCompletion:\n    pass\n",
    "lm_eval/models/api_models.py": "class JsonChatStr(str):\n    pass\n\n\nclass TemplateAPI:\n    pass\n",
    "lm_eval/__main__.py": """
import json, os, sys, urllib.request
from pathlib import Path
from lm_eval.models.api_models import TemplateAPI

args = sys.argv[1:]
model_args = dict(item.split("=", 1) for item in args[args.index("--model_args") + 1].split(","))
with open(os.environ["STUB_LM_EVAL_TRACE"], "a") as trace:
    trace.write(json.dumps({
        "argv": args,
        "tasks_found": Path(args[args.index("--tasks") + 1]).is_file(),
        "patched": getattr(TemplateAPI, "apply_chat_template", None) is not None,
    }) + "\\n")
if model_args["num_concurrent"] == os.environ.get("STUB_LM_EVAL_FAIL_AT"):
    sys.exit(3)
body = json.dumps({"model": model_args["model"], "messages": [{"role": "user", "content": "6*7?"}]})
request = urllib.request.Request(
    model_args["base_url"], data=body.encode(), headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request, timeout=10) as response:
    answer = json.load(response)["choices"][0]["message"]["content"]
output = Path(args[args.index("--output_path") + 1]) / model_args["model"].replace("/", "__")
output.mkdir(parents=True)
(output / "results_stub.json").write_text(json.dumps({"answer": answer}))
(output / "samples_gsm8k_stub.jsonl").write_text(json.dumps({"answer": answer}) + "\\n")
(output / "lm_eval.log").write_text("not an artifact")
""",
}

# The container's python3; its /logs mount lives under the test's tmp directory.
PYTHON3 = f"""#!/bin/sh
for arg; do
    shift
    case $arg in /logs/*) arg="$STUB_LOGS${{arg#/logs}}" ;; esac
    set -- "$@" "$arg"
done
exec "{sys.executable}" "$@"
"""

# The container's uv: records each call and fails those naming STUB_UV_FAIL.
UV = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_UV_TRACE"], "a") as trace:
    trace.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(1 if os.environ.get("STUB_UV_FAIL", "\\0") in " ".join(sys.argv[1:]) else 0)
"""

MULTI_NODE = {
    "IS_MULTINODE": "true", "EVAL_MAX_MODEL_LEN": "4096",
    "PREFILL_TP": "4", "PREFILL_EP": "1", "PREFILL_DP_ATTN": "false",
    "DECODE_TP": "8", "DECODE_EP": "1", "DECODE_DP_ATTN": "false", "DECODE_NUM_WORKERS": "2",
}  # fmt: skip


@pytest.fixture(autouse=True)
def recording_uv(tmp_path, monkeypatch):
    """Put the recording uv first on PATH, so no test installs into this interpreter."""
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir()
    executable(bin_dir / "uv", UV)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def openai(http_server):
    """Serves one model: listed under /v1/models, answering chat completions by POST only."""
    requests = []

    def respond(method, path):
        requests.append((method, path))
        if method == "POST":
            return 200, {"choices": [{"index": 0, "message": {"content": "42"}}]}
        if path == "/v1/models":
            return 200, {"data": [{"id": "served-model"}]}
        return (405 if path == "/v1/chat/completions" else 404), {}

    return SimpleNamespace(url=http_server(respond), requests=requests)


@pytest.fixture
def base_env(tmp_path):
    """What every eval receives: the stubs first on PYTHONPATH, and no network model lookups."""
    stubs = tmp_path / "stubs"
    for name, body in STUBS.items():
        (stubs / name).parent.mkdir(parents=True, exist_ok=True)
        (stubs / name).write_text(body)
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "PYTHONPATH": str(stubs),
        "HF_HUB_OFFLINE": "1",
        "STUB_UV_TRACE": str(tmp_path / "uv.jsonl"),
        "STUB_LM_EVAL_TRACE": str(tmp_path / "lm_eval.jsonl"),
        "OPENAI_API_KEY": "EMPTY",
        "EVAL_ONLY": "false",
        "MODEL": "org/test-model",
        "MODEL_NAME": "served-model",
    }


@pytest.fixture
def checkpoint(tmp_path):
    """A checkpoint transformers cannot load, whose config.json states an 8192-token context."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "not_registered", "max_position_embeddings": 8192, "seq_length": 4096}'
    )
    return model


@pytest.fixture
def checkout(tmp_path):
    """A checkout whose benchmarks/ are copies of the real shims, so staging stays in tmp."""
    root = tmp_path / "checkout"
    for script in ("check_env.sh", "single_node/srt_eval.sh", "multi_node/srt_eval.sh"):
        (root / "benchmarks" / script).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / "benchmarks" / script, root / "benchmarks" / script)
    (root / "infx").symlink_to(REPO_ROOT / "infx")
    (tmp_path / "bin").mkdir()
    executable(tmp_path / "bin/python3", PYTHON3)
    return root


@pytest.fixture
def single_node_env(base_env, checkpoint):
    return {
        **base_env, "MODEL_PATH": str(checkpoint), "MAX_MODEL_LEN": "10240", "CONC": "8",
        "TP": "8", "EP_SIZE": "8", "DP_ATTENTION": "true", "PP_SIZE": "1", "DCP_SIZE": "1",
        "PCP_SIZE": "1", "IS_MULTINODE": "false", "IS_AGENTIC": "0", "FRAMEWORK": "sglang",
        "PRECISION": "fp8", "SPEC_DECODING": "none", "MODEL_PREFIX": "dsr1", "RUNNER_TYPE": "h200",
        "RECIPE_FINGERPRINT": "recipe-1", "ISL": "1024", "OSL": "1024",
    }  # fmt: skip


def shim(checkout: Path, node: str, env: dict, *args: str) -> subprocess.CompletedProcess:
    """Run ``benchmarks/<node>_node/srt_eval.sh`` with the container python3 first on PATH."""
    tmp = checkout.parent
    return subprocess.run(
        ["bash", str(checkout / f"benchmarks/{node}_node/srt_eval.sh"), *args],
        env={**env, "PATH": f"{tmp / 'bin'}{os.pathsep}{env['PATH']}", "STUB_LOGS": str(tmp / "logs")},
        cwd=tmp,
        capture_output=True,
        text=True,
        check=False,
    )


def trace(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def model_args(run: dict) -> list[str]:
    """The ``--model_args`` items of one recorded lm-eval invocation."""
    return run["argv"][run["argv"].index("--model_args") + 1].split(",")


def test_single_node_shim_stages_the_eval_in_the_checkout_and_records_its_status(
    checkout, single_node_env, openai, tmp_path
):
    status = tmp_path / "infx-eval-exit-code"

    result = shim(checkout, "single", single_node_env, openai.url, str(status))

    assert result.returncode == 0, result.stdout + result.stderr
    assert status.read_text() == "0\n"
    staged = sorted(path.name for path in checkout.iterdir() if path.is_file())
    assert staged == ["meta_env.json", "results_stub.json", "samples_gsm8k_stub.jsonl"]
    assert json.loads((checkout / "results_stub.json").read_text()) == {"answer": "42"}
    assert openai.requests == [("POST", "/v1/chat/completions")]
    assert [uv_step(argv) for argv in trace(tmp_path / "uv.jsonl")][:1] == ["install lm-eval[api]"]
    [run] = trace(tmp_path / "lm_eval.jsonl")
    assert run["tasks_found"] and run["patched"]
    # MAX_MODEL_LEN 10240 is capped at the checkpoint's 8192; 4096 of it stays for the prompt.
    assert {"model=served-model", "num_concurrent=8", "max_length=8192"} <= set(model_args(run))
    assert run["argv"][run["argv"].index("--gen_kwargs") + 1].startswith("max_tokens=4096,")
    assert json.loads((checkout / "meta_env.json").read_text()) == {
        "is_multinode": False, "framework": "sglang", "precision": "fp8",
        "spec_decoding": "none", "eval_suite": "gsm8k", "recipe_fingerprint": "recipe-1",
        "tp": 8, "pp": 1, "dcp_size": 1, "pcp_size": 1, "conc": 8, "ep": 8, "dp_attention": True,
        "prefill_tp": 8, "prefill_pp": 1, "prefill_dcp_size": 1, "prefill_pcp_size": 1,
        "prefill_ep": 8, "prefill_dp_attention": True, "prefill_num_workers": 1,
        "decode_tp": 8, "decode_pp": 1, "decode_dcp_size": 1, "decode_pcp_size": 1,
        "decode_ep": 8, "decode_dp_attention": True, "decode_num_workers": 1,
        "model": "served-model", "infmax_model_prefix": "dsr1", "hw": "h200",
        "isl": "1024", "osl": "1024",
    }  # fmt: skip


def test_multi_node_shim_stages_a_batched_eval_in_the_logs_mount(
    checkout, base_env, openai, tmp_path
):
    env = {**base_env, **MULTI_NODE, "EVAL_CONC": "1 2"}

    result = shim(checkout, "multi", env, openai.url, str(checkout))

    assert result.returncode == 0, result.stdout + result.stderr
    staged = tmp_path / "logs/eval_results"
    assert sorted(path.name for path in staged.iterdir()) == [
        "meta_env.json",
        "results_stub_conc1.json",
        "results_stub_conc2.json",
        "samples_gsm8k_stub_conc1.jsonl",
        "samples_gsm8k_stub_conc2.jsonl",
    ]
    meta = json.loads((staged / "meta_env.json").read_text())
    assert {key: meta[key] for key in ("is_multinode", "conc", "eval_concs", "completed_eval_concs", "failed_eval_concs")} == {
        "is_multinode": True, "conc": 1, "eval_concs": [1, 2], "completed_eval_concs": [1, 2], "failed_eval_concs": [],
    }  # fmt: skip


@pytest.mark.parametrize(("node", "inputs", "missing"), [
    ("single", {"MAX_MODEL_LEN": ""}, ["MAX_MODEL_LEN"]),
    ("multi", {"IS_MULTINODE": "true", "PREFILL_TP": "4", "PREFILL_EP": "1"}, ["EVAL_CONC", "PREFILL_DP_ATTN", "DECODE_DP_ATTN"]),
])  # fmt: skip
def test_shims_name_every_missing_input(checkout, single_node_env, tmp_path, node, inputs, missing):
    status = tmp_path / "infx-eval-exit-code"
    target = status if node == "single" else checkout

    result = shim(checkout, node, {**single_node_env, **inputs}, "http://127.0.0.1:9", str(target))

    assert result.returncode == 1
    assert [line[4:] for line in result.stdout.splitlines() if line.startswith("  - ")] == missing
    if node == "single":
        assert status.read_text() == "1\n"
    assert trace(tmp_path / "lm_eval.jsonl") == []


def test_batched_lm_eval_defers_a_failed_concurrency_to_score_validation(
    base_env, openai, tmp_path
):
    staged = tmp_path / "eval_results"
    staged.mkdir()
    (staged / "results_stub_conc8.json").write_text("{}")  # left behind by an earlier eval
    env = {**base_env, **MULTI_NODE, "STUB_LM_EVAL_FAIL_AT": "4"}

    assert evaluate(openai.url, "1 4 8", staged, environ=env) == 0

    runs = trace(tmp_path / "lm_eval.jsonl")
    assert [[arg for arg in model_args(run) if arg.startswith("num_concurrent=")] for run in runs] == [
        ["num_concurrent=1"], ["num_concurrent=4"], ["num_concurrent=8"],
    ]  # fmt: skip
    assert sorted(path.name for path in staged.iterdir()) == [
        "meta_env.json",
        "results_stub_conc1.json",
        "results_stub_conc8.json",
        "results_stub_conc8_2.json",
        "samples_gsm8k_stub_conc1.jsonl",
        "samples_gsm8k_stub_conc8.jsonl",
    ]
    meta = json.loads((staged / "meta_env.json").read_text())
    assert {key: meta[key] for key in ("conc", "eval_concs", "completed_eval_concs", "failed_eval_concs")} == {
        "conc": 1, "eval_concs": [1, 4, 8], "completed_eval_concs": [1, 8], "failed_eval_concs": [4],
    }  # fmt: skip
    errors = validate_batch_manifest(
        str(staged / "meta_env.json"), [str(path) for path in staged.glob("results*.json")]
    )
    assert "batched eval failed for concurrency: 4" in errors


def recorder(ran: list, *, rc: int = 0, artifacts: tuple[str, ...] = ("results_x.json",)):
    """A framework that records the context it ran with and writes ``artifacts``."""

    def run(ctx):
        ran.append(ctx)
        for name in artifacts:
            (ctx.results_dir / name).write_text("{}")
        return EvalOutcome(returncode=rc, suite=ctx.suite or "default_suite")

    return run


@pytest.mark.parametrize(("env_framework", "cli_framework", "expected"), [
    (None, None, "lm-eval"),
    (None, "kimi-vendor", "kimi-vendor"),
    ("bfcl", "kimi-vendor", "bfcl"),
])  # fmt: skip
def test_the_environment_framework_overrides_the_command_line(
    monkeypatch, base_env, tmp_path, env_framework, cli_framework, expected
):
    ran = {name: [] for name in FRAMEWORKS}
    for name in FRAMEWORKS:
        monkeypatch.setitem(FRAMEWORKS, name, recorder(ran[name]))
    env = {**base_env, "IS_MULTINODE": "false", "EVAL_MAX_MODEL_LEN": "4096"}
    if env_framework:
        env["EVAL_FRAMEWORK"] = env_framework

    assert evaluate("http://127.0.0.1:9", "2", tmp_path / "out", cli_framework, environ=env) == 0
    assert [name for name, runs in ran.items() if runs] == [expected]


@pytest.mark.parametrize(("endpoint", "concurrency", "inputs", "message"), [
    ("http://127.0.0.1:9", "2", {"EVAL_FRAMEWORK": "kimi-vendor", "EVAL_SUITE": 'kimi"suite'}, "EVAL_SUITE may contain only"),
    ("http://127.0.0.1:9", "2", {"EVAL_SUITE": "gpqa_diamond"}, "EVAL_SUITE is only supported with"),
    ("http://127.0.0.1:9", "2", {"EVAL_FRAMEWORK": "no-such-eval"}, "unknown eval framework 'no-such-eval'"),
    ("http://127.0.0.1:9", "1 4", {"EVAL_FRAMEWORK": "kimi-vendor"}, "batched eval concurrency is only supported for lm-eval"),
    ("http://127.0.0.1:9", "4 0", {}, "--concurrency must be a positive integer"),
    ("http://127.0.0.1:9", " ", {}, "--concurrency must name at least one concurrency"),
    ("127.0.0.1:9", "2", {}, "--endpoint must be an http"),
    ("http://127.0.0.1:99999", "2", {}, "--endpoint must be an http"),
    ("http://127.0.0.1:9", "2", {"TP": "8x"}, "TP must be an integer"),
])  # fmt: skip
def test_invalid_requests_fail_before_any_eval(
    monkeypatch, base_env, tmp_path, endpoint, concurrency, inputs, message
):
    ran = []
    for name in FRAMEWORKS:
        monkeypatch.setitem(FRAMEWORKS, name, recorder(ran))
    env = {**base_env, "IS_MULTINODE": "false", **inputs}

    with pytest.raises(InputError, match=message):
        evaluate(endpoint, concurrency, tmp_path / "out", environ=env)
    assert ran == []
    assert not (tmp_path / "out").exists()


def test_a_failed_vendor_eval_is_staged_and_returns_its_exit_code(monkeypatch, base_env, tmp_path):
    ran = []
    artifacts = ("results_bfcl.json", "bfcl_report.json", "adapter.log")
    monkeypatch.setitem(FRAMEWORKS, "bfcl", recorder(ran, rc=7, artifacts=artifacts))
    env = {**base_env, "IS_MULTINODE": "false", "EVAL_FRAMEWORK": "bfcl"}

    assert evaluate("http://127.0.0.1:9/", "7", tmp_path / "out", environ=env) == 7

    [ctx] = ran
    assert (ctx.base_url, ctx.model, ctx.concurrency, ctx.suite) == (
        "http://127.0.0.1:9", "served-model", 7, None,
    )
    assert not ctx.results_dir.exists()
    staged = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert staged == ["bfcl_report.json", "meta_env.json", "results_bfcl.json"]
    meta = json.loads((tmp_path / "out/meta_env.json").read_text())
    assert (meta["eval_suite"], meta["conc"]) == ("default_suite", 7)


@pytest.mark.parametrize(("framework", "polled"), [("kimi-vendor", True), ("lm-eval", False)])
def test_eval_only_vendor_evals_wait_for_the_served_model(
    monkeypatch, base_env, openai, tmp_path, framework, polled
):
    seen = []

    def run(ctx):
        seen.extend(openai.requests)
        return EvalOutcome(returncode=0, suite="suite")

    monkeypatch.setitem(FRAMEWORKS, framework, run)
    env = {
        **base_env, "IS_MULTINODE": "false", "EVAL_ONLY": "true", "EVAL_FRAMEWORK": framework,
        "EVAL_MAX_MODEL_LEN": "4096", "EVAL_ENDPOINT_READY_TIMEOUT_SECONDS": "2",
        "EVAL_MODEL_STABILIZATION_SECONDS": "600",
    }  # fmt: skip

    assert evaluate(openai.url, "1", tmp_path / "out", environ=env) == 0
    # The server lists only MODEL_NAME, so readiness passes only for the name eval requests use.
    expected = [("GET", "/v1/models"), ("GET", "/v1/chat/completions")]
    assert seen == (expected if polled else [])


@pytest.mark.parametrize(("inputs", "context", "max_tokens"), [
    ({"MAX_MODEL_LEN": "2048"}, 2048, 1024),
    ({"MAX_MODEL_LEN": "10240"}, 8192, 4096),
    ({"MAX_MODEL_LEN": "0"}, 8192, 4096),
    ({"MAX_MODEL_LEN": "0", "MODEL_PATH": "/no/such/model"}, 16384, 12288),
    ({"MAX_MODEL_LEN": "2048", "EVAL_MAX_MODEL_LEN": "4096"}, 4096, 2048),
    ({"EVAL_MAX_MODEL_LEN": "4097"}, 4097, 1),
    ({"EVAL_MAX_MODEL_LEN": "30000"}, 30000, 16384),
])  # fmt: skip
def test_request_budget_is_the_benchmark_context_capped_at_the_checkpoint(
    checkpoint, inputs, context, max_tokens
):
    environ = {"MODEL": "/no/such/model", "MODEL_PATH": str(checkpoint), **inputs}

    assert lm_eval.context_length(environ) == context
    assert lm_eval.max_output_tokens(context) == max_tokens


def uv_step(argv: list[str]) -> str:
    """Name one recorded ``uv`` call of the lm-eval install into this interpreter."""
    assert argv[0] == "pip" and argv[-2:] == ["--python", sys.executable]
    assert "--break-system-packages" in argv
    spec = argv[-3]
    if argv[1] == "install" and "--no-deps" not in argv:
        assert argv[-4] == "lm-eval[api]"
        assert spec == "huggingface-hub>=1.5,<2"
        return "install lm-eval[api]"
    if argv[1] == "uninstall":
        return f"uninstall {spec}"
    if {"--no-deps", "--reinstall"} <= set(argv):
        return "git pin" if spec.startswith("git+") else "archive pin"
    return f"install {spec}"


@pytest.mark.parametrize(("image", "git", "failing", "expected"), [
    ("rocm/atom:latest", True, "git+", ["uninstall torchvision", "install lm-eval[api]", "git pin", "archive pin"]),
    ("lmsysorg/sglang:latest", True, None, ["install lm-eval[api]", "git pin"]),
    ("lmsysorg/sglang:latest", False, None, ["install lm-eval[api]", "archive pin"]),
])  # fmt: skip
def test_lm_eval_install_drops_torchvision_on_atom_and_falls_back_to_the_archive(
    base_env, tmp_path, image, git, failing, expected
):
    bin_dir = tmp_path / "path"
    bin_dir.mkdir()
    if git:
        executable(bin_dir / "git", "#!/bin/sh\n")
    env = {**base_env, "PATH": str(bin_dir), "IMAGE": image}
    if failing:
        env["STUB_UV_FAIL"] = failing

    lm_eval.install(env)

    assert [uv_step(argv) for argv in trace(tmp_path / "uv.jsonl")] == expected


def test_lm_eval_install_only_warns_when_uv_is_unavailable(base_env, monkeypatch, capfd):
    def unavailable() -> str:
        raise BenchError("uv installation did not create /scratch/uv")

    monkeypatch.setattr(uv, "find", unavailable)

    lm_eval.install(base_env)

    assert "WARN: uv installation did not create /scratch/uv" in capfd.readouterr().err
