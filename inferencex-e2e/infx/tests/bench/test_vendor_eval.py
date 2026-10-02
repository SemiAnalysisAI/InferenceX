"""``infx.bench.eval.vendor``: the real runner with stub interpreters and adapters."""

from __future__ import annotations

import json
import os
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from infx.bench.env import InputError
from infx.bench.eval import FRAMEWORKS, vendor
from infx.bench.eval.context import EvalContext, EvalOutcome
from infx.tests.bench.stubs import executable

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "http://127.0.0.1:8888"

PYTHON_STUB = r'''
import json
import os
import shutil
import sys
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps({"python": sys.argv[0], "argv": argv}) + "\n")
if argv[0] == "-c":
    sys.exit(int(os.environ["STUB_VERSION_RC"]))
if argv[:2] == ["-m", "pip"]:
    returncode = int(os.environ.get("STUB_PIP_RC", "0"))
    if returncode == 0 and "--prefix" in argv:
        uv = Path(argv[argv.index("--prefix") + 1], "bin", "uv")
        uv.parent.mkdir(parents=True)
        shutil.copy(os.environ["STUB_UV"], uv)
    sys.exit(returncode)
if argv[:2] == ["-m", "venv"]:
    python = Path(argv[-1], "bin", "python")
    python.parent.mkdir(parents=True)
    shutil.copy(__file__, python)
    sys.exit(0)
env = dict(os.environ)
if "STUB_SITE" in env:
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env["STUB_SITE"], env.get("PYTHONPATH")]))
os.execve(sys.executable, [sys.executable, *argv], env)
'''

UV_STUB = r'''
import json
import os
import shutil
import sys
from pathlib import Path

argv = sys.argv[1:]
record = {"uv": argv, **{name: os.environ.get(name) for name in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")}}
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")
python = Path(argv[-1], "bin", "python")
python.parent.mkdir(parents=True)
shutil.copy(os.environ["STUB_PYTHON"], python)
'''

ADAPTER_STUB = r'''
import json
import os
import sys
import time
from pathlib import Path

script = Path(sys.argv[0]).name
argv = sys.argv[1:]
if "--integration-error" in argv or argv[:1] == ["failure"]:
    mode = "failure"
elif script.startswith("_") or argv[:1] == ["prepare-source"] or "--install-runtime" in argv:
    mode = "prepare"
else:
    mode = "run"
with open(os.environ["STUB_LOG"], "a") as log:
    record = {"script": script, "mode": mode, "argv": argv, "PYTHONPATH": os.environ.get("PYTHONPATH")}
    log.write(json.dumps(record) + "\n")
plans = json.loads(os.environ["STUB_PLAN"])
if mode == "failure" and f"{script} failure" not in plans:
    real = os.path.join(os.environ["STUB_REAL_EVALS"], script)
    os.execv(sys.executable, [sys.executable, real, *argv])
plan = plans.get(f"{script} {mode}", {})
time.sleep(plan.get("sleep", 0))
for name in plan.get("writes", ()):
    Path(argv[argv.index("--output-dir") + 1], name).write_text(json.dumps({"argv": argv}))
if "--bfcl-project-root" in argv:
    project = Path(argv[argv.index("--bfcl-project-root") + 1])
    for name, content in plan.get("project", {}).items():
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_text(content)
    if "project_symlink" in plan:
        (project / plan["project_symlink"]).symlink_to(os.environ["STUB_LOG"])
sys.exit(plan.get("rc", 0))
'''

# What each real adapter publishes on success.
PUBLISHES = {
    "kimi_vendor_eval.py": ["results_kimi_vendor_2026-01-01.json", "kimi_vendor_report.json"],
    "minimax_provider_eval.py": ["results_minimax_vendor_2026-01-01.json", "minimax_vendor_report.json"],
    "minimax_m3_full_eval.py": [
        "results_minimax_vendor_full_2026-01-01.json",
        "minimax_vendor_report.json",
        "minimax_vendor_results.jsonl",
    ],
    "bfcl_adapter.py": ["results_bfcl.json", "bfcl_report.json"],
}
# The image python3 is too old, and pip cannot install uv.
PROVISIONING_FAILS = {"STUB_VERSION_RC": "1", "STUB_PIP_RC": "7"}


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class Stub:
    def __init__(self, root: Path) -> None:
        self.evals = root / "evals"
        self.evals.mkdir()
        for name in (*PUBLISHES, "_kimi_verifier_archive.py"):
            (self.evals / name).write_text(ADAPTER_STUB)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        self.log = root / "calls.jsonl"
        self.results = root / "results"
        self.results.mkdir()
        python = executable(bin_dir / "python3", f"#!{sys.executable}\n{PYTHON_STUB}")
        uv = executable(root / "uv", f"#!{sys.executable}\n{UV_STUB}")
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "STUB_LOG": str(self.log),
            "STUB_PYTHON": str(python),
            "STUB_UV": str(uv),
            "STUB_REAL_EVALS": str(REPO_ROOT / "infx" / "evals"),
            "STUB_VERSION_RC": "0",
        }

    def context(
        self, suite: str | None = None, *, plan: dict[str, dict[str, Any]] | None = None, **env: str
    ) -> EvalContext:
        plans = {f"{script} run": {"writes": names} for script, names in PUBLISHES.items()}
        return EvalContext(
            base_url=BASE_URL,
            model="served-model",
            concurrency=1,
            context_length=0,
            results_dir=self.results,
            suite=suite,
            env={**self.env, "STUB_PLAN": json.dumps({**plans, **(plan or {})}), **env},
        )

    def calls(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def adapter_calls(self, script: str, mode: str) -> list[dict[str, Any]]:
        return [c for c in self.calls() if c.get("script") == script and c["mode"] == mode]

    def interpreter_calls(self, argv_prefix: list[str]) -> list[dict[str, Any]]:
        n = len(argv_prefix)
        return [c for c in self.calls() if "python" in c and c["argv"][:n] == argv_prefix]

    def result(self, pattern: str) -> dict[str, Any]:
        [path] = self.results.glob(pattern)
        return json.loads(path.read_text())


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stub:
    stub = Stub(tmp_path)
    monkeypatch.setattr(vendor, "EVALS", stub.evals)
    return stub


def test_kimi_runs_the_verified_checkout_with_an_isolated_runtime(stub: Stub) -> None:
    ctx = stub.context(
        MODEL_PREFIX="dsv4", PYTHONPATH="/image/site", OPENAI_API_KEY="sk-caller-key-4711"
    )

    assert FRAMEWORKS["kimi-vendor"](ctx) == EvalOutcome(0, "kimi_tool_call_schema")

    [pip] = stub.interpreter_calls(["-m", "pip"])
    [checkout] = stub.adapter_calls("_kimi_verifier_archive.py", "prepare")
    [run] = stub.adapter_calls("kimi_vendor_eval.py", "run")
    runtime = _flag(pip["argv"], "--target")
    assert run["PYTHONPATH"] == os.pathsep.join([runtime, str(REPO_ROOT), "/image/site"])
    assert _flag(run["argv"], "--verifier-dir") == checkout["argv"][-1]
    assert _flag(run["argv"], "--base-url") == f"{BASE_URL}/v1"
    assert _flag(run["argv"], "--model") == "served-model"
    assert _flag(run["argv"], "--output-dir") == str(stub.results)
    assert _flag(run["argv"], "--task-name") == "kimi_tool_call_schema"
    assert _flag(run["argv"], "--model-prefix") == "dsv4"
    assert "sk-caller-key-4711" not in json.dumps([call["argv"] for call in stub.calls()])
    assert not stub.adapter_calls("kimi_vendor_eval.py", "failure")
    assert not Path(checkout["argv"][-1]).exists()


@pytest.mark.parametrize(
    ("suite", "adapter"),
    [(None, "minimax_provider_eval.py"), ("minimax_m3_full", "minimax_m3_full_eval.py")],
)
def test_minimax_runs_stock_sources_with_pinned_dependencies(
    stub: Stub, suite: str | None, adapter: str
) -> None:
    outcome = FRAMEWORKS["minimax-vendor"](stub.context(suite))

    assert outcome == EvalOutcome(0, suite or "minimax_m3_smoke")
    [source] = stub.adapter_calls("minimax_m3_full_eval.py", "prepare")
    [pip] = stub.interpreter_calls(["-m", "pip"])
    [run] = stub.adapter_calls(adapter, "run")
    assert run["argv"][0] == "run"
    assert _flag(run["argv"], "--source-dir") == _flag(source["argv"], "--source-dir")
    assert _flag(run["argv"], "--dependency-dir") == _flag(pip["argv"], "--target")
    assert _flag(run["argv"], "--python") == "python3"


def test_bfcl_runs_in_a_venv_that_sees_the_image_site_packages(stub: Stub) -> None:
    assert FRAMEWORKS["bfcl"](stub.context()) == EvalOutcome(0, "bfcl_smoke")

    [venv] = stub.interpreter_calls(["-m", "venv"])
    assert "--system-site-packages" in venv["argv"]
    adapter = str(stub.evals / "bfcl_adapter.py")
    interpreters = [call["python"] for call in stub.interpreter_calls([adapter])]
    assert interpreters == [str(Path(venv["argv"][-1], "bin", "python"))] * 2
    [install] = stub.adapter_calls("bfcl_adapter.py", "prepare")
    [run] = stub.adapter_calls("bfcl_adapter.py", "run")
    assert install["argv"][0] == "--install-runtime"
    assert _flag(run["argv"], "--suite") == "bfcl_smoke"
    assert not Path(_flag(run["argv"], "--bfcl-project-root")).exists()
    assert not Path(venv["argv"][-1]).exists()
    assert not (stub.results / vendor.BFCL_ARCHIVE).exists()


def test_too_old_image_python_runs_the_adapter_in_a_pinned_uv_venv(stub: Stub) -> None:
    outcome = FRAMEWORKS["kimi-vendor"](stub.context(STUB_VERSION_RC="1"))

    assert outcome.returncode == 0
    [uv] = [call for call in stub.calls() if "uv" in call]
    venv = Path(uv["uv"][-1])
    assert "--system-site-packages" not in uv["uv"]
    adapter = str(stub.evals / "kimi_vendor_eval.py")
    [run] = stub.interpreter_calls([adapter])
    assert run["python"] == str(venv / "bin" / "python")
    assert Path(uv["UV_CACHE_DIR"]).parent == venv.parent
    assert Path(uv["UV_PYTHON_INSTALL_DIR"]).parent == venv.parent
    assert not venv.parent.exists()


@pytest.mark.parametrize(("framework", "suite", "inputs", "rc", "message"), [
    ("kimi-vendor", "kimi_tool_call_schema", PROVISIONING_FAILS, 7,
     "Kimi Vendor Verifier Python runtime preparation failed with exit code 7"),
    ("minimax-vendor", "minimax_m3_smoke", PROVISIONING_FAILS, 7,
     "MiniMax Provider Verifier Python runtime preparation failed with exit code 7"),
    ("minimax-vendor", "minimax_m3_full", PROVISIONING_FAILS, 7,
     "MiniMax M3 full Python runtime preparation failed with exit code 7"),
    ("bfcl", "bfcl_vllm_minimax_m3", PROVISIONING_FAILS, 7,
     "BFCL Python runtime preparation failed with exit code 7"),
    ("kimi-vendor", "kimi_tool_call_schema_full", {"STUB_PIP_RC": "12"}, 12,
     "Kimi Vendor Verifier dependency installation failed with exit code 12"),
    ("bfcl", "bfcl_vllm_kimi", {"plan": {"bfcl_adapter.py prepare": {"rc": 6}}}, 6,
     "BFCL dependency installation failed with exit code 6"),
])  # fmt: skip
def test_setup_failure_is_recorded_by_the_real_adapter_on_python310_and_skips_the_run(
    stub: Stub,
    tmp_path: Path,
    framework: str,
    suite: str,
    inputs: dict[str, Any],
    rc: int,
    message: str,
) -> None:
    # Python 3.10 has no datetime.UTC; the real adapters must still write the failure.
    python310 = tmp_path / "python310"
    python310.mkdir()
    (python310 / "sitecustomize.py").write_text(
        "import datetime\nvars(datetime).pop('UTC', None)\n"
    )

    outcome = FRAMEWORKS[framework](stub.context(suite, STUB_SITE=str(python310), **inputs))

    assert outcome == EvalOutcome(rc, suite)
    failure = stub.result("results*.json")
    assert failure["integration_error"]["message"] == message
    assert failure["results"] and all(task.startswith(suite) for task in failure["results"])
    assert not [call for call in stub.calls() if call.get("mode") == "run"]
    assert not (stub.results / vendor.BFCL_ARCHIVE).exists()


@pytest.mark.parametrize(("framework", "adapter", "writes", "message"), [
    ("kimi-vendor", "kimi_vendor_eval.py", [], "Kimi Vendor Verifier evaluation failed with exit code 3"),
    ("minimax-vendor", "minimax_provider_eval.py", [],
     "MiniMax Provider Verifier evaluation failed with exit code 3"),
    # Half a BFCL publication is not a published result.
    ("bfcl", "bfcl_adapter.py", ["results_bfcl.json"], "BFCL evaluation failed with exit code 3"),
    # A complete publication is the adapter's own verdict.
    ("kimi-vendor", "kimi_vendor_eval.py", PUBLISHES["kimi_vendor_eval.py"], None),
])  # fmt: skip
def test_a_failed_run_is_recorded_as_an_integration_error_unless_the_adapter_published_it(
    stub: Stub, framework: str, adapter: str, writes: list[str], message: str | None
) -> None:
    plan = {f"{adapter} run": {"rc": 3, "writes": writes}}

    outcome = FRAMEWORKS[framework](stub.context(plan=plan))

    assert outcome.returncode == 3
    assert stub.result("results*.json").get("integration_error", {}).get("message") == message


def test_unwritable_failure_result_is_reported_and_keeps_the_setup_code(
    stub: Stub, capfd: pytest.CaptureFixture[str]
) -> None:
    ctx = stub.context(plan={"kimi_vendor_eval.py failure": {"rc": 5}}, STUB_PIP_RC="12")

    assert FRAMEWORKS["kimi-vendor"](ctx).returncode == 12

    assert "failed to write the Kimi Vendor Verifier failure artifact (exit code 5)" in (
        capfd.readouterr().err
    )
    assert list(stub.results.iterdir()) == []


def test_suite_deadline_kills_the_adapter_and_records_exit_124(stub: Stub) -> None:
    bfcl = vendor.PROVIDERS["bfcl"]
    provider = replace(bfcl, suites={"bfcl_smoke": replace(bfcl.suites["bfcl_smoke"], timeout_s=1)})
    ctx = stub.context(plan={"bfcl_adapter.py run": {"sleep": 60}})
    started = time.monotonic()

    assert vendor.run(provider, ctx) == EvalOutcome(124, "bfcl_smoke")

    assert time.monotonic() - started < 30
    assert stub.result("results_bfcl.json")["integration_error"]["message"] == (
        "BFCL evaluation failed with exit code 124"
    )


def test_bfcl_full_suite_archives_the_upstream_tree_after_a_failed_run(stub: Stub) -> None:
    project = {"result/run/generation.json": "{}\n", "score/run/score.json": "{}\n"}
    run = {"rc": 2, "writes": PUBLISHES["bfcl_adapter.py"], "project": project}

    outcome = FRAMEWORKS["bfcl"](stub.context("bfcl_vllm_kimi", plan={"bfcl_adapter.py run": run}))

    assert outcome == EvalOutcome(2, "bfcl_vllm_kimi")
    with tarfile.open(stub.results / vendor.BFCL_ARCHIVE) as archive:
        assert archive.getnames() == [
            "result",
            "result/run",
            "result/run/generation.json",
            "score",
            "score/run",
            "score/run/score.json",
        ]
    assert not stub.adapter_calls("bfcl_adapter.py", "failure")


def test_bfcl_archive_failure_fails_a_passing_run_and_keeps_its_scores(
    stub: Stub, capfd: pytest.CaptureFixture[str]
) -> None:
    run = {"writes": PUBLISHES["bfcl_adapter.py"], "project_symlink": "escape.json"}

    outcome = FRAMEWORKS["bfcl"](
        stub.context("bfcl_vllm_minimax_m3", plan={"bfcl_adapter.py run": run})
    )

    assert outcome == EvalOutcome(1, "bfcl_vllm_minimax_m3")
    assert "refusing to archive symbolic link: escape.json" in capfd.readouterr().err
    # Neither the archive nor its temporary file is left behind.
    assert sorted(path.name for path in stub.results.iterdir()) == sorted(
        PUBLISHES["bfcl_adapter.py"]
    )


def test_unknown_suite_is_rejected_before_any_work(stub: Stub) -> None:
    with pytest.raises(InputError, match="unsupported BFCL suite 'minimax_m3_smoke'"):
        FRAMEWORKS["bfcl"](stub.context("minimax_m3_smoke"))

    assert stub.calls() == []
    assert list(stub.results.iterdir()) == []


def test_archive_tree_is_byte_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "project"
    for name, content in (
        ("result/run/BFCL_v4_simple_python_result.json", '{"id":"simple_python_0"}\n'),
        ("score/run/BFCL_v4_simple_python_score.json", '{"accuracy":1.0}\n'),
        ("test_case_ids_to_generate.json", '{"simple_python":["simple_python_0"]}\n'),
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content)
    vendor.archive_tree(root, tmp_path / "first.tar.gz")
    for path in root.rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
    vendor.archive_tree(root, tmp_path / "second.tar.gz")

    first = (tmp_path / "first.tar.gz").read_bytes()
    assert first == (tmp_path / "second.tar.gz").read_bytes()
    with tarfile.open(tmp_path / "first.tar.gz") as archive:
        assert archive.getnames() == [
            "result",
            "result/run",
            "result/run/BFCL_v4_simple_python_result.json",
            "score",
            "score/run",
            "score/run/BFCL_v4_simple_python_score.json",
            "test_case_ids_to_generate.json",
        ]
        member = archive.getmember("score/run/BFCL_v4_simple_python_score.json")
        assert (member.uid, member.gid, member.uname, member.gname, member.mtime) == (
            0, 0, "", "", 0,
        )  # fmt: skip
