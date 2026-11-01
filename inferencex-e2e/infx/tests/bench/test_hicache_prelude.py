"""The opt-in qualification must fail closed without changing canonical evaluation."""

import json
from pathlib import Path

import pytest
import yaml

from infx.bench import eval as evaluator
from infx.bench.env import InputError
from infx.bench.eval import hicache_prelude as prelude
from infx.bench.eval import host_restore_probe, stage
from infx.launch.context import LaunchError
from infx.launch.drivers.srt import submit


@pytest.fixture
def probe_env():
    return {
        prelude.FLAG: "1",
        prelude.PINS[0]: "registry/image:tag@sha256:" + "a" * 64,
        prelude.PINS[1]: "b" * 40,
        prelude.PINS[2]: "c" * 40,
        "EVAL_ONLY": "true",
        "MODEL": "org/checkpoint",
        "MODEL_NAME": "served-model",
        "MODEL_PATH": "/model",
    }


@pytest.fixture
def mounted_logs(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "fingerprint_agg_w0.json").write_text("{}")
    (logs / "node_agg_w0.out").write_text("real boot log fixture")
    monkeypatch.setattr(prelude, "LOG_DIR", logs)
    return logs


def test_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(host_restore_probe, "main", lambda _: pytest.fail("probe ran"))
    monkeypatch.setattr(prelude, "flush_cache", lambda _: pytest.fail("flush ran"))
    assert prelude.run("http://unused", tmp_path, {}) == 0
    assert list(tmp_path.iterdir()) == []
    assert prelude.recipe_environment({}, eval_only=False) == {}
    assert prelude.recipe_environment({prelude.FLAG: "0"}, eval_only=False) == {}


@pytest.mark.parametrize("value", ["yes", "TRUE", "", "2", " false "])
def test_invalid_flags_rejected(value):
    with pytest.raises(InputError, match="must be"):
        prelude.enabled({prelude.FLAG: value})


@pytest.mark.parametrize("flag", [None, "0", "false"])
def test_orphaned_pins_rejected(flag, probe_env):
    if flag is None:
        probe_env.pop(prelude.FLAG)
    else:
        probe_env[prelude.FLAG] = flag
    with pytest.raises(InputError):
        prelude.recipe_environment(probe_env, eval_only=True)


def test_enabled_performance_rejected(probe_env):
    with pytest.raises(InputError, match="EVAL_ONLY"):
        prelude.recipe_environment(probe_env, eval_only=False)


@pytest.mark.parametrize("pin", prelude.PINS)
def test_bad_or_missing_pins_rejected(pin, probe_env):
    probe_env[pin] = "main"
    with pytest.raises(InputError):
        prelude.recipe_environment(probe_env, eval_only=True)
    del probe_env[pin]
    with pytest.raises(InputError):
        prelude.recipe_environment(probe_env, eval_only=True)


def test_bridge_and_post_eval_allowlist(tmp_path, probe_env):
    recipe = tmp_path / "bound.yaml"
    recipe.write_text(yaml.safe_dump({"benchmark": {"env": {
        **probe_env, "EVAL_UNREVIEWED": "must-not-forward", "SECRET": "not-forwarded",
    }}}))
    image = probe_env[prelude.PINS[0]]
    forwarded = submit.eval_prelude_environment(recipe, eval_only=True, image=image)
    assert set(forwarded) == {prelude.FLAG, *prelude.PINS}
    runtime_env = {"EVAL_ONLY": "true", **forwarded}
    args = submit.eval_args(runtime_env, submit.SINGLE_NODE_EVAL_COMMAND)
    names = json.loads(args[-1].split("=", 1)[1])
    assert set(forwarded) <= set(names)
    assert "EVAL_UNREVIEWED" not in names
    assert args[1] == f"post_eval.command={submit.SINGLE_NODE_EVAL_COMMAND}"
    with pytest.raises(LaunchError, match="EVAL_ONLY"):
        submit.eval_prelude_environment(recipe, eval_only=False, image=image)
    with pytest.raises(LaunchError, match="actual launch image"):
        submit.eval_prelude_environment(recipe, eval_only=True, image="wrong-image")


def test_missing_logs_fail_closed(tmp_path, monkeypatch, probe_env):
    monkeypatch.setattr(prelude, "LOG_DIR", tmp_path / "missing")
    monkeypatch.setattr(host_restore_probe, "main", lambda _: pytest.fail("probe ran"))
    assert prelude.run("http://server", tmp_path, probe_env) == 1
    report = json.loads((tmp_path / prelude.REPORT_NAME).read_text())
    assert report["status"] == "PRELUDE_FAILED"


@pytest.mark.parametrize(
    ("probe_code", "status"),
    [(1, "FAIL"), (2, "INCONCLUSIVE"), (0, "PLAN_ONLY"), (1, "DIAGNOSTIC_PASS")],
)
def test_probe_nonpass_stops_before_flush(
    tmp_path, monkeypatch, mounted_logs, probe_env, probe_code, status
):
    def probe(argv):
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"status": status}))
        return probe_code

    monkeypatch.setattr(host_restore_probe, "main", probe)
    monkeypatch.setattr(prelude, "flush_cache", lambda _: pytest.fail("flush ran"))
    assert prelude.run("http://server", tmp_path, probe_env) != 0


@pytest.mark.parametrize("body", [None, "{", "[]"])
def test_missing_or_malformed_report_stops(tmp_path, monkeypatch, mounted_logs, probe_env, body):
    def probe(argv):
        if body is not None:
            Path(argv[argv.index("--output") + 1]).write_text(body)
        return 0

    monkeypatch.setattr(host_restore_probe, "main", probe)
    monkeypatch.setattr(prelude, "flush_cache", lambda _: pytest.fail("flush ran"))
    assert prelude.run("http://server", tmp_path, probe_env) == 1


def test_probe_failure_never_starts_canonical(tmp_path, monkeypatch, mounted_logs, probe_env):
    def probe(argv):
        Path(argv[argv.index("--output") + 1]).write_text(
            json.dumps({"status": "DIAGNOSTIC_FAIL"})
        )
        return 1

    monkeypatch.setattr(host_restore_probe, "main", probe)
    monkeypatch.setattr(evaluator.meta, "build", lambda *a, **kw: {})
    monkeypatch.setattr(evaluator.lm_eval, "prepare", lambda _: pytest.fail("canonical prepared"))
    monkeypatch.setattr(prelude, "flush_cache", lambda _: pytest.fail("flush ran"))
    monkeypatch.setattr(evaluator, "_single", lambda *a: pytest.fail("canonical ran"))
    assert evaluator.evaluate("http://server", "64", tmp_path, environ=probe_env) == 1
    assert (mounted_logs / "eval_results" / prelude.REPORT_NAME).is_file()


@pytest.mark.parametrize("flush_fails", [False, True])
def test_pass_then_flush_then_unchanged_canonical(
    tmp_path, monkeypatch, mounted_logs, probe_env, flush_fails
):
    events = []
    original = dict(probe_env)

    def probe(argv):
        events.append("probe")
        assert "--execute" in argv
        assert argv[argv.index("--model") + 1] == "served-model"
        assert argv[argv.index("--expected-model") + 1] == "/model"
        assert argv[argv.index("--max-seconds") + 1] == "1800"
        assert "tp_rank=0" in argv and "pp_rank=0" in argv
        assert not any(x.startswith(("model_name=", "dp_rank=")) for x in argv)
        assert argv[argv.index("--server-fingerprint") + 1] == str(
            mounted_logs / "fingerprint_agg_w0.json"
        )
        Path(argv[argv.index("--output") + 1]).write_text(
            json.dumps({"status": "DIAGNOSTIC_PASS", "evidence": "retained"})
        )
        return 0

    def flush(endpoint):
        events.append("flush")
        assert endpoint == "http://server"
        if flush_fails:
            raise RuntimeError("flush refused")

    def canonical(framework, context, conc, destination, environ):
        events.append("canonical")
        assert framework is evaluator.FRAMEWORKS["lm-eval"]
        assert conc == 64 and destination == tmp_path
        assert environ == original
        ctx = context(concurrency=conc, results_dir=tmp_path)
        assert ctx.model == "served-model" and ctx.base_url == "http://server"
        assert ctx.context_length == 1048576 and ctx.env == original
        return 0

    monkeypatch.setattr(host_restore_probe, "main", probe)
    monkeypatch.setattr(prelude, "flush_cache", flush)
    monkeypatch.setattr(evaluator.meta, "build", lambda *a, **kw: {})
    monkeypatch.setattr(evaluator.lm_eval, "prepare", lambda environ: 1048576)
    monkeypatch.setattr(evaluator, "_single", canonical)
    code = evaluator.evaluate("http://server/", "64", tmp_path, environ=probe_env)
    assert code == (1 if flush_fails else 0)
    assert events == (["probe", "flush"] if flush_fails else ["probe", "flush", "canonical"])
    assert probe_env == original
    report = json.loads((tmp_path / prelude.REPORT_NAME).read_text())
    assert report["evidence"] == "retained"
    assert (mounted_logs / "eval_results" / prelude.REPORT_NAME).read_bytes() == (
        tmp_path / prelude.REPORT_NAME
    ).read_bytes()
    assert report["canonical_cache_flush"]["passed"] is not flush_fails
    assert any(Path(prelude.REPORT_NAME).match(pattern) for pattern in stage.ARTIFACTS)


@pytest.mark.parametrize("status,body", [(200, b"wrong"), (503, b"Cache flushed.")])
def test_flush_requires_exact_success(monkeypatch, status, body):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, _):
            return body

    response = Response()
    response.status = status

    def open_request(request, timeout):
        assert request.method == "POST"
        assert request.full_url == "http://server/flush_cache?timeout=30"
        assert timeout == 40
        return response

    monkeypatch.setattr(prelude, "urlopen", open_request)
    with pytest.raises(RuntimeError):
        prelude.flush_cache("http://server/")
