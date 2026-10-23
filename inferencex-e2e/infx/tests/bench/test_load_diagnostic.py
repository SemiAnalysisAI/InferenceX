import importlib.util
from importlib.machinery import ModuleSpec
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import ModuleType, SimpleNamespace

import pytest

from infx.bench import load_diagnostic as diag


def test_gpu_and_process_accounting(tmp_path):
    device = tmp_path / "drm/card4/device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text("1000")
    (device / "mem_info_vram_used").write_text("777")
    (tmp_path / "drm/card4-DP-1").mkdir()
    values = diag.gpu_snapshot(tmp_path / "drm")
    assert list(values) == ["card4"]
    assert values["card4"]["mem_info_vram_used"] == "777"
    proc = tmp_path / "proc/123"
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    (proc / "fd/7").symlink_to("/dev/kfd")
    (proc / "fdinfo/7").write_text("drm-memory-vram: 555 kB")
    (proc / "comm").write_text("worker")
    (proc / "cgroup").write_text("0::/slurm/job_11/step_2")
    result = diag.gpu_processes(tmp_path / "proc")
    assert result["inaccessible_processes"] == 0
    assert len(result["processes"]) == 1
    assert result["processes"][0]["pid"] == 123
    assert result["processes"][0]["fds"]["7"]["info"] == "drm-memory-vram: 555 kB"


@pytest.fixture
def torch(monkeypatch):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: 4, mem_get_info=lambda device: (400, 1000),
                             memory_stats=lambda device: {"allocated_bytes.all.current": 500,
                                                          "reserved_bytes.all.current": 550}),
        distributed=SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 4),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.mark.parametrize("fail", [False, True])
def test_iterator_preserves_values_failures_and_close(tmp_path, torch, fail):
    shard = tmp_path / "weights.safetensors"
    shard.write_bytes(b"12345")
    closed = []
    tensor = SimpleNamespace(shape=(2, 3), dtype="bf16", numel=lambda: 6, element_size=lambda: 2)

    def original(files):
        try:
            yield "layer.0", tensor
            if fail:
                raise RuntimeError("allocation failed")
            yield "layer.1", tensor
        finally:
            closed.append(True)

    wrapped = diag.trace_weights_iterator(original, logdir=tmp_path)([str(shard)])
    assert next(wrapped) == ("layer.0", tensor)
    if fail:
        with pytest.raises(RuntimeError, match="allocation failed"):
            next(wrapped)
    else:
        assert list(wrapped) == [("layer.1", tensor)]
    assert closed == [True]
    records = [json.loads(line) for line in next(tmp_path.glob("load-memory-*.jsonl")).read_text().splitlines()]
    assert records[0]["free"] == 400
    assert records[0]["files"] == [{"path": str(shard), "bytes": 5}]
    assert records[-1]["event"] == "iterator_closed"
    assert records[-2]["event"] == ("iterator_error" if fail else "iterator_complete")


def test_early_iterator_close_releases_original(tmp_path, torch):
    closed = []

    def original(files):
        try:
            yield "w", SimpleNamespace(shape=(1,), dtype="fp32", numel=lambda: 1, element_size=lambda: 4)
        finally:
            closed.append(True)

    wrapped = diag.trace_weights_iterator(original, logdir=tmp_path)([])
    next(wrapped)
    wrapped.close()
    assert closed == [True]


def test_install_and_execute_instrumented_module(tmp_path, monkeypatch, torch):
    package = tmp_path / "fake_vllm"
    folder = package / "model_executor/model_loader"
    folder.mkdir(parents=True)
    target = folder / "weight_utils.py"
    target.write_text("def fastsafetensors_weights_iterator(files):\n    yield from ()\n")
    loader = tmp_path / "fastsafetensors"
    loader.mkdir()
    (loader / "parallel_loader.py").write_text("VERSION = 123\n")
    specs = {"vllm": SimpleNamespace(origin=str(package / "__init__.py")),
             "fastsafetensors": SimpleNamespace(origin=str(loader / "__init__.py"))}
    monkeypatch.setattr(diag.importlib.util, "find_spec", lambda name: specs[name])
    diag.install(logdir=tmp_path)
    # Load the actual appended module and run its generator with a local log sink.
    parent = ModuleType("diagnostic_fixture")
    parent.__path__ = [str(folder)]
    parent.__spec__ = ModuleSpec("diagnostic_fixture", None, is_package=True)
    monkeypatch.setitem(sys.modules, "diagnostic_fixture", parent)
    spec = importlib.util.spec_from_file_location("diagnostic_fixture.weight_utils", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper = sys.modules["diagnostic_fixture._k3_load_diagnostic"]
    monkeypatch.setitem(helper.trace_weights_iterator.__kwdefaults__, "logdir", tmp_path)
    spec.loader.exec_module(module)
    assert list(module.fastsafetensors_weights_iterator([])) == []
    records = [json.loads(line) for line in next(tmp_path.glob("load-memory-*.jsonl")).read_text().splitlines()]
    assert [row["event"] for row in records] == ["iterator_start", "iterator_complete", "iterator_closed"]
    with pytest.raises(RuntimeError, match="unexpected weight_utils"):
        diag.install(logdir=tmp_path)


def test_smoke_sends_only_one_short_request(monkeypatch, capsys):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"Hello"}}]}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        monkeypatch.setenv("SRT_FRONTEND_HOST", "127.0.0.1")
        monkeypatch.setenv("SRT_FRONTEND_PORT", str(server.server_port))
        monkeypatch.setenv("MODEL", "test-model")
        diag.smoke()
        assert len(requests) == 1
        assert requests[0][0] == "/v1/chat/completions"
        assert requests[0][1]["max_tokens"] == 16
        assert "diagnostic_complete" in capsys.readouterr().out
    finally:
        server.shutdown()
        worker.join(2)
        server.server_close()


def test_preserve_retains_raw_data_and_summarizes_truncated_log(tmp_path):
    workspace = tmp_path / "workspace"
    logs = workspace / "LOGS"
    logs.mkdir(parents=True)
    output = tmp_path / "slurm/outputs/12"
    (output / "logs").mkdir(parents=True)
    (workspace / "srt-submission.json").write_text(json.dumps({"output_dir": str(output)}))
    raw = '{"free": 123, "event": "tensor_yield"}\n{broken'
    (logs / "load-memory-test.jsonl").write_text(raw)
    diag.preserve(workspace, "42")
    saved = tmp_path / "slurm/k3-startup-diagnostics/42"
    assert (saved / "LOGS/load-memory-test.jsonl").read_text() == raw
    assert json.loads((saved / "loader-summary.json").read_text())[0]["min_free_bytes"] == 123
    assert (output / "logs/perf-analysis.md").exists()
