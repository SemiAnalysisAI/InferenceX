"""Behavioral contracts for prepared clients; no GPU, server, or package install required."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from infx.benchmarks.agentx import (
    build_argv as agentx_argv,
)
from infx.benchmarks.agentx import (
    normalize,
    replay_environment,
    validate_scenario,
    verify_corpus,
)
from infx.benchmarks.agentx import (
    run as run_agentx,
)
from infx.benchmarks.common import read_json, run_child
from infx.benchmarks.eval import (
    build_argv as eval_argv,
)
from infx.benchmarks.eval import (
    eval_metadata,
    packaged_task_path,
    stage_outputs,
    validate_outputs,
)
from infx.benchmarks.identity import verify_runtime
from infx.benchmarks.spec import AgentXSpec, EvalSpec


def bound_file(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def spec_inputs(tmp_path):
    identity = tmp_path / "identity.json"
    identity.write_text(
        '{"dataset_resolution":{"metadata":{"hf_dataset_name":"semianalysisai/cc-traces-weka-062126"}},'
        '"distributions":{"aiperf":{"direct_url":{"vcs_info":{"commit_id":'
        '"754356e9a39acc6cc6afb242d123bb57c3fb6f75"}}}}}'
    )
    dataset = tmp_path / "hub/datasets--semianalysisai--cc-traces-weka-062126"
    reference = dataset / "refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("f" * 40)
    data = dataset / "snapshots" / ("f" * 40) / "traces.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"trace_id":"one"}\n')
    return {
        "schema_version": 1,
        "runtime": {
            "python": sys.executable,
            "identity": bound_file(identity),
            "distributions": ["aiperf"],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_CACHE": str(tmp_path / "hub"),
                "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
                "AIPERF_DATASET_MMAP_CACHE_DIR": str(tmp_path / "mmap"),
            },
            "env_unset": ["UNWANTED_CLIENT_SETTING"],
            "assets": [bound_file(reference), bound_file(data)],
            "timeout_seconds": 10,
            "terminate_grace_seconds": 1,
        },
        "metadata": {
            "hw": "cluster:h100-fixture",
            "model": "fixture/model",
            "model_prefix": "fixture",
            "image": "fixture@sha256:" + "1" * 64,
            "framework": "vllm",
            "precision": "fp4",
            "spec_decoding": "mtp",
            "tp": 8,
            "pp": 1,
            "dcp_size": 1,
            "pcp_size": 1,
            "ep": 1,
            "dp_attention": False,
            "total_cpu_dram_gb": 1024,
            "recipe_fingerprint": "2" * 64,
        },
        "concurrency": 28,
    }


@pytest.fixture
def agentx_spec(spec_inputs):
    return AgentXSpec.model_validate(
        {
            **spec_inputs,
            "result_filename": "agg_fixture",
            "tokenizer": "fixture/model",
            "dataset_revision": "f" * 40,
            "dataset_loader": "semianalysis_cc_traces_weka_062126",
            "dataset_repository": "semianalysisai/cc-traces-weka-062126",
            "dataset_entries": 393,
            "duration_seconds": 3600,
            "warmup_requests_per_lane": 10,
            "warmup_grace_seconds": 1800,
            "trace_idle_gap_cap_seconds": 300,
            "live_failed_request_threshold": 0.1,
            "failed_request_threshold": 0.1,
            "random_seed": 42,
            "required_server_metric_prefix": "vllm:",
        }
    )


def export_config():
    return {
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "metadata": {
            "scenario": "inferencex-agentx-mvp",
            "submission_valid": True,
            "dataset": {
                "source_type": "public_dataset",
                "loader": "semianalysis_cc_traces_weka_062126",
                "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
                "hf_split": "train",
                "num_dataset_entries": 393,
            },
            "metric_duration_coverage": [
                {
                    "expected_duration_seconds": 3600.0,
                    "required_ratio": 0.95,
                    "ttft_ratio": 0.96,
                    "inter_token_latency_ratio": 0.2,
                }
            ],
        },
        "input_config": {
            "models": {"items": [{"name": "fixture/model"}]},
            "endpoint": {
                "urls": ["http://worker.example:9123"],
                "type": "chat",
                "path": "/v1/chat/completions",
                "streaming": True,
                "use_server_token_count": True,
            },
            "tokenizer": {"name": "fixture/model"},
            "phases": [
                {
                    "kind": "profiling",
                    "type": "concurrency",
                    "timing_mode": "agentic_replay",
                    "duration": 3600,
                    "concurrency": 28,
                    "trajectory_start_min_ratio": 0.25,
                    "trajectory_start_max_ratio": 0.75,
                    "system_idle_gap_cap_seconds": 10,
                    "warmup_requests_per_lane": 10,
                    "agentic_warmup_grace_period": 1800,
                    "failed_request_threshold": 0.1,
                }
            ],
            "datasets": [
                {
                    "dataset": "semianalysis_cc_traces_weka_062126",
                    "entries": 393,
                    "random_seed": 42,
                    "trace_idle_gap_cap_seconds": 300,
                }
            ],
        },
    }


def record(start, end, tokens, *, phase="profiling", error=None):
    return {
        "metadata": {
            "request_start_ns": start,
            "request_end_ns": end,
            "benchmark_phase": phase,
        },
        "metrics": {
            "output_sequence_length": {"value": tokens, "unit": "tokens"},
            "input_sequence_length": {"value": 10, "unit": "tokens"},
        },
        "error": error,
    }


def raw_agentx(root):
    raw = root / "aiperf_artifacts"
    raw.mkdir(parents=True)
    (raw / "profile_export.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                record(1_000_000_000, 3_000_000_000, 30),
                record(4_000_000_000, 7_000_000_000, 60),
                record(0, 8_000_000_000, 500, phase="warmup"),
                record(0, 9_000_000_000, 900, error={"type": "cancelled"}),
            ]
        )
        + "\n"
    )
    (raw / "profile_export_aiperf.json").write_text(json.dumps(export_config()))
    (raw / "server_metrics_export.json").write_text(
        '{"vllm:request_success_total": {}}'
    )
    (raw / "server_metrics_export.csv").write_text(
        "metric,value\nvllm:request_success_total,2\n"
    )
    return raw


def test_one_child_process_retains_raw_and_existing_normalization(
    tmp_path, agentx_spec, monkeypatch
):
    fixture = tmp_path / "producer"
    raw_agentx(fixture)
    external = tmp_path / "installed child"
    external.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,shutil,sys\n"
        "if '--distribution' in sys.argv:\n"
        f" print(pathlib.Path({agentx_spec.runtime.identity.path!r}).read_text())\n"
        "else:\n"
        " out=pathlib.Path(sys.argv[sys.argv.index('--output-artifact-dir')+1])\n"
        f" shutil.copytree({str(fixture / 'aiperf_artifacts')!r},out)\n"
        " (out.parent/'child-observed.json').write_text(json.dumps({'argv':sys.argv[1:],"
        "'ambient':os.environ.get('UNWANTED_CLIENT_SETTING'),"
        "'poison':os.environ.get('AIPERF_DATASET_RANDOM_SEED'),"
        "'cwd':os.getcwd()}))\n"
    )
    external.chmod(0o755)
    spec = agentx_spec.model_copy(
        update={
            "runtime": agentx_spec.runtime.model_copy(update={"python": str(external)})
        }
    )
    monkeypatch.setenv("UNWANTED_CLIENT_SETTING", "leaked")
    monkeypatch.setenv("AIPERF_DATASET_RANDOM_SEED", "1")
    output = tmp_path / "outputs with spaces"
    assert run_agentx(spec, "http://worker.example:9123", output) == 0
    observed = read_json(output / "child-observed.json")
    assert observed["argv"].count("profile") == 1
    assert observed["ambient"] is None and observed["poison"] is None
    assert observed["cwd"] == str(output)
    aggregate = read_json(output / "agg_fixture.json")
    # Two successful requests produce 90 tokens over their six-second span.
    assert (
        aggregate["request_metrics"]["throughput"]["output"]["tokens_per_second"] == 15
    )
    assert aggregate["request_metrics"]["throughput"]["duration_seconds"] == 6
    assert aggregate["request_accounting"]["records_warmup_dropped"] == 1
    assert aggregate["request_accounting"]["records_error_dropped"] == 1
    assert aggregate["num_gpus"] == 8 and aggregate["is_multinode"] is False
    assert (
        read_json(output / "diagnostics/client-audit.json")["status"]["returncode"] == 0
    )


def test_literal_argv_and_resolved_remote_endpoint(tmp_path, agentx_spec):
    argv = agentx_argv(
        agentx_spec, "http://remote.example:9009/", tmp_path / "literal ; $(no)"
    )
    assert argv[argv.index("--url") + 1] == "http://remote.example:9009"
    assert (
        argv[argv.index("--server-metrics") + 1] == "http://remote.example:9009/metrics"
    )
    assert argv[argv.index("--output-artifact-dir") + 1].endswith(
        "literal ; $(no)/aiperf_artifacts"
    )
    assert "--max-context-length" not in argv
    assert "--unsafe-override" not in argv


def test_normalizer_uses_native_serving_log_context(tmp_path, agentx_spec, monkeypatch):
    output = tmp_path / "output"
    raw_agentx(output)
    logs = tmp_path / "native-logs"
    logs.mkdir()
    (logs / "node_agg_w0.out").write_text("INFO GPU KV cache size: 1,234,567 tokens\n")
    # A different point's old log must not override the supplied native context.
    (tmp_path / "watchtower-other.out").write_text(
        "INFO GPU KV cache size: 8,000,000 tokens\n"
    )
    monkeypatch.setenv("SRT_LOG_DIR", str(logs))
    normalized = read_json(normalize(agentx_spec, output))
    assert normalized["kv_cache_pool_tokens"] == 1_234_567


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda value: value["input_config"]["phases"][0].update(
                warmup_requests_per_lane=1
            ),
            "warmup_requests",
        ),
        (
            lambda value: value["input_config"]["datasets"][0].update(
                max_context_length=100
            ),
            "context cap",
        ),
        (
            lambda value: value["metadata"].update(submission_valid=False),
            "scenario validity",
        ),
        (
            lambda value: value["metadata"]["metric_duration_coverage"][0].update(
                ttft_ratio=0.94
            ),
            "neither TTFT",
        ),
    ],
)
def test_scenario_canonical_contract_is_independent_of_scenario_verdict(
    agentx_spec, change, message
):
    aggregate = export_config()
    change(aggregate)
    errors = validate_scenario(aggregate, agentx_spec, "http://worker.example:9123")
    assert any(message in error for error in errors)


def test_mutated_prepared_dataset_fails_before_client(agentx_spec):
    data = Path(agentx_spec.runtime.assets[-1].path)
    data.write_text("changed")
    with pytest.raises(ValueError, match="prepared file missing or changed"):
        verify_runtime(agentx_spec.runtime, dataset_loader=agentx_spec.dataset_loader)


def test_unbound_dataset_content_and_ambient_scenario_override_are_rejected(
    agentx_spec,
):
    Path(agentx_spec.runtime.assets[-1].path).with_name("extra.json").write_text("{}")
    with pytest.raises(ValueError, match="absent from the prepared asset list"):
        verify_corpus(agentx_spec)
    runtime = agentx_spec.runtime.model_copy(
        update={"env": {**agentx_spec.runtime.env, "AIPERF_UNSAFE_OVERRIDE": "true"}}
    )
    with pytest.raises(ValueError, match="unqualified"):
        replay_environment(agentx_spec.model_copy(update={"runtime": runtime}))


def test_nonfinite_json_and_duplicate_keys_fail(tmp_path):
    path = tmp_path / "value.json"
    for text in ('{"metric": 1e400}', '{"metric":NaN}', '{"metric":1,"metric":2}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


def test_client_timeout_kills_process_group_and_preserves_log(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('started',flush=True); time.sleep(30)"
    result = run_child(
        [sys.executable, "-c", code],
        env=os.environ,
        cwd=tmp_path,
        log=tmp_path / "client.log",
        timeout_seconds=0.2,
        terminate_grace_seconds=0.1,
    )
    assert result == {
        "returncode": -signal.SIGKILL,
        "cancelled_by_signal": None,
        "timed_out": True,
        "orphaned_descendants": False,
    }
    assert (tmp_path / "client.log").read_text() == "started\n"


def test_repeated_term_is_forwarded_and_does_not_restart_shutdown(tmp_path):
    ready = tmp_path / "ready"
    child_code = (
        "import pathlib,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
    )
    parent_code = (
        "import json,os,sys; from pathlib import Path; "
        f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r}); "
        "from infx.benchmarks.common import run_child; "
        f"status=run_child([sys.executable,'-c',{child_code!r}],env=os.environ,"
        "cwd=Path.cwd(),log=Path('child.log'),timeout_seconds=10,terminate_grace_seconds=0.2); "
        "Path('status.json').write_text(json.dumps(status))"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], cwd=tmp_path)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        parent.send_signal(signal.SIGTERM)
        time.sleep(0.03)
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=2) == 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
    status = read_json(tmp_path / "status.json")
    assert status["cancelled_by_signal"] == signal.SIGTERM
    assert status["timed_out"] is False
    assert status["returncode"] == -signal.SIGKILL


def test_successful_child_with_surviving_writer_is_not_accepted(tmp_path):
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    code = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
        "print('leader exited',flush=True)"
    )
    status = run_child(
        [sys.executable, "-c", code],
        env=os.environ,
        cwd=tmp_path,
        log=tmp_path / "client.log",
        timeout_seconds=5,
        terminate_grace_seconds=0.1,
    )
    assert status["returncode"] == 0
    assert status["orphaned_descendants"] is True


@pytest.fixture
def eval_case(tmp_path, spec_inputs):
    documents = {
        str(index): {"question": f"What is {index}+1?", "answer": f"#### {index + 1}"}
        for index in range(1319)
    }
    identities = {
        key: hashlib.sha256(
            json.dumps(doc, indent=2, ensure_ascii=False).encode()
        ).hexdigest()
        for key, doc in documents.items()
    }
    identity_path = tmp_path / "documents.json"
    identity_path.write_text(json.dumps(identities))
    inputs = {
        **spec_inputs,
        "runtime": {**spec_inputs["runtime"], "distributions": ["lm-eval"]},
        "task": bound_file(packaged_task_path()),
        "document_identities": bound_file(identity_path),
        "task_name": "gsm8k",
        "expected_documents": 1319,
        "max_length": 16384,
        "max_tokens": 12288,
        "minimum_score": 0.9,
    }
    spec = EvalSpec.model_validate(inputs)
    task = yaml.safe_load(packaged_task_path().read_text())
    task["generation_kwargs"].update(max_tokens=12288, temperature=0, top_p=1)
    result = {
        "config": {
            "model": "local-chat-completions",
            "limit": None,
            "model_args": {
                "model": "fixture/model",
                "base_url": "http://worker:9000/v1/chat/completions",
                "num_concurrent": 28,
                "max_length": 16384,
                "tokenized_requests": False,
            },
            "gen_kwargs": {"max_tokens": 12288, "temperature": 0, "top_p": 1},
        },
        "configs": {"gsm8k": task},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 1319}},
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 1318 / 1319,
                "exact_match,flexible-extract": 1318 / 1319,
                "exact_match_stderr,strict-match": 0.01,
            }
        },
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result))
    samples = []
    # Reversed filter order must preserve coverage and score checks.
    for name in ("flexible-extract", "strict-match"):
        for key, doc in documents.items():
            samples.append(
                {
                    "doc_id": int(key),
                    "filter": name,
                    "doc": doc,
                    "doc_hash": identities[key],
                    "target": doc["answer"],
                    "target_hash": hashlib.sha256(doc["answer"].encode()).hexdigest(),
                    "exact_match": 0.0 if key == "0" else 1.0,
                }
            )
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
    return spec, result_path, sample_path, result, samples


def test_real_eval_contract_accepts_complete_split_and_canonical_aggregate(
    eval_case, tmp_path
):
    spec, results, samples, _, _ = eval_case
    assert validate_outputs(spec, "http://worker:9000", [results], [samples]) == []
    argv = eval_argv(spec, "http://worker:9000", tmp_path)
    assert (
        "base_url=http://worker:9000/v1/chat/completions"
        in argv[argv.index("--model_args") + 1]
    )
    assert "max_length=16384" in argv[argv.index("--model_args") + 1]
    assert (
        argv[argv.index("--gen_kwargs") + 1] == "max_tokens=12288,temperature=0,top_p=1"
    )
    metadata = eval_metadata(spec, complete=True)
    assert (
        metadata["num_gpus"],
        metadata["prefill_num_workers"],
        metadata["decode_num_workers"],
    ) == (8, 0, 0)
    assert metadata["deployment"] == {
        "kind": "aggregate",
        "nodes": 1,
        "serving_gpus": 8,
        "tp": 8,
        "ep": 1,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["config"]["model_args"].update(max_length=1048576), "max_length"),
        (lambda r: r["config"].update(limit=0.1), "full-split"),
        (
            lambda r: r["results"]["gsm8k"].update({"exact_match,strict-match": 1.0}),
            "disagrees with raw samples",
        ),
    ],
)
def test_eval_rejects_self_consistent_wrong_budget_smoke_and_score(
    eval_case, mutation, match
):
    spec, result_path, sample_path, result, _ = eval_case
    mutation(result)
    result_path.write_text(json.dumps(result))
    assert any(
        match in error
        for error in validate_outputs(
            spec, "http://worker:9000", [result_path], [sample_path]
        )
    )


def test_eval_rejects_duplicate_missing_and_changed_document(eval_case):
    spec, result_path, sample_path, _, samples = eval_case
    samples[-1] = samples[0]
    samples[1]["doc"]["question"] = "unrelated task"
    sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
    errors = validate_outputs(spec, "http://worker:9000", [result_path], [sample_path])
    assert any("duplicate eval sample" in error for error in errors)
    assert any("coverage incomplete" in error for error in errors)
    assert any("document bytes/hash" in error for error in errors)


def test_eval_rejects_nonfinite_secondary_metric(eval_case):
    spec, result_path, sample_path, result, _ = eval_case
    result["results"]["gsm8k"]["exact_match_stderr,strict-match"] = float("inf")
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="non-finite"):
        validate_outputs(spec, "http://worker:9000", [result_path], [sample_path])


def test_staging_keeps_partial_evidence_and_rejects_collision(eval_case, tmp_path):
    spec, _, _, _, _ = eval_case
    raw = tmp_path / "harness/nested"
    raw.mkdir(parents=True)
    (raw / "results_failure.json").write_bytes(b'{"failed": true}\n')
    (raw / "samples_partial.jsonl").write_bytes(b'{"doc_id":0}\n')
    results, samples = stage_outputs(spec, tmp_path)
    assert results[0].read_bytes() == b'{"failed": true}\n'
    assert samples[0].read_bytes() == b'{"doc_id":0}\n'
    with pytest.raises(FileExistsError):
        stage_outputs(spec, tmp_path)
