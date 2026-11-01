import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from infx.bench.eval import host_restore_probe as p

SHA = "baa9befa381511cd4d56f83980951e5a45fffc6d"
FP = {
    "env": {},
    "pip_packages": {
        "freeze": [
            f"-e git+https://github.com/sgl-project/sglang.git@{SHA}#egg=sglang&subdirectory=python"
        ]
    },
}
POOL_LINE = "Attached hybrid pool stack: pools=KV + INDEXER + MAMBA"
BOOT = (
    POOL_LINE
    + "\nserver_args={'model_path': 'model', 'revision': 'pin', 'served_model_name': 'model'}"
)
CAPACITY_BOOT = BOOT + "\nmax_total_num_tokens=1000000"
INFO = {
    "model_path": "model",
    "revision": "pin",
    "served_model_name": "model",
    "enable_hierarchical_cache": True,
    "enable_metrics": True,
    "enable_cache_report": True,
    "speculative_num_steps": 5,
    "context_length": 1048576,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 6,
    "hicache_host_memory_fraction": 0.70,
    "hicache_io_backend": "kernel",
    "hicache_mem_layout": "page_first",
    "hicache_write_policy": "write_through",
    "kv_cache_dtype": "fp8_e4m3",
}


class Fake:
    def __init__(self, restore=True, corrupt=False, drop=False, pressure_tokens=1000) -> None:
        self.restore, self.corrupt, self.drop = restore, corrupt, drop
        self.probes, self.pressure, self.loads = 0, 0, 0
        self.pressure_tokens = pressure_tokens

    def text_metrics(self):
        values = [
            ('load_back_tokens_total{pool="kv"}', self.loads * 18000),
            ('load_back_tokens_total{pool="mamba"}', self.loads),
            ("load_back_duration_seconds_count", self.loads),
            ("load_back_bytes_total", self.loads * 8192),
            ('prefill_effective_tokens_total{mode="host_hit"}', self.loads * 18000),
            (
                'hicache_dropped_tokens_total{pool="kv",reason="host_pressure"}',
                1 if self.drop and self.pressure else 0,
            ),
            ("kv_available_tokens", 1000000),
            ("kv_evictable_tokens", 0),
            ("kv_used_tokens", 0),
            ("mamba_available_tokens", 1000),
            ("mamba_evictable_tokens", 0),
            ("mamba_used_tokens", 0),
        ]
        return "\n".join(f"sglang:{name} {value}" for name, value in values)

    def metrics(self):
        return p.parse_metrics(self.text_metrics())

    def chat(self, payload):
        text = payload["messages"][0]["content"]
        if text.startswith("Unrelated pressure"):
            self.pressure += 1
            content, cached, host = "OK", 0, 0
        else:
            self.probes += 1
            if self.pressure and self.restore:
                self.loads += 1
            selected = set(
                map(int, re.search(r"markers for entries ([0-9, ]+), one", text)[1].split(","))
            )
            markers = [
                marker
                for marker, index in re.findall(r"(MARK_[0-9a-f]+_(\d+))\b", text)
                if int(index) in selected
            ]
            content = "\n".join(markers)
            cached = 0 if self.probes == 1 else 19000
            host = 18000 if self.pressure and self.restore else 0
            if self.corrupt and self.probes >= 3:
                content = "wrong"
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {
                "prompt_tokens": self.tokenize(text, "model"),
                "prompt_tokens_details": {"cached_tokens": cached},
            },
            "sglext": {
                "cached_tokens_details": {"device": cached - host, "host": host},
                "spec_tokens_details": {"spec_verify_ct": 12, "spec_accept_length": 3.2},
            },
        }

    def tokenize(self, prompt, model):
        return self.pressure_tokens if prompt.startswith("Unrelated pressure") else 20000


class Tests(unittest.TestCase):
    def probe(self, fake, **changes):
        limits = p.Limits(
            pressure_chars=100,
            pressure_batch=1,
            restore_rounds=2,
            max_pressure_requests=8,
            max_prompt_tokens=262144,
            max_requests=64,
            pressure_target_tokens=1000,
        )
        for key, value in changes.items():
            setattr(limits, key, value)
        return p.Probe(fake, "model", limits)

    def test_pass_requires_two_real_restores(self) -> None:
        probe = self.probe(Fake())
        assert probe.run()["proven_host_restores"] == 2
        assert len([e for e in probe.events if e.get("proven_host_restore")]) == 2

    def test_cached_tokens_without_host_counters_not_pass(self) -> None:
        probe = self.probe(Fake(restore=False))
        with pytest.raises(p.InconclusiveError, match="pressure budget"):
            probe.run()

    def test_aggregate_host_counters_without_request_host_not_pass(self) -> None:
        fake = Fake()
        original = fake.chat

        def no_host(payload):
            result = original(payload)
            result["sglext"]["cached_tokens_details"]["host"] = 0
            return result

        fake.chat = no_host
        with pytest.raises(p.InconclusiveError, match="pressure budget"):
            self.probe(fake).run()

    def test_no_actual_mtp_evidence_inconclusive(self) -> None:
        fake = Fake()
        original = fake.chat

        def no_spec(payload):
            result = original(payload)
            result["sglext"].pop("spec_tokens_details")
            return result

        fake.chat = no_spec
        with pytest.raises(p.InconclusiveError, match="speculative"):
            self.probe(fake).run()

    def test_corrupted_facts_fail(self) -> None:
        with pytest.raises(p.IncorrectError, match="missing 4"):
            self.probe(Fake(corrupt=True)).run()

    def test_corrupt_response_without_host_proof_inconclusive(self) -> None:
        with pytest.raises(p.InconclusiveError):
            self.probe(Fake(restore=False, corrupt=True)).run()

    def test_generation_length_is_inconclusive(self) -> None:
        fake = Fake()
        original = fake.chat

        def limited(payload):
            result = original(payload)
            result["choices"][0]["finish_reason"] = "length"
            return result

        fake.chat = limited
        with pytest.raises(p.InconclusiveError, match="length"):
            self.probe(fake).run()

    def test_drop_inconclusive(self) -> None:
        with pytest.raises(p.InconclusiveError, match="without backup"):
            self.probe(Fake(drop=True)).run()

    def test_request_budget(self) -> None:
        with pytest.raises(p.InconclusiveError, match="request budget"):
            self.probe(Fake(), max_requests=2).run()

    def test_token_budget(self) -> None:
        with pytest.raises(p.InconclusiveError, match="token budget"):
            self.probe(Fake(), max_prompt_tokens=10).run()

    def test_rank_ambiguity_not_summed(self) -> None:
        m = p.parse_metrics('sglang:x{rank="0"} 7\nsglang:x{rank="1"} 7\n')
        with pytest.raises(p.InconclusiveError):
            p.delta(m, m, "x", {})
        assert p.delta(m, m, "x", {"rank": "0"}) == 0

    def test_counter_reset(self) -> None:
        with pytest.raises(p.InconclusiveError, match="reset"):
            p.delta(p.parse_metrics("sglang:x 9"), p.parse_metrics("sglang:x 2"), "x", {})

    def test_optional_absent_metric(self) -> None:
        assert p.delta({}, {}, "x", {}, required=False) is None

    def test_registered_first_counter_child_starts_zero(self) -> None:
        before = p.parse_metrics("# TYPE sglang:x counter\n")
        after = p.parse_metrics('# TYPE sglang:x counter\nsglang:x{pool="kv"} 12\n')
        assert p.delta(before, after, "x", {"pool": "kv"}) == 12

    def test_registered_histogram_first_count_uses_base_family(self) -> None:
        before = p.parse_metrics("# TYPE sglang:load_back_duration_seconds histogram\n")
        after = p.parse_metrics(
            "# TYPE sglang:load_back_duration_seconds histogram\n"
            'sglang:load_back_duration_seconds_count{tp_rank="0"} 1\n'
        )
        assert p.delta(before, after, "load_back_duration_seconds_count", {"tp_rank": "0"}) == 1

    def test_partial_marker_is_not_exact_recall(self) -> None:
        fake = Fake()
        original = fake.chat

        def suffix_markers(payload):
            result = original(payload)
            if fake.probes >= 3 and not payload["messages"][0]["content"].startswith("Unrelated"):
                result["choices"][0]["message"]["content"] = re.sub(
                    r"(MARK_[a-f0-9]+_\d+)",
                    r"\1_WRONG",
                    result["choices"][0]["message"]["content"],
                )
            return result

        fake.chat = suffix_markers
        with pytest.raises(p.IncorrectError, match="missing 4"):
            self.probe(fake).run()

    def test_unregistered_new_counter_not_assumed_zero(self) -> None:
        with pytest.raises(p.InconclusiveError):
            p.delta(p.parse_metrics(""), p.parse_metrics("sglang:x 12"), "x", {})

    def test_capacity_plan_bounds(self) -> None:
        limits = p.Limits()
        plan = p.capacity_plan(Fake().metrics(), {}, CAPACITY_BOOT, limits)
        assert plan["pressure_tokens_per_round"] == 1100000
        assert plan["hard_total_prompt_token_cap"] == 3000000
        assert plan["device_mamba_states"] == 1000

    def test_capacity_insufficient_explicit_budget_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="token budget"):
            p.capacity_plan(Fake().metrics(), {}, CAPACITY_BOOT, p.Limits(max_prompt_tokens=262144))

    def test_capacity_boot_mismatch_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="startup log"):
            p.capacity_plan(Fake().metrics(), {}, BOOT, p.Limits())

    def test_float_metric_and_escaped_label(self) -> None:
        m = p.parse_metrics('sglang:x{a="m\\\\n",rank="0"} 1.25e3\n# comment')
        assert list(m.values()) == [1250]

    def test_provenance_accept(self) -> None:
        assert p.validate_provenance(FP, BOOT, SHA)["pool_lines"] == [POOL_LINE]

    def test_simulation_zero_string_still_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="simulation"):
            p.validate_provenance({**FP, "env": {"SGLANG_SIMULATE_ACC_LEN": "0"}}, BOOT, SHA)

    def test_missing_indexer_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="INDEXER"):
            p.validate_provenance(FP, "pools=KV + MAMBA", SHA)

    def test_wrong_build_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="SHA"):
            p.validate_provenance(FP, BOOT, "bad")

    def test_live_wrong_checkpoint_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="revision"):
            p.validate_server(INFO, INFO, "model", "different")

    def test_staged_model_revision_proven_by_fingerprint(self) -> None:
        info = {**INFO, "model_path": "/model", "revision": None}
        fingerprint = {**FP, "model": {"hf_revision": "pin"}}
        result = p.validate_server(info, info, "/model", "pin", "model", fingerprint, info)
        assert result["model_path"] == "/model"

    def test_staged_model_missing_fingerprint_revision_rejected(self) -> None:
        info = {**INFO, "model_path": "/model", "revision": None}
        with pytest.raises(p.InconclusiveError, match="revision"):
            p.validate_server(info, info, "/model", "pin", "model", FP, info)

    def test_staged_boot_mismatch_rejected(self) -> None:
        info = {**INFO, "model_path": "/model", "revision": None}
        with pytest.raises(p.InconclusiveError, match="startup log"):
            p.validate_server(
                info, info, "/model", "pin", "model", {**FP, "model": {"hf_revision": "pin"}}, INFO
            )

    def test_wrong_served_model_rejected(self) -> None:
        with pytest.raises(p.InconclusiveError, match="served model"):
            p.validate_server(INFO, INFO, "model", "pin", "other")

    def test_actual_qualification_contract_fail_closed(self) -> None:
        for key, wrong in (
            ("context_length", None),
            ("context_length", 262144),
            ("speculative_eagle_topk", 2),
            ("speculative_num_draft_tokens", 5),
            ("hicache_host_memory_fraction", 0.8),
            ("hicache_io_backend", "direct"),
            ("hicache_mem_layout", "layer_first"),
            ("hicache_write_policy", "write_back"),
            ("kv_cache_dtype", "bfloat16"),
        ):
            with pytest.raises(p.InconclusiveError):
                p.validate_server({**INFO, key: wrong}, INFO, "model", "pin")

    def test_unset_cli_context_uses_actual_boot_and_pinned_flash_config(self) -> None:
        info = {**INFO, "context_length": None}
        result = p.validate_server(
            info,
            info,
            "model",
            "pin",
            boot_contexts=[1048576],
            model_config={
                "architectures": ["Glm5NextForConditionalGeneration"],
                "text_config": {"max_position_embeddings": 1048576},
            },
        )
        assert result["context_length"] is None
        assert result["resolved_context_length"] == 1048576

    def test_unset_cli_context_rejects_short_runtime_or_unproven_config(self) -> None:
        info = {**INFO, "context_length": None}
        config = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"max_position_embeddings": 1048576},
        }
        for contexts, cfg in (([262144], config), ([1048576], {}), ([], config)):
            with pytest.raises(p.InconclusiveError, match="unset CLI context"):
                p.validate_server(
                    info,
                    info,
                    "model",
                    "pin",
                    boot_contexts=contexts,
                    model_config=cfg,
                )

    def test_http_cli_end_to_end(self) -> None:
        fake = Fake(pressure_tokens=128000)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def send(self, body) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body.encode())

            def do_GET(self) -> None:
                if self.path == "/metrics":
                    self.send(fake.text_metrics())
                elif self.path in ("/server_info", "/model_info"):
                    self.send(json.dumps(INFO))
                else:
                    raise AssertionError("unexpected endpoint " + self.path)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/tokenize":
                    self.send(
                        json.dumps(
                            {"count": fake.tokenize(body["messages"][0]["content"], "model")}
                        )
                    )
                else:
                    assert self.path == "/v1/chat/completions"
                    self.send(json.dumps(fake.chat(body)))

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    d = Path(directory)
                    (d / "fingerprint.json").write_text(json.dumps(FP))
                    (d / "server.log").write_text(CAPACITY_BOOT)
                    argv = [
                        "--url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--model",
                        "model",
                        "--expected-model",
                        "model",
                        "--expected-revision",
                        "pin",
                        "--expected-image",
                        "official@sha256:fixture",
                        "--server-fingerprint",
                        str(d / "fingerprint.json"),
                        "--server-boot-log",
                        str(d / "server.log"),
                        "--output",
                        str(d / "report.json"),
                        "--exclusive-server",
                    ]
                    assert p.main(argv) == 0
                    assert fake.probes == fake.pressure == 0
                    assert json.loads((d / "report.json").read_text())["status"] == "PLAN_ONLY"
                    code = p.main([*argv, "--execute"])
                    report = json.loads((d / "report.json").read_text())
                    assert code == 0
                    assert report["status"] == "DIAGNOSTIC_PASS"
                    assert report["result"]["proven_host_restores"] == 2
                    assert report["live_server"]["revision"] == "pin"
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
