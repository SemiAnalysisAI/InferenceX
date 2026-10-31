from pathlib import Path

import pytest
import yaml

from infx.serving.scenarios import FAMILIES, Point, ScenarioError, render

LEGACY = Path(__file__).parent / "legacy_scenarios"
GEN = {"top_p": 0.95, "temperature": 1.0}

# (legacy file, family, overrides, point, search, expected label)
CASES = [
    ("agentx-cc-062126", "agentx", {}, None, False, "agentx-cc-062126"),
    ("agentx-cc-062126-smoke", "agentx", {"duration": 180}, None, False, "agentx-cc-062126-smoke"),
    ("agentx-cc-062126-256k", "agentx", {"dataset": "semianalysis_cc_traces_weka_062126_256k"}, None, False,
     "agentx-cc-062126-256k"),
    ("agentx-cc-062126-1m", "agentx", {"max_context_length": None}, None, False, "agentx-cc-062126-1m"),
    ("agentx-cc-062126-topp95", "agentx", {"generation": GEN}, None, False, "agentx-cc-062126-topp95"),
    ("custom-agentic-tau2", "tau2", {}, None, False, "custom-agentic-tau2"),
    ("custom-agentic-tau2-smoke", "tau2", {"duration": 180}, None, False, "custom-agentic-tau2-smoke"),
    ("custom-agentic-livekit111868-v1.1-warm", "livekit", {}, None, False,
     "custom-agentic-livekit111868-v1.1-warm"),
    ("custom-agentic-livekit111868-v1", "livekit", {"dataset": "livekit-111868-v1"}, None, False,
     "custom-agentic-livekit111868-v1"),
    ("custom-agentic-livekit111868-v2", "livekit", {"dataset": "livekit-111868-v2"}, None, False,
     "custom-agentic-livekit111868-v2"),
    ("custom-agentic-livekit111868-smoke", "livekit", {"dataset": "livekit-111868-v2", "duration": 180}, None,
     False, "custom-agentic-livekit111868-v2-smoke"),
    ("agentx-shared-prefix-e2", "shared-prefix-e2", {}, None, False, "agentx-shared-prefix-e2"),
    ("agentx-hidden-prefix-e3", "hidden-prefix-e3", {}, None, False, "agentx-hidden-prefix-e3"),
    ("agentx-hidden-prefix-e3-smoke", "hidden-prefix-e3", {"duration": 180}, None, False,
     "agentx-hidden-prefix-e3-smoke"),
    ("sweep-isl8k-osl512-cache50", "synthetic", {}, Point("8k", "512", 50), True, "sweep-isl8k-osl512-cache50"),
    ("sweep-isl8k-osl1k-cache90", "synthetic", {}, Point("8k", "1k", 90), True, "sweep-isl8k-osl1k-cache90"),
    ("sweep-isl128k-osl512-cache0", "synthetic", {}, Point("128k", "512", 0), True, "sweep-isl128k-osl512-cache0"),
    ("sweep-isl128k-osl1k-cache0", "synthetic", {}, Point("128k", "1k", 0), True, "sweep-isl128k-osl1k-cache0"),
    ("fixed-isl8k-osl1k-cache50", "synthetic", {}, Point("8k", "1k", 50), False, "fixed-isl8k-osl1k-cache50"),
    ("fixed-isl8k-osl1k-cache0", "synthetic", {}, Point("8k", "1k", 0), False, "fixed-isl8k-osl1k-cache0"),
]


@pytest.mark.parametrize("legacy, family, overrides, point, search, label", CASES, ids=[c[0] for c in CASES])
def test_base_plus_variant_equals_the_legacy_file(legacy, family, overrides, point, search, label):
    rendered = render(family, overrides, point, search)
    assert rendered.doc == yaml.safe_load((LEGACY / f"{legacy}.yaml").read_text())
    assert rendered.label == label
    assert yaml.safe_load(rendered.yaml()) == rendered.doc


def test_every_legacy_fixture_is_covered():
    assert {path.stem for path in LEGACY.glob("*.yaml")} == {case[0] for case in CASES}


def test_kinds_and_datasets():
    assert render("agentx").kind == "public"
    tau2 = render("tau2")
    assert (tau2.kind, tau2.dataset, tau2.duration) == ("file", "tau2-cs", 1200.0)
    assert render("livekit").dataset == "livekit-111868-v1.1-warm"
    sweep = render("synthetic", point=Point("8k", "1k", 0), search=True)
    assert (sweep.kind, sweep.search) == ("synthetic", True)


def test_slo_replaces_the_sweep_filters():
    doc = render("synthetic", point=Point("8k", "1k", 0), search=True,
                 slo="time_to_first_token:p95:lt:500;inter_token_latency:p50:lt:50").doc
    assert doc["sweep"]["sla_filters"] == [
        {"metric_tag": "time_to_first_token", "stat": "p95", "op": "lt", "threshold": 500.0},
        {"metric_tag": "inter_token_latency", "stat": "p50", "op": "lt", "threshold": 50.0},
    ]


@pytest.mark.parametrize("args, message", [
    (("agentx", {"warmup": 1}), "overrides: unknown key"),
    (("agentx", {"dataset": "tau2-cs"}), "overrides.dataset: agentx takes a public"),
    (("tau2", {"dataset": "semianalysis_cc_traces_weka_062126"}), "is not in the dataset lock"),
    (("agentx", {"duration": 0}), "overrides.duration"),
    (("agentx", {"generation": {"top_k": 5}}), "overrides.generation"),
    (("synthetic", {}), "workload: synthetic needs"),
    (("synthetic", {"dataset": "x"}, Point("8k", "1k", 0)), "overrides: synthetic only takes duration"),
    (("synthetic", {}, Point("16k", "1k", 0)), "workload.isl"),
    (("synthetic", {}, Point("8k", "1k", 0), False, "time_to_first_token:avg:lt:1"), "slo: only applies"),
    (("agentx", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic"),
    (("nope",), "scenario: 'nope' is not one of"),
])
def test_bad_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_greenspec_renders_the_selection_and_has_no_duration():
    rendered = render("greenspec", {"subset": "agentic_tools", "lite": True, "requests": 50})
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert (rendered.kind, rendered.dataset, rendered.label) == ("file", "greenspec-v1.1.1", "greenspec-agentic_tools-lite-n50")
    assert dataset["format"] == "raw_payload" and dataset["path"] == "${IX_CUSTOM_DATASET_FILE}"
    [phase] = rendered.doc["benchmark"]["phases"]
    assert phase["requests"] == 50 and "duration" not in phase and rendered.duration == 0.0
    assert "ignore_eos" not in rendered.doc["benchmark"]["endpoint"].get("extra", {})
    assert render("greenspec").label == "greenspec-diverse"


@pytest.mark.parametrize("args, message", [
    (("greenspec", {"subset": "coding"}), "overrides.subset"),
    (("greenspec", {"duration": 180}), "greenspec takes"),
    (("greenspec", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic"),
])
def test_bad_greenspec_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_speedbench_renders_the_config_dataset_and_has_no_duration():
    rendered = render("speedbench", {"config": "throughput_8k", "category": "mixed", "requests": 30})
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert (rendered.kind, rendered.dataset, rendered.label) == (
        "file", "speedbench-throughput-8k", "speedbench-throughput_8k-mixed-n30")
    assert dataset["name"] == "speedbench-throughput-8k"
    assert dataset["format"] == "raw_payload" and dataset["path"] == "${IX_CUSTOM_DATASET_FILE}"
    [phase] = rendered.doc["benchmark"]["phases"]
    assert phase["requests"] == 30 and "duration" not in phase and rendered.duration == 0.0
    assert "ignore_eos" not in rendered.doc["benchmark"]["endpoint"].get("extra", {})
    default = render("speedbench")
    assert (default.label, default.dataset) == ("speedbench-qualitative", "speedbench-qualitative")


@pytest.mark.parametrize("args, message", [
    (("speedbench", {"category": "mixed"}), "overrides.category"),
    (("speedbench", {"subset": "diverse"}), "speedbench takes"),
    (("speedbench", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic, not speedbench"),
])
def test_bad_speedbench_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_warmup_requests_per_lane_override():
    def profiling(doc):
        return next(p for p in doc["benchmark"]["phases"] if p.get("kind") == "profiling")

    off = render("livekit", {"warmup_requests_per_lane": 0})
    assert "warmupRequestsPerLane" not in profiling(off.doc)
    assert off.label == "custom-agentic-livekit111868-v1.1-warm-nowarmup"
    on = render("livekit", {"warmup_requests_per_lane": 3})
    assert profiling(on.doc)["warmupRequestsPerLane"] == 3
    assert on.label == "custom-agentic-livekit111868-v1.1-warm-warm3"
    with pytest.raises(ScenarioError, match="overrides.warmup_requests_per_lane"):
        render("livekit", {"warmup_requests_per_lane": True})


def test_families_have_base_files():
    from infx.serving.scenarios import SCENARIOS_DIR
    assert {path.stem for path in SCENARIOS_DIR.glob("*.yaml")} == set(FAMILIES)
