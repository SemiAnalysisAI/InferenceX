import json
from pathlib import Path

from gate_isb1 import build_gate_report, load_rows, main


def make_row(
    *,
    result_filename: str,
    model: str,
    hw: str,
    framework: str,
    support_status: str,
    effective_max_context_depth: int,
    context_pressure_class: str,
    context_status: str,
    requires_log_review: bool = False,
    context_pressure_suspicious: bool = False,
    completed_sessions: int = 2,
    total_sessions: int = 2,
    session_throughput_sps: float = 1.0,
    benchmark_certification_status: str = "dataset_replay_verified",
):
    return {
        "benchmark_type": "isb1_replay",
        "result_filename": result_filename,
        "artifact_stems": {
            "processed": f"isb1_{result_filename}",
            "raw_replay": f"replay_{result_filename}",
            "server_logs": f"server_logs_{result_filename}",
            "gpu_metrics": f"gpu_metrics_{result_filename}",
        },
        "infmax_model_prefix": model,
        "hw": hw,
        "framework": framework,
        "support_status": support_status,
        "effective_max_context_depth": effective_max_context_depth,
        "context_pressure_class": context_pressure_class,
        "context_pressure_signal": {
            "status": context_status,
            "requires_log_review": requires_log_review,
        },
        "context_pressure_suspicious": context_pressure_suspicious,
        "completed_sessions": completed_sessions,
        "total_sessions": total_sessions,
        "session_throughput_sps": session_throughput_sps,
        "benchmark_certification_status": benchmark_certification_status,
    }


def test_build_gate_report_passes_with_sglang_observability_gap():
    rows = [
        make_row(
            result_filename="dsr1_control_b200_vllm",
            model="dsr1",
            hw="b200-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=9416,
            context_pressure_class="standard",
            context_status="not_applicable",
        ),
        make_row(
            result_filename="gptoss_control_h100_vllm",
            model="gptoss",
            hw="h100-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=9416,
            context_pressure_class="standard",
            context_status="not_applicable",
        ),
    ]

    for hw in ("b200-cw-1", "h100-cw-1", "h200-cw-1"):
        for framework in ("vllm", "sglang"):
            rows.append(
                make_row(
                    result_filename=f"qwen_131k_{hw}_{framework}",
                    model="qwen3.5",
                    hw=hw,
                    framework=framework,
                    support_status="reviewed_preview",
                    effective_max_context_depth=131272,
                    context_pressure_class="standard",
                    context_status="not_applicable",
                )
            )
            rows.append(
                make_row(
                    result_filename=f"qwen_500k_{hw}_{framework}",
                    model="qwen3.5",
                    hw=hw,
                    framework=framework,
                    support_status="reviewed_preview",
                    effective_max_context_depth=524288,
                    context_pressure_class="extended_500k",
                    context_status="ok" if framework == "vllm" else "observability_gap",
                    requires_log_review=framework == "sglang",
                )
            )

    rows.extend(
        [
            make_row(
                result_filename="qwen_1m_b200_vllm",
                model="qwen3.5",
                hw="b200-cw-1",
                framework="vllm",
                support_status="reviewed_preview",
                effective_max_context_depth=1048576,
                context_pressure_class="extended_1m",
                context_status="ok",
            ),
            make_row(
                result_filename="qwen_1m_b200_sglang",
                model="qwen3.5",
                hw="b200-cw-1",
                framework="sglang",
                support_status="reviewed_preview",
                effective_max_context_depth=1048576,
                context_pressure_class="extended_1m",
                context_status="observability_gap",
                requires_log_review=True,
            ),
        ]
    )

    report = build_gate_report(rows)

    assert report["overall"] == "pass"
    assert all(gate["status"] == "pass" for gate in report["gates"])
    qwen_500k_gate = next(gate for gate in report["gates"] if gate["id"] == "qwen_500k")
    assert qwen_500k_gate["review_required_rows"]
    assert any(
        row["result_filename"] == "qwen_500k_b200-cw-1_sglang"
        for row in qwen_500k_gate["review_required_rows"]
    )


def test_build_gate_report_fails_control_lane_and_preserves_artifact_refs():
    rows = [
        make_row(
            result_filename="dsr1_control_b200_vllm",
            model="dsr1",
            hw="b200-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=9416,
            context_pressure_class="standard",
            context_status="not_applicable",
            completed_sessions=1,
            total_sessions=2,
            session_throughput_sps=0.0,
        )
    ]

    report = build_gate_report(rows)

    assert report["overall"] == "fail"
    control_gate = next(gate for gate in report["gates"] if gate["id"] == "control_lanes")
    assert control_gate["status"] == "fail"
    assert control_gate["failing_rows"][0]["result_filename"] == "dsr1_control_b200_vllm"
    assert control_gate["failing_rows"][0]["artifact_stems"]["server_logs"] == "server_logs_dsr1_control_b200_vllm"
    assert "completed_sessions == total_sessions" in control_gate["failing_rows"][0]["failed_criteria"]
    assert "session_throughput_sps > 0" in control_gate["failing_rows"][0]["failed_criteria"]


def test_build_gate_report_fails_when_qwen_131k_coverage_is_missing():
    rows = [
        make_row(
            result_filename="qwen_131k_b200_vllm",
            model="qwen3.5",
            hw="b200-cw-1",
            framework="vllm",
            support_status="reviewed_preview",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
        )
    ]

    report = build_gate_report(rows)

    assert report["overall"] == "fail"
    qwen_131k_gate = next(gate for gate in report["gates"] if gate["id"] == "qwen_131k")
    assert qwen_131k_gate["status"] == "fail"
    assert ["b200", "sglang"] in qwen_131k_gate["missing_coverage"]
    assert ["h200", "vllm"] in qwen_131k_gate["missing_coverage"]


def test_build_gate_report_handles_no_rows():
    report = build_gate_report([])

    assert report["overall"] == "partial"
    assert all(gate["status"] == "no_rows" for gate in report["gates"])


def test_gate_main_strict_returns_nonzero_on_failure(tmp_path):
    payload = [
        make_row(
            result_filename="dsr1_control_b200_vllm",
            model="dsr1",
            hw="b200-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=9416,
            context_pressure_class="standard",
            context_status="not_applicable",
            completed_sessions=1,
            total_sessions=2,
        )
    ]
    report_path = tmp_path / "agg_isb1.json"
    report_path.write_text(json.dumps(payload))

    assert load_rows(report_path)[0]["result_filename"] == "dsr1_control_b200_vllm"
    assert main([str(report_path), "--strict"]) == 1
