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
    mechanism: str = "baseline",
    mechanism_variant: str | None = "none",
    quality_eval_id: str | None = None,
    quality_eval_status: str | None = None,
    draft_model_id: str | None = None,
    speculative_acceptance_rate: float | None = None,
    mechanism_eval_registered: bool | None = True,
    quality_eval_registered: bool | None = None,
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
        "mechanism": mechanism,
        "mechanism_variant": mechanism_variant,
        "quality_eval_id": quality_eval_id,
        "quality_eval_status": quality_eval_status,
        "draft_model_id": draft_model_id,
        "speculative_acceptance_rate": speculative_acceptance_rate,
        "mechanism_eval_validation": {
            "mechanism_eval_registered": mechanism_eval_registered,
            "quality_eval_registered": quality_eval_registered,
            "quality_eval_status_known": True,
            "issues": [],
        },
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



def test_mechanism_gate_passes_for_baseline_rows():
    rows = [
        make_row(
            result_filename="baseline_b200_vllm",
            model="dsr1",
            hw="b200-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=9416,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="baseline",
            mechanism_variant="none",
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    # Baseline rows enter the mechanism filter but pass every criterion trivially
    # — no compression mechanism, no speculative draft required, no quality eval required.
    assert mechanism_gate["status"] == "pass"
    assert mechanism_gate["matched_rows"] == 1
    assert mechanism_gate["failing_rows"] == []


def test_mechanism_gate_fails_supported_fp8_without_completed_eval():
    rows = [
        make_row(
            result_filename="dsr1_fp8kv_h100_vllm",
            model="dsr1",
            hw="h100-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="kv_quantization",
            mechanism_variant="fp8_e4m3",
            quality_eval_id="ruler_v1",
            quality_eval_status="pending",
            quality_eval_registered=True,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "fail"
    assert mechanism_gate["failing_rows"]
    failed = mechanism_gate["failing_rows"][0]
    assert any(
        "supported+compression" in criterion for criterion in failed["failed_criteria"]
    )


def test_mechanism_gate_passes_reviewed_preview_fp8_without_eval():
    rows = [
        make_row(
            result_filename="qwen_fp8kv_b200_sglang",
            model="qwen3.5",
            hw="b200-cw-1",
            framework="sglang",
            support_status="reviewed_preview",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="kv_quantization",
            mechanism_variant="fp8_e4m3",
            quality_eval_id=None,
            quality_eval_status=None,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "pass"


def test_mechanism_gate_passes_supported_fp8_with_completed_registered_eval():
    rows = [
        make_row(
            result_filename="dsr1_fp8kv_h100_vllm",
            model="dsr1",
            hw="h100-cw-1",
            framework="vllm",
            support_status="supported",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="kv_quantization",
            mechanism_variant="fp8_e4m3",
            quality_eval_id="ruler_v1",
            quality_eval_status="completed",
            quality_eval_registered=True,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "pass"


def test_mechanism_gate_fails_unregistered_variant():
    rows = [
        make_row(
            result_filename="weird_variant_b200_vllm",
            model="qwen3.5",
            hw="b200-cw-1",
            framework="vllm",
            support_status="reviewed_preview",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="kv_quantization",
            mechanism_variant="made_up_variant",
            mechanism_eval_registered=False,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "fail"
    failed = mechanism_gate["failing_rows"][0]
    assert "mechanism_variant registered" in failed["failed_criteria"]


def test_mechanism_gate_fails_speculative_without_draft_model():
    rows = [
        make_row(
            result_filename="spec_no_draft_h100_vllm",
            model="dsr1",
            hw="h100-cw-1",
            framework="vllm",
            support_status="reviewed_preview",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="speculative_decoding",
            mechanism_variant="eagle3",
            draft_model_id=None,
            speculative_acceptance_rate=None,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "fail"
    failed = mechanism_gate["failing_rows"][0]
    assert any(
        "speculative_decoding requires draft fields" in criterion
        for criterion in failed["failed_criteria"]
    )


def test_mechanism_gate_passes_speculative_with_full_fields():
    rows = [
        make_row(
            result_filename="spec_h100_vllm",
            model="dsr1",
            hw="h100-cw-1",
            framework="vllm",
            support_status="reviewed_preview",
            effective_max_context_depth=131272,
            context_pressure_class="standard",
            context_status="not_applicable",
            mechanism="speculative_decoding",
            mechanism_variant="eagle3",
            draft_model_id="eagle3-draft-v1",
            speculative_acceptance_rate=0.78,
        )
    ]
    report = build_gate_report(rows)
    mechanism_gate = next(
        gate for gate in report["gates"] if gate["id"] == "mechanism_compression_quality"
    )
    assert mechanism_gate["status"] == "pass"
