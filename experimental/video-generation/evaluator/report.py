"""Portable, script-free presentation of a verified MVP comparison."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import quote


_CSS = """
:root{color-scheme:light;--ink:#132b37;--muted:#526875;--line:#d7e2e5;--paper:#fff;--wash:#f3f7f8;--accent:#086b66;--pass:#086b48;--fail:#a7313e;--warn:#805b11}
*{box-sizing:border-box}body{margin:0;background:var(--wash);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink)}
a{color:var(--accent)}main{max-width:1280px;margin:auto;padding:42px 36px 70px}.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:750;color:var(--muted)}
header{display:flex;justify-content:space-between;align-items:start;gap:24px;margin-bottom:22px}h1{font-size:clamp(28px,4vw,44px);letter-spacing:-.035em;line-height:1.1;margin:10px 0 16px}h2{font-size:23px;letter-spacing:-.02em;margin:0 0 12px}h3{font-size:18px;margin:0 0 10px}p{margin:0 0 12px}.sub{color:var(--muted);max-width:850px}
.download{white-space:nowrap;padding:10px 15px;border:1px solid #adc5ca;border-radius:8px;background:white;text-decoration:none;font-weight:650;font-size:13px;margin-top:22px}
.banner{padding:18px 21px;border-radius:10px;border:1px solid #d5b968;background:#fff8df;margin:0 0 20px}.banner.live{border-color:#93c8c4;background:#edf9f6}.banner strong{display:block;font-size:17px}.banner p{font-size:13px;margin:5px 0 0}
.decision{display:flex;gap:15px;align-items:center;margin:24px 0}.decision h2{margin:0}.badge{display:inline-block;border-radius:6px;padding:3px 9px;background:#edf3f5;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.045em;white-space:nowrap}.badge.pass{color:var(--pass);background:#e1f4e9}.badge.fail{color:var(--fail);background:#fbe9eb}.badge.inconclusive{color:var(--warn);background:#fff0c8}.badge.descriptive,.badge.not_applicable{color:var(--muted);background:#e8eff2}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:27px}.card,.panel{border:1px solid var(--line);border-radius:11px;background:var(--paper);padding:20px}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:28px;font-weight:730;letter-spacing:-.03em;margin:7px 0 5px;overflow-wrap:anywhere}.card .detail{font-size:12px;color:var(--muted)}
.section{margin-top:30px}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:14px}.section-head h2{margin:0}.section-head p{font-size:12px;margin:0;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.kv{display:grid;grid-template-columns:145px minmax(0,1fr);gap:8px 16px;font-size:12px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.mono,code{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.pin{font-size:11px}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;text-align:left}th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.02em}th,td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child,th:first-child{padding-left:0}tbody tr:last-child td{border-bottom:0}.number{font-variant-numeric:tabular-nums;white-space:nowrap}.reason{color:var(--muted)}
.case{background:white;border:1px solid var(--line);border-radius:12px;margin-bottom:22px;overflow:hidden}.case-heading{padding:21px 23px 17px;border-bottom:1px solid var(--line)}.case-title{display:flex;justify-content:space-between;gap:20px;align-items:center}.case-title h3{margin:0}.case-meta{font-size:11px;color:var(--muted);margin:6px 0 10px}.prompt{font-size:14px;margin:0;overflow-wrap:anywhere}.media-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:20px 23px 10px}figure{margin:0;min-width:0}.media-label{font-size:11px;font-weight:750;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;color:var(--muted)}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0c151e;border-radius:7px}.no-media{aspect-ratio:16/9;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:20px;background:#f8eff0;border:1px dashed #e1b9bd;border-radius:7px;text-align:center;overflow-wrap:anywhere}.no-media p{font-size:12px;max-width:390px}.no-media strong{margin-bottom:8px}figcaption{font-size:11px;color:var(--muted);margin:8px 0;overflow-wrap:anywhere}.case-body{padding:5px 23px 20px}.case-body details{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}summary{cursor:pointer;font-size:12px;font-weight:650}.notes{font-size:12px;color:var(--muted);margin:12px 0 0;padding-left:19px}.notes li+li{margin-top:7px}.foot{margin-top:35px;padding-top:18px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}
@media(max-width:900px){main{padding:26px 20px 45px}.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}.kv{grid-template-columns:130px minmax(0,1fr)}}
@media(max-width:600px){main{padding:20px 12px 35px}header{display:block}.download{display:inline-block;margin:0 0 8px}.cards{gap:8px}.card{padding:14px}.card .value{font-size:24px}.media-grid{grid-template-columns:1fr;padding:15px}.case-heading{padding:17px 15px}.case-body{padding:5px 15px 15px}.section-head{display:block}.decision{align-items:start;flex-wrap:wrap}.kv{grid-template-columns:110px minmax(0,1fr)}}
@media print{body{background:white}main{max-width:none;padding:10px}.download{display:none}.case{break-inside:avoid}details{display:block}.banner{print-color-adjust:exact}.cards{grid-template-columns:repeat(4,1fr)}}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "not measured"
    return f"{value:,.{digits}f}"


def _percent(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "not measured"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _badge(status: Any) -> str:
    status = str(status)
    css = status if status in {"pass", "fail", "inconclusive", "descriptive", "not_applicable"} else "inconclusive"
    return f'<span class="badge {css}">{_escape(status.replace("_", " "))}</span>'


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_artifact(source: Path, digest: str, assets: Path) -> Path:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("report artifact requires a valid SHA256")
    source = source.resolve(strict=True)
    if not source.is_file() or _digest(source) != digest:
        raise ValueError("report source artifact changed after comparison")
    suffix = source.suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        suffix = ".media"
    target = assets / f"{digest}{suffix}"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or _digest(target) != digest:
            raise ValueError("existing content-addressed report asset has different content")
        return target
    # Exclusive creation prevents replacement of a preexisting report asset.
    try:
        with source.open("rb") as origin, target.open("xb") as destination:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: origin.read(1024 * 1024), b""):
                destination.write(chunk)
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            target.unlink()
            raise ValueError("report source changed while copying")
    except FileExistsError:
        if target.is_symlink() or _digest(target) != digest:
            raise ValueError("report asset collision")
    return target


def _check_table(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        observed = check.get("observed")
        threshold = check.get("threshold")
        if check.get("exact_match"):
            observed_text = "exact match"
        elif observed is None:
            observed_text = "—"
        elif isinstance(observed, (int, float)):
            observed_text = _number(observed, 4)
        else:
            observed_text = _escape(observed)
        threshold_text = _number(threshold, 4) if threshold is not None else "—"
        rows.append(
            f'<tr><td class="mono">{_escape(check.get("name", ""))}</td>'
            f'<td>{_badge(check.get("status", "inconclusive"))}</td>'
            f'<td class="number">{observed_text}</td><td class="number">{threshold_text}</td>'
            f'<td class="reason">{_escape(check.get("reason", ""))}</td></tr>'
        )
    return '<div class="table-wrap"><table><thead><tr><th>Check</th><th>Decision</th><th>Observed</th><th>Threshold</th><th>Explanation</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def _transformation_label(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value or "not recorded")[:300]
    labels = {
        "none_original_bytes_preserved": "Original bytes preserved",
        "exact_file_copy": "Exact file copy",
        "deliberate_audio_mute": "Deliberately muted audio",
    }
    parts = [labels.get(value.get("kind"), str(value.get("kind") or "Declared transformation").replace("_", " "))]
    for key, label in (
        ("compressed_video_bitexact", "compressed video bit-exact"),
        ("decoded_video_identical", "decoded video identical"),
        ("audio_sample_count_preserved", "audio sample count preserved"),
    ):
        if key in value:
            parts.append(f"{label}: {'yes' if value[key] is True else 'no' if value[key] is False else 'unverified'}")
    return "; ".join(parts)


def _configuration(label: str, run: dict) -> str:
    configuration = run.get("configuration", {})
    provenance = run.get("provenance") or {}
    fields = [
        ("Run", run.get("run_id", "not recorded")),
        ("Hardware", configuration.get("hardware_label", "not recorded")),
        ("Runtime", configuration.get("runtime", "not recorded")),
        ("Runtime revision", configuration.get("runtime_revision", "not recorded")),
        ("Model", configuration.get("model_id", "not recorded")),
        ("Model revision", configuration.get("model_revision", "not recorded")),
        ("Identity", configuration.get("identity_verification", "operator-declared")),
        ("Run SHA256", run.get("run_bundle_sha256", "not recorded")),
    ]
    if provenance:
        fields.extend([
            ("Source", provenance.get("source_url") or provenance.get("source_filename", "not recorded")),
            ("Source SHA256", provenance.get("source_sha256", "not recorded")),
            ("Attribution", provenance.get("attribution_status", "unverified")),
            ("Transformation", _transformation_label(provenance.get("transformation"))),
            ("Contract origin", provenance.get("contract_origin", "not recorded")),
        ])
    body = "".join(f'<dt>{_escape(name)}</dt><dd class="mono">{_escape(value)}</dd>' for name, value in fields)
    return f'<section class="panel"><h3>{_escape(label)}</h3><dl class="kv">{body}</dl></section>'


def _media(label: str, observation: dict) -> str:
    path = observation.get("artifact_path")
    media = observation.get("media") or {}
    video = media.get("video") or {}
    audio = media.get("audio") or {}
    if path:
        body = f'<video controls preload="metadata" playsinline src="{_escape(quote(path, safe="/"))}">Your browser cannot play this format. <a href="{_escape(quote(path, safe="/"))}">Download the clip</a>.</video>'
    else:
        reason = observation.get("error") or observation.get("analysis_error") or "No media artifact was produced."
        body = f'<div class="no-media"><strong>No playable artifact</strong><p>{_escape(reason)}</p></div>'
    latency = observation.get("latency_seconds")
    boundary = 'downloaded media' if observation.get('latency_boundary') == 'submit_to_downloaded_media' else 'validated media'
    details = [f'{_number(latency)} s submit → {boundary}' if latency is not None else 'generation timing not measured']
    if video.get("width") and video.get("height"):
        details.append(f'{video["width"]}×{video["height"]} · {video.get("frame_count", "?")} frames')
    if audio.get("present"):
        details.append(f'{audio.get("channels", "?")} audio channels · {audio.get("sample_rate_hz", "?")} Hz')
    elif media:
        details.append("no audio stream")
    caption = " · ".join(_escape(part) for part in details)
    return f'<figure><div class="media-label">{_escape(label)}</div>{body}<figcaption>{caption}</figcaption></figure>'


def _case(slot: dict, *, imported: bool = False) -> str:
    metrics = slot.get("metrics", {})
    psnr = "exact decoded match" if metrics.get("video_identical") is True else _number(metrics.get("video_psnr_db")) + " dB"
    pairs = [
        ("Video PSNR", psnr, "same-request RGB fidelity, not generative quality"),
        ("Video mean absolute error", _number(metrics.get("video_mae"), 6), "normalized RGB, 0–1"),
        ("Audio spectral cosine", _number(metrics.get("audio_spectral_cosine"), 5), "spectral similarity; not semantic or perceptual quality"),
        ("Audio RMS ratio", _number(metrics.get("audio_rms_ratio"), 5), "candidate ÷ baseline; threshold uses worst channel"),
        ("Request latency change", _percent(metrics.get("latency_increase_fraction"), signed=True), "paired request; run-level gate uses median population"),
    ]
    if imported:
        pairs[-1] = ("Generation latency", "Not applicable", "Imported media; no inference timing was measured")
    metric_rows = "".join(f'<tr><td>{_escape(label)}</td><td class="number">{_escape(value)}</td><td class="reason">{_escape(note)}</td></tr>' for label, value, note in pairs)
    notes = "".join(f'<li>{_escape(note)}</li>' for note in slot.get("notes", []))
    note_list = f'<ul class="notes">{notes}</ul>' if notes else ""
    opened = " open" if slot.get("status") != "pass" else ""
    seed_label = "pairing ID seed (not a generation seed)" if imported else "seed"
    return (
        '<article class="case"><div class="case-heading"><div class="case-title">'
        f'<h3>{_escape(slot.get("case_id", "Case"))}</h3>{_badge(slot.get("status", "inconclusive"))}</div>'
        f'<div class="case-meta"><span class="mono">{_escape(slot.get("slot_id", ""))}</span> · {seed_label} {_escape(slot.get("seed", ""))} · repetition {_escape(slot.get("repetition", ""))}</div>'
        f'<p class="prompt">{_escape(slot.get("prompt", ""))}</p></div>'
        f'<div class="media-grid">{_media("Baseline", slot.get("baseline", {}))}{_media("Candidate", slot.get("candidate", {}))}</div>'
        f'<div class="case-body"><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Observed</th><th>Interpretation</th></tr></thead><tbody>{metric_rows}</tbody></table></div>'
        f'<details{opened}><summary>Per-slot checks and failure reasons</summary>{_check_table(slot.get("checks", []))}{note_list}</details></div></article>'
    )


def _render(comparison: dict, json_name: str) -> str:
    evidence = comparison.get("evidence_kind", "unknown")
    imported = comparison.get("comparison_scope") == "media_fidelity_only"
    title = "H3 sample media comparison" if imported else "H3 runtime comparison"
    titles = {
        "fixture": ("Synthetic fixture evidence — not an H3 result", "These test clips and fixture timings exercise the harness. They do not measure MiniMax model performance or quality."),
        "operator_endpoint": ("Endpoint results — H3 execution is not independently verified", "Clips were collected from a configured endpoint. Only a separate controlled-GPU job receipt can establish observed runtime, model files, GPU use, and resource cleanup."),
        "live_h3": ("Legacy endpoint results — H3 execution is not independently verified", "This older bundle used the label live_h3, but its model, hardware, and runtime identities were only operator-declared. The label is not proof that H3 ran."),
        "imported_media": ("Imported-media evidence — no H3 inference run or timing measurement", "These source clips demonstrate media validation and paired fidelity only. Model attribution is operator-supplied; decoded media properties are a post-hoc contract, not proof of prompt compliance."),
        "mixed": ("Mixed evidence — this comparison is inconclusive", "The baseline and candidate use different evidence kinds. Do not interpret this as a controlled live-model benchmark."),
    }
    evidence_title, evidence_note = titles.get(evidence, ("Unverified evidence", "The report does not establish where these artifacts were generated."))
    status = comparison.get("overall_status", "inconclusive")
    decision = {"pass": "Declared checks passed", "fail": "Regression checks failed", "inconclusive": "Comparison needs more evidence"}.get(status, "Comparison needs more evidence")
    baseline = comparison.get("baseline", {})
    candidate = comparison.get("candidate", {})
    left, right = baseline.get("summary", {}), candidate.get("summary", {})
    summary = comparison.get("summary", {})
    measurement = comparison.get("measurement", {})
    policy = comparison.get("policy", {})
    if imported and status == "fail":
        transformation = (candidate.get("provenance") or {}).get("transformation") or {}
        decision = "Controlled audio defect detected" if isinstance(transformation, dict) and transformation.get("kind") == "deliberate_audio_mute" else "Media checks failed"
    mode = measurement.get("performance_mode", "not recorded").replace("_", " ")
    cards = [
        ("Candidate median latency", f'{_number(right.get("latency_median_seconds"))} s', f'Baseline {_number(left.get("latency_median_seconds"))} s · {_percent(measurement.get("latency_increase_fraction"), signed=True)}'),
        ("Candidate verified-valid clips", f'{right.get("valid", "?")} / {right.get("scheduled", "?")}', f'{_percent(right.get("verified_technical_success_fraction", right.get("technical_success_rate")))} verified yield · failures stay in denominator'),
        ("Matched valid media pairs", str(summary.get("matched_valid_pairs", "?")), f'{summary.get("measurement_slots", "?")} scheduled measurement slots · warmups excluded'),
        ("Candidate throughput", f'{_number(right.get("valid_clips_per_second"), 4)}', 'valid clips / recorded measurement wall second · not saturation capacity'),
    ]
    if imported:
        cards = [
            ("Generation latency", "Not measured", "Imported clips do not establish model speed"),
            ("Candidate valid media", f'{right.get("valid", "?")} / {right.get("scheduled", "?")}', "Full-stream decode and declared media contract"),
            ("Matched valid media pairs", str(summary.get("matched_valid_pairs", "?")), "Same-source paired fidelity; not a model-quality score"),
            ("Generation throughput", "Not measured", "No inference job or hardware utilization was measured"),
        ]
    card_html = "".join(f'<div class="card"><div class="label">{_escape(label)}</div><div class="value">{_escape(value)}</div><div class="detail">{_escape(detail)}</div></div>' for label, value, detail in cards)
    unavailable = int(left.get("evaluator_unavailable", 0)) + int(right.get("evaluator_unavailable", 0))
    unavailable_note = (
        f'<aside class="banner"><strong>Media evaluation unavailable for {unavailable} observation(s)</strong>'
        '<p>Verified-valid yield is incomplete. Unassessed media is not evidence of model failure; inspect the per-slot evaluator errors.</p></aside>'
        if unavailable else ""
    )
    policy_rows = [
        ("Policy", policy.get("policy_id")),
        ("Calibration", policy.get("calibration_status")),
        ("Max median latency increase", _percent(policy.get("max_latency_increase_fraction"))),
        ("Min video PSNR", _number(policy.get("min_video_psnr_db")) + " dB (exact matches pass separately)"),
        ("Min audio spectral cosine", _number(policy.get("min_audio_spectral_cosine"), 5)),
        ("Max channel RMS ratio error", _number(policy.get("max_audio_rms_ratio_error"), 5)),
    ]
    policy_body = "".join(f'<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>' for label, value in policy_rows)
    measurement_rows = [
        ("Plan", comparison.get("plan_id")),
        ("Plan SHA256", comparison.get("plan_sha256")),
        ("Timing boundary", measurement.get("boundary")),
        ("Concurrency", measurement.get("concurrency")),
        ("Performance use", mode),
        ("Timing evidence", measurement.get("timing_evidence", "see evidence kind and source run")),
        ("Statistical claim", measurement.get("statistical_claim")),
    ]
    measurement_body = "".join(f'<dt>{_escape(label)}</dt><dd class="mono">{_escape(value)}</dd>' for label, value in measurement_rows)
    limitations = "".join(f'<li>{_escape(note)}</li>' for note in comparison.get("limitations", []))
    cases = "".join(_case(slot, imported=imported) for slot in comparison.get("slots", []))
    banner_class = "banner live" if evidence == "live_h3" else "banner"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; media-src \'self\' file:; img-src \'self\' data:; base-uri \'none\'; form-action \'none\'">'
        f'<title>{title} · evidence report</title>'
        f'<style>{_CSS}</style></head><body><main><header><div><div class="eyebrow">Video generation benchmark · execution MVP</div>'
        f'<h1>{title}</h1><p class="sub">' + ('A paired, artifact-backed view of an imported source clip and controlled transformations. Generation performance is not measured.' if imported else 'A paired, artifact-backed view of media validity, implementation fidelity, and recorded end-to-end performance.') + '</p></div>'
        f'<a class="download" href="{_escape(quote(json_name))}" download>Download evidence JSON ↗</a></header>'
        f'<aside class="{banner_class}"><strong>{_escape(evidence_title)}</strong><p>{_escape(evidence_note)}</p></aside>'
        f'<div class="decision">{_badge(status)}<h2>{_escape(decision)}</h2></div>'
        f'<p class="sub">Policy calibration: <strong>{_escape(policy.get("calibration_status", "unknown"))}</strong>. No release qualification is claimed. Performance mode: {_escape(mode)}.</p>'
        f'<div class="cards">{card_html}</div>{unavailable_note}'
        f'<section class="section"><div class="section-head"><h2>Side-by-side evidence</h2><p>Use each player’s controls to inspect video and native audio</p></div>{cases}</section>'
        '<section class="section"><div class="section-head"><h2>Configurations and pins</h2><p>Supplied by the operator · artifacts checked against SHA256</p></div>'
        f'<div class="grid2">{_configuration("Baseline", baseline)}{_configuration("Candidate", candidate)}</div></section>'
        f'<section class="section"><div class="section-head"><h2>Run-level decisions</h2><p>Missing values are not treated as zero</p></div><div class="panel">{_check_table(comparison.get("checks", []))}</div></section>'
        '<section class="section"><div class="section-head"><h2>Declared policy and measurement</h2></div><div class="grid2">'
        f'<div class="panel"><h3>Explicit thresholds</h3><dl class="kv">{policy_body}</dl></div><div class="panel"><h3>Measurement boundary</h3><dl class="kv">{measurement_body}</dl></div></div></section>'
        f'<section class="section panel"><h2>What this report does not claim</h2><ul class="notes">{limitations}</ul></section>'
        f'<footer class="foot">Generated {_escape(comparison.get("created_at", ""))}. Report version {_escape(comparison.get("bundle_version", ""))}. '
        'This report is read-only and contains no scripts or remote dependencies. Share the HTML, its comparison JSON, and the adjacent assets folder together.</footer>'
        '</main></body></html>'
    )


def write_report(comparison: dict, output_path: Path) -> None:
    """Write HTML, downloadable JSON, and immutable content-addressed media.

    Artifact hashes are checked again immediately before copying. Existing assets
    are reused only when their content matches. Existing HTML/JSON, including
    symlinks, is never overwritten; exclusive creation also guards racing writes.
    """
    if comparison.get("bundle_type") != "mvp_comparison":
        raise ValueError("write_report requires an MVP comparison")
    output_path = Path(output_path).absolute()
    if output_path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("report output_path must have an .html or .htm extension")
    json_path = output_path.with_suffix(".comparison.json")
    for target in (output_path, json_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"report output already exists: {target}")
    assets = output_path.parent / f"{output_path.stem}_assets"
    if assets.is_symlink():
        raise ValueError("report assets directory must not be a symlink")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        # Reserve both files before making assets. A post-preflight collision must
        # not clobber either file or produce an apparently complete report.
        with ExitStack() as stack:
            streams = {}
            for target in (output_path, json_path):
                stream = stack.enter_context(target.open("x", encoding="utf-8"))
                metadata = os.fstat(stream.fileno())
                created.append((target, (metadata.st_dev, metadata.st_ino)))
                streams[target] = stream
            assets.mkdir(exist_ok=True)
            portable = copy.deepcopy(comparison)
            for slot in portable.get("slots", []):
                for label in ("baseline", "candidate"):
                    observation = slot.get(label, {})
                    source = observation.get("artifact_path")
                    if source:
                        asset = _copy_artifact(Path(source), str(observation.get("sha256", "")), assets)
                        observation["artifact_path"] = asset.relative_to(output_path.parent).as_posix()
                        observation["artifact_path_base"] = "report_directory"
                        if isinstance(observation.get("media"), dict) and "path" in observation["media"]:
                            observation["media"]["path"] = observation["artifact_path"]
            for label in ("baseline", "candidate"):
                provenance = portable.get(label, {}).get("provenance")
                if isinstance(provenance, dict) and provenance.get("source_path"):
                    provenance.setdefault("source_filename", Path(str(provenance["source_path"])).name)
                    del provenance["source_path"]
            portable["report"] = {"html": output_path.name, "portable_assets": assets.name, "scripts": False}
            payload = json.dumps(portable, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            document = _render(portable, json_path.name)
            streams[json_path].write(payload)
            streams[output_path].write(document)
    except Exception:
        for target, identity in created:
            try:
                metadata = target.stat(follow_symlinks=False)
                if (metadata.st_dev, metadata.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
