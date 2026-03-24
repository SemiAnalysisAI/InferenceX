import json
import os
import re
import sys
from pathlib import Path

import psycopg2
from tabulate import tabulate


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def colorize_delta(delta_val, pct_val, higher_is_better=True, fmt=".4f"):
    """Format a colored delta string using LaTeX color syntax for GitHub markdown."""
    improved = (delta_val > 0) if higher_is_better else (delta_val < 0)
    regressed = (delta_val < 0) if higher_is_better else (delta_val > 0)
    delta_str = f"{delta_val:+{fmt}}"
    pct_str = f"({pct_val:+.1f}%)"
    if improved:
        return f"$\\color{{green}}\\text{{{delta_str}}}$ {pct_str}"
    elif regressed:
        return f"$\\color{{red}}\\text{{{delta_str}}}$ {pct_str}"
    return f"{delta_str} {pct_str}"


def compute_delta_str(current, baseline, higher_is_better=True, fmt=".4f"):
    """Compute a colored delta string between current and baseline values."""
    if current is None or baseline is None or baseline == 0:
        return "N/A"
    delta = current - baseline
    pct = (delta / baseline) * 100
    return colorize_delta(delta, pct, higher_is_better, fmt)


def extract_hardware(runner):
    """Strip suffixes like -multinode, -trt, -disagg from runner to get hardware name."""
    return re.split(r"-(multinode|trt|disagg)$", runner)[0].lower()


def build_config_params(result):
    """Build the DB config lookup parameters from a result JSON."""
    is_multinode = result.get("is_multinode", False)
    hw = extract_hardware(result["hw"])
    model = result["infmax_model_prefix"].lower()
    framework = result["framework"].lower()
    precision = result["precision"].lower()
    spec_method = result.get("spec_decoding", "none").lower()
    disagg = parse_bool(result.get("disagg", False))

    if is_multinode:
        return {
            "hardware": hw,
            "model": model,
            "framework": framework,
            "precision": precision,
            "spec_method": spec_method,
            "disagg": disagg,
            "is_multinode": True,
            "prefill_tp": int(result["prefill_tp"]),
            "prefill_ep": int(result["prefill_ep"]),
            "prefill_dp_attention": parse_bool(result["prefill_dp_attention"]),
            "decode_tp": int(result["decode_tp"]),
            "decode_ep": int(result["decode_ep"]),
            "decode_dp_attention": parse_bool(result["decode_dp_attention"]),
        }
    else:
        tp = int(result["tp"])
        ep = int(result["ep"])
        dp_attention = parse_bool(result["dp_attention"])
        return {
            "hardware": hw,
            "model": model,
            "framework": framework,
            "precision": precision,
            "spec_method": spec_method,
            "disagg": disagg,
            "is_multinode": False,
            "prefill_tp": tp,
            "prefill_ep": ep,
            "prefill_dp_attention": dp_attention,
            "decode_tp": tp,
            "decode_ep": ep,
            "decode_dp_attention": dp_attention,
        }


# Use LIKE prefix match on model to handle cases where DB model name
# differs from model-prefix (e.g. model-prefix "gptoss" -> DB "gptoss120b")
BASELINE_QUERY = """
    SELECT br.metrics as metrics,
           c.model as db_model
    FROM benchmark_results br
    JOIN configs c ON c.id = br.config_id
    JOIN workflow_runs wr ON wr.id = br.workflow_run_id
    WHERE c.hardware = %(hardware)s
      AND c.framework = %(framework)s
      AND c.model LIKE %(model)s || '%%'
      AND c.precision = %(precision)s
      AND c.spec_method = %(spec_method)s
      AND c.disagg = %(disagg)s
      AND c.is_multinode = %(is_multinode)s
      AND c.prefill_tp = %(prefill_tp)s
      AND c.prefill_ep = %(prefill_ep)s
      AND c.prefill_dp_attention = %(prefill_dp_attention)s
      AND c.decode_tp = %(decode_tp)s
      AND c.decode_ep = %(decode_ep)s
      AND c.decode_dp_attention = %(decode_dp_attention)s
      AND br.isl = %(isl)s
      AND br.osl = %(osl)s
      AND br.conc = %(conc)s
      AND wr.head_branch = 'main'
      AND br.error IS NULL
    ORDER BY br.date DESC
    LIMIT 1
"""

# Metrics to compare: (result_key, header_label, higher_is_better, format_spec)
# TTFT is in seconds in the result JSON, display in ms
# E2EL is in seconds
# Interactivity is in tok/s/user
METRIC_DEFS = [
    # TPUT
    ("tput_per_gpu", "TPUT/GPU", True, ".2f"),
    # TTFT (lower is better, stored in seconds, display in ms — handled specially)
    ("median_ttft", "TTFT Median (ms)", False, ".4f"),
    ("p90_ttft", "TTFT P90 (ms)", False, ".4f"),
    ("p99_ttft", "TTFT P99 (ms)", False, ".4f"),
    ("p99.9_ttft", "TTFT P99.9 (ms)", False, ".4f"),
    # Interactivity (higher is better)
    ("median_intvty", "Intvty Median", True, ".4f"),
    ("p90_intvty", "Intvty@P90 TPOT", True, ".4f"),
    ("p99_intvty", "Intvty@P99 TPOT", True, ".4f"),
    ("p99.9_intvty", "Intvty@P99.9 TPOT", True, ".4f"),
    # E2EL (lower is better, in seconds)
    ("median_e2el", "E2EL Median (s)", False, ".4f"),
    ("p90_e2el", "E2EL P90 (s)", False, ".4f"),
    ("p99_e2el", "E2EL P99 (s)", False, ".4f"),
    ("p99.9_e2el", "E2EL P99.9 (s)", False, ".4f"),
]

# Keys that are stored in seconds but should be displayed in ms
MS_DISPLAY_KEYS = {"median_ttft", "p90_ttft", "p99_ttft", "p99.9_ttft"}


def get_metric_value(data, key):
    """Get a metric value from a result dict, converting to float if present."""
    val = data.get(key)
    if val is None:
        return None
    return float(val)


def format_value(val, key, fmt):
    """Format a metric value for display, converting seconds to ms for TTFT keys."""
    if val is None:
        return "N/A"
    if key in MS_DISPLAY_KEYS:
        val = val * 1000
    return f"{val:{fmt}}"


def compute_metric_delta(current_data, baseline_data, key, higher_is_better, fmt):
    """Compute colored delta string for a metric."""
    current = get_metric_value(current_data, key)
    baseline = get_metric_value(baseline_data, key) if baseline_data else None
    if current is None or baseline is None or baseline == 0:
        return "N/A"
    # For ms-display keys, convert both to ms before computing delta
    if key in MS_DISPLAY_KEYS:
        current_display = current * 1000
        baseline_display = baseline * 1000
    else:
        current_display = current
        baseline_display = baseline
    delta = current_display - baseline_display
    pct = (delta / baseline_display) * 100
    return colorize_delta(delta, pct, higher_is_better, fmt)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    database_url = os.environ["DATABASE_URL"]

    # Load all benchmark result JSONs (files may contain a single dict or a list of dicts)
    results = []
    for path in results_dir.rglob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    print(f"Loaded {len(results)} benchmark results", file=sys.stderr)

    if not results:
        print("No benchmark results found to compare.")
        return

    conn = psycopg2.connect(database_url)
    rows = []
    matched = 0
    unmatched = 0

    for r in results:
        config_params = build_config_params(r)
        query_params = {
            **config_params,
            "isl": int(r["isl"]),
            "osl": int(r["osl"]),
            "conc": int(r["conc"]),
        }

        print(f"\nQuery params: {json.dumps({k: str(v) for k, v in query_params.items()}, indent=2)}", file=sys.stderr)

        with conn.cursor() as cur:
            cur.execute(BASELINE_QUERY, query_params)
            row = cur.fetchone()

        baseline_metrics = None
        if row:
            matched += 1
            baseline_metrics = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            print(f"  -> Matched DB model={row[1]}, tput={baseline_metrics.get('tput_per_gpu')}", file=sys.stderr)
        else:
            unmatched += 1
            print(f"  -> No baseline found", file=sys.stderr)

        is_multinode = r.get("is_multinode", False)
        if is_multinode:
            parallelism = (
                f"P(tp{r['prefill_tp']}/ep{r['prefill_ep']}) "
                f"D(tp{r['decode_tp']}/ep{r['decode_ep']})"
            )
        else:
            parallelism = f"tp{r['tp']}/ep{r['ep']}"

        row_data = {
            "model": r["infmax_model_prefix"],
            "served_model": r["model"],
            "hw": extract_hardware(r["hw"]).upper(),
            "framework": r["framework"].upper(),
            "precision": r["precision"].upper(),
            "parallelism": parallelism,
            "isl": int(r["isl"]),
            "osl": int(r["osl"]),
            "conc": int(r["conc"]),
            "result": r,
            "baseline_metrics": baseline_metrics,
        }
        if not is_multinode:
            row_data["dp_attention"] = r.get("dp_attention", False)
        rows.append(row_data)

    conn.close()

    print(f"\nSummary: {matched} matched, {unmatched} unmatched out of {len(results)} results", file=sys.stderr)

    rows.sort(key=lambda x: (x["model"], x["hw"], x["framework"], x["isl"], x["osl"], x["conc"]))

    single_node = [r for r in rows if "P(" not in r["parallelism"]]
    multi_node = [r for r in rows if "P(" in r["parallelism"]]

    # Build metric headers: for each metric, one column for value and one for delta
    metric_headers = []
    for _, label, _, _ in METRIC_DEFS:
        metric_headers.extend([label, f"{label} Delta"])

    if single_node:
        headers = [
            "Model", "Served Model", "Hardware", "Framework", "Precision",
            "ISL", "OSL", "TP", "EP", "DP Attention", "Conc",
        ] + metric_headers

        table_rows = []
        for row in single_node:
            parts = row["parallelism"]  # "tp1/ep1"
            tp_val = parts.split("/")[0].replace("tp", "")
            ep_val = parts.split("/")[1].replace("ep", "")
            config_cols = [
                row["model"],
                row["served_model"],
                row["hw"],
                row["framework"],
                row["precision"],
                row["isl"],
                row["osl"],
                tp_val,
                ep_val,
                row.get("dp_attention", False),
                row["conc"],
            ]
            metric_cols = []
            for key, _, higher_is_better, fmt in METRIC_DEFS:
                val = get_metric_value(row["result"], key)
                metric_cols.append(format_value(val, key, fmt))
                metric_cols.append(compute_metric_delta(
                    row["result"], row["baseline_metrics"], key, higher_is_better, fmt))
            table_rows.append(config_cols + metric_cols)

        print("## Single-Node Comparison vs. Most Recent\n")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
        print()

    if multi_node:
        headers = [
            "Model", "Served Model", "Hardware", "Framework", "Precision",
            "ISL", "OSL", "Prefill TP", "Prefill EP", "Decode TP", "Decode EP",
            "Conc",
        ] + metric_headers

        table_rows = []
        for row in multi_node:
            # Parse P(tp4/ep4) D(tp8/ep8)
            m = re.match(r"P\(tp(\d+)/ep(\d+)\) D\(tp(\d+)/ep(\d+)\)", row["parallelism"])
            config_cols = [
                row["model"],
                row["served_model"],
                row["hw"],
                row["framework"],
                row["precision"],
                row["isl"],
                row["osl"],
                m.group(1) if m else "",
                m.group(2) if m else "",
                m.group(3) if m else "",
                m.group(4) if m else "",
                row["conc"],
            ]
            metric_cols = []
            for key, _, higher_is_better, fmt in METRIC_DEFS:
                val = get_metric_value(row["result"], key)
                metric_cols.append(format_value(val, key, fmt))
                metric_cols.append(compute_metric_delta(
                    row["result"], row["baseline_metrics"], key, higher_is_better, fmt))
            table_rows.append(config_cols + metric_cols)

        print("## Multi-Node Comparison vs. Most Recent\n")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
