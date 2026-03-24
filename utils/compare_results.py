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


def colorize_delta(text, delta):
    """Wrap delta text in green (positive) or red (negative) using LaTeX color syntax for GitHub markdown."""
    if delta > 0:
        return f"$\\color{{green}}\\textsf{{{text}}}$"
    elif delta < 0:
        return f"$\\color{{red}}\\textsf{{{text}}}$"
    return text


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
    SELECT br.metrics->>'tput_per_gpu' as tput_per_gpu,
           br.metrics->>'median_intvty' as median_intvty,
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

        if row:
            matched += 1
            print(f"  -> Matched DB model={row[2]}, tput={row[0]}, intvty={row[1]}", file=sys.stderr)
        else:
            unmatched += 1
            print(f"  -> No baseline found", file=sys.stderr)

        current_tput = float(r["tput_per_gpu"])
        baseline_tput = float(row[0]) if row else None

        if baseline_tput is not None and baseline_tput > 0:
            delta = current_tput - baseline_tput
            pct = (delta / baseline_tput) * 100
            tput_delta_str = colorize_delta(f"{delta:+.2f} ({pct:+.1f}%)", delta)
        else:
            tput_delta_str = "N/A (no baseline)"

        current_intvty = float(r["median_intvty"]) if "median_intvty" in r else None
        baseline_intvty = float(row[1]) if row and row[1] else None

        if current_intvty is not None and baseline_intvty is not None and baseline_intvty > 0:
            delta_i = current_intvty - baseline_intvty
            pct_i = (delta_i / baseline_intvty) * 100
            intvty_delta_str = colorize_delta(f"{delta_i:+.4f} ({pct_i:+.1f}%)", delta_i)
        else:
            intvty_delta_str = "N/A (no baseline)"

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
            "current": current_tput,
            "baseline": baseline_tput,
            "tput_delta_str": tput_delta_str,
            "current_intvty": current_intvty,
            "baseline_intvty": baseline_intvty,
            "intvty_delta_str": intvty_delta_str,
        }
        if not is_multinode:
            row_data["dp_attention"] = r.get("dp_attention", False)
        rows.append(row_data)

    conn.close()

    print(f"\nSummary: {matched} matched, {unmatched} unmatched out of {len(results)} results", file=sys.stderr)

    rows.sort(key=lambda x: (x["model"], x["hw"], x["framework"], x["isl"], x["osl"], x["conc"]))

    single_node = [r for r in rows if "P(" not in r["parallelism"]]
    multi_node = [r for r in rows if "P(" in r["parallelism"]]

    if single_node:
        headers = [
            "Model", "Served Model", "Hardware", "Framework", "Precision",
            "ISL", "OSL", "TP", "EP", "DP Attention", "Conc",
            "TPUT per GPU", "Baseline TPUT per GPU", "TPUT Delta",
            "Interactivity", "Baseline Interactivity", "Interactivity Delta",
        ]
        table_rows = []
        for row in single_node:
            parts = row["parallelism"]  # "tp1/ep1"
            tp_val = parts.split("/")[0].replace("tp", "")
            ep_val = parts.split("/")[1].replace("ep", "")
            table_rows.append([
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
                f"{row['current']:.4f}",
                f"{row['baseline']:.4f}" if row["baseline"] is not None else "N/A",
                row["tput_delta_str"],
                f"{row['current_intvty']:.4f}" if row["current_intvty"] is not None else "N/A",
                f"{row['baseline_intvty']:.4f}" if row["baseline_intvty"] is not None else "N/A",
                row["intvty_delta_str"],
            ])

        print("## Single-Node Comparison vs. Most Recent\n")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))
        print()

    if multi_node:
        headers = [
            "Model", "Served Model", "Hardware", "Framework", "Precision",
            "ISL", "OSL", "Prefill TP", "Prefill EP", "Decode TP", "Decode EP",
            "Conc", "TPUT per GPU", "Baseline TPUT per GPU", "TPUT Delta",
            "Interactivity", "Baseline Interactivity", "Interactivity Delta",
        ]
        table_rows = []
        for row in multi_node:
            # Parse P(tp4/ep4) D(tp8/ep8)
            m = re.match(r"P\(tp(\d+)/ep(\d+)\) D\(tp(\d+)/ep(\d+)\)", row["parallelism"])
            table_rows.append([
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
                f"{row['current']:.4f}",
                f"{row['baseline']:.4f}" if row["baseline"] is not None else "N/A",
                row["tput_delta_str"],
                f"{row['current_intvty']:.4f}" if row["current_intvty"] is not None else "N/A",
                f"{row['baseline_intvty']:.4f}" if row["baseline_intvty"] is not None else "N/A",
                row["intvty_delta_str"],
            ])

        print("## Multi-Node Comparison vs. Most Recent\n")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
