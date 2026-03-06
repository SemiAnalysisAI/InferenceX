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


BASELINE_QUERY = """
    SELECT br.metrics->>'tput_per_gpu' as tput_per_gpu
    FROM benchmark_results br
    JOIN configs c ON c.id = br.config_id
    JOIN workflow_runs wr ON wr.id = br.workflow_run_id
    WHERE c.hardware = %(hardware)s
      AND c.framework = %(framework)s
      AND c.model = %(model)s
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

    if not results:
        print("No benchmark results found to compare.")
        return

    conn = psycopg2.connect(database_url)
    rows = []

    for r in results:
        config_params = build_config_params(r)
        query_params = {
            **config_params,
            "isl": int(r["isl"]),
            "osl": int(r["osl"]),
            "conc": int(r["conc"]),
        }

        with conn.cursor() as cur:
            cur.execute(BASELINE_QUERY, query_params)
            row = cur.fetchone()

        current_tput = float(r["tput_per_gpu"])
        baseline_tput = float(row[0]) if row else None

        if baseline_tput is not None and baseline_tput > 0:
            delta = current_tput - baseline_tput
            pct = (delta / baseline_tput) * 100
            delta_str = f"{delta:+.2f} ({pct:+.1f}%)"
        else:
            delta_str = "N/A (no baseline)"

        is_multinode = r.get("is_multinode", False)
        if is_multinode:
            parallelism = (
                f"P(tp{r['prefill_tp']}/ep{r['prefill_ep']}) "
                f"D(tp{r['decode_tp']}/ep{r['decode_ep']})"
            )
        else:
            parallelism = f"tp{r['tp']}/ep{r['ep']}"

        rows.append({
            "model": r["infmax_model_prefix"],
            "hw": extract_hardware(r["hw"]).upper(),
            "framework": r["framework"],
            "precision": r["precision"],
            "parallelism": parallelism,
            "isl": int(r["isl"]),
            "osl": int(r["osl"]),
            "conc": int(r["conc"]),
            "current": current_tput,
            "baseline": baseline_tput,
            "delta_str": delta_str,
        })

    conn.close()

    rows.sort(key=lambda x: (x["model"], x["hw"], x["framework"], x["isl"], x["osl"], x["conc"]))

    headers = [
        "Model", "HW", "Framework", "Precision", "Parallelism",
        "ISL", "OSL", "Conc",
        "Current (tok/s/gpu)", "Baseline (tok/s/gpu)", "Delta",
    ]

    table_rows = [
        [
            row["model"],
            row["hw"],
            row["framework"],
            row["precision"],
            row["parallelism"],
            row["isl"],
            row["osl"],
            row["conc"],
            f"{row['current']:.2f}",
            f"{row['baseline']:.2f}" if row["baseline"] is not None else "N/A",
            row["delta_str"],
        ]
        for row in rows
    ]

    print("## Throughput Comparison vs. Most Recent\n")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
