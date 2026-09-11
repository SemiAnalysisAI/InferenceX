#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tabulate import tabulate

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infx.results.evals import (
    EVAL_RESULT_FORMAT, as_int, build_row, build_rows, is_eval_result, result_order,
    select_latest_results,
)
from infx.results.evals import result_concurrency as _result_concurrency

MODEL = "Model"
HARDWARE = "Hardware"
FRAMEWORK = "Framework"
PRECISION = "Precision"
ISL = "ISL"
OSL = "OSL"
TP = "TP"
EP = "EP"
DP_ATTENTION = "DP Attention"
CONC = "Conc"
PREFILL_TP = "Prefill TP"
PREFILL_EP = "Prefill EP"
PREFILL_DP_ATTN = "Prefill DP Attn"
PREFILL_WORKERS = "Prefill Workers"
DECODE_TP = "Decode TP"
DECODE_EP = "Decode EP"
DECODE_DP_ATTN = "Decode DP Attn"
DECODE_WORKERS = "Decode Workers"
TASK = "Task"
SCORE = "Score"
EM_STRICT = "EM Strict"
EM_FLEXIBLE = "EM Flexible"
N_EFF = "N (eff)"
SPEC_DECODING = "Spec Decode"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file and return dict, or None on error."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def find_eval_sets(root: Path) -> List[Path]:
    """Return directories that contain a meta_env.json (one set per job).

    Structure: eval_results/<artifact-name>/meta_env.json
    When download-artifact downloads a single artifact, files may be
    extracted flat into root (no subdirectory), so check root itself too.
    """
    out: List[Path] = []
    try:
        # Handle flat structure (single artifact extracted directly into root)
        if (root / 'meta_env.json').exists():
            out.append(root)
        # Handle nested structure (multiple artifacts in subdirectories)
        for d in root.iterdir():
            if d.is_dir() and (d / 'meta_env.json').exists():
                out.append(d)
    except Exception:
        pass
    return out


def result_concurrency(path: Path) -> Optional[int]:
    """Extract a batched eval concurrency from a staged result filename."""
    return _result_concurrency(path.name)


def detect_lm_eval_jsons(d: Path, batched: bool = False) -> List[Path]:
    """Return the latest collector-compatible eval result JSONs.

    Result filenames contain sortable timestamps. Mtime remains a fallback for
    legacy names, with the filename as a deterministic tie-breaker.
    """
    immediate_jsons = set(d.glob('results*.json'))
    immediate_jsons.update(
        p for p in d.glob('*.json') if p.name != 'meta_env.json'
    )
    lm_paths = []

    for p in immediate_jsons:
        data = load_json(p)
        if is_eval_result(data):
            lm_paths.append(p)

    return select_latest_results(lm_paths, batched=batched)


def pct(x: Any) -> str:
    """Format value as percentage."""
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return 'N/A'


def se(x: Any) -> str:
    """Format stderr as percentage with ± prefix."""
    try:
        return f" ±{float(x)*100:.2f}%"
    except Exception:
        return ''


def collect_eval_rows(root: Path) -> List[Dict[str, Any]]:
    """Collect logical eval rows, expanding batched artifacts by concurrency."""
    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / 'meta_env.json') or {}
        batch_concs = meta.get('eval_concs')
        batched = isinstance(batch_concs, list)
        allowed_concs: Optional[set[int]] = None
        if batched:
            completed_concs = meta.get('completed_eval_concs', batch_concs)
            if isinstance(completed_concs, list):
                allowed_concs = {as_int(conc, -1) for conc in completed_concs}

        for lm_path in detect_lm_eval_jsons(d, batched=batched):
            row_meta = meta
            if batched:
                conc = result_concurrency(lm_path)
                if conc is None or (
                    allowed_concs is not None and conc not in allowed_concs
                ):
                    continue
                row_meta = {**meta, 'conc': conc}

            rows.extend(build_rows(load_json(lm_path) or {}, row_meta, source=str(lm_path)))
    return rows


def main():
    if len(sys.argv) < 3:
        print('Usage: collect_eval_results.py <results_dir> <exp_name> [sort_by: model_prefix|hw]')
        sys.exit(1)

    root = Path(sys.argv[1])
    exp_name = sys.argv[2]

    rows = collect_eval_rows(root)

    single_node_rows = [r for r in rows if not r['is_multinode']]
    multinode_rows = [r for r in rows if r['is_multinode']]

    # Sort for stable output (default: by model_prefix)
    sort_by = sys.argv[3] if len(sys.argv) > 3 else 'model_prefix'
    single_node_sort_key = (
        (lambda r: (
            r['hw'], r['framework'], r['precision'], r.get('spec_decoding', ''),
            r['isl'], r['osl'], r['tp'], r['ep'], r['conc'],
        ))
        if sort_by == 'hw'
        else (lambda r: (
            r['model_prefix'], r['hw'], r['framework'], r['precision'],
            r.get('spec_decoding', ''), r['isl'], r['osl'],
            r['tp'], r['ep'], r['conc'],
        ))
    )
    multinode_sort_key = (
        (lambda r: (
            r['hw'], r['framework'], r['precision'], r.get('spec_decoding', ''),
            r['isl'], r['osl'],
            r['prefill_tp'], r['prefill_ep'], r['prefill_num_workers'],
            r['decode_tp'], r['decode_ep'], r['decode_num_workers'], r['conc'],
        ))
        if sort_by == 'hw'
        else (lambda r: (
            r['model_prefix'], r['hw'], r['framework'], r['precision'],
            r.get('spec_decoding', ''), r['isl'], r['osl'],
            r['prefill_tp'], r['prefill_ep'], r['prefill_num_workers'],
            r['decode_tp'], r['decode_ep'], r['decode_num_workers'], r['conc'],
        ))
    )
    single_node_rows.sort(key=single_node_sort_key)
    multinode_rows.sort(key=multinode_sort_key)

    if not rows:
        print('> No eval results found to summarize.')
    else:
        # Print table using tabulate
        MODEL_PREFIX = "Model Prefix"

        if single_node_rows:
            headers = [
                MODEL_PREFIX, HARDWARE, FRAMEWORK, PRECISION, SPEC_DECODING,
                ISL, OSL, TP, EP, CONC, DP_ATTENTION,
                TASK, SCORE, EM_STRICT, EM_FLEXIBLE, N_EFF, MODEL,
            ]
            table_rows = [
                [
                    r['model_prefix'],
                    r['hw'],
                    r['framework'].upper(),
                    r['precision'].upper(),
                    r['spec_decoding'],
                    r['isl'],
                    r['osl'],
                    r['tp'],
                    r['ep'],
                    r['conc'],
                    r['dp_attention'],
                    r['task'],
                    f"{pct(r['score'])}{se(r['score_se'])}",
                    f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                    f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                    r['n_eff'] if r['n_eff'] is not None else '',
                    r['model'],
                ]
                for r in single_node_rows
            ]
            print("### Single-Node Eval Results\n")
            print(tabulate(table_rows, headers=headers, tablefmt="github"))

        if multinode_rows:
            headers = [
                MODEL_PREFIX, HARDWARE, FRAMEWORK, PRECISION, SPEC_DECODING,
                ISL, OSL,
                PREFILL_TP, PREFILL_EP, PREFILL_DP_ATTN, PREFILL_WORKERS,
                DECODE_TP, DECODE_EP, DECODE_DP_ATTN, DECODE_WORKERS,
                CONC, TASK, SCORE, EM_STRICT, EM_FLEXIBLE, N_EFF, MODEL,
            ]
            table_rows = [
                [
                    r['model_prefix'],
                    r['hw'],
                    r['framework'].upper(),
                    r['precision'].upper(),
                    r['spec_decoding'],
                    r['isl'],
                    r['osl'],
                    r['prefill_tp'],
                    r['prefill_ep'],
                    r['prefill_dp_attention'],
                    r['prefill_num_workers'],
                    r['decode_tp'],
                    r['decode_ep'],
                    r['decode_dp_attention'],
                    r['decode_num_workers'],
                    r['conc'],
                    r['task'],
                    f"{pct(r['score'])}{se(r['score_se'])}",
                    f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                    f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                    r['n_eff'] if r['n_eff'] is not None else '',
                    r['model'],
                ]
                for r in multinode_rows
            ]
            if single_node_rows:
                print("\n")
            print("### Multi-Node Eval Results\n")
            print(tabulate(table_rows, headers=headers, tablefmt="github"))


    # Write JSON aggregate
    out_path = Path(f'agg_eval_{exp_name}.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
