#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tabulate import tabulate

# Import shared utilities from summarize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import (
    load_json, MODEL, HARDWARE, FRAMEWORK, PRECISION,
    TP, EP, CONC, DP_ATTENTION, TASK, SCORE, EM_STRICT, EM_FLEXIBLE, N_EFF,
    SPEC_DECODING
)


def find_eval_sets(root: Path) -> List[Path]:
    """Return directories that contain a meta_env.json (one set per job).
    
    Structure: eval_results/<artifact-name>/meta_env.json
    """
    out: List[Path] = []
    try:
        for d in root.iterdir():
            if d.is_dir() and (d / 'meta_env.json').exists():
                out.append(d)
    except Exception:
        pass
    return out


def detect_eval_jsons(d: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (lm_eval_json) if present.
    
    Checks immediate directory for result JSONs.
    """
    immediate_jsons = list(d.glob('results*.json')) + [
        p for p in d.glob('*.json') if p.name != 'meta_env.json'
    ]
    
    lm_path = None
    le_path = None
    
    for p in immediate_jsons:
        data = load_json(p)
        if not isinstance(data, dict):
            continue
            
        if 'lm_eval_version' in data:
            # lm-eval harness - pick latest if multiple
            if lm_path is None or p.stat().st_mtime > lm_path.stat().st_mtime:
                lm_path = p
                
    return lm_path, le_path


def extract_lm_metrics(json_path: Path) -> List[Dict[str, Any]]:
    """Extract metrics from lm-eval harness result JSON.

    Returns a list of metric dicts, one per task in the results.

    Uses explicit structure from the JSON file:
    - Task names from results keys
    - Metric name from configs.metric_list
    - Filter names from configs.filter_list
    - Values from results[task][metric,filter]
    """
    data = load_json(json_path) or {}
    results = data.get('results', {})
    configs = data.get('configs', {})

    if not results:
        return []

    extracted = []

    for task in results.keys():
        task_results = results[task]
        task_config = configs.get(task, {})

        # Base metric: from config's metric_list
        metric_list = task_config.get('metric_list', [])
        base_metric = metric_list[0]['metric'] if metric_list else 'exact_match'

        # Filters: from config's filter_list
        filter_list = task_config.get('filter_list', [])

        strict_val, strict_se = None, None
        flex_val, flex_se = None, None
        accuracy_val, accuracy_se = None, None

        # Helper to get value/stderr pair for filtered metrics
        def get_val_se(filter_name: str) -> Tuple[Optional[float], Optional[float]]:
            val_key = f"{base_metric},{filter_name}"
            se_key = f"{base_metric}_stderr,{filter_name}"
            return task_results.get(val_key), task_results.get(se_key)

        # Extract metrics based on filter_list
        if not filter_list:
            # No filters - check for accuracy or use base metric
            if 'acc' in task_results:
                accuracy_val = task_results.get('acc')
                accuracy_se = task_results.get('acc_stderr')
            else:
                strict_val = task_results.get(base_metric)
                strict_se = task_results.get(f"{base_metric}_stderr")
        else:
            # Extract metrics for each filter
            for f in filter_list:
                fname = f['name']
                if 'strict' in fname:
                    strict_val, strict_se = get_val_se(fname)
                elif 'flex' in fname or 'extract' in fname:
                    flex_val, flex_se = get_val_se(fname)

        # N-samples (effective count)
        n_eff = data.get('n-samples', {}).get(task, {}).get('effective')

        # Model name
        model = (
            data.get('model_name')
            or task_config.get('metadata', {}).get('model')
        )

        extracted.append({
            'task': task,
            'strict': strict_val,
            'strict_se': strict_se,
            'flex': flex_val,
            'flex_se': flex_se,
            'accuracy': accuracy_val,
            'accuracy_se': accuracy_se,
            'n_eff': n_eff,
            'model': model,
            'source': str(json_path)
        })

    return extracted


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


def build_row(meta: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    """Build a result row from metadata and extracted metrics."""
    row = {
        'model_prefix': meta.get('infmax_model_prefix', 'unknown'),
        'model': m.get('model') or meta.get('model', 'unknown'),
        'hw': meta.get('hw', 'unknown').upper(),
        'framework': meta.get('framework', 'unknown').lower(),
        'precision': meta.get('precision', 'unknown').lower(),
        'spec_decoding': meta.get('spec_decoding', 'unknown'),
        'tp': int(meta.get('tp', 1)),
        'ep': int(meta.get('ep', 1)),
        'conc': int(meta.get('conc', 0)),
        'dp_attention': str(meta.get('dp_attention', "none")).lower(),
        'task': m.get('task', 'unknown'),
        'em_strict': m.get('strict'),
        'em_strict_se': m.get('strict_se'),
        'em_flexible': m.get('flex'),
        'em_flexible_se': m.get('flex_se'),
        'n_eff': m.get('n_eff'),
        'source': m.get('source'),
    }

    # Add universal score field (primary metric for unified comparison)
    if m.get('strict') is not None:
        row['score'] = m.get('strict')
        row['score_name'] = 'em_strict'
        row['score_se'] = m.get('strict_se')
    elif m.get('accuracy') is not None:
        row['score'] = m.get('accuracy')
        row['score_name'] = 'accuracy'
        row['score_se'] = m.get('accuracy_se')
    else:
        row['score'] = None
        row['score_name'] = None
        row['score_se'] = None

    return row


def main():
    if len(sys.argv) < 3:
        print('Usage: collect_eval_results.py <results_dir> <exp_name> [sort_by: model_prefix|hw]')
        sys.exit(1)

    root = Path(sys.argv[1])
    exp_name = sys.argv[2]

    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / 'meta_env.json') or {}
        lm_path, le_path = detect_eval_jsons(d)

        # Extract metrics (prefer lm-eval) - returns list for multi-task support
        if lm_path:
            metrics_list = extract_lm_metrics(lm_path)
        else:
            continue

        if not metrics_list:
            continue

        # Build row for each task in the results
        for m in metrics_list:
            row = build_row(meta, m)
            rows.append(row)

    # Sort for stable output (default: by model_prefix)
    sort_by = sys.argv[3] if len(sys.argv) > 3 else 'model_prefix'
    if sort_by == 'hw':
        rows.sort(key=lambda r: (
            r['hw'], r['framework'], r['precision'], r.get('spec_decoding', ''), r['tp'], r['ep'], r['conc']
        ))
    else:
        rows.sort(key=lambda r: (
            r['model_prefix'], r['hw'], r['framework'], r['precision'], r.get('spec_decoding', ''), r['tp'], r['ep'], r['conc']
        ))

    if not rows:
        print('> No eval results found to summarize.')
    else:
        # Print table using tabulate
        MODEL_PREFIX = "Model Prefix"
        headers = [
            MODEL_PREFIX, HARDWARE, FRAMEWORK, PRECISION, SPEC_DECODING, TP, EP, CONC, DP_ATTENTION,
            TASK, SCORE, EM_STRICT, EM_FLEXIBLE, N_EFF, MODEL
        ]

        table_rows = [
            [
                r['model_prefix'],
                r['hw'],
                r['framework'].upper(),
                r['precision'].upper(),
                r['spec_decoding'],
                r['tp'],
                r['ep'],
                r['conc'],
                r['dp_attention'],
                r['task'],
                f"{pct(r['score'])}{se(r['score_se'])}",
                f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                r['n_eff'] or '',
                r['model']
            ]
            for r in rows
        ]

        print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # Write JSON aggregate
    out_path = Path(f'agg_eval_{exp_name}.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
