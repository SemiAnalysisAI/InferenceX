#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tabulate import tabulate

# Import shared utilities from summarize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import (
    load_json, MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL,
    TP, EP, CONC, DP_ATTENTION, TASK, EM_STRICT, EM_FLEXIBLE, N_EFF
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
    """Return (lm_eval_json, lighteval_json) if present.
    
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
        elif 'config_general' in data and 'results' in data:
            # lighteval - pick latest if multiple
            if le_path is None or p.stat().st_mtime > le_path.stat().st_mtime:
                le_path = p
                
    return lm_path, le_path


def extract_lm_metrics(json_path: Path) -> Dict[str, Any]:
    """Extract metrics from lm-eval harness result JSON.
    
    Uses explicit structure from the JSON file:
    - Task name from results keys
    - Metric name from configs.metric_list
    - Filter names from configs.filter_list
    - Values from results[task][metric,filter]
    """
    data = load_json(json_path) or {}
    results = data.get('results', {})
    configs = data.get('configs', {})
    
    if not results:
        return {}
        
    # 1. Task: first key from results
    task = next(iter(results.keys()))
    
    # 2. Base metric: from config's metric_list
    metric_list = configs.get(task, {}).get('metric_list', [])
    base_metric = metric_list[0]['metric'] if metric_list else 'exact_match'
    
    # 3. Filters: from config's filter_list
    filter_list = configs.get(task, {}).get('filter_list', [])
    
    strict_val, strict_se = None, None
    flex_val, flex_se = None, None
    
    # Helper to get value/stderr pair for filtered metrics
    def get_val_se(filter_name: str) -> Tuple[Optional[float], Optional[float]]:
        val_key = f"{base_metric},{filter_name}"
        se_key = f"{base_metric}_stderr,{filter_name}"
        return results[task].get(val_key), results[task].get(se_key)

    # Extract metrics based on filter_list
    if not filter_list:
        # No filters - use base metric for strict
        strict_val = results[task].get(base_metric)
        strict_se = results[task].get(f"{base_metric}_stderr")
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
        or configs.get(task, {}).get('metadata', {}).get('model')
    )

    return {
        'task': task,
        'strict': strict_val,
        'strict_se': strict_se,
        'flex': flex_val,
        'flex_se': flex_se,
        'n_eff': n_eff,
        'model': model,
        'source': str(json_path)
    }


def extract_lighteval_metrics(json_path: Path, task_base: Optional[str] = None) -> Dict[str, Any]:
    """Extract metrics from lighteval result JSON."""
    data = load_json(json_path) or {}
    results = data.get('results', {}) or {}
    
    # Find task key
    key = None
    if task_base:
        for k in results.keys():
            if str(k).startswith(task_base):
                key = k
                break
    if key is None:
        key = next(iter(results.keys())) if results else 'unknown'
        
    r = results.get(key, {})
    em = r.get('extractive_match')
    em_se = r.get('extractive_match_stderr')

    cg = data.get('config_general', {}) or {}
    model = cg.get('model_name') or cg.get('model_config', {}).get('model_name', '')

    return {
        'task': key,
        'strict': em,
        'flex': None,
        'strict_se': em_se,
        'flex_se': None,
        'n_eff': None,
        'model': model,
        'source': str(json_path)
    }


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


def main():
    if len(sys.argv) < 3:
        print('Usage: collect_eval_results.py <results_dir> <exp_name>')
        sys.exit(1)

    root = Path(sys.argv[1])
    exp_name = sys.argv[2]

    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / 'meta_env.json') or {}
        lm_path, le_path = detect_eval_jsons(d)
        
        # Extract metrics (prefer lm-eval)
        if lm_path:
            m = extract_lm_metrics(lm_path)
        elif le_path:
            m = extract_lighteval_metrics(le_path)
        else:
            continue

        if not m:
            continue

        # Build row from meta + metrics
        row = {
            'model': m.get('model') or meta.get('model', 'unknown'),
            'hw': meta.get('hw', 'unknown').upper(),
            'framework': meta.get('framework', 'unknown').lower(),
            'precision': meta.get('precision', 'unknown').lower(),
            'isl': int(meta.get('isl', 0)),
            'osl': int(meta.get('osl', 0)),
            'tp': int(meta.get('tp', 1)),
            'ep': int(meta.get('ep', 1)),
            'conc': int(meta.get('conc', 0)),
            'dp_attention': str(meta.get('dp_attention', False)).lower(),
            'task': m.get('task', 'unknown'),
            'em_strict': m.get('strict'),
            'em_strict_se': m.get('strict_se'),
            'em_flexible': m.get('flex'),
            'em_flexible_se': m.get('flex_se'),
            'n_eff': m.get('n_eff'),
            'source': m.get('source'),
        }
        rows.append(row)

    # Sort for stable output
    rows.sort(key=lambda r: (
        r['hw'], r['framework'], r['precision'], r['isl'], r['osl'], r['tp'], r['ep'], r['conc']
    ))

    if not rows:
        print('> No eval results found to summarize.')
    else:
        # Print table using tabulate
        headers = [
            MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL, TP, EP, CONC, DP_ATTENTION, 
            TASK, EM_STRICT, EM_FLEXIBLE, N_EFF
        ]
        
        table_rows = [
            [
                r['model'],
                r['hw'],
                r['framework'].upper(),
                r['precision'].upper(),
                r['isl'],
                r['osl'],
                r['tp'],
                r['ep'],
                r['conc'],
                r['dp_attention'],
                r['task'],
                f"{pct(r['em_strict'])}{se(r['em_strict_se'])}",
                f"{pct(r['em_flexible'])}{se(r['em_flexible_se'])}",
                r['n_eff'] or ''
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
