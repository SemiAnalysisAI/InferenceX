#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_eval_sets(root: Path) -> List[Path]:
    """Return directories that contain a meta_env.json (one set per job).

    New structure: each downloaded artifact is placed under
    eval_results/<artifact-name>/ with flat files inside, e.g.:
      - meta_env.json
      - SUMMARY.md
      - results_*.json

    We first check immediate child directories for meta_env.json to avoid
    descending unnecessarily. If nothing is found (backward compatibility),
    fall back to recursive search.
    """
    out: List[Path] = []
    # Prefer immediate children (one directory per artifact)
    try:
        for d in root.iterdir():
            if d.is_dir() and (d / 'meta_env.json').exists():
                out.append(d)
    except Exception:
        pass
    if out:
        return out
    # Fallback: recursive (legacy structure)
    for p in root.rglob('meta_env.json'):
        out.append(p.parent)
    return out


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def detect_eval_jsons(d: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (lm_eval_json, lighteval_json) if present (latest by mtime).

    New structure places result JSONs flat in the artifact directory. We
    first check only the immediate directory for JSONs, then fall back to
    recursive search for backward compatibility.
    """
    def scan_jsons(paths: List[Path]) -> Tuple[List[Tuple[float, Path]], List[Tuple[float, Path]]]:
        lm: List[Tuple[float, Path]] = []
        le: List[Tuple[float, Path]] = []
        for p in paths:
            if p.name == 'meta_env.json':
                continue
            data = load_json(p)
            if not isinstance(data, dict):
                continue
            # Heuristics similar to utils/lm_eval_to_md.py
            if 'lm_eval_version' in data or 'pretty_env_info' in data:
                try:
                    lm.append((p.stat().st_mtime, p))
                except Exception:
                    lm.append((0, p))
            elif 'config_general' in data and 'results' in data:
                try:
                    le.append((p.stat().st_mtime, p))
                except Exception:
                    le.append((0, p))
            elif 'results' in data:
                # Fallback: treat as lm-eval JSON
                try:
                    lm.append((p.stat().st_mtime, p))
                except Exception:
                    lm.append((0, p))
        return lm, le

    # 1) Prefer immediate JSONs (flat structure)
    immediate_jsons = list(d.glob('results*.json')) + [p for p in d.glob('*.json') if p.name != 'meta_env.json']
    lm, le = scan_jsons(immediate_jsons)

    # 2) If nothing found, fallback to deep scan (legacy)
    if not lm and not le:
        deep_jsons = list(d.rglob('*.json'))
        lm, le = scan_jsons(deep_jsons)

    lm_path = sorted(lm, key=lambda x: x[0])[-1][1] if lm else None
    le_path = sorted(le, key=lambda x: x[0])[-1][1] if le else None
    return lm_path, le_path


def parse_pretty_env(pretty_env: str) -> str:
    try:
        lines = [l for l in pretty_env.splitlines() if l.startswith('GPU ')]
        names = [l.split(':', 1)[1].strip() for l in lines]
        if not names:
            return 'Unknown GPU'
        # Compress identical names (roughly)
        from collections import Counter
        c = Counter(names)
        return ' + '.join([f"{n}× {name}" for name, n in c.items()])
    except Exception:
        return 'Unknown GPU'


def extract_lm_metrics(json_path: Path, task: Optional[str] = None) -> Dict[str, Any]:
    data = load_json(json_path) or {}
    results = data.get('results') or {}
    # Pick task
    t = task
    if not t:
        if isinstance(results, dict) and results:
            t = next(iter(results.keys()))
        else:
            t = 'unknown'

    res = results.get(t, {}) if isinstance(results, dict) else {}
    strict = res.get('exact_match,strict-match')
    flex = res.get('exact_match,flexible-extract')
    strict_se = res.get('exact_match_stderr,strict-match')
    flex_se = res.get('exact_match_stderr,flexible-extract')

    n_eff = None
    ns = data.get('n-samples') or data.get('n_samples') or {}
    if isinstance(ns, dict):
        td = ns.get(t) or {}
        if isinstance(td, dict):
            n_eff = td.get('effective') or td.get('n_eff')

    hardware = 'Unknown GPU'
    pe = data.get('pretty_env_info')
    if isinstance(pe, str) and pe:
        hardware = parse_pretty_env(pe)

    model = (
        data.get('model_name')
        or (data.get('configs', {}).get(t, {}) or {}).get('metadata', {}).get('model')
        or (data.get('config') or {}).get('model')
        or ''
    )

    return {
        'task': t,
        'strict': strict,
        'flex': flex,
        'strict_se': strict_se,
        'flex_se': flex_se,
        'n_eff': n_eff,
        'hardware': hardware,
        'model': model,
        'source': str(json_path)
    }


def extract_lighteval_metrics(json_path: Path, task_base: Optional[str] = None) -> Dict[str, Any]:
    data = load_json(json_path) or {}
    results = data.get('results', {}) or {}
    # Choose a task key starting with task_base if provided, else 'all', else first key
    key = None
    if task_base:
        for k in results.keys():
            if str(k).startswith(task_base):
                key = k
                break
    if key is None:
        key = 'all' if 'all' in results else (next(iter(results.keys())) if results else 'unknown')
    r = results.get(key, {}) if isinstance(results, dict) else {}
    em = r.get('extractive_match')
    em_se = r.get('extractive_match_stderr')

    model = ''
    cg = data.get('config_general', {}) or {}
    model = cg.get('model_name') or cg.get('model_config', {}).get('model_name', '')

    return {
        'task': key,
        'strict': em,
        'flex': None,
        'strict_se': em_se,
        'flex_se': None,
        'n_eff': None,
        'hardware': 'Unknown GPU',
        'model': model,
        'source': str(json_path)
    }


def pct(x: Any) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return 'N/A'


def se(x: Any) -> str:
    try:
        return f" ±{float(x)*100:.2f}%"
    except Exception:
        return ''


def main():
    if len(sys.argv) < 3:
        print('Usage: collect_eval_results.py <results_dir> <exp_name>')
        sys.exit(1)

    root = Path(sys.argv[1])
    exp_name = sys.argv[2] or 'all'

    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / 'meta_env.json') or {}
        lm_path, le_path = detect_eval_jsons(d)
        # Prefer lm-eval when available, else lighteval
        if lm_path:
            m = extract_lm_metrics(lm_path)
        elif le_path:
            m = extract_lighteval_metrics(le_path)
        else:
            continue

        # Merge with meta
        row = {
            'model': m.get('model') or meta.get('model') or 'unknown',
            'hw': m.get('hardware', 'Unknown GPU'),
            'framework': (meta.get('framework') or 'unknown').lower(),
            'precision': (meta.get('precision') or 'unknown').lower(),
            'tp': int(meta.get('tp') or 1),
            'ep': int(meta.get('ep') or 1),
            'dp_attention': str(meta.get('dp_attention') or 'false'),
            'task': m.get('task') or 'unknown',
            'em_strict': m.get('strict'),
            'em_strict_se': m.get('strict_se'),
            'em_flexible': m.get('flex'),
            'em_flexible_se': m.get('flex_se'),
            'n_eff': m.get('n_eff'),
            'source': m.get('source'),
        }
        rows.append(row)

    # Sort for stable output
    rows.sort(key=lambda r: (r.get('model',''), r.get('hw',''), r.get('framework',''), r.get('precision',''), r.get('tp',0), r.get('ep',0)))

    if not rows:
        print('> No eval results found to summarize.')
    else:
        # Print Markdown summary table
        print('| Model | Hardware | Framework | Precision | TP | EP | DPA | Task | EM Strict | EM Flexible | N (eff) |')
        print('| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |')
        for r in rows:
            print(
                f"| {r['model']} "
                f"| {r['hw']} "
                f"| {r['framework'].upper()} "
                f"| {r['precision'].upper()} "
                f"| {r['tp']} "
                f"| {r['ep']} "
                f"| {r['dp_attention']} "
                f"| {r['task']} "
                f"| {pct(r['em_strict'])}{se(r['em_strict_se'])} "
                f"| {pct(r['em_flexible'])}{se(r['em_flexible_se'])} "
                f"| {r['n_eff'] or ''} |"
            )

    # Write JSON aggregate
    out_path = Path(f'agg_eval_{exp_name}.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
