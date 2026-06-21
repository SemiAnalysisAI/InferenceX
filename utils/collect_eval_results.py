#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tabulate import tabulate

# Import shared utilities from summarize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import (
    load_json, MODEL, HARDWARE, FRAMEWORK, PRECISION,
    ISL, OSL, TP, EP, CONC, DP_ATTENTION, TASK, SCORE,
    EM_STRICT, EM_FLEXIBLE, N_EFF,
    SPEC_DECODING, PREFILL_TP, PREFILL_EP, PREFILL_DP_ATTN, PREFILL_WORKERS,
    DECODE_TP, DECODE_EP, DECODE_DP_ATTN, DECODE_WORKERS
)

CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")


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
    match = CONC_SUFFIX_RE.search(path.name)
    return int(match.group(1)) if match else None


def valid_concurrency_list(
    value: object,
    *,
    allow_empty: bool = True,
) -> bool:
    """Return whether metadata contains unique positive integer concurrencies."""
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(
            isinstance(conc, int) and not isinstance(conc, bool) and conc > 0
            for conc in value
        )
        and len(set(value)) == len(value)
    )


def detect_lm_eval_jsons(d: Path, batched: bool = False) -> List[Path]:
    """Return lm-eval result JSONs from one artifact directory.

    Legacy artifacts contribute their latest result file. Batched artifacts
    contribute the latest result file for each `_concN` suffix.
    """
    immediate_jsons = set(d.glob('results*.json'))
    immediate_jsons.update(
        p for p in d.glob('*.json') if p.name != 'meta_env.json'
    )
    lm_paths = []

    for p in immediate_jsons:
        data = load_json(p)
        if not isinstance(data, dict):
            continue
        if 'lm_eval_version' in data:
            lm_paths.append(p)

    if not lm_paths:
        return []
    if not batched:
        return [max(lm_paths, key=lambda path: path.stat().st_mtime)]

    latest_by_conc: Dict[int, Path] = {}
    for path in lm_paths:
        conc = result_concurrency(path)
        if conc is None:
            continue
        current = latest_by_conc.get(conc)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest_by_conc[conc] = path
    return [latest_by_conc[conc] for conc in sorted(latest_by_conc)]


def detect_eval_jsons(d: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return the latest legacy lm-eval JSON and deprecated second slot."""
    lm_paths = detect_lm_eval_jsons(d)
    return (lm_paths[0] if lm_paths else None), None


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
    if not isinstance(data, dict):
        return []
    results = data.get('results', {})
    configs = data.get('configs', {})

    if not isinstance(results, dict) or not results:
        return []
    if not isinstance(configs, dict):
        configs = {}

    extracted = []

    for task, task_results in results.items():
        if not isinstance(task_results, dict):
            continue
        task_config = configs.get(task, {})
        if not isinstance(task_config, dict):
            task_config = {}

        # Base metric: from config's metric_list
        metric_list = task_config.get('metric_list', [])
        if (
            isinstance(metric_list, list)
            and metric_list
            and isinstance(metric_list[0], dict)
            and isinstance(metric_list[0].get('metric'), str)
        ):
            base_metric = metric_list[0]['metric']
        else:
            base_metric = 'exact_match'

        # Filters: from config's filter_list
        filter_list = task_config.get('filter_list', [])
        if not isinstance(filter_list, list):
            filter_list = []

        strict_val, strict_se = None, None
        flex_val, flex_se = None, None
        accuracy_val, accuracy_se = None, None

        # Helper to get value/stderr pair for filtered metrics
        def get_val_se(
            filter_name: Optional[str],
        ) -> Tuple[Optional[float], Optional[float]]:
            suffix = f",{filter_name}" if filter_name else ""
            val_key = f"{base_metric}{suffix}"
            se_key = f"{base_metric}_stderr{suffix}"
            return task_results.get(val_key), task_results.get(se_key)

        # Extract metrics based on filter_list
        if not filter_list:
            value, stderr = get_val_se('none')
            if value is None:
                value, stderr = get_val_se(None)
            if base_metric in {'acc', 'accuracy'}:
                accuracy_val, accuracy_se = value, stderr
            else:
                strict_val, strict_se = value, stderr
        else:
            # Extract metrics for each filter
            for filter_config in filter_list:
                if not isinstance(filter_config, dict):
                    continue
                filter_name = filter_config.get('name')
                if not isinstance(filter_name, str):
                    continue
                value, stderr = get_val_se(filter_name)
                normalized_name = filter_name.lower()
                if 'strict' in normalized_name:
                    strict_val, strict_se = value, stderr
                elif (
                    'flex' in normalized_name
                    or 'extract' in normalized_name
                ):
                    flex_val, flex_se = value, stderr
                elif base_metric in {'acc', 'accuracy'}:
                    accuracy_val, accuracy_se = value, stderr
                elif strict_val is None:
                    strict_val, strict_se = value, stderr

        # N-samples (effective count)
        sample_counts = data.get('n-samples', {})
        if not isinstance(sample_counts, dict):
            sample_counts = {}
        task_counts = sample_counts.get(task, {})
        n_eff = (
            task_counts.get('effective')
            if isinstance(task_counts, dict)
            else None
        )

        # Model name
        metadata = task_config.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        model = (
            data.get('model_name')
            or metadata.get('model')
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


def as_int(x: Any, default: int = 0) -> int:
    """Convert a metadata field to int with a fallback."""
    try:
        return int(x)
    except Exception:
        return default


def as_bool(x: Any, default: bool = False) -> bool:
    """Parse a metadata boolean stored as bool/string/int."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return str(x).lower() == 'true'


def build_row(meta: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    """Build a result row from metadata and extracted metrics."""
    is_multinode = as_bool(meta.get('is_multinode'), False)
    prefill_tp = as_int(meta.get('prefill_tp', meta.get('tp', 1)), 1)
    prefill_ep = as_int(meta.get('prefill_ep', meta.get('ep', 1)), 1)
    prefill_num_workers = as_int(meta.get('prefill_num_workers', 1), 1)
    decode_tp = as_int(meta.get('decode_tp', meta.get('tp', 1)), 1)
    decode_ep = as_int(meta.get('decode_ep', meta.get('ep', 1)), 1)
    decode_num_workers = as_int(meta.get('decode_num_workers', 1), 1)
    prefill_dp_attention = meta.get('prefill_dp_attention')
    decode_dp_attention = meta.get('decode_dp_attention')
    dp_attention = meta.get('dp_attention', 'none')

    if prefill_dp_attention is None:
        prefill_dp_attention = dp_attention
    if decode_dp_attention is None:
        decode_dp_attention = dp_attention

    if is_multinode:
        if prefill_dp_attention == decode_dp_attention:
            dp_attention = prefill_dp_attention
        else:
            dp_attention = f"prefill={str(prefill_dp_attention).lower()},decode={str(decode_dp_attention).lower()}"

    row = {
        'is_multinode': is_multinode,
        'model_prefix': meta.get('infmax_model_prefix', 'unknown'),
        'model': m.get('model') or meta.get('model', 'unknown'),
        'hw': meta.get('hw', 'unknown').upper(),
        'framework': meta.get('framework', 'unknown').lower(),
        'precision': meta.get('precision', 'unknown').lower(),
        'spec_decoding': meta.get('spec_decoding', 'unknown'),
        'isl': as_int(meta.get('isl', 0), 0),
        'osl': as_int(meta.get('osl', 0), 0),
        'tp': as_int(meta.get('tp', prefill_tp), prefill_tp),
        'ep': as_int(meta.get('ep', prefill_ep), prefill_ep),
        'prefill_tp': prefill_tp,
        'prefill_ep': prefill_ep,
        'prefill_num_workers': prefill_num_workers,
        'decode_tp': decode_tp,
        'decode_ep': decode_ep,
        'decode_num_workers': decode_num_workers,
        'conc': as_int(meta.get('conc', 0), 0),
        'dp_attention': str(dp_attention).lower(),
        'prefill_dp_attention': str(prefill_dp_attention).lower(),
        'decode_dp_attention': str(decode_dp_attention).lower(),
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
    elif m.get('flex') is not None:
        row['score'] = m.get('flex')
        row['score_name'] = 'em_flexible'
        row['score_se'] = m.get('flex_se')
    elif m.get('accuracy') is not None:
        row['score'] = m.get('accuracy')
        row['score_name'] = 'accuracy'
        row['score_se'] = m.get('accuracy_se')
    else:
        row['score'] = None
        row['score_name'] = None
        row['score_se'] = None

    return row


def collect_eval_rows(root: Path) -> List[Dict[str, Any]]:
    """Collect logical eval rows, expanding batched artifacts by concurrency."""
    rows: List[Dict[str, Any]] = []
    for d in find_eval_sets(root):
        meta = load_json(d / 'meta_env.json') or {}
        eval_exit_code = meta.get('eval_exit_code')
        if (
            not isinstance(eval_exit_code, int)
            or isinstance(eval_exit_code, bool)
            or eval_exit_code != 0
        ):
            continue

        batch_concs = meta.get('eval_concs')
        if 'eval_concs' in meta and not valid_concurrency_list(
            batch_concs,
            allow_empty=False,
        ):
            continue
        batched = valid_concurrency_list(batch_concs, allow_empty=False)
        allowed_concs: Optional[set[int]] = None
        if batched:
            completed_concs = meta.get('completed_eval_concs')
            failed_concs = meta.get('failed_eval_concs')
            if not (
                valid_concurrency_list(completed_concs)
                and valid_concurrency_list(failed_concs)
            ):
                continue
            expected_set = set(batch_concs)
            completed_set = set(completed_concs)
            failed_set = set(failed_concs)
            if (
                completed_set != expected_set
                or failed_set
                or completed_set & failed_set
            ):
                continue
            allowed_concs = completed_set

        lm_paths = detect_lm_eval_jsons(d, batched=batched)
        if batched and {
            result_concurrency(path) for path in lm_paths
        } != allowed_concs:
            continue

        for lm_path in lm_paths:
            row_meta = meta
            if batched:
                conc = result_concurrency(lm_path)
                if conc is None or (
                    allowed_concs is not None and conc not in allowed_concs
                ):
                    continue
                row_meta = {**meta, 'conc': conc}

            metrics_list = extract_lm_metrics(lm_path)
            for metrics in metrics_list:
                rows.append(build_row(row_meta, metrics))
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
                    r['n_eff'] or '',
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
                    r['n_eff'] or '',
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
