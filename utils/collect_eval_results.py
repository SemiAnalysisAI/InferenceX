#!/usr/bin/env python3
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate

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

CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")
EVAL_RESULT_FORMAT = "inferencex-eval-v1"


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
    match = CONC_SUFFIX_RE.search(path.name)
    return int(match.group(1)) if match else None


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

    def recency_key(path: Path) -> Tuple[int, str]:
        match = re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?",
            path.name,
        )
        if match:
            try:
                timestamp = match.group(0)
                base, separator, fraction = timestamp.partition(".")
                parsed = datetime.strptime(
                    base,
                    "%Y-%m-%dT%H-%M-%S",
                ).replace(tzinfo=timezone.utc)
                fractional_ns = (
                    int((fraction + "000000000")[:9])
                    if separator
                    else 0
                )
                order_ns = (
                    int(parsed.timestamp()) * 1_000_000_000
                    + fractional_ns
                )
            except ValueError:
                order_ns = path.stat().st_mtime_ns
        else:
            order_ns = path.stat().st_mtime_ns
        return order_ns, path.name

    for p in immediate_jsons:
        data = load_json(p)
        if not isinstance(data, dict):
            continue
        if data.get('result_format') == EVAL_RESULT_FORMAT or 'lm_eval_version' in data:
            lm_paths.append(p)

    if not lm_paths:
        return []
    if not batched:
        return [max(lm_paths, key=recency_key)]

    latest_by_conc: Dict[int, Path] = {}
    for path in lm_paths:
        conc = result_concurrency(path)
        if conc is None:
            continue
        current = latest_by_conc.get(conc)
        if current is None or recency_key(path) > recency_key(current):
            latest_by_conc[conc] = path
    return [latest_by_conc[conc] for conc in sorted(latest_by_conc)]


def has_invalid_effective_count(data: Dict[str, Any], task: str) -> bool:
    """Return whether a task has an explicitly invalid effective count."""
    if 'n-samples' not in data:
        return False
    sample_counts = data['n-samples']
    if not isinstance(sample_counts, dict) or task not in sample_counts:
        return True
    task_samples = sample_counts[task]
    if not isinstance(task_samples, dict) or 'effective' not in task_samples:
        return True
    effective = task_samples['effective']
    return (
        isinstance(effective, bool)
        or not isinstance(effective, (int, float))
        or not math.isfinite(effective)
        or effective <= 0
    )


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
    raw_configs = data.get('configs', {})
    configs = raw_configs if isinstance(raw_configs, dict) else {}

    if not isinstance(results, dict) or not results:
        return []

    extracted = []

    for task in results.keys():
        task_results = results[task]
        raw_task_config = configs.get(task, {})
        task_config = (
            raw_task_config if isinstance(raw_task_config, dict) else {}
        )
        raw_metadata = task_config.get('metadata', {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        model = data.get('model_name') or metadata.get('model')

        sample_counts = data.get('n-samples')
        raw_task_samples = (
            sample_counts.get(task)
            if isinstance(sample_counts, dict)
            else None
        )
        n_eff = (
            raw_task_samples.get('effective')
            if isinstance(raw_task_samples, dict)
            else None
        )
        invalid_effective_count = has_invalid_effective_count(data, task)
        integration_error = data.get('integration_error')
        if integration_error is None and invalid_effective_count:
            integration_error = {
                'type': 'InvalidEffectiveSampleCount',
                'message': f'invalid effective sample count: {n_eff!r}',
            }
        if integration_error is None and not isinstance(task_results, dict):
            integration_error = {
                'type': 'InvalidTaskResults',
                'message': f'invalid task results for {task!r}',
            }
        if integration_error is not None:
            if not isinstance(integration_error, dict):
                integration_error = {
                    'type': 'IntegrationError',
                    'message': str(integration_error),
                }
            extracted.append({
                'task': task,
                'strict': None,
                'strict_se': None,
                'flex': None,
                'flex_se': None,
                'accuracy': None,
                'accuracy_se': None,
                'n_eff': 0,
                'model': model,
                'source': str(json_path),
                'infrastructure_success': False,
                'integration_error': integration_error,
            })
            continue

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
                # SWE-bench uses resolved rate as its primary score.
                if 'strict' in fname or 'resolved' in fname:
                    strict_val, strict_se = get_val_se(fname)
                elif base_metric == 'acc' and fname == 'none':
                    accuracy_val, accuracy_se = get_val_se(fname)
                elif 'flex' in fname or 'extract' in fname:
                    flex_val, flex_se = get_val_se(fname)

        # N-samples (effective count)
        n_eff = data.get('n-samples', {}).get(task, {}).get('effective')

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
            'source': str(json_path),
            'infrastructure_success': True,
            'integration_error': None,
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
        'infrastructure_success': m.get('infrastructure_success', True),
        'integration_error': m.get('integration_error'),
    }

    if 'eval_suite' in meta:
        row['eval_suite'] = meta['eval_suite']

    # Add universal score field (primary metric for unified comparison)
    if m.get('strict') is not None:
        row['score'] = m.get('strict')
        row['score_name'] = 'em_strict'
        row['score_se'] = m.get('strict_se')
    elif m.get('accuracy') is not None:
        row['score'] = m.get('accuracy')
        row['score_name'] = 'accuracy'
        row['score_se'] = m.get('accuracy_se')
    elif m.get('flex') is not None:
        row['score'] = m.get('flex')
        row['score_name'] = 'em_flexible'
        row['score_se'] = m.get('flex_se')
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

            metrics_list = extract_lm_metrics(lm_path)
            for metrics in metrics_list:
                if metrics.get('infrastructure_success') is False:
                    rows.append(build_row(row_meta, metrics))
                    continue
                primary_score = next(
                    (
                        metrics.get(name)
                        for name in ('strict', 'accuracy', 'flex')
                        if metrics.get(name) is not None
                    ),
                    None,
                )
                invalid_primary_score = (
                    isinstance(primary_score, bool)
                    or not isinstance(primary_score, (int, float))
                    or not math.isfinite(primary_score)
                    or not 0.0 <= primary_score <= 1.0
                )
                if invalid_primary_score:
                    failed_metrics = {
                        **metrics,
                        'strict': None,
                        'strict_se': None,
                        'accuracy': None,
                        'accuracy_se': None,
                        'flex': None,
                        'flex_se': None,
                        'infrastructure_success': False,
                        'integration_error': {
                            'type': 'InvalidPrimaryScore',
                            'message': (
                                f'invalid primary score: {primary_score!r}'
                            ),
                        },
                    }
                    rows.append(build_row(row_meta, failed_metrics))
                    continue
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
