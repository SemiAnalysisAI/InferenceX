"""Eval row builders and shared rules for offline artifact readers."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_RESULT_FORMAT = "inferencex-eval-v1"
_CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")


def is_eval_result(data: object) -> bool:
    """Recognize an eval format marker without validating its metrics."""
    return isinstance(data, dict) and (
        "lm_eval_version" in data
        or data.get("result_format") == EVAL_RESULT_FORMAT
    )


def result_concurrency(name: str) -> int | None:
    """Read a trailing ``_concN`` with an optional numeric staging suffix."""
    match = _CONC_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else None


def result_order(path: Path) -> tuple[int, str]:
    """Order by filename time or legacy mtime, then name to break ties.

    Both timestamps use UTC epoch nanoseconds. Invalid filename dates fall
    back to mtime, and subnanosecond digits are truncated.
    """
    match = _TIMESTAMP_RE.search(path.name)
    if match:
        try:
            base, separator, fraction = match.group(0).partition(".")
            parsed = datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
            delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
            fractional_ns = int((fraction + "000000000")[:9]) if separator else 0
            return (
                delta.days * 86_400_000_000_000
                + delta.seconds * 1_000_000_000
                + fractional_ns,
                path.name,
            )
        except ValueError:
            pass
    return path.stat().st_mtime_ns, path.name


def select_latest_result(
    paths: Iterable[Path], *, concurrency: int | None = None,
) -> Path | None:
    """Select from recognized candidates, optionally for one concurrency.

    Return None when no candidate matches. Discovery and result-content
    validation belong to the caller; legacy ordering may read file mtimes.
    """
    candidates = (
        path for path in paths
        if concurrency is None or result_concurrency(path.name) == concurrency
    )
    return max(candidates, key=result_order, default=None)


def select_latest_results(paths: Iterable[Path], *, batched: bool = False) -> list[Path]:
    """Select one result, or one per suffixed concurrency in numeric order."""
    if not batched:
        latest = select_latest_result(paths)
        return [latest] if latest is not None else []

    latest_by_conc: dict[int, Path] = {}
    for path in paths:
        conc = result_concurrency(path.name)
        if conc is None:
            continue
        current = latest_by_conc.get(conc)
        if current is None or result_order(path) > result_order(current):
            latest_by_conc[conc] = path
    return [latest_by_conc[conc] for conc in sorted(latest_by_conc)]


_SCORE_NAMES = {"strict": "em_strict", "accuracy": "accuracy", "flex": "em_flexible"}


def is_valid_score(value: object) -> bool:
    """Accept finite numeric scores in [0, 1], excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def is_valid_effective_count(value: object) -> bool:
    """Accept positive finite sample counts, including fractional counts."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def metric_family(name: str) -> str | None:
    """Classify a filter name or metric key; strict/resolved takes precedence."""
    if "strict" in name or "resolved" in name:
        return "strict"
    if "flex" in name or "extract" in name:
        return "flex"
    return None


def _primary_metric(metrics: dict[str, Any]) -> str | None:
    return next((name for name in _SCORE_NAMES if metrics.get(name) is not None), None)


def extract_metrics(data: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    """Extract collector metrics from loaded JSON without I/O or input mutation.

    Configured filters use the last value in each family. Missing sample counts
    remain supported; invalid counts and integration failures produce failed
    metrics. Malformed metric/filter configurations retain their existing errors.
    """
    results = data.get('results', {})
    raw_configs = data.get('configs', {})
    configs = raw_configs if isinstance(raw_configs, dict) else {}
    if not isinstance(results, dict) or not results:
        return []

    extracted = []
    for task, task_results in results.items():
        raw_task_config = configs.get(task, {})
        task_config = raw_task_config if isinstance(raw_task_config, dict) else {}
        raw_metadata = task_config.get('metadata', {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        model = data.get('model_name') or metadata.get('model')
        sample_counts = data.get('n-samples')
        task_samples = sample_counts.get(task) if isinstance(sample_counts, dict) else None
        n_eff = task_samples.get('effective') if isinstance(task_samples, dict) else None

        invalid_count = 'n-samples' in data and not is_valid_effective_count(n_eff)
        integration_error = data.get('integration_error')
        if integration_error is None and invalid_count:
            integration_error = {
                'type': 'InvalidEffectiveSampleCount',
                'message': f'invalid effective sample count: {n_eff!r}',
            }
        if integration_error is None and not isinstance(task_results, dict):
            integration_error = {
                'type': 'InvalidTaskResults',
                'message': f'invalid task results for {task!r}',
            }
        metrics = {
            'task': task,
            'strict': None,
            'strict_se': None,
            'flex': None,
            'flex_se': None,
            'accuracy': None,
            'accuracy_se': None,
            'n_eff': n_eff,
            'model': model,
            'source': source,
            'infrastructure_success': integration_error is None,
            'integration_error': integration_error,
        }
        if integration_error is not None:
            if not isinstance(integration_error, dict):
                metrics['integration_error'] = {
                    'type': 'IntegrationError', 'message': str(integration_error),
                }
            metrics['n_eff'] = 0
        else:
            metric_list = task_config.get('metric_list', [])
            base_metric = metric_list[0]['metric'] if metric_list else 'exact_match'
            filter_list = task_config.get('filter_list', [])
            if not filter_list:
                metric = 'acc' if 'acc' in task_results else base_metric
                family = 'accuracy' if 'acc' in task_results else 'strict'
                metrics[family] = task_results.get(metric)
                metrics[f'{family}_se'] = task_results.get(f'{metric}_stderr')
            else:
                for filter_config in filter_list:
                    name = filter_config['name']
                    family = metric_family(name)
                    if base_metric == 'acc' and name == 'none':
                        family = 'accuracy'
                    if family is not None:
                        metrics[family] = task_results.get(f'{base_metric},{name}')
                        metrics[f'{family}_se'] = task_results.get(f'{base_metric}_stderr,{name}')
        extracted.append(metrics)
    return extracted


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


def build_row(meta: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
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

    primary = _primary_metric(m)
    row['score'] = m[primary] if primary is not None else None
    row['score_name'] = _SCORE_NAMES.get(primary)
    row['score_se'] = m.get(f'{primary}_se') if primary is not None else None

    return row


def build_rows(
    data: dict[str, Any], meta: dict[str, Any], *, source: str,
) -> list[dict[str, Any]]:
    """Build collector rows from loaded result/metadata mappings without I/O.

    Primary scores prefer strict, accuracy, then flexible metrics. An invalid
    primary produces a failed row rather than falling back to a secondary score.
    Inputs are not modified; discovery, concurrency selection and writes belong
    to the caller.
    """
    rows = []
    for metrics in extract_metrics(data, source=source):
        if metrics['infrastructure_success'] is not False:
            score = metrics.get(_primary_metric(metrics))
            if not is_valid_score(score):
                for name in _SCORE_NAMES:
                    metrics[name] = metrics[f'{name}_se'] = None
                metrics['infrastructure_success'] = False
                metrics['integration_error'] = {
                    'type': 'InvalidPrimaryScore',
                    'message': f'invalid primary score: {score!r}',
                }
        rows.append(build_row(meta, metrics))
    return rows
