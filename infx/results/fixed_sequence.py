"""Fixed-sequence result conversion and its compatibility CLI."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .metadata import parse_component_metadata
from .power import ALL_POWER_METRIC_KEYS, POWER_METRIC_SCHEMA_VERSION, with_power_metrics
from .topology import Parallelism, validate_parallelism
from utils.bench_serving.benchmark_outcome import benchmark_outcome


_BASE_ENV_VARS = (
    'RUNNER_TYPE', 'FRAMEWORK', 'PRECISION', 'SPEC_DECODING',
    'RESULT_FILENAME', 'ISL', 'OSL', 'DISAGG', 'MODEL_PREFIX', 'IMAGE'
)


def require_environment(env: Mapping[str, str], names: Iterable[str]) -> None:
    """Reject missing values in declaration order; empty strings remain present."""
    missing = [name for name in names if env.get(name) is None]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def build_result(benchmark: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Build fixed-sequence metrics without reading environment or writing files.

    Input mappings are read-only. The returned dictionary is independent and
    can be enriched by other result transformations before serialization.
    """
    require_environment(env, (key for key in _BASE_ENV_VARS if key != 'RESULT_FILENAME'))
    disagg = env['DISAGG'].lower() == 'true'
    data = {
        'hw': env['RUNNER_TYPE'],
        'conc': int(benchmark['max_concurrency']),
        'image': env['IMAGE'],
        'model': benchmark['model_id'],
        'infmax_model_prefix': env['MODEL_PREFIX'],
        'framework': env['FRAMEWORK'],
        'precision': env['PRECISION'],
        'spec_decoding': env['SPEC_DECODING'],
        'disagg': disagg,
        'recipe_fingerprint': env.get('RECIPE_FINGERPRINT', ''),
        'isl': int(env['ISL']),
        'osl': int(env['OSL']),
    }
    if 'benchmark_outcome' in benchmark:
        outcome = benchmark['benchmark_outcome']
        expected = benchmark_outcome(outcome['requested'], outcome['completed'])
        if (outcome != expected
                or benchmark.get('completed', expected['completed']) != expected['completed']
                or benchmark.get('num_prompts', expected['requested']) != expected['requested']):
            raise ValueError("Benchmark outcome does not match the recorded request counts and gate")
        data['benchmark_outcome'] = expected

    router = parse_component_metadata(env.get('ROUTER_METADATA'), 'ROUTER_METADATA')
    if router is not None:
        data['router'] = router

    kv_p2p_transfer = env.get('KV_P2P_TRANSFER')
    if kv_p2p_transfer:
        data['kv_p2p_transfer'] = kv_p2p_transfer

    is_multinode = env.get('IS_MULTINODE', 'false').lower() == 'true'

    if is_multinode:
        multinode_vars = ['PREFILL_GPUS', 'DECODE_GPUS', 'PREFILL_NUM_WORKERS', 'PREFILL_TP',
                          'PREFILL_EP', 'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP',
                          'DECODE_EP', 'DECODE_DP_ATTN']
        require_environment(env, multinode_vars)
        prefill_hardware = env.get('PREFILL_HARDWARE', '')
        decode_hardware = env.get('DECODE_HARDWARE', '')
        if bool(prefill_hardware) != bool(decode_hardware):
            raise ValueError(
                "PREFILL_HARDWARE and DECODE_HARDWARE must be specified together."
            )
        prefill_gpus = int(env['PREFILL_GPUS'])
        decode_gpus = int(env['DECODE_GPUS'])
        aggregate_gpus = int(env.get('AGGREGATE_GPUS', '0'))
        if aggregate_gpus:
            if aggregate_gpus < 0 or prefill_gpus or decode_gpus or disagg:
                raise ValueError("Aggregate GPUs require non-disaggregated topology without role GPUs")
            # Preserve the existing multinode layout fields for consumers, but
            # expose the actual shared role and validate telemetry as `agg`.
            prefill_gpus = aggregate_gpus
            data['num_aggregate_gpu'] = aggregate_gpus
        prefill_num_workers = int(env['PREFILL_NUM_WORKERS'])
        prefill = Parallelism(
            tp=int(env['PREFILL_TP']),
            pp=int(env.get('PREFILL_PP_SIZE', '1')),
            dcp_size=int(env.get('PREFILL_DCP_SIZE', '1')),
            pcp_size=int(env.get('PREFILL_PCP_SIZE', '1')),
            ep=int(env['PREFILL_EP']),
        )
        prefill_dp_attn = env['PREFILL_DP_ATTN']
        decode_num_workers = int(env['DECODE_NUM_WORKERS'])
        decode = Parallelism(
            tp=int(env['DECODE_TP']),
            pp=int(env.get('DECODE_PP_SIZE', '1')),
            dcp_size=int(env.get('DECODE_DCP_SIZE', '1')),
            pcp_size=int(env.get('DECODE_PCP_SIZE', '1')),
            ep=int(env['DECODE_EP']),
        )
        decode_dp_attn = env['DECODE_DP_ATTN']
        validate_parallelism(prefill, decode)

        total_gpus = prefill_gpus + decode_gpus
        if total_gpus <= 0:
            raise ValueError("Multinode results require at least one GPU.")
        if prefill_gpus <= 0:
            raise ValueError("Multinode results require at least one prefill GPU.")

        output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
        decode = decode.for_decode(decode_gpus)

        multi_node_data = {
            'is_multinode': True,
            **prefill.fields('prefill_'),
            'prefill_dp_attention': prefill_dp_attn,
            'prefill_num_workers': prefill_num_workers,
            **decode.fields('decode_'),
            'decode_dp_attention': decode_dp_attn,
            'decode_num_workers': decode_num_workers,
            'num_prefill_gpu': prefill_gpus,
            'num_decode_gpu': decode_gpus,
            'tput_per_gpu': float(benchmark['total_token_throughput']) / total_gpus,
            'output_tput_per_gpu': float(benchmark['output_throughput']) / output_tput_denominator,
            'input_tput_per_gpu': (float(benchmark['total_token_throughput']) - float(benchmark['output_throughput'])) / prefill_gpus,
        }
        if prefill_hardware:
            multi_node_data['prefill_hw'] = prefill_hardware
            multi_node_data['decode_hw'] = decode_hardware

        data = data | multi_node_data
    else:
        if disagg:
            raise ValueError("Disaggregated mode requires multinode setup.")

        require_environment(env, ['TP', 'EP_SIZE', 'DP_ATTENTION'])
        tp_size = int(env['TP'])
        ep_size = int(env['EP_SIZE'])
        dp_attention = env['DP_ATTENTION']
        parallelism = Parallelism(
            tp=tp_size,
            pp=int(env.get('PP_SIZE', '1')),
            dcp_size=int(env.get('DCP_SIZE', '1')),
            pcp_size=int(env.get('PCP_SIZE', '1')),
            ep=ep_size,
        )
        validate_parallelism(parallelism)
        num_gpus = parallelism.gpus_per_worker

        single_node_data = {
            'is_multinode': False,
            **parallelism.fields(),
            'dp_attention': dp_attention,
            'tput_per_gpu': float(benchmark['total_token_throughput']) / num_gpus,
            'output_tput_per_gpu': float(benchmark['output_throughput']) / num_gpus,
            'input_tput_per_gpu': (float(benchmark['total_token_throughput']) - float(benchmark['output_throughput'])) / num_gpus,
        }

        data = data | single_node_data

    for key, value in benchmark.items():
        if key.endswith('ms') and math.isfinite(float(value)):
            data[key.replace('_ms', '')] = float(value) / 1000.0
        if 'tpot' in key and math.isfinite(float(value)) and float(value) > 0:
            data[key.replace('_ms', '').replace(
                'tpot', 'intvty')] = 1000.0 / float(value)
    return data


def record_power_internal_error(
    *,
    csv_path: Path,
    bench_result: Path,
    agg_result: Path,
    validation_result: Path,
    expected_num_gpus: int,
    error: Exception,
) -> None:
    """Preserve an auditable invalid result when aggregation fails unexpectedly."""
    reasons = ["aggregation_internal_error"]
    try:
        from .power.single_node import (
            invalid_validation_payload,
            _write_json_atomic,
        )

        agg_data = json.loads(agg_result.read_text(encoding="utf-8"))
        agg_data = with_power_metrics(
            agg_data,
            metric_keys=ALL_POWER_METRIC_KEYS,
            schema_version=POWER_METRIC_SCHEMA_VERSION,
            power_valid=False,
            metrics={},
        )
        _write_json_atomic(agg_result, agg_data)

        validation_data = invalid_validation_payload(
            csv_path=csv_path,
            bench_result=bench_result,
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
        )
        validation_data["internal_error"] = {
            "type": type(error).__name__,
            "message": str(error)[:500],
        }
        _write_json_atomic(validation_result, validation_data)
    except (OSError, json.JSONDecodeError, ImportError, AttributeError) as fallback_error:
        print(
            f"[process_result] failed to preserve power validation fallback: "
            f"{fallback_error}",
            file=sys.stderr,
        )


def aggregate_power_result(
    env: Mapping[str, str],
    bench_path: Path,
    agg_path: Path,
) -> int:
    """Enrich a written fixed-sequence result, preserving best-effort failures."""
    require_power = env.get('REQUIRE_POWER', '').lower() in {'1', 'true', 'yes'}
    validation_path = Path(f"power_validation_{env['RESULT_FILENAME']}.json")
    is_multinode = env.get('IS_MULTINODE', 'false').lower() == 'true'
    if is_multinode:
        source = Path(env.get('POWER_ARTIFACT_DIR', 'LOGS/power'))
        prefill_gpus = int(env['PREFILL_GPUS'])
        decode_gpus = int(env['DECODE_GPUS'])
        aggregate_gpus = int(env.get('AGGREGATE_GPUS', '0'))
        expected_num_gpus = prefill_gpus + decode_gpus + aggregate_gpus
    else:
        candidates = [env.get('GPU_METRICS_CSV'), 'gpu_metrics.csv', '/workspace/gpu_metrics.csv']
        source = next(
            (Path(p) for p in candidates if p and Path(p).is_file()),
            Path(next(p for p in candidates if p)),
        )
        expected_num_gpus = int(env['TP']) * int(env.get('PP_SIZE', '1')) * int(env.get('PCP_SIZE', '1'))
    try:
        if is_multinode:
            from .power.multinode import run

            return run(
                source, bench_path, agg_path,
                prefill_gpus=prefill_gpus,
                decode_gpus=decode_gpus,
                aggregate_gpus=aggregate_gpus,
                expected_producer_sha=env.get('POWER_PRODUCER_SHA') or None,
                logs_root=Path(env.get('POWER_RESULT_ROOT', 'LOGS')),
                validation_result=validation_path,
                require_power=require_power,
            )
        from .power.single_node import run

        return run(
            csv_path=source,
            bench_result=bench_path,
            agg_result=agg_path,
            expected_num_gpus=expected_num_gpus,
            validation_result=validation_path,
            require_power=require_power,
        )
    except Exception as exc:  # noqa: BLE001 — preserve ordinary benchmark behavior
        print(f'[process_result] power aggregation failed: {exc}', file=sys.stderr)
        record_power_internal_error(
            csv_path=source,
            bench_result=bench_path,
            agg_result=agg_path,
            validation_result=validation_path,
            expected_num_gpus=expected_num_gpus,
            error=exc,
        )
        return int(require_power)


def process_result(env: Mapping[str, str]) -> int:
    require_environment(env, _BASE_ENV_VARS)
    result_filename = env['RESULT_FILENAME']
    bench_path = Path(f'{result_filename}.json')
    with open(bench_path) as f:
        benchmark = json.load(f)
    data = build_result(benchmark, env)
    agg_path = Path(f'agg_{result_filename}.json')
    with open(agg_path, 'w') as f:
        json.dump(data, f, indent=2)
    status = aggregate_power_result(env, bench_path, agg_path)
    validation_path = Path(f'power_validation_{result_filename}.json')
    from .power.audit import audit_summary

    result = json.loads(agg_path.read_text())
    try:
        validation = json.loads(validation_path.read_text())
        result.update(audit_summary(validation, validation_path.name))
    except (OSError, ValueError, TypeError) as exc:
        print(f'[process_result] audit summary unavailable: {exc}', file=sys.stderr)
        result['power_invalid_reasons'] = ['validation_artifact_unavailable']
        # A required run must preserve its audit as well as numeric metrics.
        status = max(status, int(env.get('REQUIRE_POWER', '').lower() in {'1', 'true', 'yes'}))
    agg_path.write_text(json.dumps(result, indent=2))
    with open(agg_path) as f:
        print(json.dumps(json.load(f), indent=2))
    return max(status, int(data.get('benchmark_outcome', {}).get('status') == 'failed'))


def process_multinode_results(env: Mapping[str, str]) -> int:
    """Process every available point and preserve sweep omissions before failing."""
    expected = {int(value) for value in env['CONC_LIST'].split()}
    if not expected or min(expected) <= 0:
        raise ValueError('CONC_LIST must contain positive concurrencies')
    points: list[dict[str, Any]] = []
    observed: set[int] = set()
    status = 0
    for path in sorted(Path('.').glob(f"{env['RESULT_FILENAME']}_*.json")):
        point: dict[str, Any] = {'source': path.name}
        try:
            match = re.search(r'_c(?:onc)?(\d+)_gpus_(\d+)(?:_ctx_(\d+)_gen_(\d+))?$', path.stem)
            if match is None:
                raise ValueError('Result filename lacks concurrency and physical GPU counts')
            concurrency, total = int(match[1]), int(match[2])
            raw = json.loads(path.read_text())
            if raw['max_concurrency'] != concurrency:
                raise ValueError('Result concurrency does not match its filename')
            if concurrency in observed:
                raise ValueError('Duplicate concurrency result within one recipe')
            observed.add(concurrency)
            point['concurrency'] = concurrency
            point_env = {**env, 'RESULT_FILENAME': path.stem, 'IS_MULTINODE': 'true'}
            disagg = env['DISAGG'].lower() == 'true'
            # Some disaggregated recipe groups include a shared-worker point.
            # Its zero decode workers and filename must both describe aggregate
            # execution; the group-level flag cannot manufacture role energy.
            aggregate = not disagg or int(env['DECODE_NUM_WORKERS']) == 0
            if aggregate:
                if match[3] is not None and (int(match[3]) != total or int(match[4]) != 0):
                    raise ValueError('Aggregate result contains separate role GPU counts')
                point_env.update(DISAGG='false', PREFILL_GPUS='0', DECODE_GPUS='0',
                                 AGGREGATE_GPUS=str(total))
            else:
                if match[3] is None:
                    raise ValueError('Disaggregated result lacks prefill/decode GPU counts')
                prefill, decode = int(match[3]), int(match[4])
                if prefill + decode != total:
                    raise ValueError('Role GPU counts do not equal total GPU count')
                point_env.update(PREFILL_GPUS=str(prefill), DECODE_GPUS=str(decode), AGGREGATE_GPUS='0')
            point['exit_code'] = process_result(point_env)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            point.update(exit_code=1, error=str(exc))
            print(f'[process_result] {path}: {exc}', file=sys.stderr)
        status = max(status, point['exit_code'])
        points.append(point)
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    status = max(status, int(bool(missing or unexpected)))
    summary = {'expected_concurrencies': sorted(expected),
               'missing_concurrencies': missing, 'unexpected_concurrencies': unexpected,
               'points': points, 'exit_code': status}
    Path(f"result_processing_{env['RESULT_FILENAME']}.json").write_text(json.dumps(summary, indent=2))
    if missing or unexpected:
        print(f'[process_result] incomplete sweep: missing={missing}, unexpected={unexpected}', file=sys.stderr)
    return status


def main() -> int:
    return process_multinode_results(os.environ) if sys.argv[1:] == ['--all'] else process_result(os.environ)


if __name__ == '__main__':
    sys.exit(main())
