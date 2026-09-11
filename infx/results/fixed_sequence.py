"""Fixed-sequence result conversion and its compatibility CLI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .metadata import parse_component_metadata
from .power import ALL_POWER_METRIC_KEYS, POWER_METRIC_SCHEMA_VERSION, with_power_metrics
from .topology import Parallelism, validate_parallelism


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

    router = parse_component_metadata(env.get('ROUTER_METADATA'), 'ROUTER_METADATA')
    if router is not None:
        data['router'] = router

    kv_p2p_transfer = env.get('KV_P2P_TRANSFER')
    if kv_p2p_transfer:
        data['kv_p2p_transfer'] = kv_p2p_transfer

    is_multinode = env.get('IS_MULTINODE', 'false').lower() == 'true'

    if is_multinode:
        # TODO: Eventually will have to have a separate condition in here for multinode disagg and
        # multinode agg. For now, just assume that multinode implies disagg.

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
        if key.endswith('ms'):
            data[key.replace('_ms', '')] = float(value) / 1000.0
        if 'tpot' in key:
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
        expected_num_gpus = prefill_gpus + decode_gpus
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


def main() -> int:
    env = os.environ
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
    with open(agg_path) as f:
        print(json.dumps(json.load(f), indent=2))
    return status


if __name__ == '__main__':
    sys.exit(main())
