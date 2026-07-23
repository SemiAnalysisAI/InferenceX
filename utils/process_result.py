import json
import os
import sys
from pathlib import Path


def get_required_env_vars(required_vars):
    """Load and validate required environment variables."""
    env_values = {}
    missing_env_vars = []

    for var_name in required_vars:
        value = os.environ.get(var_name)
        if value is None:
            missing_env_vars.append(var_name)
        env_values[var_name] = value

    if missing_env_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}")

    return env_values


def get_optional_component_metadata(env_var):
    """Parse strict optional component metadata from a JSON environment value."""
    raw_value = os.environ.get(env_var)
    if raw_value in (None, "", "null"):
        return None

    try:
        metadata = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var} must contain valid JSON") from exc

    if not isinstance(metadata, dict) or set(metadata) != {"name", "version"}:
        raise ValueError(f"{env_var} must contain exactly 'name' and 'version'")
    if not all(isinstance(metadata[key], str) and metadata[key] for key in metadata):
        raise ValueError(f"{env_var} name and version must be non-empty strings")
    return metadata


def record_power_internal_error(
    *,
    csv_path,
    bench_result,
    agg_result,
    validation_result,
    expected_num_gpus,
    error,
):
    """Preserve an auditable invalid result when aggregation fails unexpectedly."""
    reason = "aggregation_internal_error"
    try:
        agg_data = json.loads(agg_result.read_text(encoding="utf-8"))
        for key in (
            "avg_power_w",
            "avg_total_gpu_power_w",
            "total_gpu_energy_j",
            "joules_per_successful_query",
            "joules_per_input_token",
            "joules_per_output_token",
            "joules_per_total_token",
        ):
            agg_data.pop(key, None)
        agg_data["power_valid"] = 0
        agg_data.pop("power_invalid_reasons", None)
        agg_tmp = agg_result.with_suffix(agg_result.suffix + ".tmp")
        agg_tmp.write_text(json.dumps(agg_data, indent=2), encoding="utf-8")
        agg_tmp.replace(agg_result)

        validation_data = {
            "schema_version": 1,
            "power_valid": False,
            "reasons": [reason],
            "telemetry_source": str(csv_path),
            "benchmark_result": str(bench_result),
            "benchmark_window": None,
            "integration_method": (
                "per_device_trapezoidal_with_linear_boundary_interpolation"
            ),
            "expected_gpu_count": expected_num_gpus,
            "observed_gpu_count": 0,
            "observed_gpu_ids": [],
            "per_gpu_sample_counts": {},
            "per_gpu_max_sample_gap_s": {},
            "per_gpu_energy_j": {},
            "device_issues": {},
            "metrics": {},
            "internal_error": {
                "type": type(error).__name__,
                "message": str(error)[:500],
            },
        }
        validation_tmp = validation_result.with_suffix(
            validation_result.suffix + ".tmp"
        )
        validation_tmp.write_text(
            json.dumps(validation_data, indent=2), encoding="utf-8"
        )
        validation_tmp.replace(validation_result)
    except (OSError, json.JSONDecodeError) as fallback_error:
        print(
            f"[process_result] failed to preserve power validation fallback: "
            f"{fallback_error}",
            file=sys.stderr,
        )


# Base required env vars
base_env = get_required_env_vars([
    'RUNNER_TYPE', 'FRAMEWORK', 'PRECISION', 'SPEC_DECODING',
    'RESULT_FILENAME', 'ISL', 'OSL', 'DISAGG', 'MODEL_PREFIX', 'IMAGE'
])

hw = base_env['RUNNER_TYPE']
model_prefix = base_env['MODEL_PREFIX']
framework = base_env['FRAMEWORK']
precision = base_env['PRECISION']
spec_decoding = base_env['SPEC_DECODING']
disagg = base_env['DISAGG'].lower() == 'true'
result_filename = base_env['RESULT_FILENAME']
isl = base_env['ISL']
osl = base_env['OSL']
image = base_env['IMAGE']

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

data = {
    'hw': hw,
    'conc': int(bmk_result['max_concurrency']),
    'image': image,
    'model': bmk_result['model_id'],
    'infmax_model_prefix': model_prefix,
    'framework': framework,
    'precision': precision,
    'spec_decoding': spec_decoding,
    'disagg': disagg,
    'isl': int(isl),
    'osl': int(osl),
}

router = get_optional_component_metadata('ROUTER_METADATA')
if router is not None:
    data['router'] = router

kv_p2p_transfer = os.environ.get('KV_P2P_TRANSFER')
if kv_p2p_transfer:
    data['kv_p2p_transfer'] = kv_p2p_transfer

is_multinode = os.environ.get('IS_MULTINODE', 'false').lower() == 'true'

if is_multinode:
    # TODO: Eventually will have to have a separate condition in here for multinode disagg and
    # multinode agg. For now, just assume that multinode implies disagg.

    multinode_vars = ['PREFILL_GPUS', 'DECODE_GPUS', 'PREFILL_NUM_WORKERS', 'PREFILL_TP',
                      'PREFILL_EP', 'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP',
                      'DECODE_EP', 'DECODE_DP_ATTN']
    multinode_env = get_required_env_vars(multinode_vars)
    prefill_hardware = os.environ.get('PREFILL_HARDWARE', '')
    decode_hardware = os.environ.get('DECODE_HARDWARE', '')
    if bool(prefill_hardware) != bool(decode_hardware):
        raise ValueError(
            "PREFILL_HARDWARE and DECODE_HARDWARE must be specified together."
        )
    prefill_gpus = int(multinode_env['PREFILL_GPUS'])
    decode_gpus = int(multinode_env['DECODE_GPUS'])
    prefill_num_workers = int(multinode_env['PREFILL_NUM_WORKERS'])
    prefill_tp = int(multinode_env['PREFILL_TP'])
    prefill_pp = int(os.environ.get('PREFILL_PP_SIZE', '1'))
    prefill_dcp_size = int(os.environ.get('PREFILL_DCP_SIZE', '1'))
    prefill_pcp_size = int(os.environ.get('PREFILL_PCP_SIZE', '1'))
    prefill_ep = int(multinode_env['PREFILL_EP'])
    prefill_dp_attn = multinode_env['PREFILL_DP_ATTN']
    decode_num_workers = int(multinode_env['DECODE_NUM_WORKERS'])
    decode_tp = int(multinode_env['DECODE_TP'])
    decode_pp = int(os.environ.get('DECODE_PP_SIZE', '1'))
    decode_dcp_size = int(os.environ.get('DECODE_DCP_SIZE', '1'))
    decode_pcp_size = int(os.environ.get('DECODE_PCP_SIZE', '1'))
    decode_ep = int(multinode_env['DECODE_EP'])
    decode_dp_attn = multinode_env['DECODE_DP_ATTN']
    worker_parallelism = (
        prefill_pp,
        prefill_dcp_size,
        prefill_pcp_size,
        decode_pp,
        decode_dcp_size,
        decode_pcp_size,
    )
    if any(value <= 0 for value in worker_parallelism):
        raise ValueError(
            "Multinode PP, DCP, and PCP sizes must be positive integers."
        )

    total_gpus = prefill_gpus + decode_gpus
    if total_gpus <= 0:
        raise ValueError("Multinode results require at least one GPU.")
    if prefill_gpus <= 0:
        raise ValueError("Multinode results require at least one prefill GPU.")

    output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
    output_decode_tp = decode_tp if decode_gpus > 0 else 0
    output_decode_ep = decode_ep if decode_gpus > 0 else 0
    output_decode_pp = decode_pp if decode_gpus > 0 else 1
    output_decode_dcp_size = decode_dcp_size if decode_gpus > 0 else 1
    output_decode_pcp_size = decode_pcp_size if decode_gpus > 0 else 1

    multi_node_data = {
        'is_multinode': True,
        'prefill_tp': prefill_tp,
        'prefill_pp': prefill_pp,
        'prefill_dcp_size': prefill_dcp_size,
        'prefill_pcp_size': prefill_pcp_size,
        'prefill_ep': prefill_ep,
        'prefill_dp_attention': prefill_dp_attn,
        'prefill_num_workers': prefill_num_workers,
        'decode_tp': output_decode_tp,
        'decode_pp': output_decode_pp,
        'decode_dcp_size': output_decode_dcp_size,
        'decode_pcp_size': output_decode_pcp_size,
        'decode_ep': output_decode_ep,
        'decode_dp_attention': decode_dp_attn,
        'decode_num_workers': decode_num_workers,
        'num_prefill_gpu': prefill_gpus,
        'num_decode_gpu': decode_gpus,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / total_gpus,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / output_tput_denominator,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / prefill_gpus,
    }
    if prefill_hardware:
        multi_node_data['prefill_hw'] = prefill_hardware
        multi_node_data['decode_hw'] = decode_hardware

    data = data | multi_node_data
else:
    if disagg:
        raise ValueError("Disaggregated mode requires multinode setup.")

    single_node_env = get_required_env_vars(['TP', 'EP_SIZE', 'DP_ATTENTION'])
    tp_size = int(single_node_env['TP'])
    ep_size = int(single_node_env['EP_SIZE'])
    dp_attention = single_node_env['DP_ATTENTION']
    pp = int(os.environ.get('PP_SIZE', '1'))
    dcp_size = int(os.environ.get('DCP_SIZE', '1'))
    pcp_size = int(os.environ.get('PCP_SIZE', '1'))
    if pp <= 0 or dcp_size <= 0 or pcp_size <= 0:
        raise ValueError("PP_SIZE, DCP_SIZE, and PCP_SIZE must be positive integers.")
    num_gpus = tp_size * pp * pcp_size

    single_node_data = {
        'is_multinode': False,
        'tp': tp_size,
        'pp': pp,
        'dcp_size': dcp_size,
        'pcp_size': pcp_size,
        'ep': ep_size,
        'dp_attention': dp_attention,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / num_gpus,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / num_gpus,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / num_gpus,
    }

    data = data | single_node_data

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace(
            'tpot', 'intvty')] = 1000.0 / float(value)

agg_path = Path(f'agg_{result_filename}.json')
with open(agg_path, 'w') as f:
    json.dump(data, f, indent=2)

# Single-node measured power is best-effort by default. Power studies can set
# REQUIRE_POWER=1 to fail closed after the validation sidecar has been written.
_require_power = os.environ.get('REQUIRE_POWER', '').lower() in {'1', 'true', 'yes'}
_power_status = 0
if is_multinode:
    if _require_power:
        print(
            '[process_result] Power validation failed: PR1 supports only '
            'single-node non-disaggregated telemetry',
            file=sys.stderr,
        )
        _power_status = 1
else:
    _csv_candidates = [
        os.environ.get('GPU_METRICS_CSV'),
        'gpu_metrics.csv',
        '/workspace/gpu_metrics.csv',
    ]
    _csv_path = next(
        (Path(p) for p in _csv_candidates if p and Path(p).is_file()),
        Path(next(p for p in _csv_candidates if p)),
    )
    _bench_path = Path(f'{result_filename}.json')
    _validation_path = Path(f'power_validation_{result_filename}.json')
    try:
        from aggregate_power import run as _aggregate_power_run

        _power_status = _aggregate_power_run(
            csv_path=_csv_path,
            bench_result=_bench_path,
            agg_result=agg_path,
            expected_num_gpus=num_gpus,
            validation_result=_validation_path,
            require_power=_require_power,
        )
    except Exception as exc:  # noqa: BLE001 — preserve ordinary benchmark behavior
        print(f'[process_result] power aggregation failed: {exc}', file=sys.stderr)
        record_power_internal_error(
            csv_path=_csv_path,
            bench_result=_bench_path,
            agg_result=agg_path,
            validation_result=_validation_path,
            expected_num_gpus=num_gpus,
            error=exc,
        )
        if _require_power:
            _power_status = 1

with open(agg_path) as f:
    print(json.dumps(json.load(f), indent=2))

if _power_status:
    raise SystemExit(_power_status)
