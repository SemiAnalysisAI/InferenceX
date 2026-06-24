import sys
import json
import os
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

is_multinode = os.environ.get('IS_MULTINODE', 'false').lower() == 'true'

if is_multinode:
    # Detect topology from the prefill/decode GPU split. A disaggregated run keeps
    # prefill and decode in separate pools (decode GPUs > 0); an aggregated ("agg")
    # run colocates them in one pool and so reports zero decode GPUs. Classify on
    # that split rather than trusting the DISAGG env flag.
    topo_env = get_required_env_vars(['PREFILL_GPUS', 'DECODE_GPUS'])
    prefill_gpus = int(topo_env['PREFILL_GPUS'])
    decode_gpus = int(topo_env['DECODE_GPUS'])
    is_agg = decode_gpus == 0

    if is_agg:
        # Aggregated multinode has no decode pool, so the DECODE_* detail vars may be
        # absent; read all detail vars tolerantly and default to zero/empty.
        detail_env = {
            'PREFILL_NUM_WORKERS': os.environ.get('PREFILL_NUM_WORKERS', '0'),
            'PREFILL_TP': os.environ.get('PREFILL_TP', '0'),
            'PREFILL_EP': os.environ.get('PREFILL_EP', '0'),
            'PREFILL_DP_ATTN': os.environ.get('PREFILL_DP_ATTN', ''),
            'DECODE_NUM_WORKERS': os.environ.get('DECODE_NUM_WORKERS', '0'),
            'DECODE_TP': os.environ.get('DECODE_TP', '0'),
            'DECODE_EP': os.environ.get('DECODE_EP', '0'),
            'DECODE_DP_ATTN': os.environ.get('DECODE_DP_ATTN', ''),
        }
    else:
        # Disaggregated multinode requires the full prefill+decode env contract.
        detail_env = get_required_env_vars(['PREFILL_NUM_WORKERS', 'PREFILL_TP', 'PREFILL_EP',
                                            'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP',
                                            'DECODE_EP', 'DECODE_DP_ATTN'])

    prefill_num_workers = int(detail_env['PREFILL_NUM_WORKERS'])
    prefill_tp = int(detail_env['PREFILL_TP'])
    prefill_ep = int(detail_env['PREFILL_EP'])
    prefill_dp_attn = detail_env['PREFILL_DP_ATTN']
    decode_num_workers = int(detail_env['DECODE_NUM_WORKERS'])
    decode_tp = int(detail_env['DECODE_TP'])
    decode_ep = int(detail_env['DECODE_EP'])
    decode_dp_attn = detail_env['DECODE_DP_ATTN']

    total_gpus = prefill_gpus + decode_gpus
    if total_gpus <= 0:
        raise ValueError("Multinode results require at least one GPU.")
    if prefill_gpus <= 0:
        raise ValueError("Multinode results require at least one prefill GPU.")

    output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
    output_decode_tp = decode_tp if decode_gpus > 0 else 0
    output_decode_ep = decode_ep if decode_gpus > 0 else 0

    multi_node_data = {
        'is_multinode': True,
        # decode_gpus == 0 => aggregated, not disaggregated (overrides DISAGG env).
        'disagg': not is_agg,
        'prefill_tp': prefill_tp,
        'prefill_ep': prefill_ep,
        'prefill_dp_attention': prefill_dp_attn,
        'prefill_num_workers': prefill_num_workers,
        'decode_tp': output_decode_tp,
        'decode_ep': output_decode_ep,
        'decode_dp_attention': decode_dp_attn,
        'decode_num_workers': decode_num_workers,
        'num_prefill_gpu': prefill_gpus,
        'num_decode_gpu': decode_gpus,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / total_gpus,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / output_tput_denominator,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / prefill_gpus,
    }

    data = data | multi_node_data
else:
    if disagg:
        raise ValueError("Disaggregated mode requires multinode setup.")

    single_node_env = get_required_env_vars(['TP', 'EP_SIZE', 'DP_ATTENTION'])
    tp_size = int(single_node_env['TP'])
    ep_size = int(single_node_env['EP_SIZE'])
    dp_attention = single_node_env['DP_ATTENTION']

    single_node_data = {
        'is_multinode': False,
        'tp': tp_size,
        'ep': ep_size,
        'dp_attention': dp_attention,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / tp_size,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / tp_size,
    }

    data = data | single_node_data

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace(
            'tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

agg_path = Path(f'agg_{result_filename}.json')
with open(agg_path, 'w') as f:
    json.dump(data, f, indent=2)

# Best-effort: patch measured power into the agg JSON. Never fails the run.
try:
    from aggregate_power import run as _aggregate_power_run

    _csv_candidates = [
        os.environ.get('GPU_METRICS_CSV'),
        'gpu_metrics.csv',
        '/workspace/gpu_metrics.csv',
    ]
    _csv_path = next(
        (Path(p) for p in _csv_candidates if p and Path(p).is_file()),
        None,
    )
    if _csv_path is not None:
        _aggregate_power_run(
            csv_path=_csv_path,
            bench_result=Path(f'{result_filename}.json'),
            agg_result=agg_path,
        )
except Exception as exc:  # noqa: BLE001 — never block on telemetry
    print(f'[process_result] power aggregation skipped: {exc}', file=sys.stderr)
