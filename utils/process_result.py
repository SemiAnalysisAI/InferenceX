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

# Offline in-process runs (utils/bench_offline/run_offline.py) report
# per-decode-step TPOT and step throughput. Their dispatch value is
# SPEC_DECODING=offline, but the row should describe the actual run config:
# spec_decoding reflects whether MTP draft tokens were enabled.
is_offline = str(bmk_result.get('engine_mode', '')).lower() == 'offline'
if is_offline:
    spec_decoding = 'mtp' if int(bmk_result.get('mtp', 0) or 0) > 0 else 'none'

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

if is_offline:
    data['engine_mode'] = 'offline'
    # "simulated_uniform_random" = idealized expert-balance simulation;
    # "real" = honest routing. Rows
    # with different values must not be compared against each other.
    data['moe_routing'] = bmk_result.get('moe_routing', 'unknown')
    data['tpot_unit'] = bmk_result.get('tpot_unit', 'decode_step')
    if bmk_result.get('decode_steps_target') is not None:
        data['decode_steps'] = int(bmk_result['decode_steps_target'])
    if bmk_result.get('spec_tokens_per_step_observed') is not None:
        data['spec_tokens_per_step_observed'] = float(
            bmk_result['spec_tokens_per_step_observed'])

is_multinode = os.environ.get('IS_MULTINODE', 'false').lower() == 'true'

if is_offline and is_multinode:
    raise ValueError("Offline in-process results are single-node only.")

if is_multinode:
    # TODO: Eventually will have to have a separate condition in here for multinode disagg and
    # multinode agg. For now, just assume that multinode implies disagg.

    multinode_env = get_required_env_vars(['PREFILL_GPUS', 'DECODE_GPUS', 'PREFILL_NUM_WORKERS', 'PREFILL_TP',
                                          'PREFILL_EP', 'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP', 'DECODE_EP', 'DECODE_DP_ATTN'])
    prefill_gpus = int(multinode_env['PREFILL_GPUS'])
    decode_gpus = int(multinode_env['DECODE_GPUS'])
    prefill_num_workers = int(multinode_env['PREFILL_NUM_WORKERS'])
    prefill_tp = int(multinode_env['PREFILL_TP'])
    prefill_ep = int(multinode_env['PREFILL_EP'])
    prefill_dp_attn = multinode_env['PREFILL_DP_ATTN']
    decode_num_workers = int(multinode_env['DECODE_NUM_WORKERS'])
    decode_tp = int(multinode_env['DECODE_TP'])
    decode_ep = int(multinode_env['DECODE_EP'])
    decode_dp_attn = multinode_env['DECODE_DP_ATTN']

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

    if is_offline:
        # Offline decode-step semantics: throughput is requests_per_chip /
        # per-step TPOT. MTP bonus tokens are excluded from headline metrics.
        # Input throughput is zeroed so total throughput equals the decode-step
        # output throughput.
        step_tput_per_gpu = (
            (int(bmk_result['max_concurrency']) / tp_size)
            / (float(bmk_result['mean_tpot_ms']) / 1000.0))
        single_node_data = {
            'is_multinode': False,
            'tp': tp_size,
            'ep': ep_size,
            'dp_attention': dp_attention,
            'tput_per_gpu': step_tput_per_gpu,
            'output_tput_per_gpu': step_tput_per_gpu,
            'input_tput_per_gpu': 0.0,
        }
    else:
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
    # tpot→intvty conversion: only for *_ms latency fields (skips
    # non-numeric fields like tpot_unit, count fields like
    # tpot_sample_count, std_tpot_ms which can be 0, and zero-valued
    # latencies that would divide by zero).
    if (key.endswith('_ms') and 'tpot' in key and not key.startswith('std_')
            and float(value) > 0):
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
