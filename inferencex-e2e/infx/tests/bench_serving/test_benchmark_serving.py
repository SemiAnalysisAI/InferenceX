"""Exercise client result persistence without loading a model or serving requests."""

import asyncio
import json
from argparse import Namespace
from unittest.mock import AsyncMock

import pytest

from infx.bench_serving import benchmark_serving as client


def cli_args(tmp_path, **overrides):
    """``main`` arguments for a one-prompt random run saved to ``tmp_path/result.json``."""
    return Namespace(**{
        'seed': 0, 'backend': 'vllm', 'model': 'fixture', 'served_model_name': None,
        'tokenizer': None, 'tokenizer_mode': 'auto', 'base_url': 'http://unused',
        'endpoint': '/completions', 'trust_remote_code': False, 'dataset_name': 'random',
        'random_prefix_len': 0, 'random_input_len': 1, 'random_output_len': 1, 'num_prompts': 1,
        'random_range_ratio': 1, 'use_chat_template': False, 'dsv4': False,
        'random_num_workers': 1, 'goodput': None, 'logprobs': None, 'best_of': 1,
        'request_rate': float('inf'), 'burstiness': 1, 'disable_tqdm': True, 'num_warmups': 0,
        'profile': False, 'percentile_metrics': 'ttft', 'metric_percentiles': '99',
        'ignore_eos': False, 'max_concurrency': 1, 'lora_modules': None, 'save_result': True,
        'metadata': None, 'save_detailed': True, 'result_filename': 'result.json',
        'result_dir': str(tmp_path), **overrides,
    })


def stub_benchmark(monkeypatch, raw):
    """Skip tokenizer loading, prompt sampling and serving; ``benchmark`` returns ``raw``."""
    benchmark = AsyncMock(return_value=raw)
    monkeypatch.setattr(client, '_load_tokenizer', lambda *a, **kw: object())
    monkeypatch.setattr(client, 'sample_random_requests', lambda **kw: [])
    monkeypatch.setattr(client, 'benchmark', benchmark)
    monkeypatch.setattr(client.gc, 'freeze', lambda: None)
    return benchmark


@pytest.mark.parametrize('requested,completed,status', [
    (100, 100, 'passed'), (100, 95, 'passed'), (100, 94, 'failed'),
    (100, 0, 'failed'), (0, 0, 'failed'), (100, 101, 'failed'),
    (100, -1, 'failed'), (100, None, 'failed'),
])
@pytest.mark.parametrize('save_result', [True, False])
def test_client_preserves_outcome_before_failure(
    tmp_path, monkeypatch, requested, completed, status, save_result,
):
    args = cli_args(tmp_path, num_prompts=requested, save_result=save_result)
    metrics = {name: 0 for name in (
        'median_ttft_ms', 'mean_ttft_ms', 'std_ttft_ms', 'p99_ttft_ms',
        'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms', 'p99_tpot_ms',
        'median_itl_ms', 'mean_itl_ms', 'std_itl_ms', 'p99_itl_ms',
    )}
    raw = {**metrics, 'completed': completed, 'errors': ['retained diagnostic']}
    stub_benchmark(monkeypatch, raw)
    invalid = requested == 0 or completed is None or completed < 0 or completed > requested

    if status == 'failed':
        message = 'invalid request counts' if invalid else 'request failure rate'
        with pytest.raises(SystemExit, match=message):
            client.main(args)
    else:
        client.main(args)

    path = tmp_path / 'result.json'
    if not save_result:
        assert not path.exists()
        return
    saved = json.loads(path.read_text())
    assert saved['num_prompts'] == requested
    assert saved['completed'] == completed
    assert saved['errors'] == ['retained diagnostic']
    outcome = saved['benchmark_outcome']
    assert outcome['status'] == status
    assert outcome['requested'] == requested
    assert outcome['completed'] == completed
    if invalid:
        assert '0 <= completed <= requested > 0' in outcome['error']
        assert 'failed' not in outcome
    else:
        assert 'error' not in outcome
        assert outcome['failed'] == {100: 0, 95: 5, 94: 6, 0: 100}[completed]
        assert outcome['max_failure_rate'] == 0.05


def test_metadata_value_keeps_its_equals_signs(tmp_path, monkeypatch):
    stub_benchmark(monkeypatch, {'completed': 1})

    client.main(cli_args(tmp_path, metadata=['cmd=--a=b', ' tp = 8 ']))

    saved = json.loads((tmp_path / 'result.json').read_text())
    assert saved['cmd'] == '--a=b'
    assert saved['tp'] == '8'


@pytest.mark.parametrize('rate', [0.0, -2.0])
def test_non_positive_request_rate_is_rejected_before_the_run(tmp_path, monkeypatch, rate):
    benchmark = stub_benchmark(monkeypatch, {'completed': 1})

    with pytest.raises(ValueError, match=f'--request-rate must be positive, got {rate}'):
        client.main(cli_args(tmp_path, request_rate=rate))

    assert benchmark.await_count == 0


def test_missing_result_dir_is_rejected_before_the_run(tmp_path, monkeypatch):
    benchmark = stub_benchmark(monkeypatch, {'completed': 1})
    missing = tmp_path / 'missing'

    with pytest.raises(ValueError, match=f'--result-dir {missing} is not an existing directory'):
        client.main(cli_args(tmp_path, result_dir=str(missing)))

    assert benchmark.await_count == 0
    assert not missing.exists()


def test_random_sampling_rejects_zero_prompts():
    with pytest.raises(ValueError, match='--num-prompts must be positive, got 0'):
        client.sample_random_requests(
            prefix_len=0, input_len=4, output_len=4, num_prompts=0, range_ratio=1.0,
            tokenizer=Namespace(vocab_size=10), num_workers=1,
        )


class WordTokenizer:
    """One token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens):
        return Namespace(input_ids=text.split())


def test_metrics_retokenize_a_success_without_reported_usage():
    outputs = [
        client.RequestFuncOutput(
            generated_text='one two three four', success=True, latency=1.0, ttft=0.1,
        ),
        client.RequestFuncOutput(
            generated_text='ignored', success=True, latency=1.0, ttft=0.1, output_tokens=7,
        ),
    ]
    requests = [('p', 5, 4, None), ('p', 5, 7, None)]

    metrics, output_lens = client.calculate_metrics(
        requests, outputs, 2.0, WordTokenizer(), [50], {},
    )

    assert output_lens == [4, 7]
    assert metrics.total_output == 11
    assert metrics.output_throughput == 5.5
    assert metrics.mean_tpot_ms == pytest.approx((0.9 / 3 + 0.9 / 6) / 2 * 1000)


@pytest.mark.parametrize('enabled', [True, False])
def test_pytorch_records_skip_metrics_the_run_did_not_select(tmp_path, monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv('SAVE_TO_PYTORCH_BENCHMARK_FORMAT', '1')
    else:
        monkeypatch.delenv('SAVE_TO_PYTORCH_BENCHMARK_FORMAT', raising=False)
    results = {'mean_ttft_ms': 12.5, 'p90_ttft_ms': 20.0, 'completed': 3}
    result_file = tmp_path / 'result.json'

    client.save_to_pytorch_benchmark_format(Namespace(model='m'), results, str(result_file))

    records_file = tmp_path / 'result.pytorch.json'
    if not enabled:
        assert not records_file.exists()
        return
    [record] = json.loads(records_file.read_text())
    assert record['metric']['name'] == 'mean_ttft_ms'
    assert record['metric']['benchmark_values'] == [12.5]
    assert record['metric']['extra_info'] == {'p90_ttft_ms': 20.0, 'completed': 3}


def run_benchmark(monkeypatch, ttfts, **overrides):
    """Run ``benchmark`` against a fake server and clock; each prompt's request takes 2 s."""
    clock = Namespace(now=0.0)
    monkeypatch.setattr(client, 'time', Namespace(perf_counter=lambda: clock.now, time=lambda: 0))

    async def serve(request_func_input, pbar=None):
        cost = {'/start_profile': 0.0, '/stop_profile': 100.0}
        clock.now += cost.get(request_func_input.api_url.removeprefix('http://x'), 2.0)
        ttft = ttfts.get(request_func_input.prompt, 0.0)
        return client.RequestFuncOutput(success=True, latency=1.0, ttft=ttft, output_tokens=3)

    monkeypatch.setitem(client.ASYNC_REQUEST_FUNCS, 'fake', serve)
    kwargs = {
        'backend': 'fake', 'api_url': 'http://x/v1/completions', 'base_url': 'http://x',
        'model_id': 'm', 'model_name': None, 'tokenizer': None, 'logprobs': None, 'best_of': 1,
        'input_requests': [(prompt, 5, 3, None) for prompt in ttfts],
        'request_rate': float('inf'), 'burstiness': 1.0, 'disable_tqdm': True, 'num_warmups': 0,
        'profile': False, 'selected_percentile_metrics': ['ttft'], 'selected_percentiles': [99],
        'ignore_eos': False, 'goodput_config_dict': {}, 'max_concurrency': None,
        'lora_modules': None, **overrides,
    }
    return asyncio.run(client.benchmark(**kwargs))


def test_goodput_counts_requests_within_the_slo(monkeypatch):
    result = run_benchmark(
        monkeypatch, {'fast': 0.1, 'slow': 0.5}, goodput_config_dict={'ttft': 200.0},
    )

    assert result['duration'] == 4.0
    assert result['request_goodput'] == 0.25


def test_profiler_shutdown_is_not_part_of_the_measured_duration(monkeypatch):
    result = run_benchmark(monkeypatch, {'a': 0.1, 'b': 0.1}, profile=True)

    assert result['duration'] == 4.0
    assert result['request_throughput'] == 0.5
