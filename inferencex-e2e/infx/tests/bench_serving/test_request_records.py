"""Build the per-attempt block from concrete request outputs."""

import pytest

from infx.bench_serving.backend_request_func import RequestFuncOutput
from infx.bench_serving.request_records import build_request_records

RUN = {
    'requested': 4, 'warmups': 8, 'max_concurrency': 4, 'request_rate': float('inf'),
    'benchmark_start_time': 10.0, 'benchmark_start_time_unix': 1_700_000_000.0,
    'benchmark_end_time_unix': 1_700_000_003.0, 'duration': 3.0,
}


def test_each_attempt_keeps_its_outcome_and_what_the_client_observed():
    completed = RequestFuncOutput(
        success=True, latency=2.0, ttft=0.5, prompt_len=100, start_time=10.25, end_time=12.3,
        completion_evidence=7,
    )
    refused = RequestFuncOutput(
        failure_class='http_status', http_status=503, prompt_len=101, start_time=10.5,
        end_time=10.75,
    )
    cut_mid_stream = RequestFuncOutput(
        failure_class='connection', ttft=0.0123456, itl=[0.1], prompt_len=102, start_time=11.0,
        end_time=12.5,
    )
    one_token = RequestFuncOutput(
        success=True, latency=0.4, ttft=0.4, prompt_len=103, start_time=11.5, end_time=11.9,
        completion_evidence=1,
    )

    records = build_request_records(
        [completed, refused, cut_mid_stream, one_token], [4, 0, 0, 1], **RUN,
    )

    assert records['schema_version'] == 1
    assert [records['outcomes'][index] for index in records['columns'].pop('outcome')] == [
        'completed', 'http_status', 'connection', 'completed',
    ]
    assert records['columns'] == {
        'start_offset_ms': [250.0, 500.0, 1000.0, 1500.0],
        'ttft_ms': [500.0, None, 12.346, 400.0],
        'tpot_ms': [500.0, None, None, None],
        'e2el_ms': [2000.0, 250.0, 1500.0, 400.0],
        'output_len': [4, None, None, 1],
        'input_len': [100, 101, 102, 103],
        'completion_evidence': [7, 0, 0, 1],
    }
    assert records['completion_evidence_bits'] == {'done': 1, 'finish_reason': 2, 'usage': 4}
    assert records['accounting'] == {'complete': True, 'reasons': []}
    counts = records['counts']
    assert counts == {
        'requested': 4, 'dispatched': 4, 'completed': 2, 'failed': 2,
        'failed_by_class': {
            'http_status': 1, 'timeout': 0, 'connection': 1, 'stream_error': 0,
            'empty_response': 0, 'other': 0, 'incomplete_stream': 0,
        },
        'http_status': {'503': 1}, 'warmup_excluded': 8, 'retries': 0,
    }
    assert counts['dispatched'] == counts['completed'] + counts['failed']
    assert counts['failed'] == sum(counts['failed_by_class'].values())
    assert records['window'] == {
        'benchmark_start_time_unix': 1_700_000_000.0, 'benchmark_end_time_unix': 1_700_000_003.0,
        'duration': 3.0, 'duration_source': 'client_perf_counter_first_dispatch_to_last_return',
    }
    assert records['load']['max_concurrency'] == 4
    assert records['load']['request_rate'] == 'inf'


@pytest.mark.parametrize('failure_class', ['', 'not a known class'])
def test_failure_the_request_function_did_not_classify_is_other(failure_class):
    unclassified = RequestFuncOutput(failure_class=failure_class, prompt_len=7)

    records = build_request_records([unclassified], [0], **{**RUN, 'requested': 1})

    assert [records['outcomes'][index] for index in records['columns']['outcome']] == ['other']
    assert records['columns']['start_offset_ms'] == [None]
    assert records['columns']['e2el_ms'] == [None]
    assert records['counts']['failed_by_class']['other'] == 1


def test_finite_request_rate_is_reported_as_a_number():
    records = build_request_records([], [], **{**RUN, 'requested': 0, 'request_rate': 2.5})

    assert records['load']['request_rate'] == 2.5
    assert records['counts']['dispatched'] == 0


def test_attempt_that_was_requested_but_never_dispatched_leaves_the_accounting_incomplete():
    done = RequestFuncOutput(success=True, latency=1.0, ttft=0.5)

    records = build_request_records([done, done], [4, 4], **{**RUN, 'requested': 3})

    assert records['counts']['requested'] == 3
    assert records['counts']['dispatched'] == 2
    assert records['accounting'] == {'complete': False, 'reasons': ['requested_not_dispatched']}


def test_missing_output_length_leaves_the_accounting_incomplete_instead_of_raising():
    done = RequestFuncOutput(success=True, latency=1.0, ttft=0.5)

    records = build_request_records([done, done], [4], **{**RUN, 'requested': 2})

    assert records['columns']['output_len'] == [4]
    assert records['columns']['tpot_ms'] == [166.667, None]
    assert records['accounting'] == {
        'complete': False, 'reasons': ['column_length_mismatch', 'dispatched_count_mismatch'],
    }


@pytest.mark.parametrize('change,reasons', [
    (lambda counts: counts.update(completed=counts['completed'] + 1), ['outcome_count_mismatch']),
    (lambda counts: counts['failed_by_class'].update(http_status=0, other=1),
     ['failure_class_count_mismatch', 'http_status_count_mismatch']),
    (lambda counts: counts['http_status'].clear(), ['http_status_count_mismatch']),
])
def test_counts_that_disagree_with_the_outcome_column_are_an_accounting_gap(change, reasons):
    from infx.bench_serving.request_records import accounting_gaps

    refused = RequestFuncOutput(failure_class='http_status', http_status=503)
    done = RequestFuncOutput(success=True, latency=1.0, ttft=0.5)
    records = build_request_records([done, refused], [4, 0], **{**RUN, 'requested': 2})
    assert accounting_gaps(records) == []

    change(records['counts'])

    assert accounting_gaps(records) == reasons
