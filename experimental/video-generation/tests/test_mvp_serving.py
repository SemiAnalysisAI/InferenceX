"""Hand-worked serving metrics; synthetic request timings, never GPU evidence."""

from copy import deepcopy

import pytest

from evaluator.mvp_serving import settings, summarize, validate_window


def sample_run():
    records = []
    for index in range(11):
        latency = index + 1
        start = index * 12
        records.append({
            'slot_id': str(index), 'phase': 'measurement', 'attempted': True,
            'status': 'succeeded' if index < 10 else 'failed',
            'outcome': 'completed' if index < 10 else 'provider_failed',
            'submit_to_media_seconds': latency if index < 10 else None,
            'media': {'valid': True, 'video': {'duration_seconds': 8}} if index < 10 else None,
            'timing_window': {'start_monotonic_seconds': start, 'transport_end_monotonic_seconds': start + latency},
        })
    return {'configuration': {'serving': settings(1, 5)}, 'records': records,
            'measurement': {'boundary': 'submit_to_downloaded_media', 'concurrency': 1,
                            'start_monotonic_seconds': 0, 'end_monotonic_seconds': 132, 'wall_seconds': 132}}


def test_percentiles_goodput_and_failure_denominators():
    metrics = summarize(sample_run())
    latency = metrics['client_ready_latency_seconds']
    assert latency['sample_count'] == 10
    assert latency['p50'] == 5.5
    assert latency['p90'] == 9
    assert latency['p95'] is None
    assert metrics['deadline_met_valid_clips'] == 5
    assert metrics['deadline_attainment_fraction'] == 5 / 11
    assert metrics['deadline_goodput_clips_per_second'] == 5 / 132
    assert metrics['valid_video_seconds_per_second'] == 80 / 132
    assert metrics['outcomes']['provider_failed'] == 1
    assert metrics['capacity_qualified'] is False
    assert metrics['offered_request_rate_per_second'] is None


def test_missing_sample_withholds_percentiles_and_goodput():
    run = sample_run()
    run['records'][0]['submit_to_media_seconds'] = None
    metrics = summarize(run)
    assert metrics['client_ready_latency_seconds']['sample_count'] == 0
    assert metrics['client_ready_latency_seconds']['valid_clip_count'] == 10
    assert metrics['client_ready_latency_seconds']['p90'] is None
    assert metrics['deadline_goodput_clips_per_second'] is None


@pytest.mark.parametrize('defect', ['wall', 'boundary', 'outside', 'concurrency', 'empty_interval'])
def test_tampered_serving_windows_fail_closed(defect):
    run = deepcopy(sample_run())
    if defect == 'wall':
        run['measurement']['wall_seconds'] = 130
    elif defect == 'boundary':
        run['measurement']['boundary'] = 'submit_to_validated_media'
    elif defect == 'outside':
        run['records'][0]['timing_window']['transport_end_monotonic_seconds'] = 133
    elif defect == 'empty_interval':
        run['records'][0]['timing_window']['transport_end_monotonic_seconds'] = 0
    else:
        run['records'][1]['timing_window']['start_monotonic_seconds'] = .5
        run['records'][1]['timing_window']['transport_end_monotonic_seconds'] = 2.5
    with pytest.raises(ValueError):
        validate_window(run)
