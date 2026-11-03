import pytest

from infx.launch.resources import WorkerResources, worker_resources


@pytest.mark.parametrize('count,gpus', [(0, 4), (-1, 4), (True, 4), (1, 0), (1, True), (1, 1.5)])
def test_invalid_worker_requests_fail(count, gpus):
    with pytest.raises(ValueError, match='positive integer'):
        WorkerResources(count, gpus)


def test_resource_input_does_not_infer_gpus_from_engine_parallelism():
    with pytest.raises(ValueError, match='GPU_COUNT'):
        worker_resources({'IS_MULTINODE': 'false', 'TP': '4'})
    assert worker_resources({'IS_MULTINODE': 'false', 'TP': '4', 'GPU_COUNT': '16'}) == {
        'agg': WorkerResources(1, 16)
    }


def test_aggregate_multinode_request_uses_active_logical_worker_only():
    assert worker_resources({
        'IS_MULTINODE': 'true', 'DISAGG': 'false',
        'PREFILL_NUM_WORKERS': '1', 'PREFILL_GPUS_PER_WORKER': '16',
        'DECODE_NUM_WORKERS': '0', 'DECODE_GPUS_PER_WORKER': '16',
    }) == {'agg': WorkerResources(1, 16)}
