import pytest

from infx.launch.resources import WorkerResources, worker_resources


@pytest.mark.parametrize("gpus,nodes,local", [(4, 1, 4), (8, 1, 8), (16, 2, 8)])
def test_worker_placement_uses_physical_nodes(gpus, nodes, local):
    assert WorkerResources(1, gpus).placement(8) == (local, nodes)


def test_partial_workers_pack_without_splitting():
    assert WorkerResources(3, 5).nodes_required(8) == 3


def test_distributed_worker_requires_whole_nodes():
    with pytest.raises(ValueError, match="multiple"):
        WorkerResources(1, 12).placement(8)


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
