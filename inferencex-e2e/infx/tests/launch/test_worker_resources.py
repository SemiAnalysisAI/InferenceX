import pytest

from infx.launch.resources import WorkerResources, worker_resources


@pytest.mark.parametrize('gpus,nodes,local,memory', [(4, 1, 4, 4000), (8, 1, 8, 8000), (16, 2, 8, 8000)])
def test_worker_allocation_uses_physical_node_share(gpus, nodes, local, memory):
    allocation = WorkerResources(1, gpus).allocate(8000, 8)
    assert (allocation.nodes_per_worker, allocation.local_gpus, allocation.memory_mib) == (nodes, local, memory)


def test_partial_workers_pack_without_splitting_or_overallocating_memory():
    request = WorkerResources(3, 5)
    assert request.nodes_required(8) == 3
    assert request.allocate(8000, 8).memory_mib == 5000


def test_distributed_worker_requires_whole_nodes():
    with pytest.raises(ValueError, match='multiple'):
        WorkerResources(1, 12).allocate(8000, 8)


def test_separate_processes_split_one_worker_budget_with_mib_rounding():
    allocation = WorkerResources(1, 4).allocate(3079397, 8)
    assert allocation.memory_mib == 1539698
    assert allocation.step_memory_mib(1) == 384924
    with pytest.raises(ValueError, match='evenly partition'):
        allocation.step_memory_mib(3)


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
