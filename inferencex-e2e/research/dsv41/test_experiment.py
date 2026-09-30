import pytest
from research.dsv41.experiment import kernel_samples


def event(cat, name, start, duration, pid=0, tid=7):
    return dict(cat=cat, name=name, ts=start, dur=duration, pid=pid, tid=tid)


def test_kernel_sum_excludes_launch_gaps_eviction_and_other_streams():
    trace = {"traceEvents": [
        event("gpu_user_annotation", "target_sample_0", 10, 20),
        event("user_annotation", "target_sample_0", 8, 30, pid=100),
        event("kernel", "eviction", 1, 5),
        event("kernel", "multiply", 11, 2),
        event("kernel", "reduce", 22, 3),
        event("kernel", "other_stream", 15, 4, tid=8),
        event("kernel", "other_device", 15, 4, pid=1),
    ]}
    assert kernel_samples(trace, 1) == [5]


def test_missing_gpu_kernels_is_an_error():
    with pytest.raises(RuntimeError, match="No GPU kernels"):
        kernel_samples({"traceEvents": [
            event("gpu_user_annotation", "target_sample_0", 10, 20),
        ]}, 1)


def test_duplicate_annotations_are_not_counted_as_two_samples():
    with pytest.raises(RuntimeError, match="Missing or duplicate"):
        kernel_samples({"traceEvents": [
            event("gpu_user_annotation", "target_sample_0", 10, 20),
            event("gpu_user_annotation", "target_sample_0", 40, 20),
        ]}, 2)
