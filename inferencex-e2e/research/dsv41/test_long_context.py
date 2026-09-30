import pytest

from research.dsv41.long_context import steady_decode_window


def test_full_batch_window_excludes_prefill_and_counts_token_chunks():
    records = [
        {
            "success": True,
            "events": [[0, 1], [1, 5], [2, 9], [3, 13], [4, 17], [5, 21]],
        },
        {
            "success": True,
            "events": [[0.5, 1], [1.5, 5], [2.5, 9], [3.5, 13], [4.5, 17], [5.5, 21]],
        },
    ]
    result = steady_decode_window(records, burn=1, tail=1, gpu_count=4)
    assert result["duration_s"] == 1.5
    assert result["observed_tokens"] == 12
    assert result["per_request_tokens"] == [8, 4]
    assert result["equivalent_tpot_ms"] == 250
    assert result["output_tokens_per_second_per_gpu"] == 2


def test_serial_cohorts_cannot_claim_a_full_batch_decode_result():
    records = [
        {"success": True, "events": [[0, 1], [1, 5], [2, 9], [3, 13], [4, 17]]},
        {"success": True, "events": [[10, 1], [11, 5], [12, 9], [13, 13], [14, 17]]},
    ]
    with pytest.raises(ValueError, match="No common full-batch"):
        steady_decode_window(records, burn=1, tail=1, gpu_count=4)
