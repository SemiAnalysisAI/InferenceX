"""Describe fixed-window progress without asserting scheduler causality."""

from bisect import bisect_right
from itertools import pairwise
from statistics import mean, median, pstdev


def progress_diagnostics(records, start, end):
    if not records or end <= start:
        raise ValueError("A nonempty cohort and positive interval are required")
    events = [r["events"] for r in records]
    for seq in events:
        if not seq or seq[0][0] > start:
            raise ValueError("Each stream must cover the interval start")
        if any(b[0] < a[0] or b[1] < a[1] for a, b in pairwise(seq)):
            raise ValueError("Progress must be monotonic")
    times = [[t for t, n in seq] for seq in events]

    def interval(lo, hi):
        counts = [
            seq[bisect_right(ts, hi) - 1][1] - seq[bisect_right(ts, lo) - 1][1]
            for seq, ts in zip(events, times)
        ]
        avg = mean(counts)
        ordered = sorted(counts)
        return {
            "start": lo,
            "end": hi,
            "per_request_tokens": counts,
            "tokens_per_second": sum(counts) / (hi - lo),
            "min_tokens": min(counts),
            "median_tokens": median(counts),
            "max_tokens": max(counts),
            "mean_tokens": avg,
            "coefficient_of_variation": pstdev(counts) / avg if avg else None,
            "zero_progress_requests": counts.count(0),
            "largest_adjacent_token_gap": max(
                (b - a for a, b in pairwise(ordered)), default=0
            ),
        }

    return {
        "publication_status": "held_pending_representativeness_review",
        "qualification": "Client-observed progress; sampled running counts do not prove uniform execution.",
        "whole": interval(start, end),
        "fixed_quarters": [
            interval(start + (end - start) * i / 4, start + (end - start) * (i + 1) / 4)
            for i in range(4)
        ],
    }
