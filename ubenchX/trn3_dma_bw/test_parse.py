import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ubenchX.trn3_dma_bw import parse

ROW = 16384
TOTAL = 4 * ROW


def packet(queue, nbytes, start, end, engine=0):
    return {"queue_type": queue, "transfer_bytes": nbytes, "start_ts": start, "end_ts": end,
            "duration_ns": end - start, "engine_idx": engine}


def write_trace(tmp_path, packets, exec_start=0, exec_end=1000):
    trace = tmp_path / "parquet_exec_4" / "sg00"
    trace.mkdir(parents=True)
    pd.DataFrame(packets).to_parquet(trace / "DmaPacket.parquet")
    pd.DataFrame([{"execution_start_ts": exec_start, "execution_end_ts": exec_end}]).to_parquet(
        trace / "ExecutionInfo.parquet")
    return tmp_path / "parquet_exec_4"


def burst(queue):
    return [packet(queue, ROW, 100 * i, 100 * (i + 1), engine=i) for i in range(4)]


def test_static_burst_ignores_instruction_fetches_completion_writes_and_output_store(tmp_path):
    packets = burst("input")
    packets[-1]["transfer_bytes"] = ROW + 4
    packets += [packet("input", 4, 400, 401), packet("instruction", 10 * TOTAL, 0, 50),
                packet("output", 2048, 500, 510)]
    row = parse.parse_trace({"kernel": "load_static", "total_bytes": TOTAL}, write_trace(tmp_path, packets))

    assert row["GBps"] == pytest.approx(TOTAL / 400)
    assert row["bytes_ok"]
    assert row["pkt_bytes"] == ROW
    assert row["packets"] == 4
    assert row["trimmed_bytes"] == 0
    assert row["dma_engines"] == 4
    assert row["kernel_GBps_lb"] == pytest.approx(TOTAL / 1000)


def test_swdge_load_trims_output_store_that_shares_the_burst_queue(tmp_path):
    packets = burst("software_dynamic") + [packet("software_dynamic", 8192, 450, 480),
                                           packet("software_dynamic", 8192, 480, 500)]
    row = parse.parse_trace({"kernel": "load_swdge", "total_bytes": TOTAL}, write_trace(tmp_path, packets))

    assert row["trimmed_bytes"] == 2 * 8192
    assert row["window_us"] == pytest.approx(0.4)
    assert row["GBps"] == pytest.approx(TOTAL / 400)
    assert row["bytes_ok"]


def test_trailing_bytes_that_overlap_the_burst_raise(tmp_path):
    packets = burst("software_dynamic") + [packet("software_dynamic", 8192, 350, 420)]
    pq = write_trace(tmp_path, packets)

    with pytest.raises(RuntimeError, match="overlap the burst"):
        parse.parse_trace({"kernel": "load_swdge", "total_bytes": TOTAL}, pq)
