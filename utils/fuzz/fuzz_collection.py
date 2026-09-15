import json
import os
import tempfile
from pathlib import Path

from hypothesis import given, strategies as st

from infx.results.collect_eval_results import collect_eval_rows
from test_validate_reusable_sweep_artifacts import raw_eval_result

from cases import CONCURRENCIES


@given(concs=CONCURRENCIES, score=st.floats(0, 1, allow_nan=False), batched=st.booleans(),
       stale_mtime=st.booleans(), fractional_ns=st.integers(1, 999_999_999))
def test_eval_collection_uses_latest_result_per_concurrency(concs, score, batched, stale_mtime, fractional_ns):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        meta = {"conc": concs[0]}
        if batched:
            meta.update(eval_concs=concs, completed_eval_concs=concs, failed_eval_concs=[])
        (root / "meta_env.json").write_text(json.dumps(meta))
        for conc in concs:
            old = root / f"results_1970-01-01T00-00-01.000000000_conc{conc}.json"
            new = root / f"results_1970-01-01T00-00-01.{fractional_ns:09}_conc{conc}.json"
            old.write_text(json.dumps(raw_eval_result(1 - score)))
            new.write_text(json.dumps(raw_eval_result(score)))
            os.utime(old, ns=(9_000_000_000, 9_000_000_000))
            if stale_mtime:
                os.utime(new, ns=(0, 0))
        rows = collect_eval_rows(root)
    assert [row["conc"] for row in rows] == (sorted(concs) if batched else [concs[0]])
    assert [row["score"] for row in rows] == [score] * len(rows)
    assert all(row["infrastructure_success"] for row in rows)
