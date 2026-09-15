import json
import os
import tempfile
from pathlib import Path

from hypothesis import given, strategies as st

from infx.results.collect_eval_results import collect_eval_rows
from infx.workflows.validate_reusable_sweep_artifacts import validate_agentic_artifacts, validate_eval_artifacts, validate_fixed_artifacts
from test_validate_reusable_sweep_artifacts import (
    agentic_result, fixed_result, multinode_eval_result, raw_eval_result,
    write_agentic_artifacts, write_raw_batched_eval_artifact,
)

from cases import CONCURRENCIES


@given(concs=CONCURRENCIES, agentic=st.booleans(), duplicate=st.booleans(), missing_raw=st.booleans())
def test_reusable_artifacts_preserve_all_identities(concs, agentic, duplicate, missing_raw):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        aggregate = root / "results_bmk"
        aggregate.mkdir()
        rows = [agentic_result(conc) if agentic else fixed_result(conc) for conc in concs]
        if agentic:
            for conc in concs:
                write_agentic_artifacts(root, conc)
            if missing_raw:
                next(root.glob("agentic_*")).rmdir()
        if duplicate:
            rows.append(rows[0])
        (aggregate / "agg_bmk.json").write_text(json.dumps(rows))
        errors = (validate_agentic_artifacts if agentic else validate_fixed_artifacts)(root)
        assert bool(errors) is (duplicate or agentic and missing_raw)


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


@given(concs=CONCURRENCIES, data=st.data(), legacy=st.booleans(), duplicate=st.booleans(),
       invalid_score=st.booleans())
def test_eval_reuse_requires_complete_valid_batches(concs, data, legacy, duplicate, invalid_score):
    completed = data.draw(st.sets(st.sampled_from(concs)), label="completed concurrencies")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        aggregate = root / "eval_results_all"
        aggregate.mkdir()
        rows = [multinode_eval_result(conc) for conc in completed]
        if duplicate and rows:
            rows.append(rows[0])
        (aggregate / "agg_eval_all.json").write_text(json.dumps(rows))
        write_raw_batched_eval_artifact(root, concs, completed_concs=sorted(completed),
                                       failed_concs=sorted(set(concs) - completed))
        raw = root / "eval_gptoss_8k1k_batch"
        if legacy:
            meta_path = raw / "meta_env.json"
            meta = json.loads(meta_path.read_text())
            meta.pop("failed_eval_concs")
            meta_path.write_text(json.dumps(meta))
        if invalid_score:
            for path in raw.glob("results*.json"):
                path.write_text(json.dumps(raw_eval_result(-0.01)))
        errors = validate_eval_artifacts(root)
    assert bool(errors) is (completed != set(concs) or duplicate or invalid_score)
