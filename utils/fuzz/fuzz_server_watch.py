import pytest
from hypothesis import given, strategies as st

from infx.bench_serving import server_watch


@given(workers=st.lists(st.booleans(), min_size=1, max_size=20), reverse=st.booleans(),
       engine=st.sampled_from(["sglang::scheduler", "EngineCore", "VllmWorker", "TPWorker", "trtllm-worker"]),
       data=st.data())
def test_server_snapshot_tracks_owned_engines_through_transient_parents(workers, reverse, engine, data):
    processes = ["100 1 server-wrapper", "900 1 unrelated-server"]
    expected = {"100": "start-100"}
    states = {100: ("S", "start-100"), 900: ("S", "start-900")}
    for index, worker in enumerate(workers):
        pid = 101 + index
        parent = data.draw(st.integers(100, pid - 1), label=f"parent of {pid}")
        command = engine if worker else "tokenizer"
        processes.extend([f"{pid} {parent} {command}", f"{pid + 800} {parent + 800} {command}"])
        states[pid] = ("S", f"start-{pid}")
        states[pid + 800] = ("S", f"start-{pid + 800}")
        if worker:
            expected[str(pid)] = f"start-{pid}"
    if reverse:
        processes.reverse()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server_watch.subprocess, "check_output", lambda *args, **kwargs: "\n".join(processes))
        patch.setattr(server_watch, "process_state", states.get)
        required = server_watch.snapshot(100)
        assert required == expected
        assert server_watch.healthy(required)
        for pid in states.keys() - {int(pid) for pid in expected}:
            states[pid] = None
        assert server_watch.healthy(required)
        victim = int(data.draw(st.sampled_from(sorted(expected)), label="required process"))
        failure = data.draw(st.sampled_from([None, ("Z", expected[str(victim)]),
                                             ("S", "different-start")]), label="worker failure")
        states[victim] = failure
        assert not server_watch.healthy(required)
        if failure is None or failure[0] == "Z":
            with pytest.raises(ValueError, match="Server exited"):
                server_watch.snapshot(100)
