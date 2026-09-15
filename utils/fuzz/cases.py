from pathlib import Path

from hypothesis import strategies as st

from test_process_changelog import planning_inputs


ROOT = Path(__file__).resolve().parents[2]
CONCURRENCIES = st.lists(st.integers(1, 256), min_size=1, max_size=6, unique=True)
TEXT = st.text(alphabet="abcXYZ0123-_.=;'\"$()`[]*?|& \t中文é🙂", min_size=1, max_size=200)


def recipe(topology: str, concs: list[int], offload: bool = False) -> tuple[dict, dict]:
    master, runners = planning_inputs()
    config = master["single" if topology == "single" else "multi"]
    if topology == "aggregate":
        config["disagg"] = False
        config.pop("kv-p2p-transfer")
    for scenario, groups in config["scenarios"].items():
        for group in groups:
            space = group["search-space"][0]
            space["conc-list"] = list(concs)
            if topology == "aggregate":
                space["worker"] = space.pop("prefill")
                space.pop("decode")
                space["num-nodes"] = 1
            if scenario == "agentic-coding":
                space["kv-offloading"] = "dram" if offload else "none"
                if offload:
                    group["dram-utilization"] = 0.5
                    space["kv-offload-backend"] = {"name": "default"}
    return {"fixture": config}, runners


def rows_in_plan(plan: dict):
    for family in ("single_node", "multi_node"):
        for rows in plan.get(family, {}).values():
            yield from rows
    for bucket in ("evals", "agentic_evals", "multinode_evals", "multinode_agentic_evals"):
        yield from plan.get(bucket, [])
