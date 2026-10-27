"""Plan the AgentX speculative acceptance gate from a sweep search space."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

from infx.srt_slurm.al_gate import main, plan, summarize  # noqa: E402

EAGLE = {"speculative-algorithm": "EAGLE"}


@pytest.fixture(autouse=True)
def shared_workload(tmp_path: Path) -> None:
    shared = tmp_path / "configs/srt-recipes"
    shared.mkdir(parents=True)
    command = {"benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/srt_agentic.sh"}}
    (shared / "agentic.yaml").write_text(yaml.safe_dump(command))
    (shared / "telemetry-native.yaml").write_text("{}\n")


def write_golden(directory: Path, concurrency: int | None = 2) -> Path:
    curve: dict[str, Any] = {"deepseek-v4-pro": {"thinking_on": {2: 2.0, 3: 2.4}}}
    if concurrency is not None:
        curve["collection"] = {"concurrency": concurrency, "renderer": "chat_template"}
    directory.mkdir(exist_ok=True)
    (directory / "dsv4_mtp.yaml").write_text(yaml.safe_dump(curve))
    return directory


def single_node_recipe(variants: dict[int, int | None]) -> dict[str, Any]:
    """An aggregated SGLang AgentX recipe whose override for each concurrency drafts K tokens."""
    overrides = {
        f"override_c{conc}": {
            "benchmark": {"env": {"CONC": str(conc)}},
            "roles": {"agg": {"args": {} if k is None else {**EAGLE, "speculative-num-steps": k}}},
        }
        for conc, k in variants.items()
    }
    base = {
        "name": "agentx", "engine": "sglang",
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4,
                          "args": {"tensor-parallel-size": 4}}},
    }  # fmt: skip
    return {"schema": 2, "base": base, **overrides}


def single_node_row(conc: int) -> dict[str, Any]:
    return {
        "model": "test/model", "model-prefix": "dsv4", "image": "test:tag", "precision": "fp8",
        "framework": "sglang", "runner": "cluster:test", "tp": 4, "pp": 1, "dcp-size": 1,
        "pcp-size": 1, "ep": 1, "dp-attn": False, "spec-decoding": "mtp", "conc": conc,
        "srt-recipe": "recipe.yaml", "kv-offloading": "none", "exp-name": f"dsv4_conc{conc}", "queue-token": f"q{conc}",
    }  # fmt: skip


def multi_node_row(conc: int, decode_tp: int, config: str, framework: str = "dynamo-sglang"):
    return {
        "model": "test/model", "model-prefix": "dsv4", "framework": framework,
        "runner": "cluster:test", "spec-decoding": "mtp", "conc": [conc],
        "srt-recipe": config, "image": "test:tag", "precision": "fp8", "kv-offloading": "none",
        "exp-name": f"dsv4_c{conc}", "prefill": {"num-worker": 1, "tp": 8},
        "decode": {"num-worker": 1, "tp": decode_tp},
    }  # fmt: skip


def write_recipe(workspace: Path, name: str, recipe: dict[str, Any]) -> str:
    """Mirror a multi-node recipe into the workspace and return its CONFIG_FILE."""
    path = workspace / "benchmarks/multi_node/srt-slurm-recipes/test" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe))
    return f"benchmarks/multi_node/srt-slurm-recipes/test/{name}"


def sglang_decode(tokens: int) -> dict[str, Any]:
    decode = {"args": {**EAGLE, "speculative-num-steps": tokens}}
    return {"schema": 2, "roles": {"prefill": {}, "decode": decode}}


def plan_multi_node(rows: list[dict[str, Any]], workspace: Path, golden_dir: Path) -> dict:
    return plan({"multi_node": {"agentic": rows}}, workspace, "thinking_on", golden_dir=golden_dir)


@pytest.mark.parametrize(("golden_concurrency", "representative"), [(2, 2), (64, 4)])
def test_gates_split_by_topology_and_draft_length(
    tmp_path: Path, golden_concurrency: int, representative: int
) -> None:
    golden_dir = write_golden(tmp_path / "golden", golden_concurrency)
    recipe = single_node_recipe({1: 3, 2: 3, 4: 3, 8: 2, 16: None})
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    c16, c32, wide = (write_recipe(tmp_path, f"{n}.yaml", sglang_decode(3)) for n in "abc")
    multi = [multi_node_row(16, 8, c16), multi_node_row(32, 8, c32), multi_node_row(16, 16, wide)]
    multi.append(multi_node_row(64, 8, "benchmarks/multi_node/srt-slurm-recipes/test/missing.yaml"))
    rows = [single_node_row(conc) for conc in (1, 2, 4, 8, 16)]
    rows[-1]["spec-decoding"] = "none"
    search_space = {"single_node": {"agentic": rows}, "multi_node": {"agentic": multi}}
    gates = plan(search_space, tmp_path, "thinking_on", golden_dir=golden_dir)

    # The non-drafting concurrency-16 point needs no gate.
    three, two = sorted(gates["single_node"], key=lambda gate: -gate["al-gate"]["draft-tokens"])
    gate_id = three["al-gate"].pop("id")
    # The smallest point at or above the golden concurrency represents its group, else the
    # largest one.
    assert three["conc"] == representative
    assert three["al-gate"] == {
        "curve": "dsv4_mtp",
        "draft-tokens": 3,
        "golden-acceptance-length": 2.4,
        "concurrency": golden_concurrency,
        "covers": [1, 2, 4],
    }
    assert (two["conc"], two["al-gate"]["covers"]) == (8, [8])
    assert two["al-gate"]["golden-acceptance-length"] == 2.0
    # Same topology at two concurrencies shares one gate; a wider decode is its own topology.
    multi = sorted(gates["multi_node"], key=lambda gate: gate["decode"]["tp"])
    assert [(g["decode"]["tp"], g["al-gate"]["covers"]) for g in multi] == [
        (8, [[16], [32]]),
        (16, [[16]]),
    ]
    ids = {gate_id, two["al-gate"]["id"], *(gate["al-gate"]["id"] for gate in multi)}
    assert len(ids) == 4 and all(len(i) == 12 and int(i, 16) >= 0 for i in ids)
    [unresolved] = gates["unresolved"]
    assert (unresolved["point"], unresolved["conc"]) == ("dsv4_c64", [64])
    assert "FileNotFoundError" in unresolved["reason"]
    assert "missing.yaml" in unresolved["reason"]
    # The id names the deployment and drafting, whichever of its points a sweep includes.
    alone = {"single_node": {"agentic": [single_node_row(4)]}}
    gates = plan(alone, tmp_path, "thinking_on", golden_dir=golden_dir)
    assert gates["single_node"][0]["al-gate"]["id"] == gate_id


@pytest.mark.parametrize(
    ("tokens", "concurrency", "reason"),
    [
        (5, 2, "planning error: ValueError: No golden acceptance for dsv4_mtp/thinking_on/5"),
        (3, None, "golden curve dsv4_mtp records no collection settings to replay"),
    ],
)
def test_points_the_gate_cannot_replay_are_unresolved(
    tmp_path: Path, tokens: int, concurrency: int | None, reason: str
) -> None:
    golden_dir = write_golden(tmp_path / "golden", concurrency)
    row = multi_node_row(16, 8, write_recipe(tmp_path, "k.yaml", sglang_decode(tokens)))
    gates = plan_multi_node([row], tmp_path, golden_dir)
    [entry] = gates["unresolved"]
    assert (gates["multi_node"], entry["point"], entry["conc"]) == ([], "dsv4_c16", [16])
    assert entry["reason"].startswith(reason)


def vllm_recipe(dp: int, tp: int, gpus_per_node: int | None, role: str = "agg", pp: int = 1):
    """A vLLM recipe whose generation worker runs ``dp`` replicas of TP x PP GPUs."""
    args = {"speculative-config": '{"method": "mtp", "num_speculative_tokens": 3}',
            "data-parallel-size": dp, "tensor-parallel-size": tp,
            "pipeline-parallel-size": pp}  # fmt: skip
    recipe = {"schema": 2, "engine": {"type": "vllm"}, "roles": {role: {"args": args}}}
    if gpus_per_node is not None:
        recipe["resources"] = {"gpus_per_node": gpus_per_node}
    return recipe


@pytest.mark.parametrize(
    ("framework", "recipe", "cluster_gpus", "exempt"),
    [
        # Eight data-parallel ranks over two 4-GPU nodes run as two processes.
        ("dynamo-vllm", vllm_recipe(dp=8, tp=1, gpus_per_node=4), 8, True),
        ("dynamo-vllm", vllm_recipe(dp=16, tp=1, gpus_per_node=4, role="decode"), 8, True),
        ("dynamo-vllm", vllm_recipe(dp=2, tp=2, pp=2, gpus_per_node=4), 8, True),
        # One 8-GPU node runs a single process for every rank.
        ("dynamo-vllm", vllm_recipe(dp=8, tp=1, gpus_per_node=8), 4, False),
        # A replica wider than a node keeps a single API leader for every rank.
        ("dynamo-vllm", vllm_recipe(dp=2, tp=8, gpus_per_node=4), 4, False),
        # Without resources.gpus_per_node, srtctl uses the cluster's node size.
        ("dynamo-vllm", vllm_recipe(dp=8, tp=1, gpus_per_node=None), 4, True),
        ("dynamo-vllm", vllm_recipe(dp=8, tp=1, gpus_per_node=None), 8, False),
        # Only Dynamo hands the benchmark one leader's metrics.
        ("vllm-disagg", vllm_recipe(dp=8, tp=1, gpus_per_node=4), 4, False),
    ],
)
def test_dynamo_vllm_data_parallel_workers_across_nodes_are_exempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    framework: str, recipe: dict[str, Any], cluster_gpus: int, exempt: bool,
) -> None:  # fmt: skip
    cluster = {"gpus-per-node": cluster_gpus, "arch": "x86_64", "scheduler": "slurm",
               "slurm": {"partition": "batch", "exclusive": True}}  # fmt: skip
    runners = {"labels": {"cluster:test": ["test_0"]}, "clusters": {"test": cluster}}
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump(runners))
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))
    row = multi_node_row(8, 8, write_recipe(tmp_path, "dp.yaml", recipe), framework)
    gates = plan_multi_node([row], tmp_path, write_golden(tmp_path / "golden"))
    reasons = [entry["reason"] for entry in gates["unresolved"]]
    assert (len(gates["multi_node"]), len(reasons)) == ((0, 1) if exempt else (1, 0))
    assert all("only the leader's metrics" in reason for reason in reasons)


def test_frameworks_the_gate_cannot_measure_are_exempt(tmp_path: Path) -> None:
    # Exempt points are reported without reading their recipes, which need not exist.
    rows = [
        multi_node_row(conc, 8, "benchmarks/multi_node/srt-slurm-recipes/test/missing.yaml", framework)
        for conc, framework in ((4, "dynamo-trt"), (1, "tilert"))
    ]
    gates = plan_multi_node(rows, tmp_path, write_golden(tmp_path / "golden"))
    trt, tilert = gates["unresolved"]
    assert (gates["multi_node"], trt["conc"], tilert["conc"]) == ([], [4], [1])
    assert "TRT-LLM metrics" in trt["reason"]
    assert "rejects streaming /v1/completions" in tilert["reason"]


def test_a_missing_module_fails_the_plan_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Override-format recipes are expanded by srtctl; its import fails in a broken environment.
    config = write_recipe(tmp_path, "override.yaml", {"base": sglang_decode(3), "override_a": {}})
    monkeypatch.setitem(sys.modules, "srtctl.core.config", None)
    golden_dir = write_golden(tmp_path / "golden")
    with pytest.raises(ImportError):
        plan_multi_node([multi_node_row(8, 8, config)], tmp_path, golden_dir)


def test_summary_lists_verdicts_missing_gates_and_unresolved_points() -> None:
    unresolved = [{"point": f"p{c}", "conc": c, "reason": "not mirrored"} for c in (1, 2, 4, 8)]
    unresolved.append({"point": "z", "conc": [1], "reason": "exempt: no metrics"})
    planned = {"single_node": [{}, {}], "multi_node": [{}], "unresolved": unresolved}
    gate = {"curve": "dsv4_mtp", "draft_tokens": 3, "golden_acceptance_length": 2.4}
    below = {"acceptance_length": 2.1, "ratio": 0.875, "status": "below"}
    error = {"acceptance_length": None, "ratio": None, "status": "error"}
    results = [
        {**gate, **below, "result_filename": "c2"},
        {**gate, **error, "result_filename": "c1"},
    ]
    lines = summarize(planned, results).splitlines()
    assert [line for line in lines if line.startswith("| `")] == [
        "| `c1` | dsv4_mtp | 3 | 2.40 | - | - | error |",
        "| `c2` | dsv4_mtp | 3 | 2.40 | 2.10 | 0.875 | below |",
    ]
    assert "1 planned gate(s) have no result; see their logs." in lines
    assert "### Unresolved points (5, no gate)" in lines
    assert [line for line in lines if line.startswith("- ")] == [
        "- not mirrored: 4 point(s), e.g. `p1` (conc 1), `p2` (conc 2), `p4` (conc 4)",
        "- exempt: no metrics: 1 point(s), e.g. `z` (conc [1])",
    ]


def test_cli_plans_and_summarizes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    search_space = tmp_path / "search-space.json"
    search_space.write_text('{"single_node": {"agentic": []}, "multi_node": {}}')
    output = tmp_path / "plan.json"
    argv = ["plan", str(search_space), "--workspace", str(tmp_path), "--output", str(output)]
    assert main([*argv, "--thinking-mode", "thinking_on"]) == 0
    assert json.loads(output.read_text()) == {"single_node": [], "multi_node": [], "unresolved": []}
    # Each gate's artifact downloads into its own directory, and the plan may be missing.
    results = tmp_path / "results" / "algate_0123456789ab"
    results.mkdir(parents=True)
    (results / "al_gate_x.json").write_text('{"result_filename": "x", "status": "pass"}')
    assert main(["summarize", str(tmp_path / "results"), "--plan", str(tmp_path / "none")]) == 0
    assert "| `x` | None | None | - | - | - | pass |" in capsys.readouterr().out
