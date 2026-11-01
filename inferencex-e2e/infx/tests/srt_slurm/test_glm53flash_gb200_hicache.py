"""Keep the experimental GB200 NVFP4 child paired and its ON eval explicit."""

from pathlib import Path

from infx.clusters import load_inventory
from infx.clusters.slurm import slurm_settings
from infx.matrix.plan import build_plan
from infx.srt_slurm.generate import (
    _apply_acceptance_and_validate,
    _cluster_fabric,
    bound_variant,
    point_environment,
)
from infx.srt_slurm.synthetic_acceptance import acceptance_length
from infx.srt_slurm.variants import expand_variants
from infx.srt_slurm.workload import resolve_fabric


ROOT = Path(__file__).resolve().parents[3]
PREFIX = "glm5.3flash-fp4-gb200-sglang-agentic-mtp-hicache"


def test_gb200_hicache_frontier_and_real_verification_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "utils/srt-slurm/src"))
    inventory = load_inventory(ROOT / "configs/runners.yaml")
    cluster = inventory.cluster_for(inventory.labels["cluster:gb200-nv"][0])
    settings = slurm_settings(cluster).srt_slurm
    assert settings is not None
    common = {
        "description": ["Experimental GB200 NVFP4 child plan validation"],
        "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/4714",
        "scenario-type": ["agentic-coding"],
    }
    plan = build_plan(
        [
            {**common, "config-keys": [f"{PREFIX}-frontier"], "no-evals": True},
            {
                **common,
                "config-keys": [f"{PREFIX}-quality"],
                "evals-only": True,
                "all-evals": True,
            },
        ],
        base_ref="HEAD",
        head_ref="HEAD",
        config_files=[str(ROOT / "configs/nvidia-master.yaml")],
        runner_config=str(ROOT / "configs/runners.yaml"),
    ).model_dump(by_alias=True)
    performance = plan["single_node"]["agentic"]
    assert {(row["kv-offloading"], row["conc"]) for row in performance} == {
        *(("none", conc) for conc in (24, 32, 40, 48, 64)),
        *(("dram", conc) for conc in (32, 48, 64)),
    }
    assert len(performance) == 8
    assert not any(row.get("run-eval") for row in performance)
    evaluations = plan["agentic_evals"]
    assert len(evaluations) == 1
    assert evaluations[0]["kv-offloading"] == "dram"
    assert evaluations[0]["conc"] == 64
    assert evaluations[0]["run-eval"] is True

    for row in [*performance, *evaluations]:
        environment = point_environment(row)
        is_eval = row in evaluations
        if is_eval:
            environment.update(RUN_EVAL="true", EVAL_ONLY="true")
        _, recipe = bound_variant(
            row, environment, ROOT, expand=expand_variants
        )
        recipe = resolve_fabric(recipe, _cluster_fabric(cluster))
        _apply_acceptance_and_validate(
            recipe, environment, row["srt-recipe"], settings.extra
        )
        args = recipe["roles"]["agg"]["args"]
        env = recipe["roles"]["agg"]["env"]
        assert args["max-running-requests"] == 2 * row["conc"]
        assert args["cuda-graph-max-bs-decode"] == 2 * row["conc"]
        assert args["context-length"] == 1048576
        assert args["quantization"] == "modelopt_fp4"
        assert args["speculative-num-steps"] == 5
        assert args["speculative-num-draft-tokens"] == 6
        assert env["SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE"] == "0"
        assert env["SGLANG_GLM_NEXTN_MOE_PTPC"] == "0"
        assert "hicache-size" not in args
        assert "hicache-ratio" not in args
        if row["kv-offloading"] == "dram":
            assert row["kv-offload-backend"]["name"] == "hicache"
            assert args["enable-hierarchical-cache"] is True
            assert args["hicache-host-memory-fraction"] == 0.75
            assert args["hicache-write-policy"] == "write_through"
            assert args["hicache-io-backend"] == "kernel"
            assert args["hicache-mem-layout"] == "page_first"
        else:
            assert "enable-hierarchical-cache" not in args
            assert not any(name.startswith("hicache-") for name in args)
        measured = acceptance_length(recipe, "sglang", environment)
        if is_eval:
            assert measured is None
            assert not any(name.startswith("SGLANG_SIMULATE_ACC_") for name in env)
        else:
            assert measured == 3.603536
            assert env["SGLANG_SIMULATE_ACC_LEN"] == "3.603536"
            assert row["duration"] == 3600
