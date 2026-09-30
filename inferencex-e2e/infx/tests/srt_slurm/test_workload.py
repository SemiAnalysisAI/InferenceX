"""Runtime workload binding preserves native recipe tuning and special containers."""

from copy import deepcopy

import pytest

from infx.srt_slurm.workload import bind_workload


def runtime(**changes):
    return {
        "IMAGE": "registry/server:new", "MODEL": "org/new-model", "IS_AGENTIC": "0",
        "ISL": "1024", "OSL": "128", "CONC_LIST": "4 8", "IS_MULTINODE": "true",
        **changes,
    }


def test_native_binding_replaces_workload_and_keeps_tuning_and_distinct_images():
    recipe = {
        "model": {"path": "hf:org/old-model", "container": "old-image", "precision": "fp8"},
        "identity": {"model": {"repo": "org/old-model"}, "container": {"image": "old-image"}},
        "roles": {
            "prefill": {"container": "special-prefill", "args": {"tensor-parallel-size": 8}},
            "decode": {"args": {"max-num-seqs": 32, "speculative-model": "org/draft"}},
        },
        "frontend": {"container_image": "router:fixed"},
        "benchmark": {"type": "aiperf", "isl": 32, "osl": 16, "concurrencies": [1, 2, 4]},
    }
    original = deepcopy(recipe)
    bound = bind_workload(recipe, runtime())
    assert bound["model"] == {
        "path": "hf:org/new-model", "container": "registry/server:new", "precision": "fp8",
    }
    assert bound["identity"] == {
        "model": {"repo": "org/new-model"}, "container": {"image": "registry/server:new"},
    }
    assert bound["benchmark"] == {
        "type": "aiperf", "isl": 1024, "osl": 128, "concurrencies": [4, 8], "env": {},
    }
    assert bound["roles"] == {
        "prefill": {"container": "special-prefill", "args": {"tensor-parallel-size": 8}},
        "decode": {"args": {"max-num-seqs": 32, "speculative-model": "org/draft"}},
    }
    assert bound["frontend"]["container_image"] == "router:fixed"
    assert recipe == original


def test_multinode_custom_client_receives_selected_list_and_lengths():
    recipe = {"model": {}, "benchmark": {"type": "custom", "env": {
        "MODEL": "org/old", "IMAGE": "old", "ISL": "1", "OSL": "2", "CONC": "99",
        "CONC_LIST": "1 2 4", "CLIENT_BACKEND": "openai-chat",
    }}}
    bound = bind_workload(recipe, runtime())
    assert bound["benchmark"]["env"] == {
        "MODEL": "org/new-model", "IMAGE": "registry/server:new", "ISL": "1024", "OSL": "128",
        "CONC_LIST": "4 8", "CLIENT_BACKEND": "openai-chat",
    }
    assert bound["benchmark"]["concurrencies"] == [4, 8]


def test_removed_workload_fields_are_populated_for_single_node_custom_client():
    bound = bind_workload(
        {"benchmark": {"type": "custom"}}, runtime(CONC_LIST="8", IS_MULTINODE="false")
    )
    assert bound["model"] == {"path": "hf:org/new-model", "container": "registry/server:new"}
    assert bound["benchmark"] == {"type": "custom", "env": {
        "MODEL": "org/new-model", "ISL": "1024", "OSL": "128", "CONC": "8",
    }}


def test_agentic_keeps_trace_lengths_and_requires_one_concurrency():
    recipe = {"benchmark": {"type": "custom", "env": {
        "ISL": "trace-input", "OSL": "trace-output", "CONC_LIST": "2",
    }}}
    env = runtime(IS_AGENTIC="1", CONC_LIST="8")
    del env["ISL"], env["OSL"]
    bound = bind_workload(recipe, env)
    assert bound["benchmark"]["env"] == {
        "ISL": "trace-input", "OSL": "trace-output", "MODEL": "org/new-model",
        "CONC": "8", "CONC_LIST": "8",
    }
    with pytest.raises(ValueError, match="exactly one concurrency"):
        bind_workload(recipe, {**env, "CONC_LIST": "4 8"})


def test_model_references_follow_new_model_without_changing_aliases_or_drafts():
    recipe = {
        "model": {"path": "hf:org/old-model"},
        "engine": {"type": "trtllm", "served_model_name": "org/old-model"},
        "roles": {
            "prefill": {"args": {
                "served-model-name": "org/old-model", "tokenizer-path": "org/old-model",
                "speculative-model": "org/old-model",
            }, "env": {"DYN_TRTLLM_SERVED_MODEL_NAME": "org/client-id"}},
            "decode": {"args": {
                "served-model-name": "stable-alias", "tokenizer-path": "/mounted-tokenizer",
                "speculative_config": {"model": "org/draft"},
            }},
        },
        "benchmark": {"type": "custom", "env": {
            "MODEL": "org/client-id", "SERVED_MODEL_NAME": "stable-alias", "TOKENIZER": "org/old-model",
        }},
    }
    bound = bind_workload(recipe, runtime(CONC_LIST="4"))
    assert bound["engine"]["served_model_name"] == "org/new-model"
    assert bound["roles"]["prefill"] == {"args": {
        "served-model-name": "org/new-model", "tokenizer-path": "org/new-model",
        "speculative-model": "org/old-model",
    }, "env": {"DYN_TRTLLM_SERVED_MODEL_NAME": "org/new-model"}}
    assert bound["roles"]["decode"]["args"] == {
        "served-model-name": "stable-alias", "tokenizer-path": "/mounted-tokenizer",
        "speculative_config": {"model": "org/draft"},
    }
    assert bound["benchmark"]["env"]["SERVED_MODEL_NAME"] == "stable-alias"
    assert bound["benchmark"]["env"]["TOKENIZER"] == "org/new-model"


@pytest.mark.parametrize("name", ["MODEL", "IMAGE", "IS_AGENTIC", "ISL", "OSL", "CONC_LIST"])
def test_missing_or_empty_runtime_inputs_fail_before_binding(name):
    env = runtime()
    del env[name]
    with pytest.raises(ValueError, match="Missing runtime input"):
        bind_workload({"benchmark": {}}, env)
    message = "Missing runtime input" if name == "CONC_LIST" else f"Missing runtime input: {name}"
    with pytest.raises(ValueError, match=message):
        bind_workload({"benchmark": {}}, {**env, name: " "})


@pytest.mark.parametrize("changes,message", [
    ({"CONC_LIST": "0"}, "positive integer"),
    ({"CONC_LIST": "2 2"}, "unique"),
    ({"CONC_LIST": "4 8", "CONC": "4"}, "must match"),
    ({"ISL": "-1"}, "positive integer"),
    ({"OSL": "1.5"}, "positive integer"),
    ({"IS_AGENTIC": "true"}, "IS_AGENTIC"),
])
def test_invalid_workload_inputs_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        bind_workload({"benchmark": {}}, runtime(**changes))


def test_scalar_concurrency_input_and_matching_list_are_supported():
    env = runtime(CONC="4", CONC_LIST="4")
    assert bind_workload({"benchmark": {}}, env)["benchmark"]["concurrencies"] == [4]
    del env["CONC_LIST"]
    assert bind_workload({"benchmark": {}}, env)["benchmark"]["concurrencies"] == [4]


def test_multinode_empty_scalar_concurrency_does_not_shadow_the_list():
    bound = bind_workload({"benchmark": {}}, runtime(CONC="", CONC_LIST="4 8"))
    assert bound["benchmark"]["concurrencies"] == [4, 8]
    single = bind_workload({"benchmark": {}}, runtime(CONC="4", CONC_LIST=""))
    assert single["benchmark"]["concurrencies"] == [4]


def test_cluster_model_alias_does_not_rewrite_intentional_served_name():
    bound = bind_workload({
        "model": {"path": "stable-alias"},
        "engine": {"type": "trtllm", "served_model_name": "stable-alias"},
        "roles": {"agg": {"args": {"served-model-name": "stable-alias"}}},
        "benchmark": {"type": "custom", "env": {"MODEL": "org/old-model"}},
    }, runtime(CONC_LIST="4"))
    assert bound["model"]["path"] == "hf:org/new-model"
    assert bound["engine"]["served_model_name"] == "stable-alias"
    assert bound["roles"]["agg"]["args"]["served-model-name"] == "stable-alias"
    assert bound["benchmark"]["env"]["MODEL"] == "org/new-model"


def test_identity_model_supplies_references_when_cluster_path_is_an_alias():
    bound = bind_workload({
        "model": {"path": "cluster-alias"},
        "identity": {"model": {"repo": "org/old-model"}},
        "engine": {"served_model_name": "org/old-model"},
        "frontend": {"env": {"MODEL": "org/old-model", "SERVED_MODEL_NAME": "cluster-alias"}},
        "roles": {"agg": {"args": {
            "tokenizer-path": "org/old-model", "served-model-name": "cluster-alias",
        }}},
        "benchmark": {"type": "sa-bench"},
    }, runtime())
    assert bound["engine"]["served_model_name"] == "org/new-model"
    assert bound["frontend"]["env"] == {
        "MODEL": "org/new-model", "SERVED_MODEL_NAME": "cluster-alias",
    }
    assert bound["roles"]["agg"]["args"] == {
        "tokenizer-path": "org/new-model", "served-model-name": "cluster-alias",
    }
    assert bound["identity"] == {
        "model": {"repo": "org/new-model"}, "container": {"image": "registry/server:new"},
    }


def test_removed_identity_fields_are_filled_while_frameworks_are_preserved():
    bound = bind_workload({
        "identity": {"frameworks": [{"name": "vllm", "version": "0.1"}]},
        "benchmark": {"type": "sa-bench"},
    }, runtime())
    assert bound["identity"] == {
        "frameworks": [{"name": "vllm", "version": "0.1"}],
        "model": {"repo": "org/new-model"},
        "container": {"image": "registry/server:new"},
    }
