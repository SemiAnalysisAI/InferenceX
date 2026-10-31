import textwrap

import pytest

from infx.serving.schema import ServingError, load_serving, runner_gpus, variants
from infx.clusters import load_inventory

BASE = """\
name: qwen-tp1
runner: cluster:hcm-hgx02
image: vllm/vllm-openai:v0.29.0
model: Qwen/Qwen3.8-27B
gpus: 1
serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144
"""


def yaml_of(extra: str) -> str:
    return BASE + textwrap.dedent(extra)


def errors(text: str) -> str:
    with pytest.raises(ServingError) as caught:
        load_serving(text)
    return str(caught.value)


def test_engine_only_agentx_with_smoke_override():
    spec = load_serving(yaml_of("scenario: agentx\noverrides: {duration: 180}\nconc: [8, 16]\n"))
    assert spec.conc == [8, 16]
    [variant] = variants(spec)
    assert variant.label == "agentx-cc-062126-smoke"


def test_synthetic_grid_expands_to_points():
    spec = load_serving(
        yaml_of(
            "scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: [0, 50], search: true}\nconc: 1\n"
        )
    )
    assert [v.label for v in variants(spec)] == [
        "sweep-isl8k-osl1k-cache0",
        "sweep-isl8k-osl1k-cache50",
        "sweep-isl32k-osl1k-cache0",
        "sweep-isl32k-osl1k-cache50",
    ]


def test_osl_numbers_read_as_labels():
    spec = load_serving(
        yaml_of("scenario: synthetic\nworkload: {isl: 8k, osl: 512, cache: 0}\nconc: [4]\n")
    )
    assert variants(spec)[0].label == "fixed-isl8k-osl512-cache0"


@pytest.mark.parametrize(
    "extra, message",
    [
        ("scenario: agentx\noverrides: {warmup: 2}\nconc: 8\n", "overrides: unknown key(s) warmup"),
        ("scenario: synthetic\nconc: 1\n", "workload: synthetic needs"),
        (
            "scenario: agentx\nworkload: {isl: 8k, osl: 1k, cache: 0}\nconc: 1\n",
            "workload/slo: only for scenario synthetic",
        ),
        (
            "scenario: synthetic\nworkload: {isl: 8k, osl: 1k, cache: 0, search: true}\nconc: [8, 16]\n",
            "conc: workload.search finds the concurrency itself",
        ),
        (
            "scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: 0}\nconc: [8, 16]\n",
            "conc: a workload grid runs one job per point",
        ),
        ("scenario: agentx\nconc: 8\nenv: {PORT: '1'}\n", "env.PORT: reserved"),
        ("scenario: agentx\nconc: 8\nduration: 180\n", "duration: set it with overrides"),
        (
            "scenario: agentx\nconc: 8\nreplicas: 2\n",
            "replicas: gpus 1 is not a multiple of replicas 2",
        ),
        ("scenario: agentx-cc-062126\nconc: 8\n", "scenario:"),
    ],
)
def test_bad_yaml_names_field_and_fix(extra, message):
    assert message in errors(yaml_of(extra))


def test_runner_must_be_a_vng_cluster_slot_or_pool():
    inventory = load_inventory()
    assert runner_gpus("cluster:hcm-hgx02", inventory) == 8
    assert runner_gpus("hcm-hgx02_01", inventory) == 8
    assert runner_gpus("h200-hcm", inventory) == 4  # the smallest node of the pool
    assert runner_gpus("h200-han", inventory) == 1
    with pytest.raises(ServingError, match="runner: h100-cw_00 is not a VNG serving runner"):
        runner_gpus("h100-cw_00", inventory)


def test_gpus_above_the_node_fail():
    text = BASE.replace("runner: cluster:hcm-hgx02", "runner: cluster:hcm-hgx01").replace(
        "gpus: 1", "gpus: 8"
    )
    assert "gpus: 8 > 4 GPUs per node" in errors(text + "scenario: agentx\nconc: 8\n")


ROUTER = """\
scenario: tau2
conc: 8
router:
  image: vllm/vllm-openai:v0.29.0
  serve: vllm-router --port {{router_port}} --worker-urls {{engine_urls}} --prometheus-port {{router_metrics_port}}
"""


def test_router_metrics_port_can_be_picked_at_run_time():
    assert load_serving(yaml_of(ROUTER)).router.metrics_port is None


def test_router_metrics_port_fixed_and_placeholder_is_ambiguous():
    text = yaml_of(ROUTER.replace("router:\n", "router:\n  metrics_port: 29000\n"))
    assert "router.metrics_port: drop it" in errors(text)


def test_router_metrics_port_placeholder_in_env_is_ambiguous_with_a_fixed_port():
    router = ROUTER.replace("--prometheus-port {{router_metrics_port}}", "")
    router = router.replace(
        "router:\n", "router:\n  metrics_port: 29000\n  env: {METRICS: '{{router_metrics_port}}'}\n"
    )
    assert "router.metrics_port: drop it" in errors(yaml_of(router))


def test_image_may_be_a_squash_file_on_the_node():
    local = BASE.replace("image: vllm/vllm-openai:v0.29.0", "image: /data/images/vllm-v0.29.0.sqsh")
    assert (
        load_serving(local + "scenario: tau2\nconc: 8\n").image == "/data/images/vllm-v0.29.0.sqsh"
    )
    pool = local.replace("runner: cluster:hcm-hgx02", "runner: h200-han")
    assert "use runner: cluster:<id> of the node that holds the file" in errors(
        pool + "scenario: tau2\nconc: 8\n"
    )
    shell = BASE.replace("image: vllm/vllm-openai:v0.29.0", "image: /data/images/$(id).sqsh")
    assert "image: only letters" in errors(shell + "scenario: tau2\nconc: 8\n")


REMOTE = """\
name: gpt5-remote
runner: cluster:hcm-rtx01
endpoint:
  url: https://api.openai.com/v1/
  model: gpt-5
  api-key-secret: OPENAI_API_KEY
scenario: agentx
conc: [4]
"""


def test_remote_endpoint_needs_no_engine_fields():
    spec = load_serving(REMOTE)
    assert spec.endpoint.url == "https://api.openai.com"
    assert spec.endpoint.model == "gpt-5"
    assert spec.endpoint.api_key_secret == "OPENAI_API_KEY"
    assert spec.endpoint.tokenizer == "builtin"
    assert spec.image is None and spec.gpus is None


def test_remote_tokenizer_accepts_encodings_and_hf_ids():
    for tokenizer in ("o200k_harmony", "zai-org/GLM-5.2"):
        text = REMOTE.replace("  api-key-secret", f"  tokenizer: {tokenizer}\n  api-key-secret")
        assert load_serving(text).endpoint.tokenizer == tokenizer


@pytest.mark.parametrize(
    "extra, message",
    [
        ("image: vllm/vllm-openai:v0.29.0\n", "image: not used with endpoint:"),
        ("gpus: 1\n", "gpus: not used with endpoint:"),
        ("model: Qwen/Qwen3.8-27B\n", "model: not used with endpoint:"),
        ("router: {image: x, serve: y}\n", "router: not used with endpoint:"),
    ],
)
def test_remote_rejects_engine_fields(extra, message):
    assert message in errors(REMOTE + extra)


@pytest.mark.parametrize(
    "old, new, message",
    [
        (
            "url: https://api.openai.com/v1/",
            "url: api.openai.com",
            "endpoint.url: must start with http:// or https://",
        ),
        (
            "api-key-secret: OPENAI_API_KEY",
            "api-key-secret: openai-key",
            "endpoint.api-key-secret: a GitHub secret name",
        ),
        ("model: gpt-5", "model: gpt 5", "endpoint.model: no whitespace or quotes"),
        (
            "api-key-secret: OPENAI_API_KEY",
            "tokenizer: gpt5",
            "endpoint.tokenizer: builtin, cl100k_base",
        ),
        (
            "api-key-secret: OPENAI_API_KEY",
            "apikey: x",
            "endpoint.apikey: unknown key (allowed: url, model, api-key-secret, tokenizer, tls-verify)",
        ),
    ],
)
def test_remote_field_errors(old, new, message):
    assert message in errors(REMOTE.replace(old, new))


def test_engine_yaml_without_image_says_what_is_missing():
    text = BASE.replace("image: vllm/vllm-openai:v0.29.0\n", "")
    assert "image: required (or use endpoint: for a remote endpoint)" in errors(
        text + "scenario: agentx\nconc: 8\n"
    )


ARMS = """\
arms:
  base: {}
  mtp:
    append: |
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
    env: {X: "1"}
scenario: agentx
conc: 4
"""


def test_arms_are_engine_variants_of_the_shared_serve():
    from infx.serving.schema import arm_specs

    spec = load_serving(yaml_of(ARMS))
    arms = dict(arm_specs(spec))
    assert list(arms) == ["base", "mtp"]
    assert arms["base"].serve == spec.serve and "--speculative-config" not in arms["base"].serve
    assert arms["mtp"].serve.startswith(spec.serve.rstrip()) and "--speculative-config" in arms["mtp"].serve
    assert arms["mtp"].env == {"X": "1"} and arms["base"].env == {}
    assert arm_specs(load_serving(yaml_of("scenario: agentx\nconc: 4\n"))) == [(None, load_serving(yaml_of("scenario: agentx\nconc: 4\n")))]


@pytest.mark.parametrize("extra, message", [
    ("arms: {base: {}, other: {}}\nscenario: agentx\nconc: 4\n", "arms.other: the same engine command and env as arms.base"),
    ("arms: {Base: {}}\nscenario: agentx\nconc: 4\n", "arm names use lowercase"),
    ("arms: {base: {extra: 1}}\nscenario: agentx\nconc: 4\n", "unknown key"),
    ("arms: {}\nscenario: agentx\nconc: 4\n", "arms"),
    ("arms: {a: {append: --numa-node 0}}\nscenario: agentx\nconc: 4\n", "arms.a.serve: drop --numa-node"),
])
def test_bad_arms_fail_with_one_line_each(extra, message):
    assert message in errors(yaml_of(extra))


def test_arms_do_not_combine_with_a_remote_endpoint():
    text = ("name: r\nrunner: cluster:hcm-hgx01\nendpoint: {url: 'http://h:8000/v1', model: m}\n"
            "arms: {a: {}}\nscenario: agentx\nconc: 4\n")
    assert "arms: not used with endpoint:" in errors(text)
