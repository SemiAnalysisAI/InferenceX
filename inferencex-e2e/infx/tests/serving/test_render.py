import shlex
import textwrap

import pytest

from infx.serving.render import infer_framework, render, split_serve
from infx.serving.schema import RouterSpec, ServingError, ServingSpec


def spec(serve: str, **extra) -> ServingSpec:
    fields = dict(
        name="x", runner="cluster:hcm-hgx02", image="img", model="Qwen/Qwen3.8-27B",
        gpus=1, serve=serve, scenario="agentx", conc=[8],
    )
    fields.update(extra)
    return ServingSpec.model_validate(fields)


def test_render_keeps_quoted_json_and_continuations():
    serve = textwrap.dedent("""\
        vllm serve Qwen/Qwen3.8-27B \\
          --max-model-len 262144 \\
          --default-chat-template-kwargs '{"enable_thinking": true}'
    """)
    out = render(spec(serve))
    assert shlex.split(out["serve-cmd"]) == [
        "vllm", "serve", "Qwen/Qwen3.8-27B", "--max-model-len", "262144",
        "--default-chat-template-kwargs", '{"enable_thinking": true}',
    ]


def test_render_drops_shell_comments():
    serve = "vllm serve Qwen/Qwen3.8-27B --max-model-len 262144  # long context\n  --max-num-seqs 64\n# tuned\n"
    assert shlex.split(render(spec(serve))["serve-cmd"]) == [
        "vllm", "serve", "Qwen/Qwen3.8-27B", "--max-model-len", "262144", "--max-num-seqs", "64",
    ]


@pytest.mark.parametrize("flags", [
    "--host 0.0.0.0 --port 8889",
    "--port=8889 --host=0.0.0.0",
    "--port 8889 --tp 1 --host 127.0.0.1",
])
def test_render_strips_host_and_port_forms(flags):
    argv = shlex.split(render(spec(f"vllm serve Qwen/Qwen3.8-27B {flags} --max-num-seqs 64"))["serve-cmd"])
    assert not [a for a in argv if a.startswith(("--host", "--port"))]
    assert "8889" not in argv and "0.0.0.0" not in argv and "127.0.0.1" not in argv
    assert argv[-2:] == ["--max-num-seqs", "64"]


def test_render_moves_leading_env_assignments():
    out = render(spec("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FOO=1 vllm serve Qwen/Qwen3.8-27B",
                      env={"FOO": "2"}))
    assert shlex.split(out["serve-cmd"])[:2] == ["vllm", "serve"]
    assert out["serve-env"].splitlines() == ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "FOO=2"]


def test_sglang_gets_enable_metrics_once():
    out = render(spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --tp 1"))
    assert out["framework"] == "sglang"
    assert shlex.split(out["serve-cmd"]).count("--enable-metrics") == 1
    already = render(spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --enable-metrics"))
    assert shlex.split(already["serve-cmd"]).count("--enable-metrics") == 1


def test_served_model_name_both_forms():
    assert render(spec("vllm serve Qwen/Qwen3.8-27B --served-model-name q38"))["served-model-name"] == "q38"
    assert render(spec("vllm serve Qwen/Qwen3.8-27B --served-model-name=q38"))["served-model-name"] == "q38"
    assert render(spec("vllm serve Qwen/Qwen3.8-27B"))["served-model-name"] == ""


PASTE_KV = '--kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'
QUOTED_KV = """--kv-events-config='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'"""


def dyn(engine: str = "python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --tp=8 --enable-metrics",
        router: str = "python3 -m dynamo.frontend --http-host=0.0.0.0 --http-port=8888 --router-mode=kv",
        **router_extra) -> ServingSpec:
    return spec(engine, runner="cluster:hcm-hgx02", gpus=8,
                router=RouterSpec(image="img-r", serve=router, **router_extra))


def test_dynamo_pair_renders_router_fields():
    out = render(dyn(router="FOO=1 python3 -m dynamo.frontend --http-host 0.0.0.0 --http-port=8888 --router-mode=kv",
                     env={"BAR": "2"}))
    assert out["framework"] == "sglang"
    assert shlex.split(out["serve-cmd"])[:3] == ["python3", "-m", "dynamo.sglang"]
    assert out["router-image"] == "img-r"
    assert shlex.split(out["router-cmd"]) == ["python3", "-m", "dynamo.frontend", "--router-mode=kv"]
    assert out["router-env"].splitlines() == ["FOO=1", "BAR=2"]
    assert out["router-profile"] == "dynamo"
    assert out["router-metrics-port"] == ""


def test_engine_only_has_empty_router_fields():
    out = render(spec("vllm serve Qwen/Qwen3.8-27B"))
    assert (out["router-image"], out["router-cmd"], out["router-env"]) == ("", "", "")
    assert (out["router-profile"], out["router-metrics-port"]) == ("", "")


def test_dynamo_engine_without_router_rejected():
    with pytest.raises(ServingError, match=r"^router: required for python3 -m dynamo.sglang"):
        render(spec("python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B"))


def test_dynamo_router_with_plain_engine_rejected():
    with pytest.raises(ServingError, match=r"^serve: python3 -m dynamo.frontend needs a python3 -m dynamo.sglang engine"):
        render(dyn(engine="python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B"))


def test_dynamo_engine_with_other_router_rejected():
    with pytest.raises(ServingError, match=r"^router.serve: a python3 -m dynamo.sglang engine needs python3 -m dynamo.frontend as its router"):
        render(dyn(router="python3 -m sglang_router.launch_router --port={{router_port}}"))


def test_dynamo_router_rejects_metrics_port():
    with pytest.raises(ServingError, match=r"^router.metrics_port: not used with python3 -m dynamo.frontend"):
        render(dyn(metrics_port=18888))


GENERIC_ROUTER = (
    "python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}} "
    "--prometheus-port=18888 --worker-urls={{engine_url}} --dp-aware --policy=cache_aware"
)


def sgl_router(router: str = GENERIC_ROUTER, **router_extra) -> ServingSpec:
    return spec("python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --tp 8 --port 30000",
                runner="cluster:hcm-hgx02", gpus=8,
                router=RouterSpec(image="img-r", serve=router, **router_extra))


def test_generic_router_keeps_command_and_placeholders():
    out = render(sgl_router(metrics_port=18888, env={"A": "{{engine_url}}"}))
    assert out["framework"] == "sglang"
    assert out["router-profile"] == "generic"
    assert out["router-metrics-port"] == "18888"
    assert shlex.split(out["router-cmd"]) == shlex.split(GENERIC_ROUTER)
    assert out["router-env"] == "A={{engine_url}}"
    assert "--port" not in shlex.split(out["serve-cmd"])


def test_generic_router_without_metrics_port():
    assert render(sgl_router())["router-metrics-port"] == ""


def test_unquoted_json_flag_from_the_paste_rejected():
    with pytest.raises(ServingError) as info:
        render(dyn(engine=f"python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --enable-metrics {PASTE_KV}"))
    assert str(info.value) == (
        "serve: --kv-events-config is not valid JSON after shell quoting; "
        "wrap it in single quotes: --kv-events-config='{...}'"
    )


def test_inline_jinja_chat_template_not_treated_as_json():
    out = render(spec("vllm serve Qwen/Qwen3.8-27B --chat-template '{% for m in messages %}{{ m.content }}{% endfor %}'"))
    assert "{% for m in messages %}" in out["serve-cmd"]
    out = render(spec("vllm serve Qwen/Qwen3.8-27B --chat-template '{{ messages }}'"))
    assert "{{ messages }}" in out["serve-cmd"]


def test_dynamo_engine_without_enable_metrics_rejected():
    with pytest.raises(ServingError) as info:
        render(dyn(engine="python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B"))
    assert str(info.value) == (
        "serve: python3 -m dynamo.sglang needs --enable-metrics (the metrics gate requires sglang:* series)"
    )


def test_single_quoted_json_flag_kept_verbatim():
    out = render(dyn(engine=f"python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --enable-metrics {QUOTED_KV}"))
    assert shlex.split(out["serve-cmd"])[-1] == (
        '--kv-events-config={"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:8900"}'
    )


@pytest.mark.parametrize("argv,framework", [
    (["vllm", "serve", "m"], "vllm"),
    (["python3", "-m", "sglang.launch_server", "--model-path", "m"], "sglang"),
    (["python", "-m", "sglang.launch_server"], "sglang"),
    (["sglang", "serve", "--model-path", "m"], "sglang"),
    (["python3", "-m", "dynamo.sglang", "--model-path=m"], "sglang"),
])
def test_infer_framework(argv, framework):
    assert infer_framework(argv) == framework


def test_unknown_engine_rejected():
    with pytest.raises(ServingError, match="serve: unknown engine"):
        render(spec("trtllm-serve Qwen/Qwen3.8-27B"))


def test_vllm_disable_log_stats_rejected():
    with pytest.raises(ServingError, match="--disable-log-stats"):
        render(spec("vllm serve Qwen/Qwen3.8-27B --disable-log-stats"))


def test_unbalanced_quote_rejected():
    with pytest.raises(ServingError, match="serve: cannot parse"):
        split_serve("vllm serve m --x '{")
