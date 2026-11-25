import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.config import repository_root
from infx.srt_slurm import video

ROOT = repository_root()
ENV = {
    "IMAGE": "vllm/vllm-omni:v0.30.0@sha256:" + "f" * 64,
    "MODEL": "MiniMaxAI/MiniMax-H3",
    "PRECISION": "bf16",
    "GPU_COUNT": "2",
    "CONC_LIST": "1",
    "VIDEO_SPEC": "h3.yaml",
    "VIDEO_SET": "release-v1",
    "FRAMEWORK": "vllm-omni",
    "HF_HUB_CACHE": "/mnt/hf_hub_cache",
    "RESULT_FILENAME": "h3-native-release",
}


@pytest.mark.parametrize("spec", ["h3-v2.yaml", "h3.yaml"])
def test_fairness_requires_disabled_cross_request_caches(native_fragment, spec):
    raw, write = native_fragment
    selected = {**ENV, "FRAMEWORK": "sglang", "VIDEO_SPEC": spec, "GPU_COUNT": "4"}
    with pytest.raises(ValueError, match="--disable-conditioning-cache"):
        video.prepare(write(), selected, root=ROOT)

    raw["services"][0]["args"].append("--disable-conditioning-cache")
    recipe = video.prepare(write(raw), selected, root=ROOT)
    [server] = recipe["services"]
    assert server["args"][-1] == "--disable-conditioning-cache"
    assert server["env"]["VIDEO_CONTAINER_IMAGE"] == selected["IMAGE"]
    assert recipe["benchmark"]["env"]["VIDEO_CONTAINER_IMAGE"] == selected["IMAGE"]

    _as_vllm(raw)
    selected["FRAMEWORK"] = "vllm-omni"
    with pytest.raises(ValueError, match="OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0"):
        video.prepare(write(raw), selected, root=ROOT)

    server_env = raw["services"][0]["env"]
    server_env["OMNI_DIFFUSION_PROMPT_EMBED_CACHE"] = "1"
    with pytest.raises(ValueError, match="OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0"):
        video.prepare(write(raw), selected, root=ROOT)

    server_env["OMNI_DIFFUSION_PROMPT_EMBED_CACHE"] = "0"
    recipe = video.prepare(write(raw), selected, root=ROOT)
    assert recipe["services"][0]["env"]["OMNI_DIFFUSION_PROMPT_EMBED_CACHE"] == "0"


def test_server_port_must_be_a_tcp_port_before_submission(native_fragment):
    raw, write = native_fragment
    port = 65536
    [server] = raw["services"]
    server["args"][server["args"].index("--port") + 1] = str(port)
    server["env"]["VIDEO_SERVER_PORT"] = raw["benchmark"]["env"]["VIDEO_SERVER_PORT"] = str(port)
    server["readiness"]["http"]["port"] = port
    with pytest.raises(ValueError, match="TCP port"):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


@pytest.fixture
def native_fragment(tmp_path):
    """A small recipe whose engine settings remain owned by the recipe."""
    raw = {
        "schema": 2,
        "name": "video-fixture",
        "resources": {"gpu_type": "test-gpu"},
        "frontend": {"type": "none"},
        "services": [{
            "name": "video-server", "type": "generic", "nodes": 1, "critical": True,
            "command": ["/usr/bin/python3", "-m", "infx.bench", "video", "serve"],
            "args": [
                "--model-view", "snapshot", "--", "/usr/bin/python3", "-m",
                "evaluator.srt_selected", "serve", "--model-type", "diffusion",
                "--model-path", "{model_path}", "--model-id", "{model_id}",
                "--revision", "{model_revision}", "--port", "30283",
                "--num-gpus", "{gpus_per_node}", "--host", "0.0.0.0",
                "--tp-size", "{gpus_per_node}", "--ulysses-degree", "{gpus_per_node}",
                "--attention-backend", "fa", "--performance-mode", "speed",
            ],
            "env": {"VIDEO_SERVER_PORT": "30283", "VIDEO_RUNTIME_REVISION": "b" * 40},
            "readiness": {"http": {"port": 30283, "path": "/health"}},
        }],
        "benchmark": {
            "type": "custom",
            "command": "python3 -m infx.bench video client --request-seconds 73",
            "env": {"VIDEO_SERVER_SERVICE": "video-server", "VIDEO_SERVER_PORT": "30283"},
        },
    }
    path = tmp_path / "agg-ulysses.yaml"

    def write(recipe=raw):
        path.write_text(yaml.safe_dump(recipe))
        return path

    return raw, write


def test_selected_engine_revision_must_match_bound_point(native_fragment):
    raw, write = native_fragment
    args = raw["services"][0]["args"]
    args[args.index("--revision") + 1] = "c" * 40
    with pytest.raises(ValueError, match="--revision"):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


def _as_vllm(raw):
    [server] = raw["services"]
    server["args"] = [
        "--model-view", "h3-fl2va", "--", "vllm", "serve", "{model_path}",
        "--served-model-name", "{model_id}", "--port", "30283",
        "--num-gpus", "{gpus_per_node}", "--host", "0.0.0.0", "--omni",
        "--cache-config", '{"minimax_h3_adaln_cache": false}',
    ]
    server["env"] = {"VIDEO_SERVER_PORT": "30283", "VIDEO_RUNTIME_REVISION": "b" * 40}


@pytest.mark.parametrize(("fault", "refusal"), [
    ({"args": ["--cache-backend", "tea_cache"]}, "argument '--cache-backend'"),
    ({"args": ["--quantization=fp8"]}, "argument '--quantization=fp8'"),
    ({"args": ["-qfp8"]}, "argument '-qfp8'"),
    ({"args": ["--diffusion_attention_back", "FLASH_ATTN"]}, "argument '--diffusion_attention_back'"),
    ({"args": ["--diffusion-attention-backend", "RAINFUSION_ATTN"]}, "flag --diffusion-attention-backend="),
    ({"args": ["--cache-config", '{"minimax_h3_adaln_cache": true}']}, "flag --cache-config="),
    ({"env": {"DIFFUSION_CACHE_BACKEND": "cache_dit"}}, "variable DIFFUSION_CACHE_BACKEND"),
    ({"env": {"VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "FP8"}}, "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION='FP8'"),
    ({"preamble": "export DIFFUSION_CACHE_BACKEND=cache_dit"}, "preamble"),
])  # fmt: skip
def test_vllm_omni_recipe_binds_only_reviewed_settings(native_fragment, fault, refusal):
    raw, write = native_fragment
    _as_vllm(raw)
    [server] = raw["services"]
    server["args"] += ["--diffusion-attention-backend", "FLASH_ATTN"]
    server["env"] |= {"OMNI_DIFFUSION_PROMPT_EMBED_CACHE": "0", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "NONE"}
    video.prepare(write(raw), ENV, root=ROOT)

    server["args"] += fault.get("args", [])
    server["env"] |= fault.get("env", {})
    server.update({"preamble": fault["preamble"]} if "preamble" in fault else {})
    with pytest.raises(ValueError, match=re.escape(refusal)):
        video.prepare(write(raw), ENV, root=ROOT)


@pytest.mark.parametrize(("args", "env", "refusal"), [
    (["--quantization", "fp8"], {}, "argument '--quantization'"),
    (["--attention-backend", "sage_attn"], {}, "flag --attention-backend='sage_attn'"),
    (["--enable-torch-compile", "--enable-cache-dit"], {}, "flag --enable-torch-compile needs a value"),
    ([], {"SGLANG_CACHE_DIT_ENABLED": "true"}, "variable SGLANG_CACHE_DIT_ENABLED"),
])  # fmt: skip
def test_sglang_recipe_binds_only_reviewed_settings(native_fragment, args, env, refusal):
    raw, write = native_fragment
    [server] = raw["services"]
    server["args"].append("--disable-conditioning-cache")
    video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)

    server["args"] += args
    server["env"] |= env
    with pytest.raises(ValueError, match=re.escape(refusal)):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


def test_sglang_recipe_must_reach_the_engine_through_the_harness_and_fix_its_mode(native_fragment):
    raw, write = native_fragment
    args = raw["services"][0]["args"]
    args.append("--disable-conditioning-cache")
    mode = args.index("--performance-mode")
    del args[mode : mode + 2]
    with pytest.raises(ValueError, match="SGLang rows must set --performance-mode"):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)

    args[5:7] = ["sglang.launch_server", "--enable-teacache"]
    with pytest.raises(ValueError, match="must run python -m evaluator.srt_selected serve"):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


def test_vllm_omni_recipe_must_turn_off_a_cache_its_engine_defaults_on(native_fragment):
    raw, write = native_fragment
    _as_vllm(raw)
    [server] = raw["services"]
    server["env"]["OMNI_DIFFUSION_PROMPT_EMBED_CACHE"] = "0"
    server["args"] = server["args"][:-2]
    with pytest.raises(ValueError, match="minimax_h3_adaln_cache on by default for MiniMaxAI/MiniMax-H3"):
        video.prepare(write(raw), ENV, root=ROOT)


def test_bound_selected_service_executes_its_engine_and_retains_launch_receipts(native_fragment, tmp_path):
    raw, write = native_fragment
    # Reviewed SGLang names stand in for any recipe setting.
    raw["environment"] = {"SGLANG_USE_AITER": "literal-$DOLLAR", "TRANSFORMERS_OFFLINE": "global-override"}
    engine = tmp_path / "engine.py"
    engine.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CHILD_RECEIPT']).write_text(json.dumps({"
        "'argv': sys.argv, 'tunable': os.environ['TRANSFORMERS_OFFLINE'], "
        "'global_tunable': os.environ['SGLANG_USE_AITER'], "
        "'metadata_present': 'VIDEO_RECIPE_ENV_KEYS' in os.environ, "
        "'gpu_uuids': os.environ['VGBENCH_GPU_UUIDS']}))\n"
    )
    raw["services"][0]["args"].append("--disable-conditioning-cache")
    raw["services"][0]["env"] |= {
        "PYTHONPATH": os.pathsep.join((str(video.video_package(ROOT)), str(ROOT))),
        "TRANSFORMERS_OFFLINE": "keep-this-value",
        "HF_HUB_OFFLINE": "{model_revision}",
    }
    environment = {**ENV, "FRAMEWORK": "sglang", "HF_HUB_CACHE": str(tmp_path / "cache")}
    bound = video.prepare(write(raw), environment, root=ROOT)
    [server] = bound["services"]
    server["args"][3:7] = [sys.executable, str(engine)]
    assert bound["resources"]["gpus_per_node"] == 2
    assert server["env"]["VIDEO_SPEC_PATH"] == bound["benchmark"]["env"]["VIDEO_SPEC_PATH"] == (
        "/video-package/specs/h3.yaml"
    )
    Path(server["env"]["VIDEO_MODEL_PATH"]).mkdir(parents=True)
    output = tmp_path / "result"
    process_env = {
        **os.environ,
        **bound.get("environment", {}),
        **server["env"],
        "VIDEO_SPEC_PATH": str(video.video_package(ROOT) / "specs/h3.yaml"),
        "VIDEO_RESULT_DIR": str(output),
        "CHILD_RECEIPT": str(tmp_path / "child.json"),
    }
    bootstrap = (
        "import sys; from pathlib import Path; from evaluator import srt_selected; "
        f"srt_selected.KFD = Path({str(tmp_path / 'no-kfd')!r}); "
        "srt_selected.cuda_devices=lambda: ['GPU-00000000-0000-0000-0000-000000000001', "
        "'GPU-00000000-0000-0000-0000-000000000002']; "
        "from infx.bench.video import main; sys.exit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, "serve", *server["args"]],
        env=process_env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    child = json.loads((tmp_path / "child.json").read_text())
    launch = json.loads((output / "server/launch.json").read_text())
    spec = json.loads((output / "server/spec.json").read_text())
    expected_command = [sys.executable, str(engine), *server["args"][5:]]
    assert child["argv"] == expected_command[1:]
    flags = dict(zip(child["argv"][1:-1:2], child["argv"][2:-1:2], strict=True))
    assert flags["--tp-size"] == flags["--ulysses-degree"] == "2"
    assert launch["environment"]["HF_HUB_OFFLINE"] == launch["model_revision"]
    assert launch["argv"] == expected_command == spec["baseline"]["argv"]
    assert child["tunable"] == raw["services"][0]["env"]["TRANSFORMERS_OFFLINE"]
    assert child["global_tunable"] == raw["environment"]["SGLANG_USE_AITER"]
    assert child["metadata_present"] is False
    assert launch["environment"]["TRANSFORMERS_OFFLINE"] == child["tunable"]
    assert launch["environment"]["SGLANG_USE_AITER"] == child["global_tunable"]
    assert "VIDEO_RECIPE_ENV_KEYS" not in launch["environment"]
    assert child["gpu_uuids"].split(",") == launch["gpu_uuids"] == spec["gpu_uuids"]
    assert launch["model_revision"] == spec["baseline"]["model_revision"]
