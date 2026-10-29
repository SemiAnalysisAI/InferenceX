import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.config import repository_root
from infx.srt_slurm.generate import point_environment
from infx.srt_slurm import video

ROOT = repository_root()
ENV = {
    "IMAGE": "vllm/vllm-omni:v0.30.0",
    "MODEL": "MiniMaxAI/MiniMax-H3",
    "PRECISION": "bf16",
    "GPU_COUNT": "2",
    "CONC_LIST": "1",
    "H3_SPEC": "h3.yaml",
    "H3_SET": "release-v1",
    "FRAMEWORK": "vllm-omni",
    "HF_HUB_CACHE": "/mnt/hf_hub_cache",
    "RESULT_FILENAME": "h3-native-release",
}


def test_selected_set_reaches_the_video_binder(native_fragment):
    _, write = native_fragment
    point = {
        "scenario-type": "video-generation", "image": ENV["IMAGE"],
        "model": ENV["MODEL"], "model-prefix": "MiniMax-H3", "precision": "bf16",
        "framework": "sglang", "spec": "h3.yaml", "set": "release-v1",
        "gpus": 2, "conc-list": [1], "exp-name": "h3-native-release",
    }
    environment = point_environment(point)
    assert environment["H3_SET"] == "release-v1"
    environment["HF_HUB_CACHE"] = ENV["HF_HUB_CACHE"]
    recipe = video.prepare(write(), environment, root=ROOT)
    assert recipe["benchmark"]["env"]["H3_SET"] == "release-v1"


def test_power_requirement_reaches_the_native_telemetry_block(native_fragment):
    _, write = native_fragment
    recipe = video.prepare(
        write(), {**ENV, "FRAMEWORK": "sglang", "REQUIRE_POWER": "1"}, root=ROOT
    )
    assert recipe["telemetry"]["required"] is True
    assert recipe["benchmark"]["env"]["REQUIRE_POWER"] == "1"


def test_native_video_recipe_requires_one_server_service(native_fragment):
    raw, write = native_fragment
    raw["services"] = []
    with pytest.raises(ValueError, match="exactly one service"):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


def test_runtime_arguments_preserve_explicit_srun_values():
    options = json.dumps({"mem": "0", "cpus-per-task": "32"})
    assert video.runtime_arguments({**ENV, "SRT_SRUN_OPTIONS": options}) == [
        "--set", 'srun_options.gpus-per-node="2"',
        "--set", 'srun_options.mem="0"',
        "--set", 'srun_options.cpus-per-task="32"',
    ]
    with pytest.raises(ValueError, match="SRT_SRUN_OPTIONS"):
        video.runtime_arguments({**ENV, "SRT_SRUN_OPTIONS": json.dumps({"Mem": 1})})


def test_spec_model_path_uses_the_pinned_hub_snapshot():
    assert video.spec_model_path("/cache", "org/name", "a" * 40) == (
        "/cache/models--org--name/snapshots/" + "a" * 40
    )
    with pytest.raises(ValueError, match="40-hex"):
        video.spec_model_path("/cache", "org/name", "main")


def test_both_h3_engines_bind_the_same_selected_plan(native_fragment):
    raw, write = native_fragment
    sglang = video.prepare(write(), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)
    _as_vllm(raw)
    vllm = video.prepare(write(raw), ENV, root=ROOT)
    for recipe, runtime in ((vllm, "vllm-omni"), (sglang, "sglang")):
        [service] = recipe["services"]
        client = recipe["benchmark"]["env"]
        assert service["env"]["H3_PLAN_SHA256"] == client["H3_PLAN_SHA256"]
        assert service["env"]["H3_SET"] == client["H3_SET"] == "release-v1"
        assert service["env"]["H3_RUNTIME"] == client["H3_RUNTIME"] == runtime
        assert service["env"]["H3_RUNTIME_REVISION"] == client["H3_RUNTIME_REVISION"]
        assert client["H3_SPEC_PATH"] == "/h3-package/specs/h3.yaml"
        assert client["GPU_COUNT"] == "2"
    assert vllm["benchmark"]["env"]["H3_PLAN_SHA256"] == sglang["benchmark"]["env"]["H3_PLAN_SHA256"]


@pytest.mark.parametrize("edits", [
    {"H3_SET": "missing"},
    {"H3_SET": "../release-v1"},
    {"MODEL": "Other/model"},
    {"FRAMEWORK": "other"},
])
def test_invalid_selected_plan_or_runtime_fails_before_submission(native_fragment, edits):
    _, write = native_fragment
    with pytest.raises(ValueError):
        video.prepare(
            write(), {**ENV, "FRAMEWORK": "sglang", **edits}, root=ROOT,
        )


@pytest.fixture
def native_fragment(tmp_path):
    """A small recipe whose engine settings remain owned by the recipe."""
    raw = {
        "schema": 2,
        "name": "h3-fixture",
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
                "--attention-backend", "fixture-backend",
            ],
            "env": {
                "H3_SERVER_PORT": "30283", "VIDEO_RUNTIME_REVISION": "b" * 40,
                "MODEL_REVISION_IN_ENV": "{model_revision}", "TUNABLE_ENV": "keep-this-value",
            },
            "readiness": {"http": {"port": 30283, "path": "/health"}},
        }],
        "benchmark": {
            "type": "custom",
            "command": "python3 -m infx.bench video client --request-seconds 73",
            "env": {"H3_SERVER_SERVICE": "video-server", "H3_SERVER_PORT": "30283"},
        },
    }
    path = tmp_path / "serving-client.yaml"

    def write(recipe=raw):
        path.write_text(yaml.safe_dump(recipe))
        return path

    return raw, write


@pytest.mark.parametrize("gpus", [2, 4])
def test_native_selected_recipe_renders_bound_engine_and_preserves_tuning(native_fragment, gpus):
    raw, write = native_fragment
    recipe = video.prepare(write(), {**ENV, "FRAMEWORK": "sglang", "GPU_COUNT": str(gpus)}, root=ROOT)
    [server] = recipe["services"]
    model_path = recipe["benchmark"]["env"]["H3_MODEL_PATH"]
    args = server["args"]
    assert args[args.index("--model-path") + 1] == model_path
    assert args[args.index("--model-id") + 1] == ENV["MODEL"]
    assert args[args.index("--num-gpus") + 1] == str(gpus)
    assert args[args.index("--tp-size") + 1] == str(gpus)
    assert args[args.index("--ulysses-degree") + 1] == str(gpus)
    assert args[args.index("--attention-backend") + 1] == "fixture-backend"
    assert server["env"]["MODEL_REVISION_IN_ENV"] == args[args.index("--revision") + 1]
    assert server["env"]["TUNABLE_ENV"] == "keep-this-value"
    assert recipe["benchmark"]["command"] == raw["benchmark"]["command"]
    assert recipe["resources"]["gpus_per_node"] == gpus


@pytest.mark.parametrize(("flag", "wrong"), [
    ("--model-path", "/cache/other-snapshot"),
    ("--model-id", "Other/Model"),
    ("--revision", "c" * 40),
    ("--port", "30284"),
    ("--num-gpus", "8"),
    ("--host", None),
])
def test_selected_engine_identity_and_resources_must_match_bound_point(native_fragment, flag, wrong):
    raw, write = native_fragment
    args = raw["services"][0]["args"]
    index = args.index(flag)
    if wrong is None:
        del args[index:index + 2]
    else:
        args[index + 1] = wrong
    with pytest.raises(ValueError, match=flag):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


@pytest.mark.parametrize("edit", [
    lambda recipe: recipe["services"][0]["args"].remove("--"),
    lambda recipe: recipe["benchmark"].update(command="python3 -m infx.bench video client"),
])
def test_native_entry_has_no_implicit_engine_or_client_window(native_fragment, edit):
    raw, write = native_fragment
    edit(raw)
    with pytest.raises(ValueError):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


@pytest.mark.parametrize("scope", ["environment", "service", "benchmark"])
@pytest.mark.parametrize("name", [
    "SRT_SERVICE_VIDEO_SERVER_IPS", "SRT_MEASUREMENT_WINDOW_DIR",
    "VGBENCH_GPU_UUIDS", "CUDA_VISIBLE_DEVICES", "H3_RECIPE_ENV_KEYS",
    "H3_MODEL_PATH", "HF_TOKEN", "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
])
def test_native_recipe_rejects_reserved_and_credential_environment(native_fragment, scope, name):
    raw, write = native_fragment
    if scope == "environment":
        raw.setdefault("environment", {})[name] = "wrong"
    elif scope == "service":
        raw["services"][0]["env"][name] = "wrong"
    else:
        raw["benchmark"]["env"][name] = "wrong"
    with pytest.raises(ValueError, match=name):
        video.prepare(write(raw), {**ENV, "FRAMEWORK": "sglang"}, root=ROOT)


def test_vllm_selected_recipe_uses_explicit_fl2va_view(native_fragment):
    raw, write = native_fragment
    _as_vllm(raw)
    recipe = video.prepare(write(raw), ENV, root=ROOT)
    args = recipe["services"][0]["args"]
    assert args[:5] == ["--model-view", "h3-fl2va", "--", "vllm", "serve"]
    assert args[5] == recipe["benchmark"]["env"]["H3_MODEL_PATH"]
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.84"


def _as_vllm(raw):
    raw["services"][0]["args"] = [
        "--model-view", "h3-fl2va", "--", "vllm", "serve", "{model_path}",
        "--served-model-name", "{model_id}", "--port", "30283",
        "--num-gpus", "{gpus_per_node}", "--host", "0.0.0.0",
        "--gpu-memory-utilization", "0.84",
    ]


def test_bound_selected_service_executes_its_engine_and_retains_launch_receipts(native_fragment, tmp_path):
    raw, write = native_fragment
    raw["environment"] = {"GLOBAL_TUNABLE": "literal-$DOLLAR", "TUNABLE_ENV": "global-override"}
    engine = tmp_path / "engine.py"
    engine.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CHILD_RECEIPT']).write_text(json.dumps({"
        "'argv': sys.argv, 'tunable': os.environ['TUNABLE_ENV'], "
        "'global_tunable': os.environ['GLOBAL_TUNABLE'], "
        "'metadata_present': 'H3_RECIPE_ENV_KEYS' in os.environ, "
        "'gpu_uuids': os.environ['VGBENCH_GPU_UUIDS']}))\n"
    )
    args = raw["services"][0]["args"]
    args[3:7] = [sys.executable, str(engine)]
    raw["services"][0]["env"]["PYTHONPATH"] = os.pathsep.join(
        (str(video.video_package(ROOT)), str(ROOT))
    )
    raw["services"][0]["env"]["CHILD_RECEIPT"] = str(tmp_path / "child.json")
    environment = {**ENV, "FRAMEWORK": "sglang", "HF_HUB_CACHE": str(tmp_path / "cache")}
    bound = video.prepare(write(raw), environment, root=ROOT)
    [server] = bound["services"]
    Path(server["env"]["H3_MODEL_PATH"]).mkdir(parents=True)
    output = tmp_path / "result"
    process_env = {
        **os.environ,
        **bound.get("environment", {}),
        **server["env"],
        "H3_SPEC_PATH": str(video.video_package(ROOT) / "specs/h3.yaml"),
        "H3_RESULT_DIR": str(output),
    }
    bootstrap = (
        "import sys; from evaluator import gpu_job; "
        "gpu_job.cuda_devices=lambda: ['GPU-00000000-0000-0000-0000-000000000001', "
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
    assert launch["argv"] == expected_command == spec["baseline"]["argv"]
    assert child["tunable"] == raw["services"][0]["env"]["TUNABLE_ENV"]
    assert child["global_tunable"] == raw["environment"]["GLOBAL_TUNABLE"]
    assert child["metadata_present"] is False
    assert launch["environment"]["TUNABLE_ENV"] == child["tunable"]
    assert launch["environment"]["GLOBAL_TUNABLE"] == child["global_tunable"]
    assert "H3_RECIPE_ENV_KEYS" not in launch["environment"]
    assert child["gpu_uuids"].split(",") == launch["gpu_uuids"] == spec["gpu_uuids"]
    assert launch["model_revision"] == spec["baseline"]["model_revision"]
