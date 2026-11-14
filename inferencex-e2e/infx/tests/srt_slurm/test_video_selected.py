import json
import os
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


def test_fairness_requires_disabled_cross_request_caches(native_fragment):
    raw, write = native_fragment
    selected = {**ENV, "FRAMEWORK": "sglang", "VIDEO_SPEC": "h3-v2.yaml", "GPU_COUNT": "4"}
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
                "--attention-backend", "fixture-backend",
            ],
            "env": {
                "VIDEO_SERVER_PORT": "30283", "VIDEO_RUNTIME_REVISION": "b" * 40,
                "MODEL_REVISION_IN_ENV": "{model_revision}", "TUNABLE_ENV": "keep-this-value",
            },
            "readiness": {"http": {"port": 30283, "path": "/health"}},
        }],
        "benchmark": {
            "type": "custom",
            "command": "python3 -m infx.bench video client --request-seconds 73",
            "env": {"VIDEO_SERVER_SERVICE": "video-server", "VIDEO_SERVER_PORT": "30283"},
        },
    }
    path = tmp_path / "serving-client.yaml"

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
        "'metadata_present': 'VIDEO_RECIPE_ENV_KEYS' in os.environ, "
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
    flags = dict(zip(child["argv"][1::2], child["argv"][2::2], strict=True))
    assert flags["--tp-size"] == flags["--ulysses-degree"] == "2"
    assert launch["environment"]["MODEL_REVISION_IN_ENV"] == launch["model_revision"]
    assert launch["argv"] == expected_command == spec["baseline"]["argv"]
    assert child["tunable"] == raw["services"][0]["env"]["TUNABLE_ENV"]
    assert child["global_tunable"] == raw["environment"]["GLOBAL_TUNABLE"]
    assert child["metadata_present"] is False
    assert launch["environment"]["TUNABLE_ENV"] == child["tunable"]
    assert launch["environment"]["GLOBAL_TUNABLE"] == child["global_tunable"]
    assert "VIDEO_RECIPE_ENV_KEYS" not in launch["environment"]
    assert child["gpu_uuids"].split(",") == launch["gpu_uuids"] == spec["gpu_uuids"]
    assert launch["model_revision"] == spec["baseline"]["model_revision"]


def test_sol_engine_binds_its_snapshot_and_disables_conditioning_cache(native_fragment):
    raw, write = native_fragment
    raw["services"][0]["args"] = [
        "--model-view", "snapshot", "--", "/opt/sol-engine/venv/bin/python3",
        "/opt/sol-engine/docker/gb200/serve.py", "--model-path", "{model_path}",
        "--served-model-name", "{model_id}", "--port", "30283",
        "--num-gpus", "{gpus_per_node}", "--host", "0.0.0.0", "--attention", "dense",
    ]
    selected = {**ENV, "FRAMEWORK": "sol-engine", "VIDEO_SPEC": "h3-v2.yaml", "GPU_COUNT": "4"}
    with pytest.raises(ValueError, match="--disable-conditioning-cache"):
        video.prepare(write(raw), selected, root=ROOT)

    raw["services"][0]["args"].append("--disable-conditioning-cache")
    recipe = video.prepare(write(raw), selected, root=ROOT)
    [server] = recipe["services"]
    model_path = server["env"]["VIDEO_MODEL_PATH"]
    assert model_path.endswith("/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe")
    command = server["args"][server["args"].index("--") + 1:]
    assert command[command.index("--model-path") + 1] == model_path
    assert command[command.index("--served-model-name") + 1] == "MiniMaxAI/MiniMax-H3"
    assert command[command.index("--num-gpus") + 1] == "4"
    assert server["env"]["VIDEO_RUNTIME"] == recipe["benchmark"]["env"]["VIDEO_RUNTIME"] == "sol-engine"

    args = raw["services"][0]["args"]
    args[args.index("--model-path") + 1] = "/elsewhere/MiniMax-H3"
    with pytest.raises(ValueError, match="--model-path"):
        video.prepare(write(raw), selected, root=ROOT)
