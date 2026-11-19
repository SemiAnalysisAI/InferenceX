"""Launchable srt-slurm recipes for tests that plan or fingerprint matrix rows."""

import shutil
from pathlib import Path

import yaml

from infx.matrix.validation import VIDEO_SPEC_DIR
from infx.srt_slurm.workload import NATIVE_TELEMETRY_BLOCK, SHARED_BLOCKS, TELEMETRY_BLOCK, VIDEO_BLOCK

ROOT = Path(__file__).resolve().parents[2]
VIDEO_MODEL = "example/video-model"
VIDEO_SPEC = "fixture.yaml"
VIDEO_SET = "fixture-set"
VIDEO_PORT = "30283"


def shared_blocks() -> dict[str, str]:
    """Each lane's shared block (its benchmark client) and the DCGM telemetry block, by path."""
    blocks = {
        block.as_posix(): yaml.safe_dump({"benchmark": {"type": "custom", "command": (
            "bash srt_agentic.sh" if agentic else f"bash {'multi' if multinode else 'single'}.sh"
        )}})
        for (agentic, multinode), block in SHARED_BLOCKS.items()
    }  # fmt: skip
    blocks[TELEMETRY_BLOCK.as_posix()] = yaml.safe_dump(
        {"telemetry": {"enabled": True, "dcgm_exporter": {"container_image": "dcgm"}}}
    )
    blocks[NATIVE_TELEMETRY_BLOCK.as_posix()] = yaml.safe_dump(
        {"telemetry": {"enabled": True, "required": False}}
    )
    blocks[VIDEO_BLOCK.as_posix()] = yaml.safe_dump({
        "frontend": {"type": "none"},
        "benchmark": {
            "type": "custom", "command": "python3 -m infx.bench video client --request-seconds 60",
            "env": {"VIDEO_SERVER_SERVICE": "video-server", "VIDEO_SERVER_PORT": VIDEO_PORT},
        },
    })  # fmt: skip
    return blocks


def write_shared_blocks(project: Path) -> None:
    for path, text in shared_blocks().items():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(text)


def write_video_specs(repository: Path) -> None:
    """Put the workload helper and a one-case spec serving ``VIDEO_MODEL`` beside the project."""
    specs = repository / VIDEO_SPEC_DIR
    specs.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT.parent / VIDEO_SPEC_DIR / "video_spec.py", specs / "video_spec.py")
    (specs / VIDEO_SPEC).write_text(yaml.safe_dump({
        "schema_version": 1,
        "model": {"id": VIDEO_MODEL, "revision": "a" * 40},
        "generation": {"num_inference_steps": 1},
        "prompts": {"fixture": {"text": "a fixture prompt", "requires_motion": False, "requires_sound": False}},
        "sets": {VIDEO_SET: {
            "plan_id": "fixture-plan", "repetitions": 1, "warmup_runs": 0,
            "cases": [{"case_id": "fixture", "prompt": "fixture", "seed": 1}],
        }},
        "reference": {},
    }))  # fmt: skip


def fixed_seq_entry(recipe: str, concurrencies: list[int]) -> dict:
    """A master entry with one TP4 fixed-sequence point per concurrency, served by ``recipe``."""
    return {
        "image": "example/image:1", "model": "org/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "cluster:fixture",
        "multinode": False, "srt-recipe-dir": "fixture",
        "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [
            {"tp": 4, "conc-list": concurrencies, "srt-recipe": recipe},
        ]}]},
    }  # fmt: skip


def video_entry(recipe: str = "video.yaml") -> dict:
    """A master entry with one 2-GPU video point of ``VIDEO_SPEC``'s set served by ``recipe``."""
    return {
        "image": "example/video:1", "model": VIDEO_MODEL, "model-prefix": "h3",
        "precision": "bf16", "framework": "sglang", "runner": "cluster:fixture",
        "multinode": False, "srt-recipe-dir": "fixture",
        "scenarios": {"video-generation": [{"spec": VIDEO_SPEC, "set": VIDEO_SET, "search-space": [
            {"gpus": 2, "conc-list": [1], "srt-recipe": recipe},
        ]}]},
    }  # fmt: skip


def single_node_fragment(tp: int, **args) -> dict:
    """A single-node sglang fragment serving TP ``tp`` points without speculation."""
    return {"schema": 2, "engine": "sglang", "roles": {"agg": {
        "nodes": 1, "workers": 1, "gpus": tp, "args": {"tensor-parallel-size": tp, **args},
    }}}  # fmt: skip


def video_fragment(**args: str) -> dict:
    """A native sglang video fragment the binder accepts, with extra engine ``args`` appended."""
    return {"schema": 2, "name": "video-fixture", "resources": {"gpu_type": "fixture"}, "services": [{
        "name": "video-server", "type": "generic", "nodes": 1, "critical": True,
        "command": ["python3", "-m", "infx.bench", "video", "serve"],
        "args": [
            "--model-view", "snapshot", "--", "python3", "-m", "evaluator.srt_selected", "serve",
            "--model-path", "{model_path}", "--model-id", "{model_id}",
            "--revision", "{model_revision}", "--num-gpus", "{gpus_per_node}",
            "--disable-conditioning-cache", "--host", "0.0.0.0", "--port", VIDEO_PORT,
            *(f"--{flag}={value}" for flag, value in args.items()),
        ],
        "env": {"VIDEO_SERVER_PORT": VIDEO_PORT, "VIDEO_RUNTIME_REVISION": "b" * 40},
        "readiness": {"http": {"port": int(VIDEO_PORT), "path": "/health"}},
    }]}  # fmt: skip
