"""Generate offline Diffusers reference media; this is not a serving benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from specs import h3_spec

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "transformer",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_cases(plan: dict, names: list[str]) -> list[dict]:
    cases = [{**plan["warmup_case"], "role": "warmup"}]
    cases += [{**case, "role": "measured"} for case in plan["cases"]]
    if names == ["all"]:
        return cases
    known = {case["case_id"] for case in cases}
    if any(Path(name).name != name or name in {".", ".."} for name in known):
        raise ValueError("case IDs must be plain directory names")
    if not names or len(set(names)) != len(names) or not set(names) <= known:
        raise ValueError("select distinct case IDs from the frozen plan, or only 'all'")
    return [case for case in cases if case["case_id"] in names]


def runtime_identity(plan: dict) -> dict:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise ValueError(f"set {name}=1 before importing the reference runtime")
    versions = {}
    for line in (REFERENCE / "requirements.in").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, expected = line.split("==")
            versions[name] = importlib.metadata.version(name)
            if versions[name] != expected:
                raise ValueError(f"{name}: expected {expected}, installed {versions[name]}")
    direct = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
    source = json.loads(direct or "{}")
    revision = plan["fairness"]["reference_identity"]["diffusers_revision"]
    expected_url = f"https://github.com/huggingface/diffusers/archive/{revision}.tar.gz"
    if source.get("url") != expected_url:
        raise ValueError(
            "install the exact Diffusers archive from reference/requirements-linux.txt"
        )
    versions["diffusers"] = importlib.metadata.version("diffusers")
    return {
        "versions": versions,
        "diffusers_source": source,
        "linux_lock_sha256": digest(REFERENCE / "requirements-linux.txt"),
    }


def schedules(plan: dict) -> dict:
    from diffusers import MiniMaxH3Scheduler

    contract = plan["fairness"]["schedule"]
    result = {}
    for modality in ("video", "audio"):
        scheduler = MiniMaxH3Scheduler(
            shift=contract[f"{modality}_flow_shift"], stochastic_sampling=False
        )
        scheduler.set_timesteps(contract["sigma_points"], device="cpu")
        if len(scheduler.timesteps) != contract["denoising_updates"]:
            raise ValueError("the pinned scheduler does not implement the selected update count")
        raw = scheduler.sigmas.numpy().tobytes()
        result[modality] = {
            "sigmas": scheduler.sigmas.tolist(),
            "float32_le_sha256": hashlib.sha256(raw).hexdigest(),
            "updates": len(scheduler.timesteps),
        }
    return result


def save_media(
    directory: Path,
    video: np.ndarray,
    audio: np.ndarray,
    generation: dict,
    encoding: dict,
    *,
    threads: int,
    audio_bitrate: int,
) -> dict:
    import av
    import numpy as np

    expected = (generation["frame_count"], 3, generation["height"], generation["width"])
    if video.shape != expected or audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("unexpected reference video/audio shape")
    if not np.isfinite(video).all() or not np.isfinite(audio).all() or audio.shape[1] == 0:
        raise ValueError("reference contains non-finite or empty media")
    if video.min() < 0 or video.max() > 1:
        raise ValueError("reference video must be in [0, 1]")
    directory.mkdir()
    np.save(directory / "video.npy", video, allow_pickle=False)
    np.save(directory / "audio.npy", audio, allow_pickle=False)
    with av.open(str(directory / "reference.mp4"), "w") as container:
        stream = container.add_stream(
            encoding["video_codec"],
            rate=generation["fps"],
            options={"preset": encoding["preset"], "crf": str(encoding["crf"])},
        )
        stream.width, stream.height = generation["width"], generation["height"]
        stream.pix_fmt = encoding["pixel_format"]
        stream.thread_count = threads
        sound = container.add_stream(encoding["audio_codec"], rate=encoding["audio_sample_rate_hz"])
        sound.layout, sound.bit_rate = "stereo", audio_bitrate
        for index, pixels in enumerate(video):
            frame = av.VideoFrame.from_ndarray(
                (pixels.transpose(1, 2, 0) * 255).astype(np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, generation["fps"])
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio), format="fltp", layout="stereo"
        )
        frame.sample_rate = encoding["audio_sample_rate_hz"]
        frame.pts, frame.time_base = 0, Fraction(1, frame.sample_rate)
        for packet in sound.encode(frame):
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)
    return {p.name: digest(p) for p in sorted(directory.iterdir())}


def generate(args: argparse.Namespace, plan: dict, cases: list[dict], identity: dict) -> None:
    import torch
    from diffusers import ComponentsManager, ModularPipeline

    snapshot = args.snapshot.resolve()
    if (
        snapshot.name != plan["model_revision"]
        or not (snapshot / "modular_model_index.json").is_file()
    ):
        raise ValueError("use the complete pinned HF snapshot, not an engine-specific model view")
    if any(not (snapshot / name).is_dir() for name in COMPONENTS):
        raise ValueError("the pinned snapshot is missing a required T2VA component")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "starting",
        "quality_status": "unjudged",
        "release_qualified": False,
        "plan": plan,
        "plan_sha256": h3_spec.plan_sha256(plan),
        "runtime": identity,
        "model_snapshot": {
            "path": str(snapshot),
            "requested_revision": plan["model_revision"],
            "weights_verified_by_helper": False,
            "metadata_sha256": {
                str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.json"))
            },
            "qualification_requires": "matching external model staging/checksum receipt",
        },
        "selection": [case["case_id"] for case in cases],
        "cases": [],
        "execution": {
            "dtype": args.dtype,
            "device": args.device,
            "offload_reserve": args.offload_reserve,
            "attention_backend": args.attention_backend,
            "text_encoder_attention": "sdpa",
            "encoder_threads": args.encoder_threads,
            "audio_bitrate": args.audio_bitrate,
            "rgb8_conversion": "truncate_float32_times_255",
        },
        "schedule": schedules(plan),
        "rng_draw_order_evidence": "pinned_source; no initial-noise capture",
    }
    receipt = args.output / "reference.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    try:
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            str(snapshot), workflow="t2va", components_manager=manager, local_files_only=True
        )
        pipe.load_components(
            pretrained_model_name_or_path=str(snapshot),
            revision=None,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation={"text_encoder": "sdpa"},
        )
        if any(getattr(pipe, name, None) is None for name in COMPONENTS):
            raise RuntimeError(
                "Diffusers did not load every required T2VA component; inspect the retained log"
            )
        for name in ("transformer", "vae", "audio_vae"):
            getattr(pipe, name).set_attention_backend(args.attention_backend)
        manager.enable_auto_cpu_offload(
            device=args.device, memory_reserve_margin=args.offload_reserve
        )
        pipe.scheduler.set_shift(plan["fairness"]["schedule"]["video_flow_shift"])
        pipe.audio_scheduler.set_shift(plan["fairness"]["schedule"]["audio_flow_shift"])
        if (
            pipe.scheduler.config.stochastic_sampling
            or pipe.audio_scheduler.config.stochastic_sampling
        ):
            raise ValueError("reference requires the non-stochastic base scheduler")
        generation = plan["generation"]
        record["status"] = "generating"
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        for case in cases:
            generator = torch.Generator(device="cpu").manual_seed(case["seed"])
            before = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
            payload = {
                "prompt": case["prompt"],
                "height": generation["height"],
                "width": generation["width"],
                "num_frames": generation["frame_count"],
                "num_inference_steps": plan["fairness"]["schedule"]["sigma_points"],
                "generator": generator,
                "output_type": "pt",
            }
            if not set(payload) <= set(pipe.blocks.input_names):
                raise ValueError("reference payload contains an unsupported pipeline input")
            with torch.inference_mode():
                result = pipe(**payload, output=["videos", "audio", "sampling_rate"])
            if result["sampling_rate"] != generation["audio_sample_rate_hz"]:
                raise ValueError("unexpected decoded audio sample rate")
            for modality, scheduler in (("video", pipe.scheduler), ("audio", pipe.audio_scheduler)):
                if scheduler.sigmas.cpu().tolist() != record["schedule"][modality]["sigmas"]:
                    raise ValueError(
                        "executed pipeline schedule differs from the selected reference"
                    )
            hashes = save_media(
                args.output / case["case_id"],
                result["videos"][0].cpu().float().numpy(),
                result["audio"][0].cpu().float().numpy(),
                generation,
                plan["fairness"]["encoding"],
                threads=args.encoder_threads,
                audio_bitrate=args.audio_bitrate,
            )
            record["cases"].append(
                {
                    **case,
                    "sha256": hashes,
                    "rng_state_before_sha256": before,
                    "rng_state_after_sha256": hashlib.sha256(
                        generator.get_state().numpy().tobytes()
                    ).hexdigest(),
                }
            )
            record["status"] = "generating"
            receipt.write_text(json.dumps(record, indent=2) + "\n")
        record["status"] = "selection_complete"
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        files = sorted(p for p in args.output.rglob("*") if p.is_file())
        (args.output / "SHA256SUMS").write_text(
            "".join(f"{digest(p)}  {p.relative_to(args.output)}\n" for p in files)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check installed imports, plan and CPU schedules; load no weights",
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bfloat16"])
    parser.add_argument("--attention-backend", choices=["native"])
    parser.add_argument("--offload-reserve")
    parser.add_argument("--encoder-threads", type=int)
    parser.add_argument("--audio-bitrate", type=int)
    args = parser.parse_args()
    plan = h3_spec.plan_for(h3_spec.load_spec(args.spec), args.set)
    cases = select_cases(plan, args.case)
    identity = runtime_identity(plan)
    from diffusers import ComponentsManager, ModularPipeline  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

    if args.check:
        print(
            json.dumps(
                {
                    "runtime": identity,
                    "plan_sha256": h3_spec.plan_sha256(plan),
                    "selected_cases": [c["case_id"] for c in cases],
                    "schedule": schedules(plan),
                },
                indent=2,
            )
        )
        return
    if any(
        getattr(args, name) is None
        for name in (
            "snapshot",
            "output",
            "device",
            "dtype",
            "attention_backend",
            "offload_reserve",
            "encoder_threads",
            "audio_bitrate",
        )
    ):
        parser.error(
            "generation requires explicit snapshot, output, device, dtype, attention, offload and encoder settings"
        )
    if not args.device.startswith("cuda") or args.encoder_threads < 1 or args.audio_bitrate < 1:
        parser.error("use a CUDA device and positive encoder settings")
    generate(args, plan, cases, identity)


if __name__ == "__main__":
    main()
