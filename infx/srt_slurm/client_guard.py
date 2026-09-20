"""Verify prepared wrapper/files inside the container before starting a client."""

from __future__ import annotations

import argparse
from pathlib import Path

from infx.benchmarks import agentx, eval as real_eval
from infx.benchmarks.identity import capture_identity
from infx.benchmarks.spec import AgentXSpec, EvalSpec
from infx.srt_slurm.job import read_json
from infx.srt_slurm.launch import verify_bundle


def main() -> int:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--client", choices=("agentx", "eval"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    bundle = read_json(args.bundle)
    verify_bundle(bundle)
    identity = capture_identity(bundle["site"]["wrapper_python"], ["infx"], dataset_loader=None)
    if identity != bundle["identity"]["wrapper_identity"]:
        raise ValueError("installed InferenceX wrapper changed after preparation")
    endpoint = os.environ["SRT_ENDPOINT"]
    spec = read_json(args.spec)
    if args.client == "eval":
        return real_eval.run(EvalSpec.model_validate(spec), endpoint, args.artifact_root)
    return agentx.run(AgentXSpec.model_validate(spec), endpoint, args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
