"""Serving YAML -> the points .github/workflows/gn-benchmark.yml runs.

    python -P -m infx.serving.to_matrix serving.yaml          # [{"config": row, "serving": json}, ...]
    python -P -m infx.serving.to_matrix --check serving.yaml  # the run title only

``config`` is the benchmark-tmpl.yml row; ``serving`` is SERVING_JSON for
infx/launch/drivers/serving.py, with the rendered scenario inside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from infx.clusters import RunnerInventory
from infx.serving.render import CLIENT_IMAGE, _flag_value, render, render_remote, split_serve
from infx.serving.scenarios import Rendered
from infx.serving.schema import ServingError, ServingSpec, arm_specs, load_serving, variants


def spec_decoding(serve_cmd: str) -> str:
    """The row's spec-decoding value: mtp for an MTP draft, draft_model for any other, none without one."""
    argv = split_serve(serve_cmd) if serve_cmd else []
    if config := _flag_value(argv, "--speculative-config"):  # vLLM
        try:
            method = json.loads(config).get("method", "")
        except (ValueError, AttributeError):
            method = ""
        return "mtp" if method == "mtp" else "draft_model"
    if algorithm := _flag_value(argv, "--speculative-algorithm"):  # SGLang
        return "mtp" if algorithm.upper() in ("NEXTN", "MTP") else "draft_model"
    return "none"


def _row(
    spec: ServingSpec, framework: str, conc: int, duration: int, exp_name: str, serve_cmd: str
) -> dict:
    remote = spec.endpoint
    return {
        "image": CLIENT_IMAGE if remote else spec.image, "model": remote.model if remote else spec.model,
        "model-prefix": spec.name, "precision": "remote" if remote else spec.precision,
        "framework": framework, "runner": spec.runner,
        # tp stays 1 for a remote point: upstream derives GPU_COUNT from it and divides by it;
        # the serving driver, not GPU_COUNT, decides that a remote job takes no GPU.
        "tp": 1 if remote else spec.gpus, "pp": 1, "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False,
        "spec-decoding": spec_decoding(serve_cmd), "conc": conc, "kv-offloading": "none", "total-cpu-dram-gb": 0,
        "duration": duration, "exp-name": exp_name, "scenario-type": "agentic-coding",
    }  # fmt: skip


def _serving(spec: ServingSpec, rendered: dict[str, str], variant: Rendered, text: str) -> str:
    return json.dumps(
        {
            "framework": rendered["framework"],
            "serve_cmd": rendered["serve-cmd"],
            "serve_env": rendered["serve-env"],
            "served_model_name": rendered["served-model-name"],
            "router_image": rendered["router-image"],
            "router_cmd": rendered["router-cmd"],
            "router_env": rendered["router-env"],
            "router_profile": rendered["router-profile"],
            "router_metrics_port": rendered["router-metrics-port"],
            "replicas": spec.replicas,
            "storage_image": rendered["storage-image"],
            "storage_cmd": rendered["storage-cmd"],
            "storage_env": rendered["storage-env"],
            "scenario_yaml": variant.yaml(),
            "scenario_label": variant.label,
            "dataset_kind": variant.kind,
            "dataset": variant.dataset,
            "search": variant.search,
            "yaml": text,
            "remote_url": spec.endpoint.url if spec.endpoint else "",
            "tokenizer": spec.endpoint.tokenizer if spec.endpoint else "",
            "tls_verify": spec.endpoint.tls_verify if spec.endpoint else True,
            "api_key_secret": _secret(spec),
        }
    )


def _secret(spec: ServingSpec) -> str:
    """Name of the GitHub secret holding the endpoint's key; empty when there is none."""
    return (spec.endpoint.api_key_secret or "") if spec.endpoint else ""


def build(text: str, inventory: RunnerInventory | None = None) -> dict:
    spec = load_serving(text, inventory)
    chosen = variants(spec)
    arms = arm_specs(spec)
    multi = len(chosen) > 1
    points = []
    for arm, engine in arms:
        rendered = render_remote() if spec.endpoint else render(engine)
        stem = f"{spec.name}-{arm}" if arm else spec.name
        for variant in chosen:
            for conc in spec.conc:
                if multi:
                    exp_name = f"{stem}-{variant.label}"
                else:
                    exp_name = f"{stem}_search" if variant.search else f"{stem}_c{conc}"
                row = _row(
                    spec,
                    rendered["framework"],
                    conc,
                    int(variant.duration),
                    exp_name,
                    rendered["serve-cmd"],
                )
                points.append(
                    {
                        "config": row,
                        "serving": _serving(engine, rendered, variant, text),
                        "api_key_secret": _secret(spec),
                    }
                )
    what = f"{len(chosen)} points" if multi else chosen[0].label
    if spec.arms:
        what += f" x {len(arms)} arms ({','.join(spec.arms)})"
    durations = {int(variant.duration) for variant in chosen}
    seconds = (
        "mixed"
        if len(durations) > 1
        else f"{durations.pop()}s"
        if durations != {0}
        else "whole selection"
    )  # greenspec has no duration
    concs = ",".join(str(conc) for conc in spec.conc)
    return {
        "points": points,
        "title": f"gn-benchmark {spec.name} {what} ({seconds}) c{concs} {spec.runner}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate and print the run title only"
    )
    args = parser.parse_args(argv)
    try:
        try:
            text = args.path.read_text()
        except OSError as error:
            raise ServingError(f"serving: cannot read {args.path} ({error.strerror})") from error
        result = build(text)
    except ServingError as error:
        for line in str(error).splitlines():
            print(f"ERROR {line}", file=sys.stderr)
        return 2
    print(result["title"] if args.check else json.dumps(result["points"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
