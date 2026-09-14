"""Prepare benchmark names and environment variables."""

import json
from decimal import Decimal


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        try:
            number = float(value)
        except OverflowError:
            return "Infinity" if value > 0 else "-Infinity"
        if abs(number) < 1e21:
            return format(Decimal(str(number)), "f").removesuffix(".0")
        return str(number)
    return str(value)


def _topology(row: dict) -> str:
    label = f"TP{_text(row.get('tp'))}"
    for key, suffix in (("pp", "PP"), ("dcp-size", "DCP"), ("pcp-size", "PCP"), ("ep", "EP")):
        value = _text(row.get(key))
        if value not in ("", "1"):
            label += f"/{suffix}{value}"
    return label + ("/DPA" if row.get("dp-attn") else "")


def prepare_config(row: dict) -> dict:
    agentic = row.get("scenario-type") == "agentic-coding"
    multinode = "prefill" in row
    env = {
        name: _text(row.get(key)) for name, key in (
            ("EXP_NAME", "exp-name"), ("RECIPE_FINGERPRINT", "recipe-fingerprint"),
            ("MODEL", "model"), ("MODEL_PREFIX", "model-prefix"), ("IMAGE", "image"),
            ("FRAMEWORK", "framework"), ("PRECISION", "precision"),
            ("SPEC_DECODING", "spec-decoding"), ("KV_P2P_TRANSFER", "kv-p2p-transfer"),
        )
    }
    env.update({name: "0" if agentic else _text(row.get(key)) for name, key in (
        ("ISL", "isl"), ("OSL", "osl"), ("MAX_MODEL_LEN", "max-model-len"),
    )})
    env["DISAGG"] = "false" if agentic and not multinode else _text(row.get("disagg"))
    backend = (row.get("kv-offload-backend") or {}) if agentic else {}
    env["KV_OFFLOADING"] = _text(row.get("kv-offloading")) if agentic else ""
    env["KV_OFFLOAD_BACKEND"] = _text(backend.get("name"))
    for name, metadata in (("KV_OFFLOAD_BACKEND_METADATA", backend), ("ROUTER_METADATA", row.get("router"))):
        env[name] = json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else ""

    framework = _text(row.get("framework"))
    short_framework = {"sglang": "sgl", "dynamo-sglang": "dyn-sgl", "sglang-disagg": "sgl-disagg"}.get(
        framework.lower(), framework,
    )
    parts = [f"{env['MODEL_PREFIX']} {env['PRECISION']} {row['runner']} {short_framework}"]
    if multinode:
        for role in ("prefill", "decode"):
            worker = row[role]
            env.update({f"{role.upper()}_{name}": _text(worker.get(key)) for name, key in (
                ("HARDWARE", "hardware"), ("NUM_WORKERS", "num-worker"), ("TP", "tp"),
                ("PP_SIZE", "pp"), ("DCP_SIZE", "dcp-size"), ("PCP_SIZE", "pcp-size"),
                ("EP", "ep"), ("DP_ATTN", "dp-attn"),
            )})
        disagg = row.get("disagg") is True
        parts.extend([
            (f"{env['PREFILL_NUM_WORKERS']}P " if disagg else "") + f"({_topology(row['prefill'])})",
            f"x {env['DECODE_NUM_WORKERS']}D ({_topology(row['decode'])})" if disagg else "",
        ])
    else:
        env.update({name: _text(row.get(key)) for name, key in (
            ("TP", "tp"), ("PP_SIZE", "pp"), ("DCP_SIZE", "dcp-size"),
            ("PCP_SIZE", "pcp-size"), ("EP_SIZE", "ep"), ("CONC", "conc"),
        )})
        env["TOTAL_CPU_DRAM_GB"] = _text(row.get("total-cpu-dram-gb")) if agentic else "0"
        parts.append(_topology(row))
    parts.extend([
        env["SPEC_DECODING"] if env["SPEC_DECODING"].lower() != "none" else "",
        f"{env['KV_OFFLOADING']} KV offload" if env["KV_OFFLOADING"].lower() not in ("", "none") else "",
        env["KV_OFFLOAD_BACKEND"] if env["KV_OFFLOAD_BACKEND"].lower() not in ("none", "default") else "",
    ])
    return {"name": " ".join(parts), "env": env}
