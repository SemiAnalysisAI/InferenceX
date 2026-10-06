"""``meta_env.json``: the identity, topology, and concurrency of one eval.

Score validation, result collection, and ingestion read its keys with these value types.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from infx.bench import env
from infx.results.topology import eval_topology

BATCH_KEYS = ("eval_concs", "completed_eval_concs", "failed_eval_concs")
"""The manifest of a batched eval, next to ``conc`` (its first concurrency)."""
_TRUE = {"1", "true", "yes", "on"}
_RESULT_FILENAME = re.compile(r".*_([^_]+)_([^_]+)_tp")


def _first(values: Mapping[str, str], *names: str, default: str) -> str:
    return next((values[name] for name in names if values.get(name)), default)


def _is_true(value: str) -> bool:
    return value.lower() in _TRUE


def _disaggregated(values: Mapping[str, str]) -> dict[str, str]:
    """Metadata names of a disaggregated job from the workflow's ``PREFILL_*``/``DECODE_*``."""
    tp = _first(values, "PREFILL_TP", "TP", default="1")
    prefill_ep = _first(values, "PREFILL_EP", "EP_SIZE", "EP", default="1")
    prefill_dp = _first(values, "PREFILL_DP_ATTN", default="false")
    return {
        "TP": tp,
        "PREFILL_TP": tp,
        "PREFILL_EP": prefill_ep,
        "EP_SIZE": prefill_ep,
        "PREFILL_NUM_WORKERS": _first(values, "PREFILL_NUM_WORKERS", default="1"),
        "DECODE_TP": _first(values, "DECODE_TP", default=tp),
        "DECODE_EP": _first(values, "DECODE_EP", default=prefill_ep),
        "DECODE_NUM_WORKERS": _first(values, "DECODE_NUM_WORKERS", default="1"),
        "DP_ATTENTION": prefill_dp,
        "PREFILL_DP_ATTENTION": prefill_dp,
        "DECODE_DP_ATTENTION": _first(values, "DECODE_DP_ATTN", default="false"),
    }


def build(
    environ: Mapping[str, str],
    *,
    conc: object,
    suite: str,
    batch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The document of an eval of ``suite`` at ``conc``; ``batch`` supplies its ``BATCH_KEYS``."""
    multinode = env.flag("IS_MULTINODE", env=environ)
    values = dict(environ)
    if multinode:
        # Only here: the bridge resets unset per-phase DP flags, which would overwrite a
        # single-node job's real DP_ATTENTION.
        values.update(_disaggregated(values))

    def size(field: str, *names: str) -> int:
        name = next((name for name in names if values.get(name)), None)
        if name is None:
            return 1
        if not re.fullmatch(r"[0-9]+", values[name]):
            raise env.InputError(
                f"{name} must be an integer for meta_env.json {field}, got {values[name]!r}"
            )
        return int(values[name])

    def dp(*names: str) -> bool:
        return _is_true(_first(values, *names, default="false"))

    framework, precision = values.get("FRAMEWORK", ""), values.get("PRECISION", "")
    parsed = _RESULT_FILENAME.match(values.get("RESULT_FILENAME", ""))
    if parsed is not None:
        precision = precision or parsed.group(1)
        framework = framework or parsed.group(2)
    document = {
        "is_multinode": multinode,
        "framework": framework or "unknown",
        "precision": precision or "unknown",
        "spec_decoding": values.get("SPEC_DECODING", ""),
        "eval_suite": suite,
        "recipe_fingerprint": values.get("RECIPE_FINGERPRINT", ""),
        "tp": size("tp", "TP"),
        "pp": size("pp", "PP_SIZE"),
        "dcp_size": size("dcp_size", "DCP_SIZE"),
        "pcp_size": size("pcp_size", "PCP_SIZE"),
        "conc": conc,
        **({key: batch[key] for key in BATCH_KEYS} if batch is not None else {}),
        "ep": size("ep", "EP_SIZE"),
        "dp_attention": dp("DP_ATTENTION"),
        "prefill_tp": size("prefill_tp", "PREFILL_TP", "TP"),
        "prefill_pp": size("prefill_pp", "PREFILL_PP_SIZE", "PP_SIZE"),
        "prefill_dcp_size": size("prefill_dcp_size", "PREFILL_DCP_SIZE", "DCP_SIZE"),
        "prefill_pcp_size": size("prefill_pcp_size", "PREFILL_PCP_SIZE", "PCP_SIZE"),
        "prefill_ep": size("prefill_ep", "PREFILL_EP", "EP_SIZE"),
        "prefill_dp_attention": dp("PREFILL_DP_ATTENTION", "DP_ATTENTION"),
        "prefill_num_workers": size("prefill_num_workers", "PREFILL_NUM_WORKERS"),
        "decode_tp": size("decode_tp", "DECODE_TP", "TP"),
        "decode_pp": size("decode_pp", "DECODE_PP_SIZE", "PP_SIZE"),
        "decode_dcp_size": size("decode_dcp_size", "DECODE_DCP_SIZE", "DCP_SIZE"),
        "decode_pcp_size": size("decode_pcp_size", "DECODE_PCP_SIZE", "PCP_SIZE"),
        "decode_ep": size("decode_ep", "DECODE_EP", "EP_SIZE"),
        "decode_dp_attention": dp("DECODE_DP_ATTENTION", "DP_ATTENTION"),
        "decode_num_workers": size("decode_num_workers", "DECODE_NUM_WORKERS"),
        "model": values.get("MODEL_NAME") or values.get("MODEL", ""),
        "infmax_model_prefix": values.get("MODEL_PREFIX") or "unknown",
        "hw": values.get("RUNNER_TYPE") or "unknown",
        "isl": values.get("ISL") or "0",
        "osl": values.get("OSL") or "0",
    }
    if env.optional("DISAGG", values) is not None:
        document["disagg"] = env.flag("DISAGG", values)
        document.update(eval_topology(document))
    return document


def write(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")


def refresh(path: Path, environ: Mapping[str, str]) -> None:
    """Rebuild a staged ``meta_env.json`` from ``environ``, keeping its suite, conc, and batch."""
    staged = json.loads(path.read_text())
    batch = staged if "eval_concs" in staged else None
    write(path, build(environ, conc=staged["conc"], suite=staged["eval_suite"], batch=batch))
