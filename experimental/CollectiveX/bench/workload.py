#!/usr/bin/env python3
"""Canonical CollectiveX routing workloads.

A *canonical workload* is a routing trace generated ONCE, serialized to a platform-independent
file named from its explicit routing and shape coordinates. Every promoted benchmark point consumes
the serialized arrays after validating their manifest, dtype, shape, and value constraints.

Layout on disk (one workload = two files):
  <dir>/<workload_name>.npz            topk_idx [gt,topk] int32, topk_weights [gt,topk] float32
  <dir>/<workload_name>.manifest.json  dims, routing profile, generator version, and seed

Routing and gate weights come from a stdlib integer counter, not a framework RNG. The same
parameters therefore produce the same int32/float32 bytes across PyTorch and accelerator images.
"""
from __future__ import annotations

import json
import os
import sys

WORKLOAD_SCHEMA_VERSION = 1
# Bump when the counter or serialized layout changes.
GENERATOR_VERSION = "collectivex-routing-counter-v3"
GATE_WEIGHT_FORMAT = "counter-u16-normalized-f32"
ACTIVATION_GENERATOR = "collectivex-activation-counter-v4"
_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _counter(seed: int, token: int, slot: int, attempt: int, stream: int) -> int:
    value = (
        (seed & _MASK64)
        ^ (((token + 1) * 0xD2B74407B1CE6E93) & _MASK64)
        ^ (((slot + 1) * 0xCA5A826395121157) & _MASK64)
        ^ (((attempt + 1) * 0x9E3779B185EBCA87) & _MASK64)
        ^ (((stream + 1) * 0xA24BAED4963EE407) & _MASK64)
    )
    return _mix64(value)


def canonical_routing_rows(
    global_tokens: int,
    experts: int,
    topk: int,
    routing: str,
    seed: int,
    *,
    token_offset: int = 0,
) -> tuple[list[list[int]], list[list[float]]]:
    """Generate a deterministic routing window from exact integer counters."""
    if routing != "uniform":
        raise ValueError(f"unknown routing {routing!r} (uniform)")
    if global_tokens <= 0 or experts <= 0 or topk <= 0 or topk > experts:
        raise ValueError("global_tokens/experts/topk must be positive and topk <= experts")
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError("token_offset must be a non-negative integer")

    indices: list[list[int]] = []
    weights: list[list[float]] = []
    for local_token in range(global_tokens):
        token = token_offset + local_token
        selected: list[int] = []
        used: set[int] = set()
        for slot in range(topk):
            attempt = 0
            while True:
                value = _counter(seed, token, slot, attempt, 0)
                expert = value % experts
                if expert not in used:
                    used.add(expert)
                    selected.append(expert)
                    break
                attempt += 1
                if attempt > experts * 16:
                    raise RuntimeError("counter routing could not select distinct experts")
        raw = [1 + _counter(seed, token, slot, 0, 1) % 65535 for slot in range(topk)]
        denominator = float(sum(raw))
        indices.append(selected)
        weights.append([value / denominator for value in raw])
    return indices, weights


def workload_name(routing: str, hidden: int, topk: int, experts: int,
                  ep_size: int, global_tokens: int, seed: int,
                  generator: str = GENERATOR_VERSION,
                  token_offset: int = 0) -> str:
    """Return a readable filename stem for one canonical workload."""
    if generator != GENERATOR_VERSION:
        raise ValueError(f"unsupported workload generator {generator!r}")
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError("token_offset must be a non-negative integer")
    tokens_per_rank, remainder = divmod(global_tokens, ep_size)
    if remainder or min(hidden, topk, experts, ep_size, tokens_per_rank) <= 0:
        raise ValueError("workload dimensions must be positive and EP-divisible")
    return (
        f"cxwork-v1-{routing}-h{hidden}-k{topk}-e{experts}-ep{ep_size}"
        f"-t{tokens_per_rank}-s{seed}-o{token_offset}"
    )


def build_manifest(routing, hidden, topk, experts, global_tokens, seed, experts_per_rank,
                   idx_np, weights_np):
    """Assemble the manifest dict from the (numpy) trace arrays. Pure numpy/stdlib."""
    if experts % experts_per_rank:
        raise ValueError("experts must be divisible by experts_per_rank")
    ep_size = experts // experts_per_rank
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_name": workload_name(
            routing, hidden, topk, experts, ep_size, global_tokens, seed
        ),
        "generator_version": GENERATOR_VERSION,
        "gate_weight_format": GATE_WEIGHT_FORMAT,
        "dims": {"hidden": hidden, "topk": topk, "experts": experts, "ep_size": ep_size,
                 "tokens_per_rank": int(global_tokens) // ep_size,
                 "global_tokens": int(global_tokens), "experts_per_rank": experts_per_rank},
        "routing_profile": routing,
        "seed": seed,
        "activation_profile": "canonical-counter-source-v3",
        "activation_generator": ACTIVATION_GENERATOR,
    }


def build_workload(hidden, topk, experts, routing, global_tokens, seed, experts_per_rank):
    """Generate a canonical trace. Returns (idx_np, weights_np, manifest)."""
    import numpy as np
    indices, weights = canonical_routing_rows(global_tokens, experts, topk, routing, seed)
    idx_np = np.asarray(indices, dtype=np.int32)
    w_np = np.asarray(weights, dtype=np.float32)
    manifest = build_manifest(
        routing, hidden, topk, experts, global_tokens, seed,
        experts_per_rank, idx_np, w_np,
    )
    return idx_np, w_np, manifest


def save_workload(out_dir, idx_np, weights_np, manifest) -> str:
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    name = manifest["workload_name"]
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                        topk_idx=idx_np.astype(np.int32), topk_weights=weights_np.astype(np.float32))
    with open(os.path.join(out_dir, f"{name}.manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return name


def load_workload(npz_path, verify=True):
    """Load a canonical trace (numpy + stdlib only). Returns (idx_np, weights_np, manifest).
    Raises ValueError if validation fails."""
    import numpy as np
    base = npz_path[:-4] if npz_path.endswith(".npz") else npz_path
    with open(base + ".manifest.json") as fh:
        manifest = json.load(fh)
    if manifest.get("workload_name") != os.path.basename(base):
        raise ValueError(f"workload manifest name does not match filename for {base}")
    with np.load(base + ".npz", allow_pickle=False) as archive:
        if set(archive.files) != {"topk_idx", "topk_weights"}:
            raise ValueError(f"workload archive fields differ for {base}")
        idx_np = np.ascontiguousarray(archive["topk_idx"])
        w_np = np.ascontiguousarray(archive["topk_weights"])
    if verify:
        ok, reason = verify_workload(manifest, idx_np, w_np)
        if not ok:
            raise ValueError(f"workload validation failed for {base}: {reason}")
    return idx_np, w_np, manifest


def verify_workload(manifest, idx_np, weights_np):
    """Validate manifest coordinates and serialized arrays. Returns (ok, reason)."""
    import numpy as np
    expected_fields = {
        "schema_version", "workload_name", "generator_version", "gate_weight_format", "dims",
        "routing_profile", "seed", "activation_profile", "activation_generator",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        return False, "manifest fields differ from the v1 contract"
    if (manifest["schema_version"] != WORKLOAD_SCHEMA_VERSION
            or manifest["generator_version"] != GENERATOR_VERSION
            or manifest["gate_weight_format"] != GATE_WEIGHT_FORMAT
            or manifest["routing_profile"] != "uniform"):
        return False, "manifest version or generator is unsupported"
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        return False, "manifest seed is invalid"
    dims = manifest["dims"]
    dim_fields = {"hidden", "topk", "experts", "ep_size", "tokens_per_rank",
                  "global_tokens", "experts_per_rank"}
    if not isinstance(dims, dict) or set(dims) != dim_fields:
        return False, "manifest dimensions are invalid"
    if any(isinstance(dims[key], bool) or not isinstance(dims[key], int) or dims[key] <= 0
           for key in dim_fields):
        return False, "manifest dimensions must be positive integers"
    if (dims["experts"] != dims["ep_size"] * dims["experts_per_rank"]
            or dims["global_tokens"] != dims["ep_size"] * dims["tokens_per_rank"]):
        return False, "manifest EP dimensions are inconsistent"
    shape = (dims["global_tokens"], dims["topk"])
    if (idx_np.dtype != np.int32 or weights_np.dtype != np.float32
            or idx_np.shape != shape or weights_np.shape != shape
            or not idx_np.flags.c_contiguous or not weights_np.flags.c_contiguous):
        return False, "workload array dtype, shape, or layout is invalid"
    if (np.any(idx_np < 0) or np.any(idx_np >= dims["experts"])
            or np.any(np.diff(np.sort(idx_np, axis=1), axis=1) == 0)):
        return False, "expert indices are out of range or repeated"
    if (not np.isfinite(weights_np).all() or np.any(weights_np < 0)
            or not np.allclose(weights_np.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)):
        return False, "gate weights are invalid"
    if (manifest["activation_profile"] != "canonical-counter-source-v3"
            or manifest["activation_generator"] != ACTIVATION_GENERATOR):
        return False, "activation generator is invalid"
    expected_indices, expected_weights = canonical_routing_rows(
        dims["global_tokens"], dims["experts"], dims["topk"],
        manifest["routing_profile"], manifest["seed"],
    )
    if (not np.array_equal(idx_np, np.asarray(expected_indices, dtype=np.int32))
            or not np.array_equal(weights_np, np.asarray(expected_weights, dtype=np.float32))):
        return False, "workload arrays differ from the deterministic generator"
    expected_name = workload_name(
        manifest["routing_profile"], dims["hidden"], dims["topk"], dims["experts"],
        dims["ep_size"], dims["global_tokens"], manifest["seed"],
        manifest["generator_version"],
    )
    if expected_name != manifest["workload_name"]:
        return False, "workload name differs from manifest coordinates"
    return True, "ok"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import tempfile
    # (1) readable workload-name determinism and sensitivity.
    a = workload_name("uniform", 7168, 8, 256, 8, 4096, 67)
    b = workload_name("uniform", 7168, 8, 256, 8, 4096, 67)
    c = workload_name("uniform", 7168, 8, 256, 8, 4096, 68)
    assert a == b, "workload name must be deterministic"
    assert a != c, "workload name must depend on seed"
    print(f"workload-name determinism OK (uniform={a})")
    # (2) build/save/load/verify roundtrip.
    try:
        import numpy as np  # noqa: F401
        idx, w, man = build_workload(7168, 8, 256, "uniform", 512, 67, 32)
        with tempfile.TemporaryDirectory() as d:
            name = save_workload(d, idx, w, man)
            idx2, w2, man2 = load_workload(os.path.join(d, f"{name}.npz"), verify=True)
            assert (idx2 == idx).all() and (w2 == w).all(), "roundtrip array mismatch"
            ok, reason = verify_workload(man2, idx2, w2)
            assert ok, reason
            # tamper -> must fail
            idx2[0, 0] = (int(idx2[0, 0]) + 1) % 256
            bad, _ = verify_workload(man2, idx2, w2)
            assert not bad, "verify must catch tampering"
        print(f"save/load/verify roundtrip OK (workload_name={name})")
    except ImportError:
        print("(numpy unavailable — skipped serialization roundtrip; name logic passed)")
    print("workload self-test: PASS")
    sys.exit(0)
