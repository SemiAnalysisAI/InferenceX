#!/usr/bin/env python3
"""Canonical, byte-stable CollectiveX routing workloads.

A *canonical workload* is a routing trace generated ONCE, serialized to a platform-independent
file, and referenced by an immutable `workload_id`. Every promoted benchmark point consumes the
SAME serialized bytes, so "did NVIDIA and AMD run the identical workload?" is answered by a
checksum match, not by trusting that two machines re-ran the same seeded generator.

Layout on disk (one workload = two files, basename = workload_id):
  <dir>/<workload_id>.npz            topk_idx [gt,topk] int32, topk_weights [gt,topk] float32
  <dir>/<workload_id>.manifest.json  dims, routing profile, generator version, seed, SHA-256s

Routing and gate weights come from a stdlib integer counter, not a framework RNG. The same
parameters therefore produce the same int32/float32 bytes across PyTorch and accelerator images.
"""
from __future__ import annotations

from array import array
import bisect
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import identity  # noqa: E402

WORKLOAD_SCHEMA_VERSION = 1
# Bump when the counter or byte encoding changes. The workload ID binds parameters and trace bytes.
GENERATOR_VERSION = "collectivex-routing-counter-v3"
GATE_WEIGHT_FORMAT = "counter-u16-normalized-f32"
ACTIVATION_GENERATOR = "collectivex-activation-counter-v4"
EPLB_CALIBRATION_WINDOW = "collectivex-eplb-calibration-window-v1"
EPLB_CALIBRATION_TOKEN_OFFSET = 1 << 32
_MASK64 = (1 << 64) - 1


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    if routing not in {"uniform", "zipf"}:
        raise ValueError(f"unknown routing {routing!r} (uniform|zipf)")
    if global_tokens <= 0 or experts <= 0 or topk <= 0 or topk > experts:
        raise ValueError("global_tokens/experts/topk must be positive and topk <= experts")
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError("token_offset must be a non-negative integer")

    cumulative: list[int] | None = None
    if routing == "zipf":
        total = 0
        cumulative = []
        for expert in range(experts):
            total += (1 << 32) // (expert + 1)
            cumulative.append(total)

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
                expert = (
                    value % experts
                    if cumulative is None
                    else bisect.bisect_right(cumulative, value % cumulative[-1])
                )
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


def _canonical_bytes(
    indices: list[list[int]], weights: list[list[float]]
) -> tuple[bytes, bytes]:
    idx = array("i", (value for row in indices for value in row))
    gate = array("f", (value for row in weights for value in row))
    if idx.itemsize != 4 or gate.itemsize != 4:
        raise RuntimeError("canonical workload requires 32-bit int and float arrays")
    if sys.byteorder != "little":
        idx.byteswap()
        gate.byteswap()
    return idx.tobytes(), gate.tobytes()


def trace_checksums(
    indices: list[list[int]], weights: list[list[float]]
) -> dict[str, str]:
    """Return the manifest hashes for exact logical or remapped routing rows."""
    idx_bytes, weight_bytes = _canonical_bytes(indices, weights)
    return {
        "topk_idx": _sha256(idx_bytes),
        "topk_weights": _sha256(weight_bytes),
        "trace": _sha256(idx_bytes + weight_bytes),
    }


def canonical_member(
    routing: str,
    hidden: int,
    topk: int,
    experts: int,
    ep_size: int,
    tokens_per_rank: int,
    seed: int,
    *,
    token_offset: int = 0,
) -> tuple[str, dict[str, str], list[list[int]], list[list[float]]]:
    """Derive one canonical manifest member and retain its rows for proof checks."""
    global_tokens = ep_size * tokens_per_rank
    indices, weights = canonical_routing_rows(
        global_tokens,
        experts,
        topk,
        routing,
        seed,
        token_offset=token_offset,
    )
    checksums = trace_checksums(indices, weights)
    member = compute_workload_id(
        routing,
        hidden,
        topk,
        experts,
        ep_size,
        global_tokens,
        seed,
        trace_checksum=checksums["trace"],
        token_offset=token_offset,
    )
    return member, checksums, indices, weights


def canonical_eplb_calibration_member(
    routing: str,
    hidden: int,
    topk: int,
    experts: int,
    ep_size: int,
    tokens_per_rank: int,
    seed: int,
) -> tuple[str, dict[str, str], list[list[int]], list[list[float]]]:
    """Return the EPLB calibration trace from a disjoint global-token window."""
    return canonical_member(
        routing,
        hidden,
        topk,
        experts,
        ep_size,
        tokens_per_rank,
        seed,
        token_offset=EPLB_CALIBRATION_TOKEN_OFFSET,
    )


def compute_workload_id(routing: str, hidden: int, topk: int, experts: int,
                        ep_size: int, global_tokens: int, seed: int,
                        generator: str = GENERATOR_VERSION,
                        trace_checksum: str | None = None,
                        token_offset: int = 0) -> str:
    """Deterministic ID over parameters and canonical trace bytes."""
    if generator != GENERATOR_VERSION:
        raise ValueError(f"unsupported workload generator {generator!r}")
    if type(token_offset) is not int or token_offset < 0:
        raise ValueError("token_offset must be a non-negative integer")
    if trace_checksum is None:
        indices, weights = canonical_routing_rows(
            global_tokens,
            experts,
            topk,
            routing,
            seed,
            token_offset=token_offset,
        )
        idx_bytes, weight_bytes = _canonical_bytes(indices, weights)
        trace_checksum = _sha256(idx_bytes + weight_bytes)
    key = {
        "generator": generator, "routing": routing, "hidden": hidden, "topk": topk,
        "experts": experts, "ep_size": ep_size, "global_tokens": global_tokens,
        "seed": seed, "trace_sha256": trace_checksum,
        "activation_generator": ACTIVATION_GENERATOR,
        "activation_identity": compute_activation_identity(seed, hidden),
    }
    if token_offset:
        key.update({
            "routing_window": EPLB_CALIBRATION_WINDOW,
            "token_offset": token_offset,
        })
    return identity.workload_id(key)


def compute_activation_identity(seed, hidden, generator=ACTIVATION_GENERATOR) -> str:
    """Identity of the exact counter-derived activation generator."""
    key = f"counter|seed={seed}|hidden={hidden}|gen={generator}"
    return _sha256(key.encode())


def build_manifest(routing, hidden, topk, experts, global_tokens, seed, experts_per_rank,
                   idx_np, weights_np):
    """Assemble the manifest dict from the (numpy) trace arrays. Pure numpy/stdlib."""
    if experts % experts_per_rank:
        raise ValueError("experts must be divisible by experts_per_rank")
    idx_bytes = idx_np.astype("<i4", copy=False).tobytes()
    w_bytes = weights_np.astype("<f4", copy=False).tobytes()
    ep_size = experts // experts_per_rank
    trace_checksum = _sha256(idx_bytes + w_bytes)
    wid = compute_workload_id(
        routing, hidden, topk, experts, ep_size, global_tokens, seed,
        trace_checksum=trace_checksum,
    )
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_id": wid,
        "generator_version": GENERATOR_VERSION,
        "gate_weight_format": GATE_WEIGHT_FORMAT,
        "dims": {"hidden": hidden, "topk": topk, "experts": experts, "ep_size": ep_size,
                 "tokens_per_rank": int(global_tokens) // ep_size,
                 "global_tokens": int(global_tokens), "experts_per_rank": experts_per_rank},
        "routing_profile": routing,
        "seed": seed,
        "checksums": {  # SHA-256 over the raw little-endian array bytes (int32 / float32)
            "topk_idx": _sha256(idx_bytes),
            "topk_weights": _sha256(w_bytes),   # gate-weight (value) distribution identity
            "trace": trace_checksum,
        },
        "activation_profile": "canonical-counter-source-v3",
        "activation_generator": ACTIVATION_GENERATOR,
        "activation_identity": compute_activation_identity(seed, hidden),
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
    wid = manifest["workload_id"]
    np.savez_compressed(os.path.join(out_dir, f"{wid}.npz"),
                        topk_idx=idx_np.astype(np.int32), topk_weights=weights_np.astype(np.float32))
    with open(os.path.join(out_dir, f"{wid}.manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return wid


def load_workload(npz_path, verify=True):
    """Load a canonical trace (numpy + stdlib only). Returns (idx_np, weights_np, manifest).
    Raises ValueError if verify=True and the on-disk bytes don't match the manifest checksums."""
    import numpy as np
    base = npz_path[:-4] if npz_path.endswith(".npz") else npz_path
    with open(base + ".manifest.json") as fh:
        manifest = json.load(fh)
    if manifest.get("workload_id") != os.path.basename(base):
        raise ValueError(f"workload manifest ID does not match filename for {base}")
    with np.load(base + ".npz", allow_pickle=False) as archive:
        if set(archive.files) != {"topk_idx", "topk_weights"}:
            raise ValueError(f"workload archive fields differ for {base}")
        idx_np = np.ascontiguousarray(archive["topk_idx"])
        w_np = np.ascontiguousarray(archive["topk_weights"])
    if verify:
        ok, reason = verify_workload(manifest, idx_np, w_np)
        if not ok:
            raise ValueError(f"workload checksum mismatch for {base}: {reason}")
    return idx_np, w_np, manifest


def verify_workload(manifest, idx_np, weights_np):
    """Recompute checksums and compare to the manifest. Returns (ok, reason)."""
    import numpy as np
    expected_fields = {
        "schema_version", "workload_id", "generator_version", "gate_weight_format", "dims",
        "routing_profile", "seed", "checksums", "activation_profile", "activation_generator",
        "activation_identity",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        return False, "manifest fields differ from the v1 contract"
    if (manifest["schema_version"] != WORKLOAD_SCHEMA_VERSION
            or manifest["generator_version"] != GENERATOR_VERSION
            or manifest["gate_weight_format"] != GATE_WEIGHT_FORMAT
            or manifest["routing_profile"] not in {"uniform", "zipf"}):
        return False, "manifest version or generator is unsupported"
    if (isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int)
            or not identity.is_typed_id(manifest["workload_id"], "workload")):
        return False, "manifest seed or workload ID is invalid"
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
            or manifest["activation_generator"] != ACTIVATION_GENERATOR
            or manifest["activation_identity"]
            != compute_activation_identity(
                manifest["seed"], dims["hidden"], manifest["activation_generator"]
            )):
        return False, "activation identity is invalid"
    ib = idx_np.astype("<i4", copy=False).tobytes()
    wb = weights_np.astype("<f4", copy=False).tobytes()
    cs = manifest.get("checksums", {})
    if set(cs) != {"topk_idx", "topk_weights", "trace"}:
        return False, "checksum fields are invalid"
    if _sha256(ib) != cs.get("topk_idx"):
        return False, "topk_idx hash differs"
    if _sha256(wb) != cs.get("topk_weights"):
        return False, "topk_weights hash differs"
    if _sha256(ib + wb) != cs.get("trace"):
        return False, "trace hash differs"
    wid = compute_workload_id(
        manifest["routing_profile"], manifest["dims"]["hidden"],
        manifest["dims"]["topk"], manifest["dims"]["experts"],
        manifest["dims"]["ep_size"], manifest["dims"]["global_tokens"], manifest["seed"],
        manifest.get("generator_version", GENERATOR_VERSION), trace_checksum=cs["trace"],
    )
    if wid != manifest["workload_id"]:
        return False, f"workload_id mismatch (recomputed {wid} != {manifest['workload_id']})"
    return True, "ok"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import tempfile
    # (1) workload_id determinism + sensitivity — pure stdlib, always runs.
    a = compute_workload_id("zipf", 7168, 8, 256, 8, 4096, 67)
    b = compute_workload_id("zipf", 7168, 8, 256, 8, 4096, 67)
    c = compute_workload_id("uniform", 7168, 8, 256, 8, 4096, 67)
    assert a == b, "workload_id must be deterministic"
    assert a != c, "workload_id must depend on routing"
    print(f"workload_id determinism OK (zipf={a} uniform={c})")
    # (2) build/save/load/verify roundtrip + cross-build identity — needs torch+numpy.
    try:
        import numpy as np  # noqa: F401
        idx, w, man = build_workload(7168, 8, 256, "zipf", 512, 67, 32)
        with tempfile.TemporaryDirectory() as d:
            wid = save_workload(d, idx, w, man)
            idx2, w2, man2 = load_workload(os.path.join(d, f"{wid}.npz"), verify=True)
            assert (idx2 == idx).all() and (w2 == w).all(), "roundtrip array mismatch"
            ok, reason = verify_workload(man2, idx2, w2)
            assert ok, reason
            # tamper -> must fail
            idx2[0, 0] = (int(idx2[0, 0]) + 1) % 256
            bad, _ = verify_workload(man2, idx2, w2)
            assert not bad, "verify must catch tampering"
        print(f"save/load/verify roundtrip OK (workload_id={wid})")
    except ImportError:
        print("(numpy unavailable — skipped serialization roundtrip; id logic passed)")
    print("workload self-test: PASS")
    sys.exit(0)
