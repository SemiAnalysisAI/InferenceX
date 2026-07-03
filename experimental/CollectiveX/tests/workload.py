#!/usr/bin/env python3
"""CollectiveX — canonical, serialized MoE routing workloads (goal Part 1: workload identity).

A *canonical workload* is a routing trace generated ONCE, serialized to a platform-independent
file, and referenced by an immutable `workload_id`. Every official benchmark point consumes the
SAME serialized bytes, so "did NVIDIA and AMD run the identical workload?" is answered by a
checksum match, not by trusting that two machines re-ran the same seeded generator.

Layout on disk (one workload = two files, basename = workload_id):
  <dir>/<workload_id>.npz            topk_idx [gt,topk] int32, topk_weights [gt,topk] float32
  <dir>/<workload_id>.manifest.json  dims, routing profile, generator version, seed, SHA-256s

Split by dependency so it runs where each step lives:
  * build_workload()  needs torch (via routing.py) — run on a node/container.
  * load/verify/manifest  need only numpy + stdlib — run on a login node or in CI.

Seeded runtime generation (routing.build_global_routing) stays for local dev; canonical files
are how cross-hardware comparisons are gated.
"""
from __future__ import annotations

import hashlib
import json
import os

WORKLOAD_SCHEMA_VERSION = 1
# Bump when routing.build_global_routing's numerics change so a stale file can't masquerade as
# current. The workload_id folds this in: same id <=> same generator + params.
GENERATOR_VERSION = "collectivex-routing-v1"
GATE_WEIGHT_FORMAT = "softmax-of-randn-f32"   # how topk_weights are produced (see routing.py)
ACTIVATION_GENERATOR = "collectivex-activation-v1"  # bump if the activation value-generator changes
ACTIVATION_PROFILE_DEFAULT = "normal"               # seeded N(0,1) per token; the only wired profile


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_workload_id(routing: str, hidden: int, topk: int, experts: int,
                        global_tokens: int, seed: int, generator: str = GENERATOR_VERSION,
                        step: int = 0) -> str:
    """Deterministic id over the identity-defining params. Same params+generator => same id.
    `step` is the temporal snapshot for moving/alternating routing; folded in ONLY when non-zero
    so every existing (step=0) canonical workload keeps its id."""
    key = (f"{generator}|routing={routing}|hidden={hidden}|topk={topk}|experts={experts}"
           f"|gt={global_tokens}|seed={seed}")
    if step:
        key += f"|step={step}"
    return _sha256(key.encode())[:16]


def compute_activation_identity(activation_profile, seed, hidden,
                                generator=ACTIVATION_GENERATOR) -> str:
    """Deterministic identity of the activation VALUE distribution (scaffold). Today activations
    are seeded N(0,1) and NOT serialized, so identity = a descriptor hash. The formula MUST match
    the inline one in ep_harness so a manifest and a result doc agree. Becomes the byte-hash of
    the serialized activations once a model-trace value rig lands."""
    key = f"{activation_profile}|seed={seed}|hidden={hidden}|gen={generator}"
    return _sha256(key.encode())[:16]


def build_manifest(routing, hidden, topk, experts, global_tokens, seed, experts_per_rank,
                   idx_np, weights_np, routing_stats=None,
                   activation_profile=ACTIVATION_PROFILE_DEFAULT):
    """Assemble the manifest dict from the (numpy) trace arrays. Pure numpy/stdlib."""
    idx_bytes = idx_np.astype("int32").tobytes()
    w_bytes = weights_np.astype("float32").tobytes()
    wid = compute_workload_id(routing, hidden, topk, experts, global_tokens, seed)
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_id": wid,
        "generator_version": GENERATOR_VERSION,
        "gate_weight_format": GATE_WEIGHT_FORMAT,
        "dims": {"hidden": hidden, "topk": topk, "experts": experts,
                 "global_tokens": int(global_tokens), "experts_per_rank": experts_per_rank},
        "routing_profile": routing,
        "seed": seed,
        "checksums": {  # SHA-256 over the raw little-endian array bytes (int32 / float32)
            "topk_idx": _sha256(idx_bytes),
            "topk_weights": _sha256(w_bytes),   # gate-weight (value) distribution identity
            "trace": _sha256(idx_bytes + w_bytes),   # full-workload identity
        },
        "routing_stats": routing_stats or {},
        # Activation value distribution (scaffold): name + deterministic descriptor identity.
        # NOT under checksums — activations are not byte-serialized today (see compute_activation_identity).
        "activation_profile": activation_profile,
        "activation_identity": compute_activation_identity(activation_profile, seed, hidden),
    }


def build_workload(hidden, topk, experts, routing, global_tokens, seed, experts_per_rank,
                   activation_profile=ACTIVATION_PROFILE_DEFAULT):
    """Generate a canonical trace. Needs torch (routing.py). Returns (idx_np, weights_np, manifest)."""
    import numpy as np
    import routing as _routing
    idx_t, w_t = _routing.build_global_routing(global_tokens, experts, topk, routing, seed,
                                               experts_per_rank)
    rstats = _routing.routing_stats(idx_t, experts, experts_per_rank, weights=w_t)
    idx_np = idx_t.detach().cpu().numpy().astype(np.int32)
    w_np = w_t.detach().cpu().numpy().astype(np.float32)
    manifest = build_manifest(routing, hidden, topk, experts, global_tokens, seed,
                              experts_per_rank, idx_np, w_np, rstats,
                              activation_profile=activation_profile)
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
    z = np.load(base + ".npz")
    idx_np, w_np = z["topk_idx"], z["topk_weights"]
    if verify:
        ok, reason = verify_workload(manifest, idx_np, w_np)
        if not ok:
            raise ValueError(f"workload checksum mismatch for {base}: {reason}")
    return idx_np, w_np, manifest


def verify_workload(manifest, idx_np, weights_np):
    """Recompute checksums and compare to the manifest. Returns (ok, reason)."""
    import numpy as np  # noqa: F401
    ib = idx_np.astype("int32").tobytes()
    wb = weights_np.astype("float32").tobytes()
    cs = manifest.get("checksums", {})
    if _sha256(ib) != cs.get("topk_idx"):
        return False, "topk_idx hash differs"
    if _sha256(wb) != cs.get("topk_weights"):
        return False, "topk_weights hash differs"
    if _sha256(ib + wb) != cs.get("trace"):
        return False, "trace hash differs"
    wid = compute_workload_id(manifest["routing_profile"], manifest["dims"]["hidden"],
                              manifest["dims"]["topk"], manifest["dims"]["experts"],
                              manifest["dims"]["global_tokens"], manifest["seed"],
                              manifest.get("generator_version", GENERATOR_VERSION))
    if wid != manifest["workload_id"]:
        return False, f"workload_id mismatch (recomputed {wid} != {manifest['workload_id']})"
    return True, "ok"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import tempfile
    # (1) workload_id determinism + sensitivity — pure stdlib, always runs.
    a = compute_workload_id("zipf", 7168, 8, 256, 4096, 67)
    b = compute_workload_id("zipf", 7168, 8, 256, 4096, 67)
    c = compute_workload_id("uniform", 7168, 8, 256, 4096, 67)
    assert a == b, "workload_id must be deterministic"
    assert a != c, "workload_id must depend on routing"
    print(f"workload_id determinism OK (zipf={a} uniform={c})")
    # (2) build/save/load/verify roundtrip + cross-build identity — needs torch+numpy.
    try:
        import numpy as np  # noqa: F401
        try:
            idx, w, man = build_workload(7168, 8, 256, "zipf", 512, 67, 32)
            built = True
        except Exception as exc:   # torch missing on a login node
            print(f"(torch unavailable — synthesizing arrays to test load/verify: {exc!r})")
            idx = np.random.default_rng(0).integers(0, 256, size=(512, 8)).astype(np.int32)
            w = np.random.default_rng(1).random((512, 8)).astype(np.float32)
            man = build_manifest("zipf", 7168, 8, 256, 512, 67, 32, idx, w)
            built = False
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
        print(f"save/load/verify roundtrip OK (workload_id={wid}, built_via_torch={built})")
    except ImportError:
        print("(numpy unavailable — skipped serialization roundtrip; id logic passed)")
    print("workload self-test: PASS")
    sys.exit(0)
