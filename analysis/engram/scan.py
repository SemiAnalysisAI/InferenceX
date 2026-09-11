"""Rank the suffix n-grams that coincide with strong Engram gates.

Procedure (mirrors the reference gate-activation analysis, adapted to the
production checkpoint):
  1. prefill domain corpora through the real model,
  2. record the Engram gate at every token position,
  3. keep the right tail of the gate distribution,
  4. decode the suffix 2-, 3-, and 4-grams ending at each strong position,
  5. aggregate count and mean gate per n-gram, rank by mean gate.

DeepSeek-V4.1-Flash hashes `engram_max_ngram_size - 1` tiers (2..4-grams) at
each of two insertion layers (1 and 14), so every tier the reference study
reported has a counterpart here, plus a 4-gram tier it did not have.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("engram-scan")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engram import corpora, gate_probe  # noqa: E402

NGRAM_SIZES = (2, 3, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk-tokens", type=int, default=3584)
    ap.add_argument("--max-chunks-per-domain", type=int, default=1750,
                    help="per shard; 0 = all. This is the wall-clock knob: the ten "
                         "streaming domains will always hit it, the small "
                         "reference corpora exhaust first. See README for the "
                         "chunks-to-hours mapping.")
    ap.add_argument("--probe-dir", default=os.environ.get("ENGRAM_PROBE_SCRATCH"),
                    help="Node-local scratch for the probe's .npy files. "
                         "Must not be on NFS.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Rescan domains that already have a part file on disk.")
    ap.add_argument("--strong-quantile", type=float, default=0.99)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--out", default=os.environ.get("RESULT_DIR", ".") + "/engram_scan")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Deliberately NOT under --out: that is NFS-backed (/ix), and every chunk
    # writes 16 small files (8 ranks x 2 layers) and reads them all back. The
    # probe was costing 0.52s of the 0.73s per chunk.
    probe_dir = args.probe_dir or f"/dev/shm/engram_probe_shard{args.shard}"
    os.makedirs(probe_dir, exist_ok=True)
    logger.info("engram-probe: writing gates to %s", probe_dir)
    # Workers inherit this, so the wrapper knows where to write.
    os.environ[gate_probe.PROBE_DIR_ENV] = probe_dir

    # Patch the driver, then arm the spawned TP workers -- which is where the
    # model actually lives, and where the first run captured nothing.
    gate_probe.install()
    analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info("engram-probe: worker bootstrap at %s", gate_probe.install_in_workers(analysis_dir))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    domains = corpora.all_domains()  # priority order; see corpora.PRIORITY
    logger.info("scanning %d domains: %s", len(domains), ", ".join(domains))

    # A short context keeps the sparse-attention indexer's
    # [batched-tokens, max-model-len] buffer small; this is a prefill scan, so
    # the 1M serving context buys nothing here.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
        **({"language_model_only": True} if _supports_lmo() else {}),
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    # stats[(domain, layer, n)][ngram] -> [count, gate_sum]
    stats: dict[tuple, dict[tuple[int, ...], list]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0.0])
    )
    gate_hist: dict[tuple, list] = collections.defaultdict(list)
    # Full gate distribution per (domain, layer) -- the earlier runs only kept
    # the strong tail, so the observed maximum was an artifact of selection.
    dist: dict[tuple, dict] = collections.defaultdict(_new_dist)
    seen_tokens = collections.Counter()

    for domain in domains:
        part_path = _part_path(args.out, args.shard, domain)
        if os.path.exists(part_path) and not args.no_resume:
            logger.info("domain %s: already on disk, skipping (%s)", domain, part_path)
            continue
        chunks = list(
            _take_chunks(
                corpora.iter_texts(domain),
                tokenizer,
                args.chunk_tokens,
                args.max_chunks_per_domain,
                args.shard,
                args.num_shards,
            )
        )
        if not chunks:
            logger.warning("domain %s: no chunks (source unusable or empty)", domain)
            continue
        logger.info("domain %s: %d chunks on shard %d", domain, len(chunks), args.shard)

        # Pass 1 collects gates; the threshold is per (domain, layer).
        per_chunk: list[tuple[list[int], dict[int, "object"]]] = []
        timing = collections.Counter()
        for idx, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            _clear(probe_dir)
            t1 = time.perf_counter()
            llm.generate({"prompt_token_ids": chunk}, sampling, use_tqdm=False)
            t2 = time.perf_counter()
            captured = _collect(probe_dir, len(chunk))
            t3 = time.perf_counter()
            timing["clear"] += t1 - t0
            timing["prefill"] += t2 - t1
            timing["collect"] += t3 - t2
            if not captured:
                logger.warning("domain %s chunk %d: no gates captured", domain, idx)
                continue
            per_chunk.append((chunk, {l: g.max(axis=1) for l, g in captured.items()}))
            for layer, gates in captured.items():
                # Threshold and rank on the strongest hyper-connection copy.
                gate_hist[(domain, layer)].append(gates.max(axis=1))
                _accumulate_dist(dist[(domain, layer)], gates)
            seen_tokens[domain] += len(chunk)
            timing["accumulate"] += time.perf_counter() - t3
            if idx % 25 == 0:
                logger.info("domain %s: %d/%d chunks", domain, idx + 1, len(chunks))

        if chunks:
            n = len(chunks)
            logger.info(
                "domain %s timing per chunk: prefill %.3fs collect %.3fs "
                "clear %.3fs accumulate %.3fs (total %.3fs)",
                domain, timing["prefill"] / n, timing["collect"] / n,
                timing["clear"] / n, timing["accumulate"] / n,
                sum(timing.values()) / n,
            )

        thresholds = {}
        for (dom, layer), arrays in list(gate_hist.items()):
            if dom != domain:
                continue
            import numpy as np

            flat = np.concatenate([a for a in arrays])
            thresholds[layer] = float(np.quantile(flat, args.strong_quantile))
            logger.info(
                "domain %s layer %s: strong-gate threshold q%.3f = %.4f (n=%d)",
                dom, layer, args.strong_quantile, thresholds[layer], flat.size,
            )

        # Pass 2 attributes each strong position to the n-grams ending there.
        for chunk, captured in per_chunk:
            for layer, gates in captured.items():
                thr = thresholds.get(layer)
                if thr is None:
                    continue
                for t, gate in enumerate(gates):
                    # A chunk boundary truncates the lookback, so the first
                    # max(n)-1 positions have incomplete n-grams for some tier.
                    # Skipping them keeps every tier over the same positions.
                    if t + 1 < max(NGRAM_SIZES) or gate < thr:
                        continue
                    for n in NGRAM_SIZES:
                        key = tuple(chunk[t + 1 - n : t + 1])
                        slot = stats[(domain, layer, n)][key]
                        slot[0] += 1
                        slot[1] += float(gate)

        # Write each domain the moment it finishes. The 180-minute Slurm
        # allocation killed a three-hour scan that held everything until the
        # end, losing 14 completed domains; a partial scan now keeps its work
        # and a rerun resumes from here.
        part = _render(
            {k: v for k, v in stats.items() if k[0] == domain},
            tokenizer, args.top_k, args.min_count,
        )
        part["_tokens"] = seen_tokens[domain]
        part["_gate_dist"] = {
            f"engram{layer}": _finalize_dist(acc)
            for (dom, layer), acc in dist.items()
            if dom == domain
        }
        tmp = part_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(part, handle, indent=2)
        os.replace(tmp, part_path)
        logger.info("domain %s: wrote %s", domain, part_path)
        for key in [k for k in stats if k[0] == domain]:
            del stats[key]
        for key in [k for k in gate_hist if k[0] == domain]:
            del gate_hist[key]
        for key in [k for k in dist if k[0] == domain]:
            del dist[key]

    report, tokens, dists = _merge_parts(args.out, args.shard, domains)
    report["meta"] = {
        "model": args.model,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "tokens_scanned": tokens,
        "gate_distribution": dists,
        "strong_quantile": args.strong_quantile,
        "ngram_sizes": list(NGRAM_SIZES),
    }
    path = os.path.join(args.out, f"engram_scan_shard{args.shard}.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    logger.info("wrote %s", path)
    # stdout is the channel that always survives, so emit the whole report.
    print("===ENGRAM_SCAN_JSON_BEGIN===")
    print(json.dumps(report))
    print("===ENGRAM_SCAN_JSON_END===")
    if not tokens:
        # The first run emitted an empty report and still exited 0, which read
        # as "ran fine" when the probe had in fact never fired. Fail loudly.
        logger.error("no gates captured for any domain; the probe never fired")
        return 1
    return 0





# Gate histogram over [0, 1); the gate is a sigmoid so it cannot leave that range.
_DIST_BINS = 200


def _new_dist():
    import numpy as np

    return {
        "hist": np.zeros(_DIST_BINS, dtype=np.int64),
        "max": 0.0,
        "sum": 0.0,
        "n": 0,
        "per_copy_sum": None,
        "per_copy_max": None,
    }


def _accumulate_dist(acc, gates):
    """Fold one chunk's [tokens, hc] gates into the running distribution."""
    import numpy as np

    flat = gates.reshape(-1)
    acc["hist"] += np.bincount(
        np.clip((flat * _DIST_BINS).astype(np.int64), 0, _DIST_BINS - 1),
        minlength=_DIST_BINS,
    )
    acc["max"] = max(acc["max"], float(flat.max()))
    acc["sum"] += float(flat.sum())
    acc["n"] += int(flat.size)
    copies = gates.shape[1]
    if acc["per_copy_sum"] is None:
        acc["per_copy_sum"] = np.zeros(copies, dtype=np.float64)
        acc["per_copy_max"] = np.zeros(copies, dtype=np.float64)
    acc["per_copy_sum"] += gates.sum(axis=0)
    acc["per_copy_max"] = np.maximum(acc["per_copy_max"], gates.max(axis=0))


def _finalize_dist(acc):
    if not acc["n"]:
        return {}
    import numpy as np

    hist = acc["hist"]
    cum = np.cumsum(hist) / acc["n"]
    quantiles = {
        f"q{q}": float(np.searchsorted(cum, q) + 0.5) / _DIST_BINS
        for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
    }
    tokens = acc["n"] // len(acc["per_copy_sum"])
    return {
        "max": round(acc["max"], 5),
        "mean": round(acc["sum"] / acc["n"], 5),
        "n_gate_values": acc["n"],
        **{k: round(v, 5) for k, v in quantiles.items()},
        "per_copy_mean": [round(v / tokens, 5) for v in acc["per_copy_sum"]],
        "per_copy_max": [round(float(v), 5) for v in acc["per_copy_max"]],
        "hist_bins": _DIST_BINS,
        "hist": hist.tolist(),
    }


def _part_path(out_dir, shard, domain):
    return os.path.join(out_dir, f"part_shard{shard}_{domain}.json")


def _merge_parts(out_dir, shard, domains):
    """Assemble the final report from the per-domain files on disk."""
    report, tokens, dists = {}, {}, {}
    for domain in domains:
        path = _part_path(out_dir, shard, domain)
        if not os.path.exists(path):
            logger.warning("domain %s: no part file; absent from the report", domain)
            continue
        with open(path) as handle:
            part = json.load(handle)
        tokens[domain] = part.pop("_tokens", 0)
        dists[domain] = part.pop("_gate_dist", {})
        report.update(part)
    return report, tokens, dists


def _take_chunks(pieces, tokenizer, chunk_tokens, limit, shard, num_shards):
    """Yield this shard's token chunks, tokenizing lazily as the stream is read.

    The corpora are streamed, so they cannot be tokenized up front -- a
    five-hour scan reads more text than fits in memory. Tokenizing in row
    batches keeps the fast tokenizer's batching win without materializing the
    corpus.
    """
    buffer: list[int] = []
    produced = 0
    index = 0
    batch: list[str] = []

    def flush(batch):
        nonlocal buffer, produced, index
        if not batch:
            return
        for ids in tokenizer(batch, add_special_tokens=False).input_ids:
            buffer.extend(ids)
            while len(buffer) >= chunk_tokens:
                chunk, buffer = buffer[:chunk_tokens], buffer[chunk_tokens:]
                if index % num_shards == shard:
                    yield chunk
                    produced += 1
                index += 1
                if limit and produced >= limit:
                    return

    for piece in pieces:
        batch.append(piece)
        if len(batch) < 64:
            continue
        for chunk in flush(batch):
            yield chunk
        batch = []
        if limit and produced >= limit:
            return
    for chunk in flush(batch):
        yield chunk


def _supports_lmo() -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs

        return hasattr(EngineArgs, "language_model_only")
    except Exception:
        return False


def _clear(probe_dir):
    for name in os.listdir(probe_dir):
        try:
            os.unlink(os.path.join(probe_dir, name))
        except OSError:
            pass


def _collect(probe_dir, num_tokens):
    """Stitch this prefill's gate slices into one array per Engram layer.

    Without sequence parallelism every rank holds the whole sequence, so any
    full-length slice wins. With it, ranks hold disjoint windows that have to
    be ordered by their start offset.
    """
    import numpy as np

    by_layer = collections.defaultdict(list)
    for name in os.listdir(probe_dir):
        if not name.endswith(".npy") or name.startswith("."):
            continue
        fields = name[:-4].split("_")
        try:
            layer = int(fields[0][1:])
            start = int(fields[2][1:])
        except (IndexError, ValueError):
            continue
        try:
            arr = np.load(os.path.join(probe_dir, name))
        except Exception:
            continue
        by_layer[layer].append((start, arr))

    out = {}
    for layer, entries in by_layer.items():
        full = [a for _, a in entries if a.shape[0] == num_tokens]
        if full:
            out[layer] = full[0]
            continue
        entries.sort(key=lambda item: item[0])
        merged = np.concatenate([a for _, a in entries]) if entries else None
        if merged is not None and merged.shape[0] >= num_tokens:
            out[layer] = merged[:num_tokens]
    return out


def _render(stats, tokenizer, top_k, min_count):
    out: dict[str, list] = {}
    for (domain, layer, n), table in sorted(stats.items()):
        rows = [
            {
                "ngram": tokenizer.decode(list(key)),
                "token_ids": list(key),
                "count": count,
                "avg_gate": round(total / count, 4),
            }
            for key, (count, total) in table.items()
            if count >= min_count
        ]
        rows.sort(key=lambda r: (-r["avg_gate"], -r["count"]))
        out[f"{domain}/engram{layer}/{n}gram"] = rows[:top_k]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
