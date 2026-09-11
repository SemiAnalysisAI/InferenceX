"""The four domain corpora from the Engram gate-activation analysis.

Same datasets as the reference study (WikiText-2, DailyDialog, GSM8K, MBPP) so
the n-gram tables are directly comparable; only the model differs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_CHARS = 5_000_000

# (domain, hf id, config, split, field-or-None-for-joiner)
SOURCES = (
    ("wiki", "wikitext", "wikitext-2-raw-v1", "train", "text"),
    ("chat", "daily_dialog", None, "train", "dialog"),
    ("math", "openai/gsm8k", "main", "train", ("question", "answer")),
    ("code", "google-research-datasets/mbpp", "full", "train", ("text", "code")),
)


def _render(row, field) -> str:
    if isinstance(field, tuple):
        return "\n".join(str(row[f]) for f in field if row.get(f))
    value = row[field]
    if isinstance(value, list):  # DailyDialog turns
        return "\n".join(str(v).strip() for v in value)
    return str(value)


def build(max_chars: int = MAX_CHARS) -> dict[str, str]:
    """Return {domain: text}. A domain that fails to load is skipped loudly."""
    from datasets import load_dataset

    out: dict[str, str] = {}
    for domain, path, config, split, field in SOURCES:
        try:
            kwargs = {"split": split}
            if config:
                kwargs["name"] = config
            try:
                ds = load_dataset(path, **kwargs)
            except Exception:
                ds = load_dataset(path, trust_remote_code=True, **kwargs)
            chunks, total = [], 0
            for row in ds:
                piece = _render(row, field).strip()
                if not piece:
                    continue
                chunks.append(piece)
                total += len(piece) + 1
                if total >= max_chars:
                    break
            out[domain] = "\n".join(chunks)[:max_chars]
            logger.info("corpus %s: %d chars from %s", domain, len(out[domain]), path)
        except Exception:
            logger.exception("corpus %s: FAILED to load %s; skipping", domain, path)
    if not out:
        raise RuntimeError("no domain corpora could be loaded")
    return out
