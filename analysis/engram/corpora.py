"""Domain corpora for the Engram gate-activation scan.

The first four domains match the reference study (WikiText-2, DailyDialog,
GSM8K, MBPP) so those n-gram tables stay directly comparable. The rest widen
the code side beyond MBPP's Python-only word problems: CodeSearchNet supplies
real repository functions in six languages, and Rosetta Code adds a smaller
sample of languages CodeSearchNet does not carry.

Each domain lists candidate sources in preference order; the first that loads
wins, and a domain that loads nothing is skipped loudly rather than silently
shrinking the report.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REFERENCE_CHARS = 5_000_000
CODE_CHARS = 3_000_000
SMALL_CHARS = 1_000_000

CSN = "code-search-net/code_search_net"
ROSETTA = "christopher/rosetta-code"


def _csn(lang: str):
    return (CSN, {"name": lang, "split": "train"}, "whole_func_string", None)


def _rosetta(language_name: str):
    return (
        ROSETTA,
        {"split": "train"},
        "code",
        lambda row, want=language_name: row.get("language_name") == want,
    )


# domain -> (char budget, [(path, load kwargs, field, row filter), ...])
DOMAINS: dict[str, tuple[int, list]] = {
    # --- reference study domains ---
    "wiki": (REFERENCE_CHARS, [("wikitext", {"name": "wikitext-2-raw-v1", "split": "train"}, "text", None)]),
    "chat": (REFERENCE_CHARS, [("daily_dialog", {"split": "train"}, "dialog", None)]),
    "math": (REFERENCE_CHARS, [("openai/gsm8k", {"name": "main", "split": "train"}, ("question", "answer"), None)]),
    "code_mbpp": (REFERENCE_CHARS, [("google-research-datasets/mbpp", {"name": "full", "split": "train"}, ("text", "code"), None)]),
    # --- real repository code, six languages ---
    "code_python": (CODE_CHARS, [_csn("python")]),
    "code_javascript": (CODE_CHARS, [_csn("javascript")]),
    "code_java": (CODE_CHARS, [_csn("java")]),
    "code_go": (CODE_CHARS, [_csn("go")]),
    "code_php": (CODE_CHARS, [_csn("php")]),
    "code_ruby": (CODE_CHARS, [_csn("ruby")]),
    # --- languages CodeSearchNet lacks; Rosetta is small, so budgets are too ---
    "code_c": (SMALL_CHARS, [_rosetta("C")]),
    "code_cpp": (SMALL_CHARS, [_rosetta("C++")]),
    "code_rust": (SMALL_CHARS, [_rosetta("Rust")]),
    "code_kotlin": (SMALL_CHARS, [_rosetta("Kotlin")]),
    "code_haskell": (SMALL_CHARS, [_rosetta("Haskell")]),
    "code_typescript": (SMALL_CHARS, [_rosetta("TypeScript")]),
}


def _render(row, field) -> str:
    if isinstance(field, tuple):
        return "\n".join(str(row[f]) for f in field if row.get(f))
    value = row[field]
    if isinstance(value, list):  # DailyDialog turns
        return "\n".join(str(v).strip() for v in value)
    return str(value)


def _load_one(path, kwargs, field, where, budget):
    from datasets import load_dataset

    try:
        ds = load_dataset(path, **kwargs)
    except Exception:
        ds = load_dataset(path, trust_remote_code=True, **kwargs)
    chunks, total = [], 0
    for row in ds:
        if where is not None and not where(row):
            continue
        piece = _render(row, field).strip()
        if not piece:
            continue
        chunks.append(piece)
        total += len(piece) + 1
        if total >= budget:
            break
    return "\n".join(chunks)[:budget]


def build(domains: list[str] | None = None) -> dict[str, str]:
    """Return {domain: text}, skipping domains whose sources all fail."""
    wanted = domains or list(DOMAINS)
    out: dict[str, str] = {}
    for domain in wanted:
        budget, sources = DOMAINS[domain]
        for path, kwargs, field, where in sources:
            try:
                text = _load_one(path, kwargs, field, where, budget)
            except Exception:
                logger.exception("corpus %s: source %s failed", domain, path)
                continue
            if not text:
                logger.warning("corpus %s: source %s produced no rows", domain, path)
                continue
            out[domain] = text
            logger.info("corpus %s: %d chars from %s", domain, len(text), path)
            break
        else:
            logger.error("corpus %s: no usable source; SKIPPED", domain)
    if not out:
        raise RuntimeError("no domain corpora could be loaded")
    logger.info("corpora built: %s", {k: len(v) for k, v in sorted(out.items())})
    return out
