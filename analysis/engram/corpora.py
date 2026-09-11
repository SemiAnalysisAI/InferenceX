"""Domain corpora for the Engram gate-activation scan.

The first four domains stand in for the reference study (WikiText-2,
UltraChat for open-domain dialogue, GSM8K, MBPP) so those n-gram tables stay directly comparable. CodeSearchNet
widens the code side beyond MBPP's Python-only word problems with real
repository functions in six languages. Natural-language corpora are English
and Chinese only.

Each domain lists candidate sources in preference order; the first that loads
wins, and a domain that loads nothing is skipped loudly rather than silently
shrinking the report.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REFERENCE_CHARS = 5_000_000
CODE_CHARS = 3_000_000

CSN = "code-search-net/code_search_net"


def _csn(lang: str):
    return (CSN, {"name": lang, "split": "train"}, "whole_func_string", None)


# domain -> (char budget, [(path, load kwargs, field, row filter), ...])
DOMAINS: dict[str, tuple[int, list]] = {
    # --- reference study domains ---
    "wiki": (
        REFERENCE_CHARS,
        [
            ("Salesforce/wikitext", {"name": "wikitext-2-raw-v1", "split": "train"}, "text", None),
            ("wikimedia/wikipedia", {"name": "20231101.en", "split": "train"}, "text", None),
        ],
    ),
    "chat": (
        REFERENCE_CHARS,
        [
            ("HuggingFaceH4/ultrachat_200k", {"split": "train_sft"}, "messages", None),
            ("allenai/soda", {"split": "train"}, "dialogue", None),
        ],
    ),
    "math": (REFERENCE_CHARS, [("openai/gsm8k", {"name": "main", "split": "train"}, ("question", "answer"), None)]),
    "code_mbpp": (REFERENCE_CHARS, [("google-research-datasets/mbpp", {"name": "full", "split": "train"}, ("text", "code"), None)]),
    # --- real repository code, six languages ---
    "code_python": (CODE_CHARS, [_csn("python")]),
    "code_javascript": (CODE_CHARS, [_csn("javascript")]),
    "code_java": (CODE_CHARS, [_csn("java")]),
    "code_go": (CODE_CHARS, [_csn("go")]),
    "code_php": (CODE_CHARS, [_csn("php")]),
    "code_ruby": (CODE_CHARS, [_csn("ruby")]),
}


def _render(row, field) -> str:
    if isinstance(field, tuple):
        return "\n".join(str(row[f]) for f in field if row.get(f))
    value = row[field]
    if isinstance(value, list):  # dialogue turns: plain strings or {role, content}
        turns = []
        for v in value:
            text = v.get("content", "") if isinstance(v, dict) else v
            text = str(text).strip()
            if text:
                turns.append(text)
        return "\n".join(turns)
    return str(value)


def _load_one(path, kwargs, field, where, budget):
    from datasets import load_dataset

    try:
        ds = load_dataset(path, streaming=True, **kwargs)
    except Exception:
        ds = load_dataset(path, **kwargs)
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


# Large streaming sources, added so a multi-hour scan has something to read:
# the reference-study corpora are small (MBPP is 96k chars total, GSM8K 3.9M,
# WikiText-2 ~11M) and are exhausted in minutes.
STREAM_DOMAINS: dict[str, list] = {
    "web": [
        ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT", "split": "train"}, "text", None),
        ("allenai/c4", {"name": "en", "split": "train"}, "text", None),
    ],
    "wiki_full": [
        ("wikimedia/wikipedia", {"name": "20231101.en", "split": "train"}, "text", None),
    ],
    "math_web": [
        ("open-web-math/open-web-math", {"split": "train"}, "text", None),
        ("EleutherAI/proof-pile-2", {"name": "open-web-math", "split": "train"}, "text", None),
    ],
    # --- Chinese, mirroring the English streaming domains ---
    "wiki_zh": [
        ("wikimedia/wikipedia", {"name": "20231101.zh", "split": "train"}, "text", None),
    ],
    "web_zh": [
        ("HuggingFaceFW/fineweb-2", {"name": "cmn_Hani", "split": "train"}, "text", None),
        ("opencsg/chinese-fineweb-edu", {"split": "train"}, "text", None),
    ],
    "chat_zh": [
        ("BelleGroup/train_1M_CN", {"split": "train"}, ("instruction", "output"), None),
        ("YeungNLP/firefly-train-1.1M", {"split": "train"}, ("input", "target"), None),
    ],
}


def iter_texts(domain: str):
    """Yield text pieces for a domain, unbounded, from the first source that loads.

    Streamed rather than materialized: a five-hour scan reads far more than
    fits in a char budget, and the budget in DOMAINS only ever existed to keep
    the eager build cheap.
    """
    import itertools

    from datasets import load_dataset

    sources = STREAM_DOMAINS.get(domain) or DOMAINS[domain][1]
    for path, kwargs, field, where in sources:
        try:
            rows = iter(load_dataset(path, streaming=True, **kwargs))
            first = next(rows)  # fail fast on gated / script-based / renamed
        except Exception:
            logger.exception("corpus %s: source %s unusable", domain, path)
            continue
        logger.info("corpus %s: streaming from %s", domain, path)
        for row in itertools.chain([first], rows):
            if where is not None and not where(row):
                continue
            try:
                piece = _render(row, field).strip()
            except Exception:
                continue
            if piece:
                yield piece
        logger.info("corpus %s: source %s exhausted", domain, path)
        return
    logger.error("corpus %s: no usable source; SKIPPED", domain)


# Scan order, not alphabetical: a scan can be cut short by the Slurm
# allocation limit, so the domains that carry the argument go first and the
# large streaming corpora -- which only ever add depth -- go last.
PRIORITY = [
    # the reference-study four
    "wiki", "chat", "math", "code_mbpp",
    # Chinese
    "wiki_zh", "chat_zh", "web_zh",
    # code, most-used languages first
    "code_python", "code_javascript", "code_java", "code_go", "code_php", "code_ruby",
    # depth
    "web", "wiki_full", "math_web",
]


def all_domains() -> list[str]:
    known = list(DOMAINS) + list(STREAM_DOMAINS)
    missing = [d for d in known if d not in PRIORITY]
    assert not missing, f"PRIORITY is missing {missing}"
    return [d for d in PRIORITY if d in known]
