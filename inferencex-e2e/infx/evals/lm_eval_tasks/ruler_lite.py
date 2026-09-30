"""Long-context retrieval and tracking probes in the style of RULER.

RULER (https://arxiv.org/abs/2404.06654) as bundled with lm-eval builds 500
samples per task and length, downloads its essay haystack and NLTK data at
runtime, and names its metrics after the sequence lengths, which the InferenceX
collector cannot read. This module generates a small, deterministic subset of
three RULER tasks instead:

- multikey: four needles with distinct keys; retrieve the value for one key.
- multivalue: four values share one key; retrieve all of them.
- vt: variable tracking; a four-hop assignment chain among distractor chains;
  name every variable that holds the chained value.

Each prompt has its own seeded haystack of plain sentences, so prompts share no
prefix beyond the instruction and prefix caching cannot shortcut prefill.
Lengths are measured with the served model's tokenizer when it can be loaded
(MODEL_PATH, then MODEL), otherwise estimated at four characters per token; the
doc records which. A response scores the fraction of expected strings it
contains, case-insensitively, as RULER's string_match_all does.
"""

from __future__ import annotations

import os
import random
import string
from functools import cache
from typing import Any

import datasets

LENGTHS = (65536, 131072)
SAMPLES = 50
SEED = 20260930
TASKS = ("multikey", "multivalue", "vt")
# Room for the question and the chat template around the haystack.
RESERVE = 512

_SUBJECTS = ("The river", "A lantern", "The committee", "An old map", "The orchard", "A courier",
             "The glacier", "A notebook", "The harbor", "A violinist", "The archive", "A comet")
_VERBS = ("drifted past", "was left near", "quietly described", "waited beside", "circled",
          "was painted over", "leaned against", "outlasted", "reflected", "was traded for")
_OBJECTS = ("the northern bridge", "a stack of letters", "the evening market", "an empty field",
            "the copper roof", "a broken clock", "the western gate", "a pale mountain",
            "the quiet library", "a row of lanterns", "the salt flats", "an unfinished song")
_KEY_WORDS = ("amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor",
              "indigo", "juniper", "kestrel", "lagoon", "marble", "nimbus", "onyx", "prairie",
              "quartz", "raven", "sierra", "tundra", "umber", "velvet", "willow", "zephyr")


@cache
def _tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    for source in (os.environ.get("MODEL_PATH"), os.environ.get("MODEL")):
        if not source:
            continue
        try:
            return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        except Exception:  # noqa: BLE001 - fall back to the estimate below
            continue
    return None


def _count(text: str) -> tuple[int, str]:
    tokenizer = _tokenizer()
    if tokenizer is None:
        return len(text) // 4, "chars/4"
    return len(tokenizer.encode(text, add_special_tokens=False)), "tokenizer"


@cache
def _tokens_per_sentence() -> float:
    sample = " ".join(_sentence(random.Random(i)) for i in range(400))
    return max(1.0, _count(sample)[0] / 400)


def _sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}."


def _number(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 10_000_000))


def _key(rng: random.Random) -> str:
    return "-".join(rng.sample(_KEY_WORDS, 2))


def _variable(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=5))


def _needles(task: str, rng: random.Random) -> tuple[list[str], str, list[str]]:
    """Return inserted lines, the question, and the expected answer strings."""
    if task == "multikey":
        keys = rng.sample(_KEY_WORDS, 8)
        pairs = [(f"{keys[2 * i]}-{keys[2 * i + 1]}", _number(rng)) for i in range(4)]
        lines = [f"One of the special magic numbers for {k} is: {v}." for k, v in pairs]
        key, value = pairs[rng.randrange(4)]
        question = f"What is the special magic number for {key} mentioned in the provided text?"
        return lines, question, [value]
    if task == "multivalue":
        key = _key(rng)
        values = [_number(rng) for _ in range(4)]
        lines = [f"One of the special magic numbers for {key} is: {v}." for v in values]
        question = f"What are all the special magic numbers for {key} mentioned in the provided text?"
        return lines, question, values
    names = rng.sample([_variable(rng) for _ in range(64)], 15)
    chains = [names[i * 5:(i + 1) * 5] for i in range(3)]
    lines, answer, value = [], [], None
    for index, chain in enumerate(chains):
        start = _number(rng)
        if index == 0:
            value, answer = start, chain
        lines.append(f"VAR {chain[0]} = {start}")
        lines += [f"VAR {chain[j]} = VAR {chain[j - 1]}" for j in range(1, len(chain))]
    question = (
        f"Find all variables that are assigned the value {value} in the text above, "
        "directly or through other variables. List every one of them."
    )
    return lines, question, answer


def _doc(task: str, length: int, index: int) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:{task}:{length}:{index}")
    lines, question, expected = _needles(task, rng)
    intro = (
        "Some special information is hidden within the following text. Make sure to "
        "memorize it. I will quiz you about it afterwards.\n"
    )
    outro = f"\n{question} Answer with only the requested values."
    fixed, source = _count(intro + outro + " ".join(lines))
    budget = length - RESERVE - fixed
    sentences = [_sentence(rng) for _ in range(max(1, int(budget / _tokens_per_sentence())))]
    # Spread the needles over the haystack, preserving VT chain order.
    positions = sorted(rng.sample(range(len(sentences) + 1), len(lines)))
    if task != "vt":
        rng.shuffle(lines)
    for offset, (position, line) in enumerate(zip(positions, lines, strict=True)):
        sentences.insert(position + offset, line)
    text = intro + " ".join(sentences) + outro
    tokens, source = _count(text)
    return {
        "task": task,
        "target_length": length,
        "input_tokens": tokens,
        "length_source": source,
        "input": text,
        "outputs": expected,
    }


def dataset(**_: Any) -> dict[str, datasets.Dataset]:
    docs = [
        _doc(task, length, index)
        for length in LENGTHS
        for task in TASKS
        for index in range(SAMPLES)
    ]
    return {"test": datasets.Dataset.from_list(docs, split=datasets.Split.TEST)}


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    response = (results[0] if results else "").lower()
    hits = sum(1 for expected in doc["outputs"] if expected.lower() in response)
    return {"exact_match": hits / len(doc["outputs"])}
