import json
from pathlib import Path

from prompts import (
    INFINITEBENCH_SUFFIX,
    build_exact_prompt_ids,
    load_contexts,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize, **kwargs):
        rendered = "<B>" + messages[0]["content"] + "<A>"
        return self.encode(rendered) if tokenize else rendered


class BoundaryJumpTokenizer(CharacterTokenizer):
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def apply_chat_template(self, messages, tokenize, **kwargs):
        content = messages[0]["content"]
        context = content[len(self.prefix) : -len(self.suffix)]
        rendered = "<B>" + content + "<A>"
        token_ids = self.encode(rendered)
        if context and len(context) % 2 == 0 and not context.endswith(" "):
            token_ids.append(0)
        return token_ids if tokenize else rendered


class TailTrimTokenizer:
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def encode(self, text, add_special_tokens=False):
        return [1] * ((len(text) + 1) // 2)

    def decode(self, token_ids, skip_special_tokens=True):
        return "x" * (2 * len(token_ids))

    def apply_chat_template(self, messages, tokenize, **kwargs):
        content = messages[0]["content"]
        context = content[len(self.prefix) : -len(self.suffix)]
        wrapper_tokens = len("<B>" + self.prefix + self.suffix + "<A>")
        context_tokens = sum(not character.isspace() for character in context)
        token_ids = [0] * (wrapper_tokens + context_tokens)
        return token_ids if tokenize else "<rendered>"


def test_build_exact_prompt_ids_has_no_padding():
    tokenizer = CharacterTokenizer()
    prefix = "prefix:"
    suffix = ":suffix"
    wrapper_tokens = len("<B>" + prefix + suffix + "<A>")
    target = wrapper_tokens + 37
    prompt_ids, metadata = build_exact_prompt_ids(
        "abcdefghijklmnopqrstuvwxyz" * 10,
        tokenizer,
        target_tokens=target,
        prefix=prefix,
        suffix=suffix,
    )
    assert len(prompt_ids) == target
    assert metadata["used_context_tokens"] == 37


def test_build_exact_prompt_ids_repairs_boundary_length_jump():
    prefix = "prefix:"
    suffix = ":suffix"
    tokenizer = BoundaryJumpTokenizer(prefix, suffix)
    wrapper_tokens = len("<B>" + prefix + suffix + "<A>")
    target = wrapper_tokens + 10
    prompt_ids, metadata = build_exact_prompt_ids(
        "abcdefghijklmnopqrstuvwxyz" * 10,
        tokenizer,
        target_tokens=target,
        prefix=prefix,
        suffix=suffix,
    )
    assert len(prompt_ids) == target
    assert metadata["boundary_adjustment"] == "space"
    assert metadata["boundary_adjustment_characters"] == 1
    assert metadata["context_tail_trimmed_characters"] == 0


def test_build_exact_prompt_ids_can_trim_decoded_context_tail():
    prefix = "prefix:"
    suffix = ":suffix"
    tokenizer = TailTrimTokenizer(prefix, suffix)
    wrapper_tokens = len("<B>" + prefix + suffix + "<A>")
    target = wrapper_tokens + 9
    prompt_ids, metadata = build_exact_prompt_ids(
        "abcdefghijklmnopqrstuvwxyz" * 10,
        tokenizer,
        target_tokens=target,
        prefix=prefix,
        suffix=suffix,
    )
    assert len(prompt_ids) == target
    assert metadata["boundary_adjustment"] == "none"
    assert metadata["context_tail_trimmed_characters"] == 1


def test_contexts_repeat_deterministically(tmp_path: Path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"context": value}) for value in ("alpha", "beta")
        ),
        encoding="utf-8",
    )
    assert load_contexts(dataset, 5) == [
        "alpha",
        "beta",
        "alpha",
        "beta",
        "alpha",
    ]


def test_prompt_keeps_cann_256_word_literal():
    assert "in 256 words" in INFINITEBENCH_SUFFIX
