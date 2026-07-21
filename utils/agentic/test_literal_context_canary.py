"""Tests for the literal-context serving canary."""

from utils.agentic.literal_context_canary import chat_token_count


class FakeTokenizer:
    """Return a configured chat-template result."""

    def __init__(self, result: object) -> None:
        self.result = result

    def apply_chat_template(self, *args: object, **kwargs: object) -> object:
        """Return the configured tokenization shape."""
        return self.result


def test_chat_token_count_accepts_token_list() -> None:
    """Older Transformers versions return input IDs directly."""
    assert chat_token_count(FakeTokenizer([1, 2, 3]), "prompt") == 3


def test_chat_token_count_accepts_mapping() -> None:
    """Newer Transformers versions return a BatchEncoding-like mapping."""
    tokenized = {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    assert chat_token_count(FakeTokenizer(tokenized), "prompt") == 4


def test_chat_token_count_accepts_batched_mapping() -> None:
    """A single batch dimension does not change the token count."""
    tokenized = {"input_ids": [[1, 2, 3, 4, 5]]}
    assert chat_token_count(FakeTokenizer(tokenized), "prompt") == 5
