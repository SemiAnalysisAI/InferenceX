import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import prompts


class CountingTokenizer:
    def __init__(self):
        self.truncated_inputs = []
        self.decode_calls = 0

    def encode(self, text, add_special_tokens=False, truncation=False,
               max_length=None):
        if truncation:
            self.truncated_inputs.append(text)
        tokens = list(text)
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls += 1
        return "".join(ids)


def test_build_infinitebench_prompts_caches_repeated_contexts(monkeypatch):
    tokenizer = CountingTokenizer()
    contexts = ["alpha", "beta", "alpha", "alpha", "beta"]

    monkeypatch.setattr(
        prompts,
        "load_infinitebench_contexts",
        lambda dataset_path, task, num_prompts: contexts,
    )

    built = prompts.build_infinitebench_prompts(
        dataset_path=None,
        task="infinitebench",
        input_len=512,
        output_len=16,
        num_prompts=len(contexts),
        tokenizer=tokenizer,
        use_chat_template=False,
        dsv4=False,
    )

    assert len(built) == len(contexts)
    assert tokenizer.truncated_inputs == ["alpha", "beta"]
    assert tokenizer.decode_calls == 2
    assert built[0] == built[2] == built[3]
    assert built[1] == built[4]
