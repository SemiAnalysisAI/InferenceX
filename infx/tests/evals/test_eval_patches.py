import json
import runpy
import sys
import types
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parents[3] / "infx/evals/patches"


def test_lm_eval_sitecustomize_hooks(monkeypatch):
    lm_eval = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    completions = types.ModuleType("lm_eval.models.openai_completions")
    api_models = types.ModuleType("lm_eval.models.api_models")

    class LocalChatCompletion:
        pass

    class JsonChatStr(str):
        pass

    class TemplateAPI:
        tokenizer_backend = "none"
        tokenized_requests = False

    completions.LocalChatCompletion = LocalChatCompletion
    api_models.JsonChatStr = JsonChatStr
    api_models.TemplateAPI = TemplateAPI
    models.api_models = api_models
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.openai_completions", completions)
    monkeypatch.setitem(sys.modules, "lm_eval.models.api_models", api_models)

    runpy.run_path(str(PATCH_DIR / "lm_eval_sitecustomize.py"))

    parsed = LocalChatCompletion.parse_generations(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "", "reasoning_content": "reason"},
                    }
                ]
            }
        ]
    )
    assert parsed == ["reason"]
    rendered = TemplateAPI().apply_chat_template([{"role": "user", "content": "hi"}])
    assert isinstance(rendered, JsonChatStr)
    assert json.loads(rendered) == [{"role": "user", "content": "hi"}]
