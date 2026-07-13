import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent / "patches"


def _load_patch_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_patch = _load_patch_module("patch_swebench_agent")
scoring_patch = _load_patch_module("patch_swebench_scoring")


def test_agent_patch_is_atomic_and_idempotent(tmp_path):
    target = tmp_path / "dependency.py"
    original = "alpha\nbeta\n"
    target.write_text(original)

    assert not agent_patch._patch(
        str(target),
        [("alpha", "patched-alpha", ""), ("missing", "patched-missing", "")],
        "test",
    )
    assert target.read_text() == original

    replacements = [("alpha", "patched-alpha", ""), ("beta", "patched-beta", "")]
    assert agent_patch._patch(str(target), replacements, "test")
    patched = target.read_text()
    assert patched == "patched-alpha\npatched-beta\n"
    assert agent_patch._patch(str(target), replacements, "test")
    assert target.read_text() == patched


def test_scoring_patch_is_atomic_and_idempotent(tmp_path):
    target = tmp_path / "run_evaluation_modal.py"
    target.write_text("prefix\ncpu=4,\nsuffix\n")

    assert not scoring_patch.patch(str(target), "2")
    assert target.read_text() == "prefix\ncpu=4,\nsuffix\n"

    source = "cpu=4,\n" + scoring_patch.LIFECYCLE_ANCHOR
    target.write_text(source)
    assert scoring_patch.patch(str(target), "2")
    patched = target.read_text()
    assert "cpu=2," in patched
    assert scoring_patch.CPU_MARKER in patched
    assert scoring_patch.LIFECYCLE_MARKER in patched
    assert scoring_patch.patch(str(target), "2")
    assert target.read_text() == patched


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_scoring_patch_rejects_invalid_cpu(monkeypatch, value):
    monkeypatch.setenv("SWEBENCH_EVAL_SANDBOX_CPU", value)
    with pytest.raises(ValueError):
        scoring_patch._cpu_value()


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
