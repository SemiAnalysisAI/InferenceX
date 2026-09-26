import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

from infx.evals.patches import (
    patch_swebench_agent as agent_patch,
    patch_swebench_scoring as scoring_patch,
)

PATCH_DIR = Path(__file__).resolve().parents[3] / "infx/evals/patches"


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


def test_agent_patch_closes_inherited_stdin(tmp_path):
    target = tmp_path / "swerex_modal.py"
    target.write_text(
        """import json
import subprocess

class Environment:
    def execute(self, action):
        command = action.get("command", "") if isinstance(action, dict) else action
        try:
            result = subprocess.run(
                command,
                shell=True,
                timeout=0.2,
                stdout=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "output": "timed out"}
        return {"returncode": result.returncode, "output": result.stdout}

environment = Environment()
commands = ["cat", "printf 'pipe-ok\\\\n' | cat"]
print(json.dumps([environment.execute({"command": command}) for command in commands]))
"""
    )

    assert agent_patch._patch_swerex_environment(str(target))
    patched = target.read_text()
    assert agent_patch._patch_swerex_environment(str(target))
    assert target.read_text() == patched

    process = subprocess.Popen(
        [sys.executable, str(target)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        assert process.stdin is not None
        process.stdin.close()

    assert process.stdout is not None
    assert process.stderr is not None
    output = process.stdout.read()
    errors = process.stderr.read()
    process.stdout.close()
    process.stderr.close()
    assert not errors
    assert json.loads(output) == [
        {"returncode": 0, "output": ""},
        {"returncode": 0, "output": "pipe-ok\n"},
    ]


def test_scoring_patch_is_atomic_and_idempotent(tmp_path):
    target = tmp_path / "run_evaluation_modal.py"
    target.write_text("prefix\ncpu=4,\nsuffix\n")

    assert not scoring_patch.patch(str(target), "2")
    assert target.read_text() == "prefix\ncpu=4,\nsuffix\n"

    target.write_text(
        """events = []
log_dir = "logs"

class Runner:
    def __init__(self, cpu):
        events.append(["cpu", cpu])
        self.sandbox = self

    def terminate(self):
        events.append(["terminated"])

def report(log_dir, errored):
    events.append([log_dir, errored])

def run_instance(fail):
    runner = Runner(
        cpu=4,
    )
    try:
        if fail:
            raise RuntimeError("failed")
        return "completed"
    except RuntimeError:
        report(
            log_dir=log_dir,
            errored=True,
        )


def run_instances_modal():
    return run_instance(False), run_instance(True)

outcomes = run_instances_modal()
"""
    )
    for _ in range(2):
        assert scoring_patch.patch(str(target), "2")
        result = runpy.run_path(str(target))
        assert result["outcomes"] == ("completed", None)
        assert result["events"] == [
            ["cpu", 2],
            ["terminated"],
            ["cpu", 2],
            ["logs", True],
            ["terminated"],
        ]


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
