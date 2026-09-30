"""Echoed command execution hides secrets from logs without altering the command."""

import os

from infx.launch import proc


def test_echo_redacts_secret_values_but_child_gets_real_argv(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_fixturesecret123")
    env = {**os.environ, "MODAL_TOKEN_SECRET": "as-modal-fixture"}
    result = proc.run(
        ["echo", "--hf", "hf_fixturesecret123", "Authorization=as-modal-fixture", "keep"],
        env=env, capture=True,
    )
    echoed = capsys.readouterr().err
    assert echoed == "+ echo --hf *** Authorization=*** keep\n"
    assert result.stdout == "--hf hf_fixturesecret123 Authorization=as-modal-fixture keep\n"
