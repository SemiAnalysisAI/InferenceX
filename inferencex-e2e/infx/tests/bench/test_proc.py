"""Child-process helpers: the echoed command line and start failures."""

from __future__ import annotations

from infx.bench import proc


def test_echo_masks_credentials_but_not_token_count_settings(capsys):
    env = {
        "HF_TOKEN": "hf_abcdef",
        "GITHUB_TOKEN": "ghp_123456",
        "OPENAI_API_KEY": "sk-proj-9",
        "CLIENT_SECRET": "shh-quiet",
        "DB_PASSWORD": "hunter22",
        "TOKENIZER": "/models/tok",
        "MAX_NUM_BATCHED_TOKENS": "8192",
    }
    argv = ["run", "--tokenizer", "/models/tok", "--max-tokens=8192"]
    argv += ["hf_abcdef", "ghp_123456", "api_key=sk-proj-9", "shh-quiet", "hunter22"]

    proc.echo(argv, env)

    assert capsys.readouterr().err == (
        "+ run --tokenizer /models/tok --max-tokens=8192 *** *** api_key=*** *** ***\n"
    )
