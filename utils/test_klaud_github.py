import json
import os
import shlex
import subprocess
import sys

import pytest

from infx.klaud import __main__ as klaud
from infx.klaud import claims, github


@pytest.mark.parametrize("current_head,expected", [("ours", True), ("other", False)])
def test_claim_conflict_checks_the_actual_owner(monkeypatch, current_head, expected):
    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/")).partition("?")[0]
        method = args[args.index("--method") + 1]
        if endpoint.endswith("git/commits/base") and method == "GET":
            response = {"tree": {"sha": "tree"}}
        elif endpoint.endswith("git/commits") and method == "POST":
            request = json.loads(kwargs["input"])
            assert request == {"message": '{"owner": 42}', "tree": "tree", "parents": ["base"]}
            response = {"sha": "ours"}
        elif endpoint.endswith("git/refs") and method == "POST":
            raise subprocess.CalledProcessError(1, args, stderr="reference already exists")
        elif endpoint.endswith("git/matching-refs/heads/claim") and method == "GET":
            response = [[{"ref": "refs/heads/claim", "object": {"sha": current_head}}]]
        else:
            raise AssertionError((method, endpoint))
        return subprocess.CompletedProcess(args, 0, json.dumps(response), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert claims.create("example/project", "claim", {"owner": 42}, "base") is expected


def test_delete_accepts_empty_response(monkeypatch):
    def run(args, **kwargs):
        assert args[args.index("--method") + 1] == "DELETE"
        assert json.loads(kwargs["input"]) == {}
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.write("example/project", "git/refs/heads/claim", "DELETE") == {}


@pytest.mark.parametrize("failure,reason", [
    ("command", "State unavailable or invalid; inspect GitHub before retrying"),
    ("json", "State unavailable or invalid; inspect GitHub before retrying"),
    ("shape", "GitHub listing returned an unexpected shape"),
    ("incomplete", "Incomplete GitHub listing"),
])
def test_recovery_errors_do_not_publish_raw_api_data(tmp_path, monkeypatch, capfd, failure, reason):
    responses = {
        "json": "private",
        "shape": '[{"artifacts": ["private"], "total_count": 1}]',
        "incomplete": '[{"artifacts": [], "total_count": 1, "detail": "private"}]',
    }
    command = ("printf '%s' private >&2\nexit 1" if failure == "command"
               else f"printf '%s' {shlex.quote(responses[failure])}")
    executable = tmp_path / "gh"
    executable.write_text(f"#!/bin/sh\n{command}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setattr(sys, "argv", ["klaud", "recover"])
    assert klaud.main() == 1
    captured = capfd.readouterr()
    assert captured.out == f"::error::Klaud: {reason}.\n"
    assert captured.err == ""
