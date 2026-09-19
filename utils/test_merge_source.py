from infx.workflows import merge_source


def test_approved_older_run_survives_newer_diagnostic(monkeypatch):
    paths = []

    def api(path, **kwargs):
        paths.append(path)
        if "/comments?" in path:
            return [
                [
                    {
                        "id": 1,
                        "created_at": "2026-09-01",
                        "author_association": "MEMBER",
                        "body": "/use 10",
                    }
                ]
            ]
        if "/commits?" in path:
            return [[{"sha": "approved"}, {"sha": "newer"}]]
        if path.endswith("/runs/10"):
            return {
                "id": 10,
                "head_sha": "approved",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "failure",
                "path": ".github/workflows/run-sweep.yml",
            }
        if "/runs/10/artifacts" in path:
            return [{"artifacts": [{"name": "bmk_agentic_result", "expired": False}]}]
        raise AssertionError(f"unexpected API request: {path}")

    monkeypatch.setattr(merge_source, "gh", api)
    assert merge_source.select_source("owner/repo", 2, "branch") == 10
    assert not any("workflows/run-sweep.yml/runs?" in path for path in paths)
