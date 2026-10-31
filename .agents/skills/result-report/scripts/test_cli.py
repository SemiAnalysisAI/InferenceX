import contextlib
import http.server
import io
import json
import os
import stat
import tempfile
import threading
import zipfile
import unittest
from pathlib import Path
from unittest import mock

import result_report

FIX = Path(__file__).resolve().parent / "fixtures"


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = result_report.main([str(a) for a in argv])
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_writes_report_data_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(FIX / "gh_fixed", FIX / "search_local", "--out", tmp, "--html", "--title", "demo")
            self.assertEqual(code, 0, out)
            md = (Path(tmp) / "report.md").read_text()
            data = json.loads((Path(tmp) / "data.json").read_text())
            page = (Path(tmp) / "report.html").read_text()
        self.assertIn("# demo", md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertEqual(len(data["arms"]), 2)
        self.assertIn("<svg", page)
        self.assertIn("report.md", out)

    def test_findings_replace_the_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            findings = Path(tmp) / "findings.md"
            findings.write_text("- GLM c108 reaches 734 output tok/s.\n")
            code, out = run(FIX / "gh_fixed", "--out", tmp, "--findings", findings)
            md = (Path(tmp) / "report.md").read_text()
        self.assertEqual(code, 0, out)
        self.assertIn("- GLM c108 reaches 734 output tok/s.", md)
        self.assertNotIn("KEY_FINDINGS", md)

    def test_bad_input_and_empty_dir(self):
        self.assertEqual(run("/no/such/dir")[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertEqual(run(empty, "--out", Path(tmp) / "o")[0], 1)

    def test_unknown_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(FIX / "gh_fixed", "--out", tmp, "--baseline", "nope")[0], 2)


class FetchedBytesTest(unittest.TestCase):
    def test_cumulative_bytes_survive_a_cached_rerun(self):
        import fetch
        real = fetch.discover

        def fake(src, cache, mb):
            fetch.FETCHED["bytes"] += 3_000_000
            return real(src, cache, mb)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fetch, "discover", fake):
            code, out = run(FIX / "gh_fixed", "--out", tmp)
            self.assertIn("downloaded 3.00 MB this run (total 3.00 MB)", out)
            self.assertEqual(json.loads((Path(tmp) / ".cache/fetched.json").read_text()), {"bytes": 3_000_000})
            with mock.patch.object(fetch, "discover", real):
                code, out = run(FIX / "gh_fixed", "--out", tmp)
            data = json.loads((Path(tmp) / "data.json").read_text())
            md = (Path(tmp) / "report.md").read_text()
        self.assertEqual(data["bytes_fetched"], 3_000_000)
        self.assertIn("downloaded 0.00 MB this run (total 3.00 MB)", out)
        self.assertIn("downloaded 3.00 MB", md)

    def test_corrupt_fetched_json_is_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".cache").mkdir()
            (Path(tmp) / ".cache/fetched.json").write_text("{not json")
            code, out = run(FIX / "gh_fixed", "--out", tmp)
        self.assertEqual(code, 0, out)
        self.assertIn("total 0.00 MB", out)

    def test_null_bytes_in_fetched_json_is_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".cache").mkdir()
            (Path(tmp) / ".cache/fetched.json").write_text('{"bytes": null}')
            code, out = run(FIX / "gh_fixed", "--out", tmp)
        self.assertEqual(code, 0, out)

    def test_local_folder_creates_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(FIX / "gh_fixed", "--out", tmp)
            self.assertFalse((Path(tmp) / ".cache").exists())


class ErrorPathsTest(unittest.TestCase):
    def test_missing_findings_file_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(FIX / "gh_fixed", "--out", tmp, "--findings", Path(tmp) / "nope.md")
        self.assertEqual(code, 2)
        self.assertIn("ERROR", out)
        self.assertNotIn("Traceback", out)

    def test_missing_gh_is_exit_4_but_vanished_input_file_is_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(result_report.fetch, "discover", side_effect=FileNotFoundError(2, "No such file", "gh")):
                self.assertEqual(run(FIX / "gh_fixed", "--out", tmp)[0], 4)
            with mock.patch.object(result_report.fetch, "discover", side_effect=FileNotFoundError(2, "No such file", "/x/y.json")):
                code, out = run(FIX / "gh_fixed", "--out", tmp)
        self.assertEqual(code, 2)
        self.assertNotIn("gh CLI", out)


FAKE_GH = """#!/usr/bin/env python3
import json, os, sys, zipfile
a = sys.argv[1:]
d = os.environ["FAKE_GH_DIR"]
if a[:2] == ["auth", "token"]:
    print("tok")
elif a[0] == "api" and "/jobs" in a[1]:
    print(json.dumps({"total_count": 0, "jobs": []}))
elif a[0] == "api" and "/artifacts" in a[1]:
    arts = [{"id": i, "name": n, "expired": False, "workflow_run": {"id": 5},
             "size_in_bytes": os.path.getsize(os.path.join(d, n + ".zip"))}
            for i, n in ((1, "bmk_agentic_RF"), (2, "agentic_RF"))]
    print(json.dumps({"total_count": 2, "artifacts": arts}))
elif a[:2] == ["run", "download"]:
    name, dest = a[a.index("-n") + 1], a[a.index("-D") + 1]
    zipfile.ZipFile(os.path.join(d, name + ".zip")).extractall(dest)
else:
    sys.exit("fake gh: unexpected " + " ".join(a))
"""


class _NoRangeBlob(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # the artifact zip endpoint
        self.server.hits.append(self.path)  # redirect to a blob that ignores Range
        if self.path.startswith("/repos/"):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/blob")
            self.end_headers()
            return
        body = b"not a zip, and no Range support"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FullZipFallbackTest(unittest.TestCase):
    """End to end: fake gh on PATH, blob server without Range support."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        bin_dir, self.art_dir = root / "bin", root / "zips"
        bin_dir.mkdir()
        self.art_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
        raw = FIX / "gh_fixed/agentic_glm53_c108"
        with zipfile.ZipFile(self.art_dir / "agentic_RF.zip", "w") as zf:
            for f in raw.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(raw).as_posix())
        with zipfile.ZipFile(self.art_dir / "bmk_agentic_RF.zip", "w") as zf:
            zf.write(FIX / "gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json", "glm53_c108.json")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoRangeBlob)
        self.server.hits = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_GH_DIR": str(self.art_dir),
               "GITHUB_API_URL": f"http://127.0.0.1:{self.server.server_port}"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = root / "out"

    def test_range_unsupported_under_budget_downloads_the_full_zip(self):
        code, out = run("5", "--repo", "o/r", "--out", self.out)
        self.assertEqual(code, 0, out)
        md = (self.out / "report.md").read_text()
        self.assertIn("734", md)  # GLM c108 output tok/s from the fixture
        data = json.loads((self.out / "data.json").read_text())
        self.assertGreater(data["bytes_fetched"], 0)
        self.assertIn("/blob", self.server.hits)  # the Range probe really hit the no-Range server
        self.assertEqual(data["points"][0]["conc"], 108)

    def test_range_unsupported_over_budget_exits_3_without_traceback(self):
        code, out = run("5", "--repo", "o/r", "--out", self.out, "--max-full-zip-mb", "0")
        self.assertEqual(code, 3, out)
        self.assertIn("--max-full-zip-mb", out)
        self.assertNotIn("Traceback", out)


if __name__ == "__main__":
    unittest.main()
