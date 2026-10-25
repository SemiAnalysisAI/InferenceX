import http.server
import io
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
import unittest
from pathlib import Path
from unittest import mock

import fetch

FIX = Path(__file__).resolve().parent / "fixtures"


class ParseSourceTest(unittest.TestCase):
    def test_run_url(self):
        src = fetch.parse_source("https://github.com/vngcloud/InferenceX/actions/runs/39558375380/job/1")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "vngcloud/InferenceX", "39558375380"))
        self.assertEqual(src.link, "https://github.com/vngcloud/InferenceX/actions/runs/39558375380")

    def test_bare_id_uses_default_repo(self):
        src = fetch.parse_source("123", default_repo="o/r")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "o/r", "123"))

    def test_dir(self):
        src = fetch.parse_source(str(FIX / "gh_fixed"))
        self.assertEqual(src.kind, "dir")
        self.assertEqual(src.path, (FIX / "gh_fixed").resolve())

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            fetch.parse_source("/no/such/dir")


class LocalConfigsTest(unittest.TestCase):
    def configs(self, name):
        return fetch.local_configs(fetch.parse_source(str(FIX / name)))

    def test_gh_layout_pairs_on_rf(self):
        [cfg] = self.configs("gh_fixed")
        self.assertEqual(cfg.rf, "glm53_c108")
        self.assertEqual(cfg.agg["conc"], 108)
        self.assertIn(b"--concurrency 108", cfg.files.read("benchmark_command.txt"))
        self.assertIsNone(cfg.files.read("missing.txt"))

    def test_hand_named_layout_pairs_the_lone_aggregate_with_the_lone_root(self):
        [cfg] = self.configs("search_local")
        self.assertEqual(cfg.agg["state"], "ok")
        self.assertIsNotNone(cfg.files.read("aiperf_artifacts/search_history.json"))

    def test_raw_aiperf_dir_is_a_root_without_aggregate(self):
        [cfg] = self.configs("raw_aiperf")
        self.assertIsNone(cfg.agg)
        self.assertIsNotNone(cfg.files.read("profile_export_aiperf.json"))
        self.assertIn("ccu16", cfg.rf)

    def test_aggregate_only(self):
        [cfg] = self.configs("aggonly")
        self.assertIsNone(cfg.files)
        self.assertEqual(cfg.agg["conc"], 96)

    def test_empty_dir_gives_a_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs, notes = fetch.discover(fetch.parse_source(tmp), Path(tmp) / ".cache", 100)
        self.assertEqual(configs, [])
        self.assertIn("no aggregate JSON or aiperf export", notes[0])


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


class _Handler(http.server.BaseHTTPRequestHandler):
    payloads: dict = {}
    ranges = True
    expired = False
    full_after_probe = False  # answer 200 with the whole body to any Range request after the first probe

    def do_GET(self):
        if self.expired:
            self.send_error(403)
            return
        body = self.payloads[self.path]
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        if match and match[1] != "0" and self.full_after_probe:
            match = None
        if not (self.ranges and match):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except ConnectionError:
                pass
            return
        start, end = int(match[1]), min(int(match[2]), len(body) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(body[start:end + 1])

    def log_message(self, *args):
        pass


class RemoteTest(unittest.TestCase):
    def setUp(self):
        self.big = os.urandom(2_000_000)
        _Handler.ranges = True
        _Handler.expired = False
        _Handler.full_after_probe = False
        _Handler.payloads = {
            "/raw": _zip_bytes({"server.log": self.big, "aiperf_artifacts/profile_export_aiperf.json": b'{"x": 1}'}),
            "/agg": _zip_bytes({"RF.json": json.dumps({"infmax_model_prefix": "m", "conc": 8, "request_metrics": {}}).encode()}),
        }
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_range_read_fetches_a_fraction_of_the_zip(self):
        before = fetch.FETCHED["bytes"]
        zf = zipfile.ZipFile(fetch.RangeFile(self.base + "/raw"))
        self.assertEqual(zf.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertLess(fetch.FETCHED["bytes"] - before, len(_Handler.payloads["/raw"]) // 2)

    def test_zip_reader_caches_members(self):
        art = {"id": 7, "name": "agentic_RF", "size_in_bytes": 10, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name), 100, url_for=lambda repo, i: self.base + "/raw")
        self.assertEqual(reader.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertIsNone(reader.read("nope.txt"))
        cached = Path(self.tmp.name) / "7" / "files" / "aiperf_artifacts/profile_export_aiperf.json"
        self.assertTrue(cached.is_file())

    def test_no_range_support_and_big_zip_raises_too_large(self):
        _Handler.ranges = False
        art = {"id": 8, "name": "agentic_RF", "size_in_bytes": 500_000_000, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name), 100, url_for=lambda repo, i: self.base + "/raw")
        with self.assertRaises(fetch.TooLarge):
            reader.names()

    def test_remote_configs_pairs_artifacts_and_reports_failures(self):
        arts = [
            {"id": 1, "name": "bmk_agentic_RF", "size_in_bytes": 300, "expired": False, "workflow_run": {"id": 5}},
            {"id": 2, "name": "agentic_RF", "size_in_bytes": 2_000_000, "expired": False, "workflow_run": {"id": 5}},
            {"id": 3, "name": "bmk_agentic_OLD", "size_in_bytes": 300, "expired": True, "workflow_run": {"id": 5}},
        ]
        jobs = [{"name": "bench c8", "conclusion": "success"},
                {"name": "agentic / (/data/…-3... / p2.500 | name TP8    c32", "conclusion": "failure", "id": 77,
                 "html_url": "https://github.com/o/r/actions/runs/5/job/77"}]
        fetch.FAILED_JOBS.clear()

        def fake_gh(*args):
            path = args[1]
            if "/jobs" in path:
                return json.dumps({"total_count": 2, "jobs": jobs})
            return json.dumps({"total_count": 3, "artifacts": arts})

        urls = {1: self.base + "/agg", 2: self.base + "/raw"}
        with mock.patch.object(fetch, "_gh", side_effect=fake_gh), \
                mock.patch.object(fetch, "blob_url", side_effect=lambda repo, i: urls[i]):
            configs, notes = fetch.remote_configs(fetch.parse_source("5", "o/r"), Path(self.tmp.name), 100)
            [cfg] = configs
            self.assertEqual((cfg.rf, cfg.agg["conc"]), ("RF", 8))
            # raw zip is read lazily, so the blob_url patch must still be active here
            self.assertEqual(cfg.files.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertIn("job failed: p2.500 | name TP8 c32 (failure) https://github.com/o/r/actions/runs/5/job/77",
                      " ".join(notes))
        self.assertNotIn("/data/", " ".join(notes))
        self.assertEqual(fetch.FAILED_JOBS[0]["name"], "agentic / (/data/…-3... / p2.500 | name TP8    c32")
        self.assertTrue(any("bmk_agentic_OLD" in n and "expired" in n for n in notes))

    def test_zip_reader_size_comes_from_the_directory_without_reading(self):
        reader = fetch.ZipReader("o/r", {"id": 9, "name": "agentic_RF", "size_in_bytes": 1},
                                 Path(self.tmp.name) / "c", 100, url_for=lambda r, i: self.base + "/raw")
        before = fetch.FETCHED["bytes"]
        self.assertEqual(reader.size("server.log"), len(self.big))
        self.assertIsNone(reader.size("nope"))
        self.assertLess(fetch.FETCHED["bytes"] - before, len(self.big) // 2)
        self.assertFalse((reader.cache / "files").exists())
        self.assertEqual(json.loads((reader.cache / "names.json").read_text())["server.log"], len(self.big))
        reader.close()
        again = fetch.ZipReader("o/r", {"id": 9, "name": "agentic_RF", "size_in_bytes": 1},
                                Path(self.tmp.name) / "c", 100, url_for=lambda r, i: 1 / 0)
        self.assertEqual(again.size("server.log"), len(self.big))  # from the cache, no URL needed

    def test_later_read_answered_with_200_is_range_unsupported(self):
        rf = fetch.RangeFile(self.base + "/raw")
        _Handler.full_after_probe = True
        rf.seek(len(_Handler.payloads["/raw"]) - 100)
        with self.assertRaises(OSError):
            rf.read(10)

    def test_killed_write_leaves_no_truncated_cache_entry(self):
        art = {"id": 9, "name": "agentic_RF", "size_in_bytes": 1}
        cache = Path(self.tmp.name) / "c"
        reader = fetch.ZipReader("o/r", art, cache, 100, url_for=lambda r, i: self.base + "/raw")
        member = "aiperf_artifacts/profile_export_aiperf.json"
        real = Path.write_bytes

        def dying(path, data):
            real(path, data[:3])
            raise KeyboardInterrupt

        with mock.patch.object(Path, "write_bytes", dying), self.assertRaises(KeyboardInterrupt):
            reader.read(member)
        self.assertFalse((reader.cache / "files" / member).exists())
        stale = reader.cache / "files" / (member + ".tmp")
        self.assertEqual(reader.read(member), b'{"x": 1}')  # a leftover .tmp is not mistaken for a cache hit
        self.assertFalse(stale.exists())
        reader.close()

    def test_killed_names_write_leaves_no_truncated_listing(self):
        art = {"id": 9, "name": "agentic_RF", "size_in_bytes": 1}
        cache = Path(self.tmp.name) / "c"
        reader = fetch.ZipReader("o/r", art, cache, 100, url_for=lambda r, i: self.base + "/raw")
        real = Path.write_text

        def dying(path, data, *a, **k):
            real(path, data[:3])
            raise KeyboardInterrupt

        with mock.patch.object(Path, "write_text", dying), self.assertRaises(KeyboardInterrupt):
            reader.names()
        self.assertFalse((reader.cache / "names.json").exists())
        self.assertIn("server.log", reader.names())
        reader.close()

    def test_old_list_format_names_cache_is_rebuilt_with_sizes(self):
        art = {"id": 9, "name": "agentic_RF", "size_in_bytes": 1}
        cache = Path(self.tmp.name) / "c"
        (cache / "9").mkdir(parents=True)
        (cache / "9" / "names.json").write_text(json.dumps(["server.log"]))
        reader = fetch.ZipReader("o/r", art, cache, 100, url_for=lambda r, i: self.base + "/raw")
        self.assertEqual(reader.size("server.log"), len(self.big))
        reader.close()

    def test_read_after_blob_url_expiry_re_resolves(self):
        calls = []

        def url_for(repo, i):
            calls.append(i)
            _Handler.expired = False  # a fresh URL works again
            return self.base + "/raw"

        art = {"id": 9, "name": "agentic_RF", "size_in_bytes": 10, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name), 100, url_for=url_for)
        reader.read("aiperf_artifacts/profile_export_aiperf.json")
        _Handler.expired = True
        self.assertEqual(reader.read("server.log"), self.big)
        self.assertEqual(len(calls), 2)

    def test_malicious_member_names_are_not_read_or_cached(self):
        _Handler.payloads["/evil"] = _zip_bytes({"../../escape.txt": b"x", "/abs.txt": b"y"})
        art = {"id": 10, "name": "agentic_RF", "size_in_bytes": 10, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name) / "c", 100, url_for=lambda r, i: self.base + "/evil")
        self.assertIsNone(reader.read("../../escape.txt"))
        self.assertIsNone(reader.read("/abs.txt"))
        self.assertFalse((Path(self.tmp.name) / "escape.txt").exists())
        reader.close()


class MultiArtifactTest(unittest.TestCase):
    def test_lookalike_json_dirs_are_not_aggregates(self):
        agg = (FIX / "gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json").read_bytes()
        other = (FIX / "aggonly/bmk_agentic_gemma4_c96/gemma4_c96.json").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(FIX / "gh_fixed/agentic_glm53_c108", root / "agentic_RF108")
            for d, rf, data in (("bmk_agentic_RF108", "RF108", agg), ("bmk_agentic_RF120", "RF120", other),
                                ("power_audit_RF108", "RF108", agg), ("power_audit_RF120", "RF120", other),
                                ("results_bmk", "agg_bmk", agg), ("run-stats", "run_stats", other)):
                (root / d).mkdir()
                (root / d / f"{rf}.json").write_bytes(data)
            cfgs = fetch.local_configs(fetch.parse_source(str(root)))
        self.assertEqual([c.rf for c in cfgs], ["RF108", "RF120"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bmk_agentic_A").mkdir()
            (Path(tmp) / "hand.json").write_bytes(other)
            notes = []
            fetch.local_configs(fetch.parse_source(tmp), notes)
        self.assertIn("ignored 1 aggregate-looking file(s)", notes[0])
        self.assertIsNotNone(cfgs[0].files)
        self.assertIsNone(cfgs[1].files)


class FixRound1FetchTest(unittest.TestCase):
    def test_dir_reader_size(self):
        reader = fetch.DirReader(FIX / "gh_fixed/agentic_glm53_c108")
        self.assertEqual(reader.size("benchmark_command.txt"), len(reader.read("benchmark_command.txt")))
        self.assertIsNone(reader.size("nope"))

    def test_job_tail_is_short_and_readable(self):
        name = "agentic / (/data/…-3... / p2.500 | name fp8 hardware-hcm-8x_02 sgl TP8    c1"
        self.assertEqual(fetch.job_tail(name), "p2.500 | name fp8 hardware-hcm-8x_02 sgl TP8 c1")
        self.assertTrue(fetch.job_tail("x" * 200).startswith("…"))

    def test_skipped_folders_noted_for_multi_folder_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(FIX / "raw_aiperf/glm5.1_public_ccu16_dur900", root / "conc16")
            (root / "conc1" / "logs").mkdir(parents=True)
            (root / "power_audit_x").mkdir()
            notes = []
            cfgs = fetch.local_configs(fetch.parse_source(tmp), notes)
        self.assertEqual(len(cfgs), 1)
        self.assertEqual(notes, ["skipped 1 folder(s) with no results: conc1"])

    def test_single_run_folder_has_no_skipped_note(self):
        notes = []
        fetch.local_configs(fetch.parse_source(str(FIX / "gh_fixed/agentic_glm53_c108")), notes)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
