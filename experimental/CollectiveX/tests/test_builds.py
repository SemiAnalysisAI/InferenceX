"""Source staging and backend-cache behavior without a cluster or vendor build."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import build


class BackendBuilds(unittest.TestCase):
    def test_cache_reuse_rebuild_and_failed_install_keep_the_ready_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installations = root / "installations"

            def install(cache):
                with installations.open("a") as stream:
                    stream.write("installed\n")
                (cache / "site").mkdir(parents=True)
                (cache / ".ready").touch()

            build.install_cached(root / "cache", "site", install)
            build.install_cached(root / "cache", "site", install)
            shutil.rmtree(root / "cache/site")
            build.install_cached(root / "cache", "site", install)

            def fail_install(cache):
                (cache / "site").mkdir(parents=True)
                raise RuntimeError("install failed")

            with self.assertRaisesRegex(RuntimeError, "install failed"):
                build.install_cached(root / "incomplete", "site", fail_install)
            self.assertFalse(build.cache_ready(root / "incomplete", "site"))
            self.assertEqual(installations.read_text(), "installed\ninstalled\n")
            self.assertTrue((root / "cache/.ready").is_file())
            self.assertFalse((root / "incomplete/.ready").exists())

    def test_symlink_lock_is_rejected_before_installing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache.lock").symlink_to(root / "untouched")
            with self.assertRaisesRegex(RuntimeError, "cache lock is unsafe"):
                build.install_cached(root / "cache", "site", lambda cache: (root / "installed").touch())
            self.assertFalse((root / "installed").exists())
            self.assertFalse((root / "untouched").exists())

    def test_source_staging_reuses_the_pin_and_materializes_a_separate_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "upstream"
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            (repository / "payload").write_text("pinned source\n")
            subprocess.run(["git", "-C", str(repository), "add", "payload"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "-c", "user.name=Test", "-c",
                 "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
            ).strip()
            destination = root / "stage/experimental/CollectiveX/.collx_sources"
            with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(root / "gitconfig")}):
                staged = build.stage_source(destination, "uccl", str(repository), revision, (), root / "git.log")
                build.stage_source(destination, "uccl", str(root / "absent"), revision, (), root / "git.log")
            with mock.patch.dict(build.SOURCES, {"uccl-ep": ("uccl", str(repository), revision, ())}):
                build.materialize_source(root / "build", "uccl-ep", {"COLLX_BACKEND_SOURCE_ROOT": str(destination)})
            (root / "build/payload").write_text("build change\n")
            self.assertEqual((staged / "payload").read_text(), "pinned source\n")
            self.assertEqual((root / "build/payload").read_text(), "build change\n")


if __name__ == "__main__":
    unittest.main()
