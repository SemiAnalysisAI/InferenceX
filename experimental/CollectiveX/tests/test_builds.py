"""Source staging and backend-cache behavior without a cluster or vendor build."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


class BackendBuilds(unittest.TestCase):
    def _shell(self, directory, script, **environment):
        """Run the real sourced helpers with isolated logs, git settings, and cache paths."""
        return subprocess.run(
            ["bash", "-c", 'source "$RUNTIME/common.sh"; source "$RUNTIME/build_common.sh"\n' + script],
            text=True, capture_output=True,
            env={
                **os.environ,
                "RUNTIME": str(RUNTIME),
                "TEST_ROOT": str(directory),
                "COLLX_JOB_ROOT": str(directory / "job"),
                "GIT_CONFIG_GLOBAL": str(directory / "gitconfig"),
                **environment,
            },
        )

    def test_cache_reuse_rebuild_and_failed_install_keep_the_ready_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin"
            binary.mkdir()
            # The external lock operation is stubbed; cache creation and readiness are real.
            flock = binary / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n")
            flock.chmod(0o755)
            result = self._shell(root, '''
install() {
  printf 'installed\\n' >> "$TEST_ROOT/installations"
  mkdir -p "$1/site"
  touch "$1/.ready"
}
install_backend_cache test "$TEST_ROOT/cache" site install || exit 10
install_backend_cache test "$TEST_ROOT/cache" site install || exit 11
rm -r "$TEST_ROOT/cache/site"
install_backend_cache test "$TEST_ROOT/cache" site install || exit 12
fail_install() { mkdir -p "$1/site"; return 17; }
install_backend_cache test "$TEST_ROOT/incomplete" site fail_install && exit 13
backend_cache_ready "$TEST_ROOT/incomplete" site && exit 14
exit 0
''', PATH=f"{binary}:{os.environ['PATH']}")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "installations").read_text(), "installed\ninstalled\n")
            self.assertTrue((root / "cache/.ready").is_file())
            self.assertFalse((root / "incomplete/.ready").exists())

    def test_symlink_lock_is_rejected_before_installing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache.lock").symlink_to(root / "untouched")
            result = self._shell(root, '''
install() { touch "$TEST_ROOT/installed"; }
install_backend_cache test "$TEST_ROOT/cache" site install
''')
            self.assertEqual(result.returncode, 1)
            self.assertIn("test cache lock is unsafe", result.stderr)
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
            result = self._shell(root, '''
COLLX_UCCL_REPO="$TEST_ROOT/upstream"
COLLX_UCCL_COMMIT="$REVISION"
collx_prepare_uccl_source "$TEST_ROOT/stage" || exit 10
COLLX_UCCL_REPO="$TEST_ROOT/no-such-repository"
collx_prepare_uccl_source "$TEST_ROOT/stage" || exit 11
COLLX_BACKEND_SOURCE_ROOT="$TEST_ROOT/stage/experimental/CollectiveX/.collx_sources"
collx_materialize_uccl_source "$TEST_ROOT/build" || exit 12
printf 'build change\\n' > "$TEST_ROOT/build/payload"
''', REVISION=revision)
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = root / "stage/experimental/CollectiveX/.collx_sources" / f"uccl-{revision}"
            self.assertEqual((staged / "payload").read_text(), "pinned source\n")
            self.assertEqual((root / "build/payload").read_text(), "build change\n")


if __name__ == "__main__":
    unittest.main()
