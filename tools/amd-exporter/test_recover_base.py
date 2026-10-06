import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "recover_base", Path(__file__).with_name("recover-base.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RuntimeRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "opt/runtime/lib").mkdir(parents=True)
        self.library = self.root / "opt/runtime/lib/amdsmi.so"
        self.library.write_bytes(b"retained runtime library")
        self.library.chmod(0o755)
        (self.root / "opt/rocm").symlink_to("/opt/runtime")
        (self.root / "opt/runtime/lib/amdsmi").symlink_to("amdsmi.so")
        self.files = {
            "opt/rocm": {"type": "symlink", "target": "/opt/runtime"},
            "opt/rocm/lib/amdsmi": {"type": "symlink", "target": "amdsmi.so"},
            "opt/rocm/lib/amdsmi.so": {
                "type": "file",
                "mode": 0o755,
                "sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
            },
        }

    def test_accepts_retained_bytes_and_container_absolute_symlink(self):
        module.verify_rootfs(self.root, self.files)

    def test_rejects_changed_runtime_library(self):
        self.library.write_bytes(b"different version")
        with self.assertRaisesRegex(ValueError, "Runtime bytes changed"):
            module.verify_rootfs(self.root, self.files)

    def test_rejects_changed_symlink(self):
        link = self.root / "opt/runtime/lib/amdsmi"
        link.unlink()
        link.symlink_to("other.so")
        with self.assertRaisesRegex(ValueError, "Runtime symlink changed"):
            module.verify_rootfs(self.root, self.files)

    def test_rejects_lost_executable_bit(self):
        self.library.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "executable bits changed"):
            module.verify_rootfs(self.root, self.files)

    def test_rejects_path_outside_container(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            module.container_path(self.root, "../outside")

    def test_original_runtime_config_can_be_restored(self):
        fixture = json.loads(Path(__file__).with_name("base-runtime.json").read_text())
        config = fixture["config"]
        actual = copy.deepcopy(config)
        actual.update({"Cmd": None, "User": ""})
        module.verify_config(config, actual)
        changes = module.config_changes(config)
        self.assertIn('ENTRYPOINT ["/home/amd/tools/entrypoint.sh"]', changes)
        self.assertIn('ENV GPUAGENT_EVENTS_DISABLE="1"', changes)
        actual["Env"] = ["PATH=/bin"]
        with self.assertRaisesRegex(ValueError, "Env"):
            module.verify_config(config, actual)

    def test_rejects_another_squash_in_source_receipt(self):
        fixture = {
            "official_archive_sha256": "archive",
            "official_image_config_digest": "config",
            "recovery": {
                "squash_sha256": "expected",
                "run_id": 10,
                "repository": "owner/repo",
            },
        }
        source = {
            "squash": {"sha256": "expected"},
            "archive": {"sha256": "archive"},
            "image": {"config_digest": "config"},
            "preparation": {"run_id": "10", "repository": "owner/repo"},
        }
        module.verify_provenance(source, fixture)
        source["squash"]["sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "provenance"):
            module.verify_provenance(source, fixture)


if __name__ == "__main__":
    unittest.main()
