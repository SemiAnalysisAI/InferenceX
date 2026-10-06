import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "verify_image", Path(__file__).with_name("verify-image.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ImageVerificationTest(unittest.TestCase):
    def setUp(self):
        self.base = {
            "RootFS": {"Layers": ["base", "gpu-agent"]},
            "Config": {
                "Entrypoint": ["/home/amd/tools/entrypoint.sh"],
                "Cmd": None,
                "WorkingDir": "/home/amd",
                "User": "root",
                "Env": ["PATH=/bin"],
            },
            "Os": "linux",
            "Architecture": "amd64",
        }
        self.candidate = copy.deepcopy(self.base)
        self.candidate["RootFS"]["Layers"].append("new-exporter")
        self.candidate["Config"]["Env"].append("AMD_GPU_GET_CACHE_TTL=0s")

    def test_accepts_binary_overlay(self):
        module.verify(self.base, self.candidate)

    def test_rejects_changed_base_layer(self):
        self.candidate["RootFS"]["Layers"][1] = "different-gpu-agent"
        with self.assertRaisesRegex(ValueError, "preserve every base layer"):
            module.verify(self.base, self.candidate)

    def test_rejects_changed_entrypoint(self):
        self.candidate["Config"]["Entrypoint"] = ["/bin/sh"]
        with self.assertRaisesRegex(ValueError, "Entrypoint"):
            module.verify(self.base, self.candidate)

    def test_rejects_cache_reuse(self):
        self.candidate["Config"]["Env"][-1] = "AMD_GPU_GET_CACHE_TTL=1s"
        with self.assertRaisesRegex(ValueError, "cache bypass"):
            module.verify(self.base, self.candidate)


if __name__ == "__main__":
    unittest.main()
