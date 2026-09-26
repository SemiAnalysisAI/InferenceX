"""Default repository paths used by sweep planning."""

from pathlib import Path

MASTER_CONFIGS = ["configs/amd-master.yaml", "configs/nvidia-master.yaml"]
RUNNER_CONFIG = "configs/runners.yaml"
# The matrix generator module. Revisions that predate it shipped a script, which
# historical append-only planning still runs from their own snapshot.
GENERATOR_MODULE = "infx.matrix.generate"
GENERATOR_MODULE_PATH = "infx/matrix/generate.py"
LEGACY_GENERATOR_SCRIPT = "utils/matrix_logic/generate_sweep_configs.py"


def repository_root() -> Path:
    source_root = Path(__file__).resolve().parent.parent
    return source_root if (source_root / "configs").is_dir() else Path.cwd()
