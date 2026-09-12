"""Default repository paths used by sweep planning."""

MASTER_CONFIGS = ["configs/amd-master.yaml", "configs/nvidia-master.yaml"]
RUNNER_CONFIG = "configs/runners.yaml"
# Historical revisions and their diagnostics retain the legacy generator path.
GENERATE_SWEEPS_PY_SCRIPT = "utils/matrix_logic/generate_sweep_configs.py"
