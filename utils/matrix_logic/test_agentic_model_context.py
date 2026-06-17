import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_CONTEXT_SETTING = re.compile(
    r"""^\s*["']?(?:
        --max-model-len
        |--context-length
        |max-model-len:
        |context-length:
        |MAX_MODEL_LEN\s*[:=]
    )""",
    re.MULTILINE | re.VERBOSE,
)


def test_agentic_configs_do_not_set_model_context() -> None:
    config_paths = [
        *sorted((REPO_ROOT / "benchmarks/single_node/agentic").glob("*.sh")),
        *sorted(
            (
                REPO_ROOT
                / "benchmarks/multi_node/srt-slurm-recipes"
            ).glob("**/agentic/*.yaml")
        ),
    ]

    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in config_paths
        if SERVER_CONTEXT_SETTING.search(path.read_text())
    ]

    assert not violations, (
        "Agentic configs must use the model's native context limit: "
        + ", ".join(violations)
    )


def test_agentic_benchmark_unsets_inherited_model_context() -> None:
    command = """
        set -euo pipefail
        export IS_AGENTIC=1
        export MAX_MODEL_LEN=12345
        source benchmarks/benchmark_lib.sh
        test -z "${MAX_MODEL_LEN+x}"
    """

    subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        check=True,
    )
