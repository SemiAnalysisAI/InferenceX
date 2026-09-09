import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

PATCH = Path(__file__).parent / "patches/srt-slurm-custom-metrics.patch"
TARGET = "src/srtctl/cli/mixins/benchmark_stage.py"
ORIGINAL = '''class Stage:
    def get_env(self, runner):
        env = {}
        # Add AIPerf-specific env vars for AIPerf-driven benchmarks only
        if isinstance(runner, AIPerfBenchmarkRunner):
            env.update(self._get_aiperf_server_metrics_env())
            if self.config.benchmark.aiperf_package:
                env["AIPERF_PACKAGE"] = self.config.benchmark.aiperf_package

        return env
'''


class AIPerfRunner:
    name = "AIPerf"


@pytest.mark.parametrize(
    "runner,required,expected",
    [
        (AIPerfRunner(), "", True),
        (SimpleNamespace(name="Custom"), "vllm:prompt_tokens_cached_by_source", True),
        (SimpleNamespace(name="Custom"), "", False),
        (SimpleNamespace(name="Other"), "vllm:", False),
    ],
)
def test_custom_metrics_opt_in(tmp_path, runner, required, expected):
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL)
    subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    symbols = runpy.run_path(
        str(target), init_globals={"AIPerfBenchmarkRunner": AIPerfRunner}
    )
    stage = symbols["Stage"]()
    stage.config = SimpleNamespace(
        benchmark=SimpleNamespace(
            env={"AIPERF_REQUIRED_SERVER_METRIC_PREFIX": required},
            aiperf_package=None,
        )
    )
    urls = "http://prefill:7500/metrics,http://decode:7501/metrics"
    stage._get_aiperf_server_metrics_env = lambda: {"AIPERF_SERVER_METRICS_URLS": urls}
    assert stage.get_env(runner) == (
        {"AIPERF_SERVER_METRICS_URLS": urls} if expected else {}
    )


def test_custom_metrics_patch_rejects_unexpected_source(tmp_path):
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    target.write_text("# Different upstream implementation\n")
    result = subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, capture_output=True)
    assert result.returncode != 0
    assert target.read_text() == "# Different upstream implementation\n"
