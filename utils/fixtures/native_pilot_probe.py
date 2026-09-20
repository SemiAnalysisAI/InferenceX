"""Inspect an installed native prepared job without running a scheduler or client.

Only physical node discovery and the final scheduler process boundary are
simulated. Schema loading, mounts, topology, worker commands, benchmark context,
and srun command construction use the installed runtime's real implementations.
"""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from srtctl.benchmarks.base import get_runner
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import cluster_config_scope, load_config
from srtctl.core.prepared import load_prepared
from srtctl.core.runtime import Nodes, RuntimeContext


def inspect(prepared):
    prepared = Path(prepared).resolve()
    load_prepared(prepared, verify_runtime=True)
    profile = yaml.safe_load((prepared / "profile.yaml").read_text())
    nodes = Nodes(
        head="simulated-h100",
        bench="simulated-h100",
        infra="simulated-h100",
        worker=("simulated-h100",),
    )
    with (
        cluster_config_scope(profile),
        patch.object(Nodes, "from_slurm", return_value=nodes),
        patch("srtctl.core.runtime.get_hostname_ip", return_value="127.0.0.1"),
        patch("srtctl.core.slurm.get_hostname_ip", return_value="127.0.0.1"),
        patch(
            "srtctl.cli.mixins.benchmark_stage.get_hostname_ip",
            return_value="127.0.0.1",
        ),
    ):
        config = load_config(prepared / "config.yaml", frozen=True)
        runtime = RuntimeContext.from_config(
            config, "offline-inspection", log_dir_base=prepared.parent / "inspection"
        )
        orchestrator = SweepOrchestrator(config, runtime)
        processes = orchestrator.backend_processes
        server = config.backend.build_worker_command(
            processes[0], processes, runtime, frontend_type=config.frontend.type
        )
        runner = get_runner(config.benchmark.type)
        captured = []

        def intercept_scheduler(argv, **_kwargs):
            captured.append(argv)
            return SimpleNamespace(poll=lambda: 0, returncode=0)

        with patch(
            "srtctl.core.slurm.subprocess.Popen", side_effect=intercept_scheduler
        ):
            exit_code = orchestrator._run_benchmark_script(
                runner, runtime.log_dir / "inspection.log", threading.Event()
            )
        return {
            "server_argv": server,
            "client_argv": runner.build_command(config, runtime),
            "worker_env": config.backend.get_environment_for_mode("agg"),
            "client_env": {
                **orchestrator._get_benchmark_env(runner),
                **runner.get_environment(config, runtime),
            },
            "client_env_unset": runner.get_environment_unset(config, runtime),
            "client_cwd": runner.get_working_directory(config, runtime),
            "mounts": {
                str(host): str(container)
                for host, container in runtime.container_mounts.items()
            },
            "srun_argv": captured,
            "exit_code": exit_code,
            "processes": [
                {"node": p.node, "mode": p.endpoint_mode, "gpus": sorted(p.gpu_indices)}
                for p in processes
            ],
            "time_limit": config.slurm.time_limit,
            "frontend_type": config.frontend.type,
            "discovery_services": config.services,
            "profiling": config.profiling.enabled,
        }


if __name__ == "__main__":
    print(json.dumps(inspect(sys.argv[1]), default=str))
