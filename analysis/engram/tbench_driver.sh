#!/usr/bin/env bash
# Terminal-Bench 4.0 against a locally-served DeepSeek-V4.1-Flash.
#
# Two things are unknown from the published docs and are therefore discovered
# here rather than guessed:
#
#   1. Whether Docker works on a compute node. Harbor's local backend needs it
#      ("you will first need to install Docker and have it running"), and these
#      nodes run under Slurm + pyxis/enroot where nested Docker is usually
#      unavailable. The repo's own SWE-bench path offers --modal as an
#      alternative to local Docker, which hints it is not available here.
#   2. The flags for a self-hosted OpenAI-compatible endpoint, task subsetting
#      and the agent timeout. None appear in the release notes, the README or
#      the running-tbench docs, so `harbor run --help` is dumped to the log and
#      read from there.
#
# Phase 1 (this script) establishes feasibility cheaply. Phase 2 runs the
# benchmark only if phase 1 says it can work.
set -eo pipefail

source "$(dirname "$0")/../../benchmarks/benchmark_lib.sh"
check_env_vars MODEL TP RESULT_DIR
export GPU_COUNT="$TP"
mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/tbench_probe.txt"

say() { echo "$@" | tee -a "$REPORT"; }

say "=== Terminal-Bench 4.0 feasibility probe ==="
say "--- container runtime"
for tool in docker podman enroot nvidia-container-cli; do
    if command -v "$tool" >/dev/null 2>&1; then
        say "FOUND $tool: $($tool --version 2>&1 | head -1)"
    else
        say "absent $tool"
    fi
done

say "--- docker daemon reachable?"
if command -v docker >/dev/null 2>&1; then
    if timeout 30 docker info >/dev/null 2>&1; then
        say "docker daemon OK"
        DOCKER_OK=1
    else
        say "docker present but daemon unreachable: $(timeout 30 docker info 2>&1 | tail -3 | tr '\n' ' ')"
        DOCKER_OK=0
    fi
else
    DOCKER_OK=0
fi
say "DOCKER_OK=$DOCKER_OK"

say "--- cgroup / privilege (task containers need to start processes)"
say "uid=$(id -u) $(id -un 2>/dev/null || true)"
say "unshare available: $(command -v unshare >/dev/null && echo yes || echo no)"

say "--- harbor install"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v harbor >/dev/null 2>&1; then
    python3 -m pip install -q --no-input --break-system-packages 'harbor' 2>&1 | tail -3 || true
fi
if command -v harbor >/dev/null 2>&1 || python3 -c "import harbor" 2>/dev/null; then
    say "harbor importable/CLI present"
    say "--- harbor run --help (authoritative flag list)"
    (harbor run --help 2>&1 || python3 -m harbor run --help 2>&1) | tee -a "$REPORT" | head -120
    say "--- harbor --version"
    (harbor --version 2>&1 || true) | tee -a "$REPORT"
else
    say "harbor NOT installable in this image"
fi

say "=== verdict ==="
if [[ "$DOCKER_OK" == 1 ]]; then
    say "LOCAL EXECUTION LOOKS POSSIBLE -- proceed to phase 2 on this node"
else
    say "LOCAL EXECUTION BLOCKED: no usable Docker daemon on the compute node."
    say "Terminal-Bench needs a container per task; Harbor's alternatives are"
    say "--env modal and --env daytona, both of which are hosted services that"
    say "would need credentials and would send task environments off-cluster."
    say "The model endpoint is not the obstacle -- vLLM serves fine here."
fi
say "probe written to $REPORT"
