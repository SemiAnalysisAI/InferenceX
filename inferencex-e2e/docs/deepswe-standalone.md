# Run DeepSWE Standalone

Run the DeepSWE coding-agent benchmark against an already-running OpenAI-compatible endpoint,
with the configuration the Kimi Vendor Verifier does not publish.

[DeepSWE](https://github.com/datacurve-ai/deep-swe) is 113 coding tasks, each solved by an agent
inside a sandbox and scored by the task's own tests. The
[Kimi Vendor Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier) runs it through the
[Pier](https://pypi.org/project/datacurve-pier/) platform with the kimi-code CLI as the agent and
states only "Kimi-code 0.29.0 or later". Everything else that decides a score (agent version and
settings, Pier job settings, timeouts, network policy, retries) is pinned below, as validated
against a Kimi K3.1-Pro deployment.

This is a standalone recipe, not an eval suite of `python3 -m infx.bench eval`, and no config
row selects it. Every trial is a Docker container, plus an egress proxy and a separate verifier
container, while InferenceX evals run inside the serving container of a Slurm job with no container
daemon. The repository's earlier SWE-bench Lite eval (#1947) put each instance's shell in a remote
Modal sandbox and was removed in #3675; DeepSWE differs in that the agent itself runs inside the
sandbox, so the sandbox must reach the endpoint. Run it from any host with Docker that can reach
the endpoint under test.

## Configuration

| Component | Pin | Notes |
|---|---|---|
| Tasks | `datacurve-ai/deep-swe` at `0b9fabb` | 113 task directories; each `task.toml` names a prebuilt image `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`, so no task image is built locally |
| Pier | `datacurve-pier==0.3.1`, environment `docker` | `uv tool install datacurve-pier==0.3.1` |
| Agent | kimi-code `0.40.1` (`kimi-code-linux-x64` from `code.kimi.com`) | installed as root into each sandbox by [`utils/deepswe/kimi_code_pier.py`](../utils/deepswe/kimi_code_pier.py); telemetry and auto-update off |
| Agent settings | `[thinking] enabled = true, effort = "max", keep = "all"`; `[loop_control] max_steps_per_turn = 200000`; `[subagent] timeout_ms = 259200000` | written to `/root/.kimi-code/config.toml`; a pre-written migration marker keeps `max`, which kimi-code would otherwise rewrite to `high` on a fresh home |
| Tools | ripgrep `15.2.0`, static x86_64 build, checksum-verified | kimi-code's Glob and Grep tools download ripgrep at first use, which the sandbox egress policy blocks |
| Model wiring | `KIMI_MODEL_NAME`, `KIMI_MODEL_API_KEY`, `KIMI_MODEL_BASE_URL`, `KIMI_MODEL_PROVIDER_TYPE=openai`, `KIMI_MODEL_MAX_CONTEXT_SIZE` | the agent environment; the validated run used a 1,000,000-token context |
| Pier job | `n_attempts` and `n_concurrent_trials` are inputs; `agent_timeout_multiplier: 50`; `retry.max_retries: 3` with Pier's default exclusions (agent timeouts, missing or empty reward files, verifier errors are not retried) | `n_attempts 1` is pass@1; Moonshot's K3 table reports avg@3. The validated run used concurrency 4 |
| Network | agent and verifier run `no-network`; Pier's egress proxy allows only the endpoint host and `code.kimi.com` | tasks that install packages at run time fail by design |
| Verifier | separate container, 1,800 s; reward in `verifier/reward.json` | the task's agent timeout is 10,800 s before the multiplier |
| Guards | wait up to 30 minutes for `GET /models` to answer 200 from inside the sandbox, else exit 97 before the agent starts; any `failed to run prompt: provider.*` line in the transcript exits 97 | an unreachable or failing endpoint becomes an errored trial that Pier retries, never a zero score |

## Install

On an x86_64 host with a Docker daemon the current user can use, and egress to the task image
registry, `code.kimi.com` and GitHub releases:

```bash
uv tool install datacurve-pier==0.3.1
git clone https://github.com/datacurve-ai/deep-swe.git
git -C deep-swe checkout 0b9fabb
```

## Run

```bash
set -eo pipefail
export KIMI_MODEL_BASE_URL="http://127.0.0.1:8000/v1"
export KIMI_MODEL_API_KEY="EMPTY"
export KIMI_MODEL_NAME="moonshotai/Kimi-K3.1-Pro"
export KIMI_MODEL_MAX_CONTEXT_SIZE="1000000"
export DEEPSWE_TASKS_DIR="$PWD/deep-swe/tasks"
export DEEPSWE_TASKS="all"
export DEEPSWE_N_ATTEMPTS="1"
export DEEPSWE_CONCURRENCY="4"
export DEEPSWE_OUTPUT_DIR="$PWD/results/deepswe-$(date -u +%Y%m%dT%H%M%SZ)"
bash inferencex-e2e/utils/deepswe/run_deepswe.sh
```

`DEEPSWE_TASKS` is `all` or a comma-separated list of task directory names; a one-task run such as
`abs-module-cache-flags` proves the wiring in about twenty minutes. Every input is required. The
script writes the Pier job file to `$DEEPSWE_OUTPUT_DIR/pier.yaml` with permissions `0600` because
it carries the API key, runs `pier run`, then reduces the rewards.

## Outputs

- `jobs/<run>/<task>__<id>/` per trial: `verifier/reward.json` (`reward`, and the failing-to-passing
  and passing-to-passing test ratios), `agent/kimi-code.txt` (the stream-json transcript),
  `agent/kimi-sessions/` (kimi-code's wire log, whose response ids join each trial to the server's
  request logs), `trial.log` and `config.json`.
- `deepswe_report.json`: per-task rewards, errored trials, accuracy, pass@1 per attempt, and the
  job's timing and statistics.
- `results_deepswe_<timestamp>.json` in the `inferencex-eval-v1` shape under the task name
  `kimi_deepswe`: `exact_match,strict-match` is the accuracy (mean reward over all trials),
  `n-samples` holds `113 × n_attempts` original and the scored count effective, a `limit` records a
  run that selected fewer than the 113 tasks, and an `integration_error` marks any run with an
  errored trial or fewer scored trials than the job selected. `validate_scores.py` reads it
  against the `kimi_deepswe` entry in `thresholds.yaml`, which is 0.0 like every vendor suite: the
  score is diagnostic, and only an integration error fails validation.

## Cost

The validated pass of all 113 tasks at concurrency 4 took 12 hours 10 minutes and ended with 113
trials completed, 0 errored and 0 retries. The median trial ran about 23 minutes and the longest
about an hour. Each trial reserves 2 CPUs, 8 GB of memory and 20 GB of storage; task images are
about 4.3 GB each, so a full pass needs several hundred gigabytes of Docker storage.

## Known pitfalls

- **Effort silently downgraded.** On a fresh home, kimi-code rewrites `effort = "max"` to `"high"`
  through a one-time migration. The adapter pre-writes the migration marker; without it every
  request carries `reasoning_effort: high`.
- **Glob and Grep dead.** Without the baked ripgrep, the tools fail their bootstrap download
  inside the sandbox and the agent works without search.
- **Endpoint errors scored as zero.** kimi-code exits 0 after any `failed to run prompt:
  provider.*` line (`api_error` on a failing endpoint, `connection_error` on an unreachable one),
  and the verifier then scores the untouched repository as 0. The adapter's guards turn both
  into errored trials. Keep the serving stack frozen for the whole run; a redeploy mid-run is the
  usual cause.
- **Endpoint unreachable from the sandbox.** The agent runs inside a container behind Pier's egress
  proxy, which resolves the endpoint host itself. A name that resolves only through the host's
  `/etc/hosts` or a host-only resolver is unknown inside the containers, and the readiness wait
  spends its full 30 minutes before failing the trial. Check `GET /models` from a container, not
  from the host, before a run.
- **Stopping a run.** Stop Pier itself. Sending SIGTERM to the agent makes Pier retry the trial.
- **Forced tool choice with maximum effort.** Some SGLang builds stall on `tool_choice: required`
  together with `reasoning_effort: max`
  ([sgl-project/sglang#37430](https://github.com/sgl-project/sglang/issues/37430)); check the
  serving image before a long run.

## Reference numbers

The Kimi Vendor Verifier README publishes a per-provider table for Kimi K3.1-Pro; Moonshot's own
DeepSWE entry is 0.755. Single passes at temperature 1.0 spread by about three points from run to
run, so compare averages of several passes, or report the attempt count with any single number.
