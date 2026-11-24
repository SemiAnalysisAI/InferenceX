"""The kimi-code CLI as a Pier installed agent against an OpenAI-compatible endpoint.

Pier loads it with ``--agent-import-path kimi_code_pier:KimiCode`` (this directory on
``PYTHONPATH``). The endpoint is wired through the ``KIMI_MODEL_*`` variables of the agent
environment; everything else is pinned here. See ``docs/deepswe-standalone.md``.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, ClassVar

from pier.agents.installed.base import BaseInstalledAgent, EnvVar, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.models.agent.install import AgentInstallSpec, InstallStep

if TYPE_CHECKING:
    from pier.environments.base import BaseEnvironment
    from pier.models.agent.context import AgentContext
    from pier.models.agent.network import NetworkAllowlist

KIMI_CODE_VERSION = "0.40.1"
RIPGREP_VERSION = "15.2.0"
TASK_AGENT_TIMEOUT_SEC = 10_800
AGENT_TIMEOUT_MULTIPLIER = 50
AGENT_HOME = "/root/.kimi-code"
TRANSCRIPT = "/logs/agent/kimi-code.txt"

CONFIG_TOML = (
    '[thinking]\\nenabled = true\\neffort = "max"\\nkeep = "all"\\n\\n'
    "[loop_control]\\nmax_steps_per_turn = 200000\\n\\n"
    "[subagent]\\ntimeout_ms = 259200000\\n"
)
# kimi-code runs a one-time migration on a fresh home that rewrites effort "max" to "high";
# a pre-written marker keeps "max", which is then sent verbatim as reasoning_effort.
MIGRATION_MARKER = '{"thinking-effort-max-to-high": "2026-11-03T00:00:00.000Z"}\\n'

# Exit 97 makes Pier record an errored trial, which its retry policy reruns, instead of letting
# the verifier score an untouched repository as 0.
ENDPOINT_WAIT = (
    'c=0; for i in $(seq 1 90); do c=$(curl -s -m 10 -o /dev/null -w "%{http_code}" '
    '-H "Authorization: Bearer $KIMI_MODEL_API_KEY" "$KIMI_MODEL_BASE_URL/models" 2>/dev/null); '
    '[ "$c" = 200 ] && break; echo "endpoint not ready ($c), waiting ($i/90)" >&2; sleep 20; done; '
    'if [ "$c" != 200 ]; then echo "ENDPOINT_UNREACHABLE: $KIMI_MODEL_BASE_URL/models answered '
    '$c from the sandbox" >&2; exit 97; fi;'
)
# kimi-code prints "failed to run prompt: provider.<kind>" (api_error, connection_error, ...)
# and exits 0 whenever the endpoint fails it.
PROVIDER_ERROR_GUARD = (
    f'if grep -q "failed to run prompt: provider\\." {TRANSCRIPT}; then '
    f'echo "PROVIDER_ERROR: $(grep -m1 "failed to run prompt: provider\\." {TRANSCRIPT} '
    '| cut -c1-200)" >&2; exit 97; fi'
)


class KimiCode(BaseInstalledAgent):
    """kimi-code in headless mode, installed into each task sandbox."""

    ENV_VARS: ClassVar[list[EnvVar]] = [
        EnvVar("model_name_kw", env="KIMI_MODEL_NAME", env_fallback="KIMI_MODEL_NAME"),
        EnvVar("model_api_key_kw", env="KIMI_MODEL_API_KEY", env_fallback="KIMI_MODEL_API_KEY"),
        EnvVar("model_base_url_kw", env="KIMI_MODEL_BASE_URL", env_fallback="KIMI_MODEL_BASE_URL"),
        EnvVar(
            "model_provider_type_kw",
            env="KIMI_MODEL_PROVIDER_TYPE",
            env_fallback="KIMI_MODEL_PROVIDER_TYPE",
            default="openai",
        ),
        EnvVar(
            "model_max_context_kw",
            env="KIMI_MODEL_MAX_CONTEXT_SIZE",
            type="int",
            env_fallback="KIMI_MODEL_MAX_CONTEXT_SIZE",
        ),
    ]

    @staticmethod
    def name() -> str:
        """The agent name Pier records."""
        return "kimi-code"

    def get_version_command(self) -> str | None:
        """The command that prints the installed agent version."""
        return "kimi --version"

    def install_spec(self) -> AgentInstallSpec:
        """Install the pinned binary, a static ripgrep and the thinking configuration as root."""
        binary = f"https://code.kimi.com/kimi-code/binaries/{KIMI_CODE_VERSION}/kimi-code-linux-x64"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run="apt-get update && apt-get install -y curl ca-certificates",
                ),
                InstallStep(
                    user="root",
                    run=(
                        f"mkdir -p {AGENT_HOME}/bin && "
                        f"curl -fsSL --retry 3 {binary} -o {AGENT_HOME}/bin/kimi && "
                        f"chmod +x {AGENT_HOME}/bin/kimi && "
                        f"ln -sf {AGENT_HOME}/bin/kimi /usr/local/bin/kimi"
                    ),
                ),
                # The Glob and Grep tools download ripgrep at first use, which the sandbox egress
                # policy blocks; a pinned static binary at the path kimi-code expects keeps them alive.
                InstallStep(user="root", run=_ripgrep_install()),
                InstallStep(
                    user="root",
                    run=(
                        f"mkdir -p {AGENT_HOME} && "
                        f"printf '{CONFIG_TOML}' > {AGENT_HOME}/config.toml && "
                        f"printf '{MIGRATION_MARKER}' > {AGENT_HOME}/migrations-effort.json && "
                        f"cat {AGENT_HOME}/config.toml {AGENT_HOME}/migrations-effort.json"
                    ),
                ),
                InstallStep(user="root", run=self.get_version_command()),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        """Only the endpoint and the kimi-code host are reachable from the sandbox."""
        return allowlist_from_urls(
            [self._get_env("KIMI_MODEL_BASE_URL")], default_domains=["code.kimi.com"]
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,  # noqa: ARG002  # Pier's agent protocol passes it
    ) -> None:
        """Run one task headlessly and keep the transcript and the wire log for the host."""
        escaped = shlex.quote(instruction)
        env = self.build_process_env(
            {"KIMI_DISABLE_TELEMETRY": "1", "KIMI_CODE_NO_AUTO_UPDATE": "1"}
        )
        await self.exec_as_agent(
            environment,
            command=(
                f"{ENDPOINT_WAIT} "
                f"kimi -p {escaped} --output-format stream-json "
                f"2>&1 </dev/null | stdbuf -oL tee {TRANSCRIPT}; "
                f"cp -r {AGENT_HOME}/sessions /logs/agent/kimi-sessions 2>/dev/null || true; "
                f"{PROVIDER_ERROR_GUARD}"
            ),
            env=env,
            timeout_sec=TASK_AGENT_TIMEOUT_SEC * AGENT_TIMEOUT_MULTIPLIER,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Token totals and step count from the stream-json transcript, when present."""
        transcript = self.logs_dir / "agent" / "kimi-code.txt"
        if not transcript.exists():
            return
        n_in = n_out = n_steps = 0
        for raw in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_steps += 1
            usage = record.get("usage") if isinstance(record, dict) else None
            if isinstance(usage, dict):
                n_in += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                n_out += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        if n_in:
            context.n_input_tokens = n_in
        if n_out:
            context.n_output_tokens = n_out
        if n_steps:
            context.n_agent_steps = n_steps


def _ripgrep_install() -> str:
    """The shell that installs a checksum-verified static ripgrep next to the agent binary."""
    asset = f"ripgrep-{RIPGREP_VERSION}-x86_64-unknown-linux-musl"
    url = (
        f"https://github.com/BurntSushi/ripgrep/releases/download/{RIPGREP_VERSION}/{asset}.tar.gz"
    )
    work = f"{AGENT_HOME}/rg-download"
    return (
        f"mkdir -p {AGENT_HOME}/bin {work} && cd {work} && "
        f"curl -fsSL --retry 3 -O {url} && curl -fsSL --retry 3 -O {url}.sha256 && "
        f"sha256sum -c {asset}.tar.gz.sha256 && tar -xzf {asset}.tar.gz && "
        f"mv {asset}/rg {AGENT_HOME}/bin/rg && chmod +x {AGENT_HOME}/bin/rg && "
        f"cd / && rm -rf {work} && {AGENT_HOME}/bin/rg --version | head -1"
    )
