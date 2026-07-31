"""Runtime fixes for mini-swe-agent 2.4.5 and swe-rex 1.4.0."""

import os
import sys
from pathlib import Path


def _patch(path: str, replacements: list[tuple[str, str, str]], label: str) -> bool:
    source_path = Path(path)
    source = source_path.read_text()
    pending: list[tuple[str, str]] = []
    failures: list[tuple[str, int]] = []

    for old, new, marker in replacements:
        if new in source or (marker and marker in source):
            continue
        count = source.count(old)
        if count != 1:
            failures.append((old.splitlines()[0], count))
        else:
            pending.append((old, new))

    if failures:
        for anchor, count in failures:
            print(
                f"WARN: [{label}] {source_path}: patch anchor {anchor!r} "
                f"found {count} times; no changes written",
                file=sys.stderr,
            )
        return False

    if not pending:
        print(f"[{label}] {source_path}: patch already applied")
        return True

    for old, new in pending:
        source = source.replace(old, new)
    source_path.write_text(source)
    print(f"[{label}] {source_path}: patch applied")
    return True



def _patch_swerex_environment(path: str) -> bool:
    return _patch(
        path,
        [
            (
                '        command = action.get("command", "") if isinstance(action, dict) else action\n'
                "        try:",
                '        command = action.get("command", "") if isinstance(action, dict) else action\n'
                '        command = f"exec </dev/null\\n{command}"  # close inherited server stdin (SWE-ReX #281)\n'
                "        try:",
                "# close inherited server stdin (SWE-ReX #281)",
            ),
        ],
        "swebench-agentic",
    )

def main() -> int:
    import minisweagent.run.benchmarks.swebench as mini_swebench
    import swerex.deployment.modal as rex_modal
    import minisweagent.environments.extra.swerex_modal as mini_rex_modal

    mini_ok = _patch(
        mini_swebench.__file__,
        [
            (
                "    agent = None\n    exit_status = None",
                "    agent = None\n    env = None\n    exit_status = None",
                "",
            ),
            (
                "        info = agent.run(task)\n"
                '        exit_status = info.get("exit_status")\n'
                '        result = info.get("submission")',
                "        info = agent.run(task)\n"
                '        exit_status = info.get("exit_status")\n'
                '        result = info.get("submission")\n'
                "        if not result and env is not None:\n"
                "            try:\n"
                '                _fb = env.execute("git diff")\n'
                '                _fb_out = (_fb.get("output") or "").strip()\n'
                '                if _fb.get("returncode") == 0 and _fb_out.startswith("diff --git"):\n'
                '                    result = _fb_out + "\\n"\n'
                '                    extra_info["submission_source"] = f"fallback_after_{exit_status}"\n'
                "            except Exception:\n"
                "                pass",
                "",
            ),
            (
                '        exit_status, result = type(e).__name__, ""\n'
                '        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}',
                '        exit_status, result = type(e).__name__, ""\n'
                '        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}\n'
                "        if env is not None:\n"
                "            try:\n"
                '                _fb = env.execute("git diff")\n'
                '                _fb_out = (_fb.get("output") or "").strip()\n'
                '                if _fb.get("returncode") == 0 and _fb_out.startswith("diff --git"):\n'
                '                    result = _fb_out + "\\n"\n'
                '                    extra_info["submission_source"] = f"fallback_after_{exit_status}"\n'
                "            except Exception:\n"
                "                pass",
                "",
            ),
            (
                "    finally:\n        if agent is not None:",
                "    finally:\n"
                '        if env is not None and callable(getattr(env, "stop", None)):\n'
                "            try:\n"
                "                env.stop()\n"
                "            except Exception:\n"
                "                pass\n"
                "        if agent is not None:",
                "",
            ),
        ],
        "swebench-agentic",
    )
    mini_env_ok = _patch_swerex_environment(mini_rex_modal.__file__)


    app_name = os.environ.get("SWEBENCH_MODAL_APP_NAME", "infx-evals-swe")
    rex_ok = _patch(
        rex_modal.__file__,
        [
            (
                'self._app = modal.App.lookup("swe-rex", create_if_missing=True)',
                f'self._app = modal.App.lookup("{app_name}", create_if_missing=True)  # isolate InferenceX sandboxes',
                "# isolate InferenceX sandboxes",
            ),
            (
                "        if self._sandbox is not None:\n"
                "            exit_code = await self._sandbox.poll.aio()\n"
                "            if exit_code is not None:\n"
                "                await self._sandbox.terminate.aio()",
                "        if self._sandbox is not None:\n"
                "            try:\n"
                "                await self._sandbox.terminate.aio()\n"
                "            except Exception:\n"
                "                pass",
                "",
            ),
            (
                "        await self._wait_until_alive(timeout=remaining_startup_timeout)",
                "        try:\n"
                "            await self._wait_until_alive(timeout=remaining_startup_timeout)\n"
                "        except BaseException:\n"
                "            try:\n"
                "                await self._sandbox.terminate.aio()\n"
                "            except Exception:\n"
                "                pass\n"
                "            raise",
                "",
            ),
        ],
        "swebench-agentic",
    )
    return 0 if mini_ok and mini_env_ok and rex_ok else 1


if __name__ == "__main__":
    sys.exit(main())
