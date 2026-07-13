"""Patch the installed pinned mini-swe-agent/swe-rex deps (idempotent,
anchor-checked; pinned versions keep the anchors stable). Run by
_patch_swebench_agent in benchmarks/benchmark_lib.sh.

Sandbox lifecycle (observed: batches dying at 59m59s = the 1h
runtime_timeout, billing the full hour for ~7-min instances):
  - mini-swe-agent 2.4.5: process_instance() never calls env.stop() -- not
    even on success -- so EVERY sandbox lives until runtime_timeout.
  - swe-rex 1.4.0: ModalDeployment.stop() has its poll check inverted
    (terminates only already-exited sandboxes), and start() leaks the
    sandbox when the runtime never comes alive (startup timeout).
  Best-effort: the post-generation sweep still bounds the damage if an
  anchor ever drifts.

Budget-exhaustion fallback (6/50 instances per 50-run burn all steps and
submit NOTHING -- forensics showed correct fixes can be sitting in the
tree): when an instance ends without a submission but with a live sandbox,
submit `git diff` of the tree as the patch. Empty scores zero anyway, so
this is strictly >=; guarded to require rc 0 and a real diff so an error
message can never be submitted as a patch. NOTE: LimitsExceeded does NOT
raise into process_instance -- mini's run loop absorbs InterruptAgentFlow
and returns normally with an empty submission (verified: 0 fallbacks fired
when this hook lived only on the except path) -- so the primary hook is on
the normal-return path; the except-path hook stays for real exceptions.
"""
import os, sys

SENTINEL = "inferencex sandbox cleanup"


def patch(path, replacements):
    src = open(path).read()
    if SENTINEL in src:
        print(f"[swebench-agentic] {path}: cleanup patch already applied")
        return True
    ok = True
    for old, new in replacements:
        if src.count(old) != 1:
            print(f"WARN: [swebench-agentic] {path}: patch anchor not found exactly once "
                  f"(count={src.count(old)}); skipping hunk", file=sys.stderr)
            ok = False
            continue
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"[swebench-agentic] {path}: sandbox-cleanup patch {'applied' if ok else 'PARTIALLY applied'}")
    return ok


import minisweagent.run.benchmarks.swebench as mini_sb
import swerex.deployment.modal as rex_modal

mini_ok = patch(mini_sb.__file__, [
    (
        "    agent = None\n    exit_status = None",
        "    agent = None\n    env = None  # inferencex sandbox cleanup\n    exit_status = None",
    ),
    (
        '        info = agent.run(task)\n'
        '        exit_status = info.get("exit_status")\n'
        '        result = info.get("submission")',
        '        info = agent.run(task)\n'
        '        exit_status = info.get("exit_status")\n'
        '        result = info.get("submission")\n'
        "        if not result and env is not None:  # budget-exhaustion fallback: submit the tree's diff\n"
        "            try:\n"
        '                _fb = env.execute("git diff")\n'
        '                _fb_out = (_fb.get("output") or "").strip()\n'
        '                if _fb.get("returncode") == 0 and _fb_out.startswith("diff --git"):\n'
        "                    result = _fb_out + \"\\n\"\n"
        '                    extra_info["submission_source"] = f"fallback_after_{exit_status}"\n'
        "            except Exception:\n"
        "                pass",
    ),
    (
        '        exit_status, result = type(e).__name__, ""\n'
        '        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}',
        '        exit_status, result = type(e).__name__, ""\n'
        '        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}\n'
        "        if env is not None:  # budget-exhaustion fallback: submit the tree's diff\n"
        "            try:\n"
        '                _fb = env.execute("git diff")\n'
        '                _fb_out = (_fb.get("output") or "").strip()\n'
        '                if _fb.get("returncode") == 0 and _fb_out.startswith("diff --git"):\n'
        "                    result = _fb_out + \"\\n\"\n"
        '                    extra_info["submission_source"] = f"fallback_after_{exit_status}"\n'
        "            except Exception:\n"
        "                pass",
    ),
    (
        "    finally:\n        if agent is not None:",
        "    finally:\n"
        "        if env is not None and callable(getattr(env, \"stop\", None)):\n"
        "            try:\n"
        "                env.stop()\n"
        "            except Exception:\n"
        "                pass\n"
        "        if agent is not None:",
    ),
])

APP_NAME = os.environ.get("SWEBENCH_MODAL_APP_NAME", "infx-evals-swe")
rex_ok = patch(rex_modal.__file__, [
    (
        'self._app = modal.App.lookup("swe-rex", create_if_missing=True)',
        f'self._app = modal.App.lookup("{APP_NAME}", create_if_missing=True)  # inferencex app name',
    ),
    (
        "        if self._sandbox is not None:\n"
        "            exit_code = await self._sandbox.poll.aio()\n"
        "            if exit_code is not None:\n"
        "                await self._sandbox.terminate.aio()",
        "        if self._sandbox is not None:  # inferencex sandbox cleanup\n"
        "            try:\n"
        "                await self._sandbox.terminate.aio()\n"
        "            except Exception:\n"
        "                pass",
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
    ),
])
sys.exit(0 if (mini_ok and rex_ok) else 1)
