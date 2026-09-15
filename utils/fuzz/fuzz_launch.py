import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st

from cases import CONCURRENCIES, ROOT, TEXT


@pytest.mark.parametrize("multinode", [False, True])
@pytest.mark.parametrize("scenario", ["fixed-seq-len", "agentic-coding"])
@given(concs=CONCURRENCIES, data=st.data(), historical=st.booleans(), base=TEXT,
       fingerprint=st.one_of(st.just(""), st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)))
def test_launch_publishes_identity_and_checks_outputs(multinode, scenario, concs, data, historical, base, fingerprint):
    cases = {
        "fixed-seq-len": [(False, "success", 0), (False, "missing", 1), (False, "failed", 17), (False, "empty", 0),
                          (True, "success", 0), (True, "missing", 1), (True, "failed", 17), (True, "empty", 0)],
        "agentic-coding": [(False, "success", 0), (False, "missing", 1), (False, "failed", 17), (False, "empty", 1),
                           (True, "success", 0), (True, "missing", 1), (True, "failed", 17), (True, "empty", 0)],
    }
    eval_only, outcome, expected_status = data.draw(st.sampled_from(cases[scenario]), label="result case")
    name = "benchmark-multinode-tmpl.yml" if multinode else "benchmark-tmpl.yml"
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    step = next(step for step in workflow["jobs"]["benchmark"]["steps"] if step.get("name", "").startswith("Launch "))
    script = step["run"].replace("${{ inputs.eval-only }}", str(eval_only).lower()).replace("${{ inputs.scenario-type }}", scenario)
    script = re.sub(r"\$\{\{ join\([^\n]*?additional-settings[^\n]*?\}\}", "FUZZ_SETTING=forwarded", script)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "infx").symlink_to(ROOT / "infx", target_is_directory=True)
        (root / "utils").mkdir()
        (root / "utils/agentic").symlink_to(ROOT / "utils/agentic", target_is_directory=True)
        if not historical:
            shutil.copy(ROOT / "utils/result_filename.py", root / "utils/result_filename.py")
        (root / "runners").mkdir()
        (root / "runners/launch_fixture.sh").write_text('''python3 - <<'INNER'
import json, os
from pathlib import Path
Path('received.json').write_text(json.dumps(dict(os.environ)))
for name, data in json.loads(os.environ['FUZZ_ARTIFACTS']).items():
    path = Path(name.replace('@RESULT@', os.environ['RESULT_FILENAME']))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
INNER
exit "$FUZZ_EXIT"
''')
        (root / "bin").mkdir()
        wait = root / "bin/sleep"
        wait.write_text("#!/bin/sh\nprintf 'wait\\n' >> waits\n")
        wait.chmod(0o755)
        artifacts = {}
        if outcome != "missing":
            if eval_only:
                artifacts = {"results_fixture.json": {}}
            elif multinode:
                artifacts = {f"@RESULT@_conc{conc}.json": {"num_requests_successful": int(outcome != "empty")}
                             for conc in concs}
            else:
                artifacts = {"@RESULT@.json": {}}
                if scenario == "agentic-coding":
                    artifacts["results/aiperf_artifacts/profile_export_aiperf.json"] = {
                        "request_count": {"avg": int(outcome != "empty")}}
        run = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=root,
                             capture_output=True, text=True, timeout=10, env={
            "PATH": f"{root / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}", "PYTHONPATH": "",
            "TP": "4", "PP_SIZE": "2", "PCP_SIZE": "3", "DCP_SIZE": "4", "RUNNER_NAME": "fixture_03",
            "RESULT_FILENAME_BASE": base, "RECIPE_FINGERPRINT": fingerprint, "CONC_LIST": " ".join(map(str, concs)),
            "GITHUB_ENV": str(root / "github-env"), "AIPERF_FAILED_REQUEST_THRESHOLD": "0.05",
            "FUZZ_EXIT": "17" if outcome == "failed" else "0", "FUZZ_ARTIFACTS": json.dumps(artifacts),
        })
        received = json.loads((root / "received.json").read_text())
        published = dict(line.split("=", 1) for line in (root / "github-env").read_text().splitlines())
        assert published["RESULT_FILENAME"] == received["RESULT_FILENAME"]
        if multinode:
            assert received["IS_MULTINODE"] == "true"
            assert received["FUZZ_SETTING"] == "forwarded"
        else:
            assert received["GPU_COUNT"] == published["GPU_COUNT"] == "24"
        assert run.returncode == expected_status, run.stderr
        if outcome == "missing" and not multinode and not eval_only:
            assert (root / "waits").read_text().splitlines() == ["wait"] * 10
