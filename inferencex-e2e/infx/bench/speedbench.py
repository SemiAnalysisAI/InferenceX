"""The SPEED-Bench prompts every golden collection sent, prepared once per run.

``python -m infx.bench.speedbench --output-dir DIR`` writes ``DIR/speedbench_coding.jsonl``, one
``{"question_id", "prompt"}`` per line in dataset order: the first user message of each coding
row of nvidia/SPEED-Bench's qualitative test split, as NeMo-Skills' pinned prepare.py resolves
it. It writes nothing unless their digest is ``PROMPTS_SHA256``. It needs prepare.py's imports
and about 35 GB under ``HF_HOME``; from ``inferencex-e2e/``::

    uv run --no-project --exclude-newer PT12H --python 3.12 --with datasets --with numpy \
        --with pandas --with tiktoken python -m infx.bench.speedbench --output-dir DIR

Importing this module needs only the standard library: gate containers ``load`` the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path

# The reviewed prepare.py, at the only NeMo-Skills commit that touched it.
PREPARE_PY = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/"
    "5ac8609a56ac941540b10c92e68d556e6343cd4c/nemo_skills/dataset/speed-bench/prepare.py"
)
PREPARE_PY_SHA256 = "a551be4df541474e54e21b480022b0cbb66c2da068fda61b2a64bd3223bbbed2"
# prepare.py loads SPEED-Bench's moving main branch; PROMPTS_SHA256 is of this revision.
SPEED_BENCH_REVISION = "454f88454792dfa3ccfd7ef15fff248efde44cd1"
# ``digest`` of the 80 coding prompts, in dataset order.
PROMPTS_SHA256 = "6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18"
PROMPTS_FILE = "speedbench_coding.jsonl"
# prepare.py runs as this importable module: datasets fingerprints its functions by reference.
MODULE = "infx_speedbench_prepare"


def digest(prompts: list[str]) -> str:
    """sha256 of the prompts as one UTF-8 JSON array."""
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()


def load(path: Path) -> list[str]:
    """The prompts of a prepared ``speedbench_coding.jsonl``, in dataset order."""
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line)["prompt"] for line in lines]


def prepare(script: bytes) -> list[dict[str, str]]:
    """Each coding row's id and first user message, as prepare.py's ``script`` resolves it."""
    module = sys.modules[MODULE] = types.ModuleType(MODULE)
    try:
        exec(compile(script, PREPARE_PY, "exec"), module.__dict__)  # noqa: S102
        rows = module.load_dataset(
            "nvidia/SPEED-Bench", "qualitative", split="test", revision=SPEED_BENCH_REVISION
        )
        coding = rows.filter(lambda row: row["category"] == "coding")
        # Only the coding rows reach its resolver; other categories need a gated dataset.
        resolved = module._resolve_external_data(coding, "qualitative")  # noqa: SLF001
        return [{"question_id": row["question_id"], "prompt": row["turns"][0]} for row in resolved]
    finally:
        sys.modules.pop(MODULE, None)


def main(argv: list[str] | None = None) -> int:
    """Prepare the gate's SPEED-Bench prompts into ``--output-dir``."""
    parser = argparse.ArgumentParser(prog="python -m infx.bench.speedbench", description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with urllib.request.urlopen(PREPARE_PY, timeout=60) as response:
        script = response.read()
    if (actual := hashlib.sha256(script).hexdigest()) != PREPARE_PY_SHA256:
        sys.exit(f"ERROR: {PREPARE_PY} has sha256 {actual}, not {PREPARE_PY_SHA256}")
    rows = prepare(script)
    if (actual := digest([row["prompt"] for row in rows])) != PROMPTS_SHA256:
        sys.exit(f"ERROR: {len(rows)} prompts with sha256 {actual}, not {PROMPTS_SHA256}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / PROMPTS_FILE
    lines = (json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} SPEED-Bench prompts to {path} ({actual})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
