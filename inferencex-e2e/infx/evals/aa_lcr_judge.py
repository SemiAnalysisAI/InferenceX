"""Grade AA-LCR samples with Claude Haiku 5.5 at low effort, using the AA-LCR v1.1 judge prompts.

Usage: python aa_lcr_judge.py <eval_results_dir> <out.json>
Reads every ``samples_aa_lcr*.jsonl`` below the directory; the parent artifact
name identifies the configuration.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"
SYSTEM = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""
USER = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["CORRECT", "INCORRECT"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=8)


def judge(sample: dict) -> bool:
    """Whether the judge accepts the sample's final answer; empty or failed answers are wrong."""
    candidate = (sample.get("response") or "").strip()
    if not candidate:
        return False
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": USER.format(
            question=sample["question"],
            official_answer=sample["answer"],
            candidate_answer=candidate,
        )}],
    )  # fmt: skip
    if response.stop_reason == "refusal":
        raise RuntimeError(f"judge refused question {sample['question_id']}")
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)["verdict"] == "CORRECT"


def main(root: Path, out: Path) -> int:
    """Grade every config's samples; print a Markdown table and write per-sample verdicts."""
    configs: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("samples_aa_lcr*.jsonl")):
        with path.open(encoding="utf-8") as lines:
            configs[path.parent.name] = [json.loads(line) for line in lines]
    if not configs:
        print("no AA-LCR samples found", file=sys.stderr)
        return 1
    report = {}
    print("| Config | Accuracy | ±1 SE | Per-repeat | Answered | Errors |")
    print("|---|---|---|---|---|---|")
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for name, samples in configs.items():
            verdicts = list(pool.map(judge, samples))
            per_repeat = defaultdict(list)
            for sample, verdict in zip(samples, verdicts, strict=True):
                sample["correct"] = verdict
                per_repeat[sample["repeat"]].append(verdict)
            n = len(verdicts)
            accuracy = sum(verdicts) / n
            # Questions are the sampling unit; average repeats per question first.
            by_question = defaultdict(list)
            for sample in samples:
                by_question[sample["question_id"]].append(sample["correct"])
            means = [sum(v) / len(v) for v in by_question.values()]
            mean = sum(means) / len(means)
            se = math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1) / len(means))
            repeats = " / ".join(
                f"{sum(v) / len(v):.3f}" for _, v in sorted(per_repeat.items())
            )
            answered = sum(bool((s.get("response") or "").strip()) for s in samples)
            errors = sum("error" in s for s in samples)
            print(f"| {name} | {accuracy:.3f} | {se:.3f} | {repeats} | {answered}/{n} | {errors} |")
            report[name] = {"accuracy": accuracy, "se": se, "n": n, "samples": samples}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
