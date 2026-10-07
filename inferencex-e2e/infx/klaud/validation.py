"""Verify full-sweep coverage using the dispatched matrix and result validators."""

from __future__ import annotations

import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from infx.results import artifacts, eval_artifacts

from . import github
from .github import VerificationError
from .models import OwnedCandidate, utc


def benchmark_entries(matrix: dict) -> list[dict]:
    return [
        entry
        for group in ("single_node", "multi_node")
        for rows in matrix.get(group, {}).values()
        for entry in rows
    ]


def benchmark_points(entries: list[dict]) -> set[tuple]:
    return {
        (entry["recipe-fingerprint"], int(conc), entry["image"])
        for entry in entries
        for conc in (entry["conc"] if isinstance(entry["conc"], list) else [entry["conc"]])
    }


def project_prefix(repository: str, head: str) -> str:
    """The end-to-end project directory at ``head``: ``inferencex-e2e/`` or the root."""
    tree = github.read(repository, f"git/trees/{head}")
    return "inferencex-e2e/" if "inferencex-e2e" in {item["path"] for item in tree["tree"]} else ""


def canonical_matrix(repository: str, head: str, family: str) -> dict:
    """Generate the unfiltered family from exact-head YAML using trusted local code.

    Candidate source is data only. A future generator-policy change can require manual
    inspection of an older run; it cannot silently waive points or required evals.
    """
    import yaml

    from infx.matrix.generate import generate_test_config_sweep, mark_eval_entries
    from infx.matrix.plan import recipe_fingerprint
    from infx.matrix.validation import (
        srt_recipe_references,
        validate_master_config,
        validate_runner_config,
    )

    OwnedCandidate(id="0" * 16 + "-" + "0" * 16, family=family, base=head)
    source, key = family.split(":", 1)
    prefix = project_prefix(repository, head)
    master = yaml.safe_load(github.file_at(repository, head, prefix + source))
    runners = yaml.safe_load(github.file_at(repository, head, prefix + "configs/runners.yaml"))
    # Only the selected family is relevant; retired sibling schemas may have changed.
    family_config = {key: master[key]}
    with tempfile.TemporaryDirectory(prefix="klaud-recipes-") as temp:
        # Recipes are head data too: resolve them from the candidate, not this checkout.
        root = Path(temp)
        for reference in srt_recipe_references(family_config)[key]:
            path = reference.partition(":")[0]
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(github.file_at(repository, head, prefix + path))
        entries = generate_test_config_sweep(
            SimpleNamespace(config_keys=[key]),
            validate_master_config(family_config, root),
            validate_runner_config(runners),
            root,
        )
    evals = [
        dict(row, **{"eval-only": True})
        for row in mark_eval_entries(deepcopy(entries))
        if row.get("run-eval")
    ]
    return {
        "single_node": {
            "all": [{**row, "recipe-fingerprint": recipe_fingerprint(row)} for row in entries]
        },
        "evals": evals,
    }


PRODUCER_UNAVAILABLE = "Baseline producer family cannot be regenerated"


class ProducerRegenerationError(VerificationError):
    """A producer revision's own tooling could not regenerate its family.

    The message is the fixed public reason; ``stderr`` holds the failing tool's output for
    the operator and never becomes public.
    """

    def __init__(self, stderr: str) -> None:
        super().__init__(PRODUCER_UNAVAILABLE)
        self.stderr = stderr


def producer_matrix(repository: str, head: str, family: str) -> dict:
    """Regenerate a published producer's family with the producer revision's own generator.

    This runs another revision's code, which inherits only ``INHERITED_ENV``: call it only
    where no credentials are held, as the ``regenerate-producers`` step does.
    """
    from infx.matrix.plan import recipe_fingerprint
    from infx.matrix.revision import snapshot

    OwnedCandidate(id="0" * 16 + "-" + "0" * 16, family=family, base=head)
    try:
        missing = subprocess.run(
            ["git", "cat-file", "-e", f"{head}^{{commit}}"], capture_output=True, check=False
        ).returncode
        if missing:
            subprocess.run(
                ["git", "fetch", "--quiet", "--no-tags", f"https://github.com/{repository}", head],
                capture_output=True,
                check=True,
                timeout=300,
            )
        with snapshot(head) as producer:
            rows = producer.generate([family.split(":", 1)[1]], ["--no-evals"])
    except subprocess.SubprocessError as error:
        stderr = getattr(error, "stderr", None) or b""
        text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        raise ProducerRegenerationError(text) from error
    except (OSError, ValueError) as error:
        raise ProducerRegenerationError(str(error)) from error
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProducerRegenerationError("the generator did not print a list of matrix rows")
    return {
        "single_node": {
            "all": [{**row, "recipe-fingerprint": recipe_fingerprint(row)} for row in rows]
        }
    }


def check_matrix(matrix: dict, canonical: dict, head: str, family: str) -> None:
    from pydantic import TypeAdapter

    from infx.matrix.validation import (
        MultiNodeAgenticMatrixEntry,
        MultiNodeMatrixEntry,
        SingleNodeAgenticMatrixEntry,
        SingleNodeMatrixEntry,
    )

    schema = TypeAdapter(
        SingleNodeMatrixEntry
        | SingleNodeAgenticMatrixEntry
        | MultiNodeMatrixEntry
        | MultiNodeAgenticMatrixEntry
    )

    metadata = matrix["changelog_metadata"]
    if metadata["head_ref"] != head:
        raise VerificationError("Matrix was generated from a different head")
    entries = metadata["entries"]
    if len(entries) != 1 or entries[0]["config-keys"] != [family.split(":", 1)[1]]:
        raise VerificationError("Final changelog must select exactly the candidate family")
    expected = benchmark_points(benchmark_entries(canonical))
    generated = {
        point: schema.validate_python(entry).model_dump(by_alias=True, exclude_none=True)
        for entry in benchmark_entries(canonical)
        for point in benchmark_points([entry])
    }
    for entry in benchmark_entries(matrix):
        measured = {
            key: value for key, value in entry.items() if key not in ("priority", "queue-token")
        }
        # The workflow adds schema defaults AFTER fingerprinting. Compare complete
        # settings with the independently generated recipe under that same schema.
        normalized = schema.validate_python(measured).model_dump(by_alias=True, exclude_none=True)
        if any(generated.get(point) != normalized for point in benchmark_points([entry])):
            raise VerificationError("Matrix settings do not match the canonical recipe")
    count = sum(
        len(entry["conc"]) if isinstance(entry["conc"], list) else 1
        for entry in benchmark_entries(matrix)
    )
    if count != len(benchmark_points(benchmark_entries(matrix))):
        raise VerificationError("Duplicate point in final matrix")
    if not expected or benchmark_points(benchmark_entries(matrix)) != expected:
        raise VerificationError("Final matrix omits or changes canonical benchmark points")
    if expected_evals(matrix) != expected_evals(canonical):
        raise VerificationError("Final matrix does not match canonical default evals")


def expected_evals(matrix: dict) -> set[tuple]:
    expected = set()
    for bucket in (
        "evals",
        "agentic_evals",
        "multinode_evals",
        "multinode_agentic_evals",
    ):
        for entry in matrix.get(bucket, []):
            multi = entry.get("prefill") is not None
            row = {key.replace("-", "_"): value for key, value in entry.items()}
            row.update(
                is_multinode=multi,
                hw=entry["runner"],
                model_prefix=entry["model-prefix"],
                eval_suite=entry.get("eval-suite") or "gsm8k",
                isl=entry.get("isl", 0),
                osl=entry.get("osl", 0),
                dp_attention=entry.get("dp-attn", False),
            )
            for role in ("prefill", "decode") if multi else ():
                for key, value in entry[role].items():
                    name = {"dp-attn": "dp_attention", "num-worker": "num_workers"}.get(
                        key, key.replace("-", "_")
                    )
                    row[f"{role}_{name}"] = value
            concs = (
                entry["conc"]
                if multi and entry.get("eval-all-concs")
                else [entry["eval-conc"] if multi else entry["conc"]]
            )
            expected.update(eval_artifacts.eval_key({**row, "conc": conc}) for conc in concs)
    return expected


def check_coverage(
    directory: Path, manifest: dict, run: dict, family: str, canonical: dict
) -> None:
    if (
        manifest["head"] != run["head_sha"]
        or manifest["run-id"] != run["id"]
        or manifest["run-attempt"] != run["run_attempt"]
        or manifest["full-sweep"] is not True
    ):
        raise VerificationError("Full-sweep provenance mismatch")
    matrix = manifest["matrix"]
    check_matrix(matrix, canonical, run["head_sha"], family)
    expected = benchmark_points(benchmark_entries(matrix))
    paths = list((directory / "results_bmk").glob("*.json"))
    fixed = [
        row for _, row in artifacts.json_rows(paths) if row.get("scenario_type") != "agentic-coding"
    ]
    agentic = [row for _, row in artifacts.json_rows(artifacts.agentic_point_files(directory))]
    actual = [
        (row["recipe_fingerprint"], int(row["conc"]), row["image"]) for row in fixed + agentic
    ]
    errors = artifacts.duplicate_identity_errors("benchmark", actual)
    errors += artifacts.validate_identity_set("benchmark", expected, set(actual))
    eval_artifacts.dedupe_reran_evals(directory)
    eval_rows, eval_errors = eval_artifacts.inspect_eval_artifacts(directory)
    errors += eval_errors
    errors += artifacts.validate_identity_set(
        "eval", expected_evals(matrix), {row[:-1] for row in eval_rows}
    )
    if errors:
        # Only a fixed failure code escapes: no raw artifact data in public diagnostics.
        raise VerificationError("Full-sweep result coverage or consistency failed")


def select_artifacts(inventory: list[dict], run: dict, attempt: dict) -> list[dict]:
    """Newest same-name artifact within this run; never overlay two archives.

    Retained point artifacts from successful jobs in an earlier attempt are allowed.
    The final manifest must be current. An earlier aggregate is retained only when
    none of its point producers reran. Coverage/raw agreement still must pass.
    """
    selected: dict[str, dict] = {}
    for artifact in inventory:
        name = artifact["name"]
        wanted = (
            name in ("klaud-sweep-manifest", "results_bmk")
            or name.startswith("bmk_agentic_")
            or (name.startswith("eval_") and not name.startswith("eval_server_logs_"))
        )
        if not wanted:
            continue
        if "/" in name or "\\" in name or name in (".", ".."):
            raise VerificationError("Invalid artifact name")
        producer = artifact.get("workflow_run", {})
        if producer.get("id") != run["id"] or producer.get("head_sha") != run["head_sha"]:
            raise VerificationError("Artifact run or head mismatch")
        if utc(artifact["created_at"]) < utc(run["created_at"]) or utc(
            artifact["created_at"]
        ) > utc(run["updated_at"]):
            raise VerificationError("Artifact outside run lifetime")
        previous = selected.get(name)
        if previous is None or (utc(artifact["created_at"]), artifact["id"]) > (
            utc(previous["created_at"]),
            previous["id"],
        ):
            selected[name] = artifact
    for name, artifact in selected.items():
        if artifact["expired"]:
            raise VerificationError("Latest result artifact has expired")
        if utc(artifact["created_at"]) < utc(attempt["run_started_at"]):
            if name == "klaud-sweep-manifest":
                raise VerificationError("Manifest belongs to an earlier run attempt")
            if name in ("results_bmk", "eval_results_all"):
                prefix = "bmk_" if name == "results_bmk" else "eval_"
                if any(
                    a["name"].startswith(prefix)
                    and a["name"] != name
                    and not a["name"].startswith("eval_server_logs_")
                    and utc(a["created_at"]) >= utc(attempt["run_started_at"])
                    for a in inventory
                ):
                    raise VerificationError("Stale aggregate after a point producer reran")
    if "klaud-sweep-manifest" not in selected:
        raise VerificationError("Missing final-sweep manifest")
    return list(selected.values())


def verify_sweep(repository: str, run: dict, family: str) -> tuple[dict, list[dict], list[dict]]:
    if run["status"] != "completed" or run["conclusion"] != "success":
        raise VerificationError("Final sweep has not passed")
    attempt = github.read(repository, f"actions/runs/{run['id']}/attempts/{run['run_attempt']}")
    inventory = select_artifacts(github.artifacts(repository, run["id"]), run, attempt)
    canonical = canonical_matrix(repository, run["head_sha"], family)
    with tempfile.TemporaryDirectory(prefix="klaud-validation-") as temp:
        directory = Path(temp)
        for artifact in inventory:
            github.download_json(repository, artifact, directory / artifact["name"])
        manifest = json.loads((directory / "klaud-sweep-manifest/sweep_manifest.json").read_text())
        check_coverage(directory, manifest, run, family, canonical)
        paths = list((directory / "results_bmk").glob("*.json"))
        fixed = [
            row
            for _, row in artifacts.json_rows(paths)
            if row.get("scenario_type") != "agentic-coding"
        ]
        agentic = [row for _, row in artifacts.json_rows(artifacts.agentic_point_files(directory))]
        evals = [
            row for _, row in artifacts.json_rows((directory / "eval_results_all").glob("*.json"))
        ]
        return canonical, fixed + agentic, evals
