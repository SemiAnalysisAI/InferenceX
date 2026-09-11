"""Changelog selection, historical comparisons, and complete sweep planning."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
import traceback
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import GENERATE_SWEEPS_PY_SCRIPT, MASTER_CONFIGS, RUNNER_CONFIG
from .generate import (
    EvalMode,
    freeze_config_value,
    generate_config_matrix,
    seq_len_to_str,
    trim_conc,
)
from .validation import (
    ChangelogEntry,
    ChangelogMatrixEntry,
    load_config_files,
    load_runner_file,
)

SCENARIO_TYPES = ("fixed-seq-len", "agentic-coding")


@dataclass(frozen=True)
class GenerationInputs:
    config_files: list[str]
    generator_script: str
    runner_config: str


def get_added_lines(base_ref: str, head_ref: str, filepath: str) -> str:
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--", filepath],
        capture_output=True,
        text=True,
    )

    added_lines = []
    for line in result.stdout.split("\n"):
        if line.startswith("-") and not line.startswith("---"):
            deleted_content = line[1:]
            # Allow whitespace-only or empty line deletions
            if deleted_content.strip():
                # Don't allow deletions in the changelog
                # By convention, it should act as a running log of performance changes,
                # so we only want to see additions
                raise ValueError(
                    f"Deletions are not allowed in {filepath}. "
                    f"Only additions to the changelog are permitted. "
                    f"Found deleted line: {deleted_content}"
                )
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    return "\n".join(added_lines)


def filter_eval_rows_by_prefill_ep(
    eval_rows: list[dict], min_prefill_ep: int | None
) -> list[dict]:
    """Drop multinode eval rows below a prefill EP threshold."""
    if min_prefill_ep is None:
        return eval_rows
    kept: list[dict] = []
    for row in eval_rows:
        prefill = row.get("prefill")
        if isinstance(prefill, dict):
            ep = prefill.get("ep", 1)
            try:
                if int(ep) < min_prefill_ep:
                    continue
            except (TypeError, ValueError):
                continue
        kept.append(row)
    return kept


def get_config_keys_from_master(
    config_keys: list[str], master_config: dict
) -> list[str]:
    resolved_keys = {}
    for key in config_keys:
        if "*" in key:
            pattern = re.compile(re.escape(key).replace(r"\*", ".*"))
            matched_keys = [k for k in master_config if pattern.fullmatch(k)]
            if not matched_keys:
                raise ValueError(
                    f"No config keys matched the wildcard pattern '{key}' in master configs."
                )
            for matched_key in matched_keys:
                resolved_keys.setdefault(matched_key, None)
        elif key not in master_config:
            raise ValueError(f"Config key '{key}' not found in master configs.")
        else:
            resolved_keys.setdefault(key, None)
    return list(resolved_keys)


@contextmanager
def generation_inputs_at_ref(ref: str):
    """Materialize config and generator inputs from one repository revision."""
    with tempfile.TemporaryDirectory(prefix="inferencex-append-only-") as temp_dir:
        files_result = subprocess.run(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                ref,
                "--",
                "utils/matrix_logic",
                "infx",
                *MASTER_CONFIGS,
                "configs/runners.yaml",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        repo_paths = files_result.stdout.splitlines()
        required_paths = {
            *MASTER_CONFIGS,
            "configs/runners.yaml",
            GENERATE_SWEEPS_PY_SCRIPT,
        }
        missing_paths = required_paths - set(repo_paths)
        if missing_paths:
            raise ValueError(
                f"append-only base revision is missing generation inputs: "
                f"{sorted(missing_paths)}"
            )

        for repo_path in repo_paths:
            result = subprocess.run(
                ["git", "show", f"{ref}:{repo_path}"],
                capture_output=True,
                check=True,
            )
            destination = Path(temp_dir) / repo_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.stdout)

        yield GenerationInputs(
            config_files=[str(Path(temp_dir) / path) for path in MASTER_CONFIGS],
            generator_script=str(Path(temp_dir) / GENERATE_SWEEPS_PY_SCRIPT),
            runner_config=str(Path(temp_dir) / "configs/runners.yaml"),
        )


def _matrix_curve_key(entry: dict) -> tuple:
    """Identify one curve while deliberately excluding point-level fields."""
    return tuple(
        sorted(
            (key, freeze_config_value(value))
            for key, value in entry.items()
            if key not in {"conc", "exp-name", "recipe-fingerprint"}
        )
    )


def recipe_fingerprint(entry: dict) -> str:
    """Hash the generated recipe independently of point-level concurrency/name."""
    recipe = {
        key: value
        for key, value in entry.items()
        if key not in {"conc", "exp-name", "recipe-fingerprint"}
    }
    canonical = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _matrix_visual_series_key(entry: dict) -> tuple:
    """Identify the App curve that an appended recipe must already belong to."""
    is_agentic = entry.get("scenario-type") == "agentic-coding"
    kv_offloading = entry.get("kv-offloading", "none")
    offload_mode = "off" if kv_offloading in (None, "", "none") else "on"
    prefill = entry.get("prefill") or {}
    decode = entry.get("decode") or {}
    return (
        entry.get("model"),
        entry.get("model-prefix"),
        entry.get("precision"),
        entry.get("framework"),
        entry.get("runner"),
        bool(entry.get("disagg", False)),
        "agentic_traces" if is_agentic else "single_turn",
        None if is_agentic else entry.get("isl"),
        None if is_agentic else entry.get("osl"),
        offload_mode,
        "" if is_agentic else entry.get("spec-decoding", "none"),
        prefill.get("hardware"),
        decode.get("hardware"),
    )


def _matrix_concurrencies(entry: dict) -> tuple[int, ...]:
    conc = entry.get("conc")
    if isinstance(conc, int):
        return (conc,)
    if isinstance(conc, list) and conc and all(isinstance(value, int) for value in conc):
        return tuple(conc)
    raise ValueError(f"append-only matrix entry has invalid concurrency value: {conc!r}")


def append_only_delta(base_entries: list[dict], head_entries: list[dict]) -> list[dict]:
    """Return only newly added points, rejecting any existing-point mutation.

    Generated matrix rows are the runtime contract. Grouping them without ``conc``
    and ``exp-name`` lets an existing recipe gain concurrency while also permitting
    entirely new recipe variants. Every base recipe and concurrency must remain in
    the head unchanged; the returned delta is therefore strictly additive.
    """
    base_groups: dict[tuple, set[int]] = defaultdict(set)
    head_groups: dict[tuple, set[int]] = defaultdict(set)
    for entry in base_entries:
        base_groups[_matrix_curve_key(entry)].update(_matrix_concurrencies(entry))
    for entry in head_entries:
        head_groups[_matrix_curve_key(entry)].update(_matrix_concurrencies(entry))

    if not base_groups:
        raise ValueError("append-only requires an existing curve in the base revision")

    removed_curves = base_groups.keys() - head_groups.keys()
    if removed_curves:
        raise ValueError(
            "append-only may not remove or modify existing generated recipes"
        )

    for key, base_concurrencies in base_groups.items():
        removed_points = base_concurrencies - head_groups[key]
        if removed_points:
            raise ValueError(
                "append-only may not remove existing concurrency points: "
                f"{sorted(removed_points)}"
            )

    delta: list[dict] = []
    emitted_concurrencies: dict[tuple, set[int]] = defaultdict(set)
    for entry in head_entries:
        key = _matrix_curve_key(entry)
        added = head_groups[key] - base_groups.get(key, set())
        conc = entry.get("conc")
        if isinstance(conc, int):
            if conc in added and conc not in emitted_concurrencies[key]:
                delta.append(entry)
                emitted_concurrencies[key].add(conc)
            continue
        added_in_source_order = []
        for value in conc:
            if value in added and value not in emitted_concurrencies[key]:
                added_in_source_order.append(value)
                emitted_concurrencies[key].add(value)
        if added_in_source_order:
            delta_entry = copy.deepcopy(entry)
            delta_entry["conc"] = added_in_source_order
            delta.append(delta_entry)

    if not delta:
        raise ValueError("append-only did not add any generated points")

    base_images_by_series: dict[tuple, set[str | None]] = defaultdict(set)
    for entry in base_entries:
        base_images_by_series[_matrix_visual_series_key(entry)].add(
            entry.get("image")
        )
    for entry in delta:
        series_key = _matrix_visual_series_key(entry)
        base_images = base_images_by_series.get(series_key, set())
        image = entry.get("image")
        if image is None or base_images != {image}:
            raise ValueError(
                "append-only additions must belong to an existing visual curve "
                "with one unchanged non-null image"
            )
    return delta


def validate_append_only_scope(
    base_master: dict,
    head_master: dict,
    selected_config_scenarios: dict[str, set[str]],
) -> None:
    """Reject edits outside selected existing configs and scenarios.

    Changes inside an explicitly selected scenario are checked semantically by
    ``append_only_delta`` after generating the complete base and head matrices.
    This permits arbitrary additive recipe variants while ensuring every existing
    generated point remains unchanged and present.
    """
    selected_configs = selected_config_scenarios.keys()
    all_keys = base_master.keys() | head_master.keys()
    unrelated_changes = [
        key
        for key in all_keys
        if key not in selected_configs and base_master.get(key) != head_master.get(key)
    ]
    if unrelated_changes:
        raise ValueError(
            "append-only PR changed configs not selected by its changelog entry: "
            f"{sorted(unrelated_changes)}"
        )

    for config, allowed_scenarios in selected_config_scenarios.items():
        base_config = base_master[config]
        head_config = head_master[config]
        base_scenarios = base_config.get("scenarios", {})
        head_scenarios = head_config.get("scenarios", {})
        if base_scenarios.keys() != head_scenarios.keys():
            raise ValueError(
                f"append-only added or removed a scenario in config {config!r}"
            )

        unselected_scenarios = base_scenarios.keys() - allowed_scenarios
        base_top_level = {
            key: value for key, value in base_config.items() if key != "scenarios"
        }
        head_top_level = {
            key: value for key, value in head_config.items() if key != "scenarios"
        }
        if unselected_scenarios and base_top_level != head_top_level:
            raise ValueError(
                "append-only changed config-wide fields that can affect scenarios "
                f"outside its changelog scope: {config!r}"
            )

        for scenario in base_scenarios:
            if scenario not in allowed_scenarios:
                if base_scenarios[scenario] != head_scenarios[scenario]:
                    raise ValueError(
                        "append-only changed a scenario outside its changelog scope: "
                        f"{config!r} / {scenario!r}"
                    )
                continue


def group_unseen_scenarios(
    config_keys: list[str],
    scenarios: tuple[str, ...],
    seen: dict[str, set[str]],
) -> dict[tuple[str, ...], list[str]]:
    """Claim unseen config/scenario pairs in canonical scenario and input-key order.

    Benchmark and eval callers pass separate coverage maps; grouping one must
    never suppress work in the other.
    """
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for config in config_keys:
        unseen = tuple(
            scenario for scenario in SCENARIO_TYPES
            if scenario in scenarios and scenario not in seen[config]
        )
        if unseen:
            seen[config].update(unseen)
            groups[unseen].append(config)
    return groups


def generate_matrix(
    config_keys: list[str],
    flags: list[str],
    inputs: GenerationInputs | None = None,
) -> list[dict]:
    """Run the selected generator in its own process and decode its matrix.

    The planner uses this only for historical revisions. Retain the callable's
    legacy flags, optional input override, and child diagnostics for script users.
    """
    command = _matrix_command(config_keys, flags, inputs)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr)
        raise
    return json.loads(result.stdout)


class MatrixGenerationError(ValueError):
    """A current-revision generation failure, with its CLI diagnostic context."""

    def __init__(
        self, keys: list[str], flags: list[str],
        inputs: GenerationInputs | None, cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.keys, self.flags, self.inputs, self.cause = keys, flags, inputs, cause


def build_plan(
    changelog_data: list[dict], *, base_ref: str, head_ref: str,
    config_files: list[str] | None = None, runner_config: str = RUNNER_CONFIG,
    trim: bool = False, all_evals: bool = False, evals_only: bool = False,
) -> ChangelogMatrixEntry:
    """Build the complete sweep from changelog entries and configuration paths.

    Load each current input once; load runners only when generation is needed.
    Append-only base generation uses that revision's own isolated code and inputs.
    Nothing is published until generation and final schema validation succeed.
    """
    if not changelog_data:
        raise ValueError("No valid YAML entries found in the changelog additions.")

    with ExitStack() as stack:
        parsed_entries = [ChangelogEntry.model_validate(entry) for entry in changelog_data]
        if any(entry.no_evals for entry in parsed_entries) and (
            all_evals or evals_only
        ):
            raise ValueError("no-evals entries cannot use all-evals or evals-only modifiers")
        has_append_only = any(entry.append_only for entry in parsed_entries)
        if has_append_only and not all(entry.append_only for entry in parsed_entries):
            raise ValueError(
                "append-only entries cannot share a sweep with regular changelog entries"
            )
        if has_append_only and (all_evals or evals_only):
            raise ValueError(
                "append-only sweeps cannot use all-evals or evals-only modifiers"
            )

        final_results = {
            "single_node": defaultdict(list),
            "multi_node": defaultdict(list),
            "evals": [],
            "agentic_evals": [],
            "multinode_evals": [],
            "multinode_agentic_evals": [],
            "changelog_metadata": {
                "base_ref": base_ref,
                "head_ref": head_ref,
                "entries": changelog_data,
            },
        }

        all_benchmark_results = []
        all_eval_results = []
        # Track benchmark coverage per scenario so overlapping changelog entries
        # with disjoint scenario filters do not suppress each other.
        benchmark_scenarios_seen = defaultdict(set)
        eval_scenarios_seen = defaultdict(set)

        config_files = MASTER_CONFIGS if config_files is None else config_files
        head_inputs = GenerationInputs(config_files, GENERATE_SWEEPS_PY_SCRIPT, runner_config)
        master_config = load_config_files(config_files)
        runner_data = None

        def generate_current(
            keys: list[str], mode: EvalMode, scenarios: tuple[str, ...] | None,
        ) -> list[dict]:
            nonlocal runner_data
            try:
                if runner_data is None:
                    runner_data = load_runner_file(runner_config)
                return generate_config_matrix(
                    keys, master_config, runner_data,
                    scenario_types=scenarios, eval_mode=mode,
                )
            except Exception as error:
                raise MatrixGenerationError(
                    keys, _generation_flags(mode, scenarios),
                    head_inputs if mode == "none" else None, error,
                ) from error

        resolved_entries = []
        for entry in parsed_entries:
            all_configs = get_config_keys_from_master(
                entry.config_keys, master_config
            )
            resolved_entries.append((entry, all_configs))

        base_inputs = None
        if has_append_only:
            base_inputs = stack.enter_context(generation_inputs_at_ref(base_ref))
            base_master = load_config_files(base_inputs.config_files)
            selected_config_scenarios: dict[str, set[str]] = defaultdict(set)
            for entry, configs in resolved_entries:
                for config in configs:
                    selected_config_scenarios[config].update(
                        entry.scenario_type or SCENARIO_TYPES
                    )
            selected_configs = selected_config_scenarios.keys()
            missing_from_base = selected_configs - base_master.keys()
            if missing_from_base:
                raise ValueError(
                    "append-only requires every selected config to exist in the base "
                    f"revision; missing: {sorted(missing_from_base)}"
                )
            validate_append_only_scope(
                base_master,
                master_config,
                selected_config_scenarios,
            )

        # Process all-evals entries first so their broader eval matrix wins when
        # the same config appears in multiple changelog entries.
        resolved_entries.sort(key=lambda item: not item[0].all_evals)

        for entry, all_configs in resolved_entries:
            entry_scenarios = tuple(entry.scenario_type or SCENARIO_TYPES)
            expand_all_evals = all_evals or entry.all_evals
            suppress_throughput = (
                evals_only
                or entry.evals_only
                or entry.all_evals
            )

            if not suppress_throughput:
                benchmark_groups = group_unseen_scenarios(
                    all_configs, entry_scenarios, benchmark_scenarios_seen)
                for scenarios, benchmark_configs in benchmark_groups.items():
                    selection = scenarios if scenarios != SCENARIO_TYPES else None
                    head_results = generate_current(benchmark_configs, "none", selection)
                    if entry.append_only:
                        assert base_inputs is not None
                        base_results = generate_matrix(
                            benchmark_configs, _generation_flags("none", selection), base_inputs,
                        )
                        head_results = append_only_delta(base_results, head_results)
                    all_benchmark_results.extend(head_results)

            if entry.append_only or entry.no_evals:
                continue

            eval_groups = group_unseen_scenarios(
                all_configs, entry_scenarios, eval_scenarios_seen)
            for scenarios, eval_configs in eval_groups.items():
                entry_eval_results = generate_current(
                    eval_configs, "all" if expand_all_evals else "subset", scenarios,
                )
                entry_eval_results = filter_eval_rows_by_prefill_ep(
                    entry_eval_results, entry.eval_min_prefill_ep
                )
                all_eval_results.extend(entry_eval_results)

        if trim:
            all_benchmark_results = trim_conc(all_benchmark_results)

        for result in all_benchmark_results:
            result["recipe-fingerprint"] = recipe_fingerprint(result)
            node_type = "multi_node" if result.get("prefill") is not None else "single_node"
            scenario = (
                "agentic" if result.get("scenario-type") == "agentic-coding"
                else seq_len_to_str(result["isl"], result["osl"])
            )
            final_results[node_type][scenario].append(result)

        # Fixed-sequence and AgentX eval jobs have different workflow inputs.
        for result in all_eval_results:
            prefix = "multinode_" if result.get("prefill") is not None else ""
            suffix = "agentic_evals" if result.get("scenario-type") == "agentic-coding" else "evals"
            final_results[prefix + suffix].append(result)

        # Validate final results structure
        return ChangelogMatrixEntry.model_validate(final_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", type=str, required=True)
    parser.add_argument("--head-ref", type=str, required=True)
    parser.add_argument("--changelog-file", type=str, required=True)
    parser.add_argument("--trim-conc", action="store_true")
    parser.add_argument(
        "--all-evals",
        action="store_true",
        help="Expand every changelog entry's eval selection without changing throughput.",
    )
    parser.add_argument(
        "--evals-only",
        action="store_true",
        help="Suppress throughput for every changelog entry without expanding eval selection.",
    )
    args = parser.parse_args()

    added_yaml = get_added_lines(args.base_ref, args.head_ref, args.changelog_file)

    if not added_yaml.strip():
        raise ValueError("No additions found in the changelog file.")

    changelog_data = yaml.safe_load(added_yaml)

    try:
        result = build_plan(
            changelog_data, base_ref=args.base_ref, head_ref=args.head_ref,
            trim=args.trim_conc, all_evals=args.all_evals, evals_only=args.evals_only,
        )
    except MatrixGenerationError as error:
        # Preserve the legacy child diagnostics and failure status at the CLI.
        stderr = "".join(traceback.format_exception(error.cause))
        print(stderr)
        raise subprocess.CalledProcessError(
            1, _matrix_command(error.keys, error.flags, error.inputs), stderr=stderr,
        ) from None
    print(result.model_dump_json(by_alias=True, exclude_none=True))


def _generation_flags(mode: EvalMode, scenarios: tuple[str, ...] | None) -> list[str]:
    """Serialize only at the historical process / legacy diagnostic boundary."""
    flags = ["--no-evals"] if mode == "none" else ["--evals-only"]
    if mode == "all":
        flags.append("--all-evals")
    if scenarios is not None:
        flags.extend(["--scenario-type", *scenarios])
    return flags


def _matrix_command(config_keys: list[str], flags: list[str], inputs: GenerationInputs | None) -> list[str]:
    command = [
        "python3",
        inputs.generator_script if inputs else GENERATE_SWEEPS_PY_SCRIPT,
        "test-config",
        "--config-keys",
        *config_keys,
        "--config-files",
        *(inputs.config_files if inputs else MASTER_CONFIGS),
    ]
    if inputs is not None:
        command.extend(["--runner-config", inputs.runner_config])
    command.extend(flags)
    return command


if __name__ == "__main__":
    main()
