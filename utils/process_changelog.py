import argparse
import copy
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml
from constants import GENERATE_SWEEPS_PY_SCRIPT, MASTER_CONFIGS
from matrix_logic.generate_sweep_configs import seq_len_to_str
from matrix_logic.validation import (
    ChangelogEntry,
    ChangelogMatrixEntry,
    load_config_files,
)

SCENARIO_TYPES = ("fixed-seq-len", "agentic-coding")
APPEND_ONLY_ALLOWED_FILES = {"perf-changelog.yaml", *MASTER_CONFIGS}
CONCURRENCY_CONFIG_FIELDS = {"conc-list", "conc-start", "conc-end"}


def _freeze_config_value(value):
    """Convert JSON-shaped config values into deterministic hashable values."""
    if isinstance(value, dict):
        return tuple(
            sorted((key, _freeze_config_value(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze_config_value(item) for item in value)
    return value


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


def trim_conc(entries: list[dict]) -> list[dict]:
    """Trim each parallelism config's concurrency sweep to its lowest point.

    Non-full-sweep PRs only need a single concurrency point per parallelism
    config to validate a change runs end-to-end, so the shared cluster stays
    clear. Push-to-main and ``full-sweep-enabled`` PRs skip this reduction.

    The retained value is the minimum configured concurrency — independent of
    the source ordering of ``conc-list`` / ``conc-start``.

    Input comes from ``json.loads(subprocess.stdout)`` so ``conc`` is always
    ``int`` (single-node) or ``list`` (multi-node). Other fields may contain
    nested dictionaries or lists, such as KV-offload backend metadata.

    - Single-node entries: group by every configuration field other than
      ``conc`` and the generated ``exp-name``, then keep only the entry with
      the lowest ``conc`` per group.
    - Multi-node entries: trim the ``conc`` list in place to ``[min(conc)]``.
    """
    groups: dict[tuple, list[int]] = {}
    out: list[dict] = []

    for entry in entries:
        if entry.get("prefill") is not None:
            conc = entry.get("conc")
            if isinstance(conc, list) and len(conc) > 1:
                entry = {**entry, "conc": [min(conc)]}
            out.append(entry)
            continue

        key = tuple(
            sorted(
                (k, _freeze_config_value(v))
                for k, v in entry.items()
                if k not in {"conc", "exp-name"}
            )
        )
        groups.setdefault(key, []).append(len(out))
        out.append(entry)

    drop: set[int] = set()
    for idxs in groups.values():
        if len(idxs) > 1:
            keep = min(idxs, key=lambda i: out[i]["conc"])
            drop.update(i for i in idxs if i != keep)
    return [e for i, e in enumerate(out) if i not in drop]


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


def get_changed_files(base_ref: str, head_ref: str) -> set[str]:
    """Return repository-relative paths changed by the requested sweep diff."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, head_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in result.stdout.splitlines() if line}


@contextmanager
def config_files_at_ref(ref: str):
    """Materialize the master configs from ``ref`` for matrix generation."""
    with tempfile.TemporaryDirectory(prefix="inferencex-append-only-") as temp_dir:
        paths = []
        for config_file in MASTER_CONFIGS:
            result = subprocess.run(
                ["git", "show", f"{ref}:{config_file}"],
                capture_output=True,
                check=True,
            )
            destination = Path(temp_dir) / config_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.stdout)
            paths.append(str(destination))
        yield paths


def _matrix_curve_key(entry: dict) -> tuple:
    """Identify one curve while deliberately excluding point-level fields."""
    return tuple(
        sorted(
            (key, _freeze_config_value(value))
            for key, value in entry.items()
            if key not in {"conc", "exp-name"}
        )
    )


def _matrix_concurrencies(entry: dict) -> tuple[int, ...]:
    conc = entry.get("conc")
    if isinstance(conc, int):
        return (conc,)
    if isinstance(conc, list) and conc and all(isinstance(value, int) for value in conc):
        return tuple(conc)
    raise ValueError(f"append-only matrix entry has invalid concurrency value: {conc!r}")


def append_only_delta(base_entries: list[dict], head_entries: list[dict]) -> list[dict]:
    """Return only newly added points, rejecting any existing-curve mutation.

    Generated matrix rows are the runtime contract. Grouping them without ``conc``
    and ``exp-name`` catches image, launcher, topology, arguments, duration, and
    scenario changes while allowing only concurrency-set expansion.
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
    new_curves = head_groups.keys() - base_groups.keys()
    if removed_curves or new_curves:
        raise ValueError(
            "append-only may not add, remove, or modify curve logic; only concurrency "
            "points may be added"
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
        added = head_groups[key] - base_groups[key]
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
        raise ValueError("append-only did not add any concurrency points")
    return delta


def validate_append_only_scope(
    base_ref: str,
    head_ref: str,
    base_master: dict,
    head_master: dict,
    selected_config_scenarios: dict[str, set[str]],
) -> None:
    """Reject code changes and unrelated config edits in an append-only PR."""
    unexpected_files = get_changed_files(base_ref, head_ref) - APPEND_ONLY_ALLOWED_FILES
    if unexpected_files:
        raise ValueError(
            "append-only PRs may change only perf-changelog.yaml and master config "
            f"files; unexpected changes: {sorted(unexpected_files)}"
        )

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
        if base_config.keys() != head_config.keys():
            raise ValueError(
                f"append-only changed top-level fields for config {config!r}"
            )
        for field in base_config.keys() - {"scenarios"}:
            if base_config[field] != head_config[field]:
                raise ValueError(
                    f"append-only changed non-scenario field {field!r} in {config!r}"
                )

        base_scenarios = base_config.get("scenarios", {})
        head_scenarios = head_config.get("scenarios", {})
        if base_scenarios.keys() != head_scenarios.keys():
            raise ValueError(
                f"append-only added or removed a scenario in config {config!r}"
            )
        for scenario in base_scenarios:
            if scenario not in allowed_scenarios:
                if base_scenarios[scenario] != head_scenarios[scenario]:
                    raise ValueError(
                        "append-only changed a scenario outside its changelog scope: "
                        f"{config!r} / {scenario!r}"
                    )
                continue
            _validate_concurrency_only_structure(
                base_scenarios[scenario],
                head_scenarios[scenario],
                f"{config}.{scenario}",
            )


def _validate_concurrency_only_structure(base, head, path: str) -> None:
    """Require identical config structure except at explicit concurrency fields."""
    if isinstance(base, dict) and isinstance(head, dict):
        base_keys = base.keys() - CONCURRENCY_CONFIG_FIELDS
        head_keys = head.keys() - CONCURRENCY_CONFIG_FIELDS
        if base_keys != head_keys:
            raise ValueError(f"append-only changed config structure at {path}")
        for key in base_keys:
            _validate_concurrency_only_structure(base[key], head[key], f"{path}.{key}")
        return
    if isinstance(base, list) and isinstance(head, list):
        if len(base) != len(head):
            raise ValueError(f"append-only changed config structure at {path}")
        for index, (base_item, head_item) in enumerate(zip(base, head)):
            _validate_concurrency_only_structure(
                base_item, head_item, f"{path}[{index}]"
            )
        return
    if base != head:
        raise ValueError(f"append-only changed non-concurrency value at {path}")


def main():
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

    if not changelog_data:
        raise ValueError("No valid YAML entries found in the changelog additions.")

    parsed_entries = [ChangelogEntry.model_validate(entry) for entry in changelog_data]
    has_append_only = any(entry.append_only for entry in parsed_entries)
    if has_append_only and not all(entry.append_only for entry in parsed_entries):
        raise ValueError(
            "append-only entries cannot share a sweep with regular changelog entries"
        )
    if has_append_only and (args.all_evals or args.evals_only):
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
            "base_ref": args.base_ref,
            "head_ref": args.head_ref,
            "entries": changelog_data,
        },
    }

    all_benchmark_results = []
    all_eval_results = []
    # Track benchmark coverage per scenario so overlapping changelog entries
    # with disjoint scenario filters do not suppress each other.
    benchmark_scenarios_seen = defaultdict(set)
    eval_scenarios_seen = defaultdict(set)

    master_config = load_config_files(MASTER_CONFIGS)
    resolved_entries = []
    for entry in parsed_entries:
        all_configs = get_config_keys_from_master(
            entry.config_keys, master_config
        )
        resolved_entries.append((entry, all_configs))

    base_config_context = None
    base_config_files = None
    if has_append_only:
        base_config_context = config_files_at_ref(args.base_ref)
        base_config_files = base_config_context.__enter__()
        base_master = load_config_files(base_config_files)
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
            args.base_ref,
            args.head_ref,
            base_master,
            master_config,
            selected_config_scenarios,
        )

    # Process all-evals entries first so their broader eval matrix wins when
    # the same config appears in multiple changelog entries.
    resolved_entries.sort(key=lambda item: not item[0].all_evals)

    for entry, all_configs in resolved_entries:
        entry_scenarios = tuple(entry.scenario_type or SCENARIO_TYPES)
        expand_all_evals = args.all_evals or entry.all_evals
        suppress_throughput = (
            args.evals_only
            or entry.evals_only
            or entry.all_evals
        )

        if not suppress_throughput:
            # Generate benchmark entries (no evals)
            benchmark_groups = defaultdict(list)
            for config in all_configs:
                unseen_scenarios = tuple(
                    scenario for scenario in SCENARIO_TYPES
                    if (
                        scenario in entry_scenarios
                        and scenario not in benchmark_scenarios_seen[config]
                    )
                )
                if unseen_scenarios:
                    benchmark_scenarios_seen[config].update(unseen_scenarios)
                    benchmark_groups[unseen_scenarios].append(config)

            for scenarios, benchmark_configs in benchmark_groups.items():
                head_cmd = [
                    "python3",
                    GENERATE_SWEEPS_PY_SCRIPT,
                    "test-config",
                    "--config-keys",
                    *benchmark_configs,
                    "--config-files",
                    *MASTER_CONFIGS,
                    "--no-evals",
                ]
                if scenarios != SCENARIO_TYPES:
                    head_cmd.extend(["--scenario-type", *scenarios])
                try:
                    result = subprocess.run(
                        head_cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    head_results = json.loads(result.stdout)
                    if entry.append_only:
                        base_cmd = head_cmd.copy()
                        config_files_index = base_cmd.index("--config-files") + 1
                        base_cmd[
                            config_files_index:config_files_index + len(MASTER_CONFIGS)
                        ] = base_config_files
                        base_result = subprocess.run(
                            base_cmd,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        head_results = append_only_delta(
                            json.loads(base_result.stdout), head_results
                        )
                except subprocess.CalledProcessError as e:
                    print(e.stderr)
                    raise
                all_benchmark_results.extend(head_results)

        if entry.append_only:
            continue

        eval_groups = defaultdict(list)
        for config in all_configs:
            unseen_scenarios = tuple(
                scenario for scenario in SCENARIO_TYPES
                if (
                    scenario in entry_scenarios
                    and scenario not in eval_scenarios_seen[config]
                )
            )
            if unseen_scenarios:
                eval_scenarios_seen[config].update(unseen_scenarios)
                eval_groups[unseen_scenarios].append(config)

        for scenarios, eval_configs in eval_groups.items():
            eval_flags = ["--evals-only"]
            if expand_all_evals:
                eval_flags.append("--all-evals")
            base_cmd = [
                "python3",
                GENERATE_SWEEPS_PY_SCRIPT,
                "test-config",
                "--config-keys",
                *eval_configs,
                "--config-files",
                *MASTER_CONFIGS,
                *eval_flags,
                "--scenario-type",
                *scenarios,
            ]
            try:
                eval_result = subprocess.run(
                    base_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                raise
            entry_eval_results = json.loads(eval_result.stdout)
            entry_eval_results = filter_eval_rows_by_prefill_ep(
                entry_eval_results, entry.eval_min_prefill_ep
            )
            all_eval_results.extend(entry_eval_results)

    if base_config_context is not None:
        base_config_context.__exit__(None, None, None)

    if args.trim_conc:
        all_benchmark_results = trim_conc(all_benchmark_results)

    for result in all_benchmark_results:
        if result.get("scenario-type") == "agentic-coding":
            if result.get("prefill") is not None:
                final_results["multi_node"]["agentic"].append(result)
            else:
                final_results["single_node"]["agentic"].append(result)
        elif "prefill" in result and result["prefill"] is not None:
            seq_len_str = seq_len_to_str(result["isl"], result["osl"])
            final_results["multi_node"][seq_len_str].append(result)
        else:
            seq_len_str = seq_len_to_str(result["isl"], result["osl"])
            final_results["single_node"][seq_len_str].append(result)

    # Agentic GSM8K eval rows go to their own bucket so run-sweep.yml can dispatch
    # them with agentic inputs (scenario-type, kv-offloading, ...) instead of
    # the fixed-seq-len inputs (isl/osl/max-model-len) they don't have. Same
    # split applies on the multi-node side (multinode_evals vs
    # multinode_agentic_evals).
    single_node_evals = [e for e in all_eval_results if e.get("prefill") is None]
    multi_node_evals = [e for e in all_eval_results if e.get("prefill") is not None]
    final_results["evals"] = [
        e for e in single_node_evals
        if e.get("scenario-type") != "agentic-coding"
    ]
    final_results["agentic_evals"] = [
        e for e in single_node_evals
        if e.get("scenario-type") == "agentic-coding"
    ]
    final_results["multinode_evals"] = [
        e for e in multi_node_evals
        if e.get("scenario-type") != "agentic-coding"
    ]
    final_results["multinode_agentic_evals"] = [
        e for e in multi_node_evals
        if e.get("scenario-type") == "agentic-coding"
    ]

    # Validate final results structure
    validated = ChangelogMatrixEntry.model_validate(final_results)
    print(validated.model_dump_json(by_alias=True, exclude_none=True))


if __name__ == "__main__":
    main()
