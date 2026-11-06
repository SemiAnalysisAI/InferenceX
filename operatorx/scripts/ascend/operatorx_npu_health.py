from __future__ import annotations

import math
import os
import re

PCI = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
VISIBILITY_VARIABLES = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
)
MAPPING_COLUMNS = re.compile(
    r"NPU ID|Slot ID|Chip Logic ID|Chip Phy-ID|Chip ID|Chip Name"
)


def mapping_rows(text):
    lines = text.splitlines()
    headers = [
        i for i, line in enumerate(lines) if "NPU ID" in line and "Chip Name" in line
    ]
    if len(headers) != 1:
        raise ValueError("Expected one recognized npu-smi mapping header")
    header = lines[headers[0]]
    columns = MAPPING_COLUMNS.findall(header)
    remainder = MAPPING_COLUMNS.sub("", header).replace("|", "").strip()
    if (
        remainder
        or len(columns) != len(set(columns))
        or not {"NPU ID", "Chip ID", "Chip Name"} <= set(columns)
    ):
        raise ValueError("Unrecognized npu-smi mapping columns")
    if columns[-1] != "Chip Name" or not (
        {"Chip Logic ID", "Chip Phy-ID"} & set(columns)
    ):
        raise ValueError("Mapping must provide a logical or physical chip ID")
    rows = []
    for line in lines[headers[0] + 1 :]:
        line = line.strip().strip("|").strip()
        if not line or set(line) <= set("+-=|"):
            continue
        fields = line.split(maxsplit=len(columns) - 1)
        if len(fields) != len(columns):
            raise ValueError("Malformed npu-smi mapping row")
        values = dict(zip(columns, fields))
        if values["Chip Name"].lower() in {"mcu"}:
            continue
        if not re.fullmatch(r"(?:Ascend\S*|910B[1-4])", values["Chip Name"], re.I):
            raise ValueError("Unknown device in npu-smi mapping")
        if any(
            not value.isdecimal() for key, value in values.items() if key != "Chip Name"
        ):
            raise ValueError("Mapping IDs must be nonnegative integers")
        rows.append(
            {
                "board": int(values["NPU ID"]),
                "chip": int(values["Chip ID"]),
                "logical_device_id": int(values["Chip Logic ID"])
                if "Chip Logic ID" in values
                else None,
                "physical_device_id": int(values["Chip Phy-ID"])
                if "Chip Phy-ID" in values
                else None,
                "slot_id": int(values["Slot ID"]) if "Slot ID" in values else None,
                "name": values["Chip Name"],
            }
        )
    if not rows or len({(r["board"], r["chip"]) for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate board/chip mapping")
    for key in ("logical_device_id", "physical_device_id"):
        ids = [r[key] for r in rows if r[key] is not None]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {key} mapping")
    return rows


def _cells(line):
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _processes(text, board):
    match = re.search(r"Process\s+id", text, re.I)
    if match is None:
        raise ValueError("Missing recognized process-table header")
    lines = text[match.start() :].splitlines()
    container_ids = "Process id in container" in lines[0]
    records, empty_boards = [], set()
    for line in lines[1:]:
        if "|" not in line:
            continue
        cells = _cells(line)
        if not any(cells):
            continue
        empty = re.fullmatch(
            r"No running processes found(?: in NPU\s+(\d+))?", cells[0], re.I
        )
        if len(cells) == 1 and empty:
            empty_boards.add(int(empty[1]) if empty[1] is not None else None)
            continue
        identity = re.fullmatch(r"(\d+)(?:\s+(\d+))?", cells[0])
        if not identity or len(cells) < 4 or not cells[1].isdecimal():
            raise ValueError("Unrecognized process-table row")
        record = {
            "board": int(identity[1]),
            "chip": int(identity[2]) if identity[2] else None,
            "pid": int(cells[1]),
        }
        if container_ids:
            if len(cells) < 5 or not cells[-1].isdecimal():
                raise ValueError(
                    "Container PID column is present but PID is unavailable"
                )
            record.update(
                host_pid=int(cells[1]), container_pid=int(cells[-1]), pid=int(cells[-1])
            )
        records.append(record)
    selected = [record for record in records if record["board"] == board]
    if selected and (None in empty_boards or board in empty_boards):
        raise ValueError("Contradictory empty and occupied process status")
    if not selected and None not in empty_boards and board not in empty_boards:
        raise ValueError("No explicit process status for selected board")
    return selected


def status_rows(text, board):
    if "Health" not in text or "Bus-Id" not in text:
        raise ValueError("Missing Health or Bus-Id status header")
    board_only = "NPU Util(%)" in text
    if board_only == ("AICore(%)" in text):
        raise ValueError("Unrecognized or ambiguous utilization header")
    id_header = re.search(r"\bChip\s+(Phy-ID|Device)\b", text)
    chip_only = bool(re.search(r"\|\s*Chip\s*\|\s*Bus-Id\s*\|", text))
    if not board_only and not id_header and not chip_only:
        raise ValueError("Unrecognized chip/status-ID header")
    current = None
    devices, health = [], []
    for line in text.splitlines():
        if re.search(r"Process\s+id", line, re.I):
            break
        if "|" not in line:
            continue
        cells = _cells(line)
        combined = re.fullmatch(r"(\d+)\s+(?:Ascend\S*|910B[1-4])", cells[0], re.I)
        separate = (
            len(cells) > 2
            and cells[0].isdecimal()
            and re.fullmatch(r"(?:Ascend\S*|910B[1-4])", cells[1], re.I)
        )
        if combined or separate:
            current = int(combined[1]) if combined else int(cells[0])
            state = cells[1] if combined else cells[2]
            if current == board:
                if state != "OK":
                    raise ValueError(f"Selected board Health is {state!r}, expected OK")
                health.append(state)
        buses = [i for i, cell in enumerate(cells) if PCI.search(cell)]
        if not buses:
            continue
        if current is None or len(buses) != 1:
            raise ValueError("Cannot associate PCI status with one board")
        index = buses[0]
        if PCI.fullmatch(cells[index]) is None or index + 1 >= len(cells):
            raise ValueError("Malformed PCI/utilization row")
        util = re.match(r"^(\d+(?:\.\d+)?)(?:\s|$)", cells[index + 1])
        if (
            util is None
            or not math.isfinite(float(util[1]))
            or not 0 <= float(util[1]) <= 100
        ):
            raise ValueError("Invalid utilization value")
        chip, physical, reported = None, None, None
        if not board_only:
            pair = re.fullmatch(r"(\d+)" if chip_only else r"(\d+)\s+(\d+)", cells[0])
            if pair is None:
                raise ValueError("Malformed chip/status-ID row")
            chip = int(pair[1])
            reported = None if chip_only else int(pair[2])
            physical = reported if id_header and id_header[1] == "Phy-ID" else None
        elif any(cells[i] for i in range(index)):
            raise ValueError("Unexpected identity columns in board-only PCI row")
        if current == board:
            devices.append(
                {
                    "board": board,
                    "chip": chip,
                    "physical_device_id": physical,
                    "reported_device_id": reported,
                    "reported_device_column": "none"
                    if board_only
                    else "Chip-only"
                    if chip_only
                    else id_header[1],
                    "pci": cells[index],
                    "aicore_percent": float(util[1]),
                    "utilization_metric": "NPU Util(%)" if board_only else "AICore(%)",
                    "health": "OK",
                }
            )
    if (
        not health
        or not devices
        or len({row["pci"].lower() for row in devices}) != len(devices)
    ):
        raise ValueError("Missing selected board health/status or duplicate PCI rows")
    if board_only and len(devices) != 1:
        raise ValueError("Board-only status must identify exactly one PCI device")
    if not board_only and len({row["chip"] for row in devices}) != len(devices):
        raise ValueError("Duplicate chip status")
    return {"devices": devices, "processes": _processes(text, board), "health": "OK"}


def _acl_identity(record, logical_id, physical_id):
    required = (
        "logical_device_id",
        "physical_device_id",
        "logical_to_physical_rc",
        "physical_to_logical_rc",
        "roundtrip_logical_device_id",
    )
    if not isinstance(record, dict) or any(
        type(record.get(key)) is not int for key in required
    ):
        raise ValueError(
            "ACL runtime logical/physical cross-check is required with integer IDs and return codes"
        )
    if record["logical_to_physical_rc"] != 0 or record["physical_to_logical_rc"] != 0:
        raise ValueError("ACL runtime logical/physical query failed")
    if (
        record["logical_device_id"] != logical_id
        or record["physical_device_id"] != physical_id
        or record["roundtrip_logical_device_id"] != logical_id
    ):
        raise ValueError(
            "ACL runtime logical/physical mapping disagrees with requested device or SMI physical ID"
        )
    user_keys = (
        "user_device_id",
        "user_to_logical_rc",
        "user_to_physical_rc",
        "physical_to_user_rc",
        "logical_to_user_rc",
        "user_logical_device_id",
        "user_physical_device_id",
        "roundtrip_user_device_id",
    )
    if any(key in record for key in user_keys):
        if any(type(record.get(key)) is not int for key in user_keys):
            raise ValueError("ACL runtime user-device mapping is incomplete")
        if any(record[key] != 0 for key in user_keys if key.endswith("_rc")):
            raise ValueError("ACL runtime user-device query failed")
        if (
            record["user_device_id"] != logical_id
            or record["user_logical_device_id"] != logical_id
            or record["user_physical_device_id"] != physical_id
            or record["roundtrip_user_device_id"] != record["user_device_id"]
        ):
            raise ValueError(
                "ACL runtime user-device mapping differs from the requested unremapped index"
            )
    return dict(record)


def validate_identity(
    parsed,
    mapped,
    board,
    chip,
    logical_id,
    expected_pci=None,
    *,
    acl_runtime_mapping=None,
):
    expected = [row for row in mapped if row["board"] == board]
    selected_mapping = [row for row in expected if row["chip"] == chip]
    if len(selected_mapping) != 1:
        raise ValueError("Requested board/chip mapping not present exactly once")
    mapping = selected_mapping[0]
    logical = mapping["logical_device_id"]
    if logical is not None and logical != logical_id:
        raise ValueError("Requested logical-device mapping differs from npu-smi")
    acl = None
    if logical is None:
        if mapping["physical_device_id"] is None:
            raise ValueError("Neither logical nor physical identity is available")
        acl = _acl_identity(
            acl_runtime_mapping, logical_id, mapping["physical_device_id"]
        )
    elif acl_runtime_mapping is not None:
        if mapping["physical_device_id"] is None:
            raise ValueError(
                "Cannot cross-check ACL mapping without an SMI physical ID"
            )
        acl = _acl_identity(
            acl_runtime_mapping, logical_id, mapping["physical_device_id"]
        )
    rows = parsed["devices"]
    if any(row["chip"] is None for row in rows):
        if len(expected) != 1 or len(rows) != 1 or rows[0]["board"] != board:
            raise ValueError("Board-only PCI status cannot identify a multi-chip board")
        selected = dict(rows[0], chip=chip)
    else:
        if {(row["board"], row["chip"]) for row in rows} != {
            (row["board"], row["chip"]) for row in expected
        }:
            raise ValueError("Status rows do not cover every mapped chip on the board")
        selected = dict(next(row for row in rows if row["chip"] == chip))
    if selected.get("health") != "OK" or parsed.get("health") != "OK":
        raise ValueError("Identity requires healthy status evidence")
    physical = selected.get("physical_device_id")
    if (
        physical is not None
        and mapping["physical_device_id"] is not None
        and physical != mapping["physical_device_id"]
    ):
        raise ValueError("Status physical ID differs from mapping")
    if expected_pci and selected["pci"].lower() != expected_pci.lower():
        raise ValueError("Selected board/chip PCI address changed")
    selected.update(
        logical_device_id=logical_id,
        physical_device_id=mapping["physical_device_id"]
        if mapping["physical_device_id"] is not None
        else physical,
        identity_basis="SMI plus bidirectional ACL mapping"
        if acl
        else "SMI logical mapping",
    )
    if acl is not None:
        selected["acl_runtime_mapping"] = acl
    return selected


def validate_health_query(capture, board):
    fields = _query_fields(capture)
    if (
        not {"NPU ID", "Health Status", "Error Code", "Error Information"}
        <= fields.keys()
    ):
        raise ValueError("Health query omitted required status/error fields")
    if fields["NPU ID"] != str(board) or fields["Health Status"] != "OK":
        raise ValueError("Health-query identity/status mismatch")
    if (
        fields["Error Code"] not in {"NA", "0", "0x0", "0x00000000"}
        or fields["Error Information"] != "NA"
    ):
        raise ValueError("Health query reports an error or unrecognized error state")
    return {
        "board": board,
        "health": "OK",
        "error_code": fields["Error Code"],
        "error_information": fields["Error Information"],
    }


def _query_fields(capture):
    if (
        not isinstance(capture, dict)
        or type(capture.get("returncode")) is not int
        or capture["returncode"] != 0
    ):
        raise ValueError("npu-smi query did not succeed")
    fields = {}
    for line in capture.get("stdout", "").splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or key.strip() in fields:
            raise ValueError("Malformed or duplicate diagnostic-query field")
        fields[key.strip()] = value.strip()
    return fields


def validate_diagnostic_queries(captures, board):
    required = ("health", "current-fault-event", "ecc", "ecc-enable", "pcie-err")
    selected = {}
    for capture in captures:
        argv = capture.get("argv", [])
        if "-t" not in argv or argv.index("-t") + 1 >= len(argv):
            continue
        kind = argv[argv.index("-t") + 1]
        if kind not in required:
            continue
        if (
            argv != ["npu-smi", "info", "-t", kind, "-i", str(board)]
            or kind in selected
        ):
            raise ValueError("Ambiguous diagnostic query or wrong board")
        selected[kind] = capture
    if set(selected) != set(required):
        raise ValueError("Missing required health/fault/ECC/PCIe queries")
    result = {"health": validate_health_query(selected["health"], board)}
    fault = _query_fields(selected["current-fault-event"])
    if fault != {"Fault Event Count": "0", "Fault Event Information": "NA"}:
        raise ValueError("Current fault-event state is not explicitly clean")
    result["current-fault-event"] = {"fault_event_count": 0}
    enabled = _query_fields(selected["ecc-enable"])
    if enabled != {"HBM ECC-Enable Status": "True"}:
        raise ValueError("HBM ECC is not explicitly enabled")
    result["ecc-enable"] = {"enabled": True}
    counters = {
        "ecc": (
            "HBM Single Bit Error Count",
            "HBM Double Bit Error Count",
            "HBM Single Bit Aggregate Total Err Cnt",
            "HBM Double Bit Aggregate Total Err Cnt",
            "HBM Single Bit Isolated Pages Count",
            "HBM Double Bit Isolated Pages Count",
            "HBM Single Bit Next-Isolated Pages Count",
            "HBM Double Bit Next-Isolated Pages Count",
        ),
        "pcie-err": (
            "TX Error Count",
            "RX Error Count",
            "Lcrc Error Count",
            "Ecrc Error Count",
            "Retry Count",
        ),
    }
    for kind, names in counters.items():
        fields = _query_fields(selected[kind])
        if set(fields) != {"NPU ID", *names} or fields["NPU ID"] != str(board):
            raise ValueError(f"{kind} query has missing/unknown fields or wrong board")
        if any(fields[name] != "0" for name in names):
            raise ValueError(f"{kind} counters are not all zero")
        result[kind] = {name: 0 for name in names}
    return result


def collect_acl_runtime_mapping(logical_id, *, library_path="libascendcl.so"):
    import ctypes

    if type(logical_id) is not int or logical_id < 0:
        raise ValueError("logical_id must be a nonnegative integer")
    if any(os.environ.get(key) for key in VISIBILITY_VARIABLES):
        raise ValueError(
            "Resolve device visibility/remapping before ACL identity collection"
        )
    lib = ctypes.CDLL(library_path)
    lib.aclInit.argtypes, lib.aclInit.restype = [ctypes.c_char_p], ctypes.c_int
    lib.aclFinalize.argtypes, lib.aclFinalize.restype = [], ctypes.c_int
    names = (
        "aclrtGetUserDevIdByLogicDevId",
        "aclrtGetLogicDevIdByUserDevId",
        "aclrtGetPhyDevIdByUserDevId",
        "aclrtGetPhyDevIdByLogicDevId",
        "aclrtGetLogicDevIdByPhyDevId",
        "aclrtGetUserDevIdByPhyDevId",
    )
    for name in names:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = (
            [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)],
            ctypes.c_int,
        )
    initialized = int(lib.aclInit(None))
    if initialized != 0:
        raise RuntimeError(
            f"aclInit returned {initialized}; preflight must own an uninitialized ACL runtime"
        )
    record = {
        "library": library_path,
        "acl_init_rc": initialized,
        "logical_device_id": logical_id,
        "queries": [],
    }
    failure = None

    def query(name, value):
        output = ctypes.c_int32(-1)
        rc = int(getattr(lib, name)(value, ctypes.byref(output)))
        record["queries"].append(
            {"api": name, "input": value, "returncode": rc, "output": output.value}
        )
        if rc != 0 or output.value < 0:
            raise RuntimeError(
                f"{name}({value}) returned rc={rc}, output={output.value}"
            )
        return output.value, rc

    try:
        record["user_device_id"], record["logical_to_user_rc"] = query(
            names[0], logical_id
        )
        record["user_logical_device_id"], record["user_to_logical_rc"] = query(
            names[1], record["user_device_id"]
        )
        record["user_physical_device_id"], record["user_to_physical_rc"] = query(
            names[2], record["user_device_id"]
        )
        record["physical_device_id"], record["logical_to_physical_rc"] = query(
            names[3], logical_id
        )
        record["roundtrip_logical_device_id"], record["physical_to_logical_rc"] = query(
            names[4], record["physical_device_id"]
        )
        record["roundtrip_user_device_id"], record["physical_to_user_rc"] = query(
            names[5], record["physical_device_id"]
        )
        _acl_identity(record, logical_id, record["physical_device_id"])
    except Exception as exc:
        failure = exc
    finally:
        record["acl_finalize_rc"] = int(lib.aclFinalize())
    if failure is not None:
        raise RuntimeError(
            f"ACL identity query failed: {failure}; aclFinalize={record['acl_finalize_rc']}"
        ) from failure
    if record["acl_finalize_rc"] != 0:
        raise RuntimeError(f"aclFinalize returned {record['acl_finalize_rc']}")
    return record
