import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from infx.results.power.single_node import run
from infx.results.power.native_multinode import run as run_native
from test_aggregate_power import _write_amd_csv, _write_nvidia_csv
from test_aggregate_power_multinode import build_package
from test_native_multinode_power import _package as native_package


@given(gpus=st.integers(1, 8), scale=st.integers(1, 4), ramp=st.booleans(), amd=st.booleans(),
       strict=st.booleans(), missing_gpu=st.booleans(),
       bad_count=st.sampled_from([None, 'completed', 'total_input_tokens', 'total_output_tokens']))
def test_power_integration_preserves_energy_units_and_failure_artifacts(gpus, scale, ramp, amd, strict, missing_gpu, bad_count):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        csv, bench, aggregate, audit = [root / name for name in ("power.csv", "bench.json", "agg.json", "audit.json")]
        samples = [(1_700_000_000 + t, gpu, (100 + 20 * t if ramp else 400) * scale)
                   for t in range(0, 11, 2) for gpu in range(gpus - int(missing_gpu))]
        (_write_amd_csv if amd else _write_nvidia_csv)(csv, samples)
        benchmark = {"benchmark_start_time_unix": 1_700_000_000, "benchmark_end_time_unix": 1_700_000_010,
                     "duration": 10, "completed": 10, "total_input_tokens": 100, "total_output_tokens": 100}
        if bad_count:
            benchmark[bad_count] = 10**310
        bench.write_text(json.dumps(benchmark))
        aggregate.write_text(json.dumps({"model": "preserved", "total_gpu_energy_j": -999}))
        code = run(csv, bench, aggregate, expected_num_gpus=gpus, validation_result=audit, require_power=strict)
        result, validation = json.loads(aggregate.read_text()), json.loads(audit.read_text())
    assert result["model"] == "preserved"
    invalid = missing_gpu or bad_count is not None
    assert code == int(strict and invalid)
    assert validation["power_valid"] is not invalid
    assert result["power_valid"] == int(not invalid)
    if invalid:
        assert "total_gpu_energy_j" not in result
        assert validation["reasons"]
    else:
        assert result["total_gpu_energy_j"] == pytest.approx((2000 if ramp else 4000) * gpus * scale)
        assert result["avg_power_w"] == pytest.approx((200 if ramp else 400) * scale)
        assert result["joules_per_successful_query"] == pytest.approx((200 if ramp else 400) * gpus * scale)


@given(value=st.one_of(st.none(), st.booleans(), st.integers(), st.text(), st.lists(st.integers()),
                      st.binary(max_size=128).map(lambda value: b'\xff' + value),
                      st.tuples(st.sampled_from(['benchmark_start_time_unix', 'benchmark_end_time_unix', 'duration']),
                                st.integers(10**309, 10**400)).map(lambda item: {
                                    'benchmark_start_time_unix': 1, 'benchmark_end_time_unix': 2,
                                    'duration': 1, item[0]: item[1]})), strict=st.booleans())
def test_malformed_benchmark_still_emits_a_power_audit(value, strict):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bench, aggregate, audit = [root / name for name in ("bench.json", "agg.json", "audit.json")]
        bench.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        aggregate.write_text('{"model":"preserved","total_gpu_energy_j":999}')
        code = run(root / "missing.csv", bench, aggregate, validation_result=audit, require_power=strict)
        result, validation = json.loads(aggregate.read_text()), json.loads(audit.read_text())
    assert code == int(strict)
    assert validation["power_valid"] is False
    assert ("invalid_benchmark_window" if isinstance(value, dict) else "invalid_benchmark_result") in validation["reasons"]
    assert result["model"] == "preserved"
    assert result["power_valid"] == 0
    assert "total_gpu_energy_j" not in result


@given(scale=st.integers(1, 4), publication_valid=st.booleans())
def test_multinode_power_roles_keep_whole_deployment_totals(scale, publication_valid):
    with tempfile.TemporaryDirectory() as directory:
        package = build_package(Path(directory), power_fn=lambda host, index, timestamp: (400 if host == "node-p" else 300) * scale,
                                publication_valid=publication_valid)
        package.run()
        result = package.agg()
        audit = package.sidecar()
    assert result["power_valid"] == int(publication_valid)
    assert audit["power_valid"] is publication_valid
    if publication_valid:
        assert result["total_gpu_energy_j"] == 84000 * scale
        assert result["prefill_gpu_energy_j"] == 48000 * scale
        assert result["decode_gpu_energy_j"] == 36000 * scale
    else:
        assert "total_gpu_energy_j" not in result
        assert "producer_verdict_mismatch" in audit["reasons"]


@given(vendor=st.sampled_from(['amd', 'nvidia']), scale=st.integers(1, 20), strict=st.booleans(),
       topology=st.sampled_from(['disaggregate', 'aggregate', 'prefill', 'decode']),
       problem=st.sampled_from([None, 'clock', 'start', 'end', 'node-count']))
def test_native_power_binds_serving_devices_to_roles_and_rejects_invalid_manifests(vendor, scale, strict, topology, problem):
    with tempfile.TemporaryDirectory() as directory:
        root, bench, aggregate = native_package(Path(directory), vendor)
        for rank in range(2):
            node = root / f'node-{rank}'
            manifest = json.loads((node / 'manifest.json').read_text())
            if topology != 'disaggregate':
                manifest['role'] = topology
            if rank == 0 and problem:
                field, value = {'clock': ('clock_synchronized', False),
                                'start': ('collection_start_unix', 10**310),
                                'end': ('collection_end_unix', 10**310),
                                'node-count': ('expected_num_nodes', 10**100)}[problem]
                manifest[field] = value
            (node / 'manifest.json').write_text(json.dumps(manifest))
            (node / 'gpu_metrics.csv').write_text('timestamp,gpu,power\n' + ''.join(
                f'{tick},0,{(100 if rank == 0 else 300) * scale}\n{tick},1,900\n' for tick in range(5)))
        counts = {'disaggregate': (1, 1, 0), 'aggregate': (0, 0, 2), 'prefill': (2, 0, 0), 'decode': (0, 2, 0)}[topology]
        code = run_native(root, bench, aggregate, expected_prefill_gpus=counts[0], expected_decode_gpus=counts[1],
                          expected_aggregate_gpus=counts[2], require_power=strict)
        result = json.loads(aggregate.read_text())
        audit = json.loads((Path(directory) / 'power_validation_result.json').read_text())
    assert code == int(strict and problem is not None)
    assert audit['power_valid'] is (problem is None)
    assert result['power_valid'] == int(problem is None)
    if problem:
        assert 'avg_power_w' not in result
        assert 'prefill_gpu_energy_j' not in result
        assert audit['reasons']
    else:
        assert result['total_gpu_energy_j'] == 800 * scale
        assert result['avg_power_w'] == 200 * scale
        for role, energy in [('prefill', 200), ('decode', 600)]:
            if topology in ['disaggregate', role]:
                assert result[f'{role}_gpu_energy_j'] == (energy if topology == 'disaggregate' else 800) * scale
            else:
                assert f'{role}_gpu_energy_j' not in result
