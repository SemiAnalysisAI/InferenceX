import calc_success_rate as success_rate
import pytest
import yaml


@pytest.mark.parametrize(
    "runners,expected",
    [
        (
            {
                "labels": {
                    "cluster:zeta": ["self-hosted"],
                    "gpu": ["self-hosted"],
                    "cluster:alpha": ["self-hosted"],
                },
                "hardware": {"legacy": {}},
            },
            ["alpha", "zeta"],
        ),
        ({"cluster:zeta": [], "unrelated": [], "cluster:alpha": []}, ["alpha", "zeta"]),
        ({"labels": {"gpu": []}, "hardware": {"legacy": {}}}, ["legacy"]),
        ({}, []),
    ],
    ids=["cluster-labels-take-precedence", "flat-labels", "legacy-fallback", "empty"],
)
def test_load_hardware_labels_normalizes_supported_layouts(
    tmp_path, monkeypatch, runners, expected
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "runners.yaml").write_text(yaml.safe_dump(runners, sort_keys=False))
    monkeypatch.setattr(success_rate, "__file__", str(tmp_path / "utils" / "calc_success_rate.py"))

    assert success_rate.load_hardware_labels() == expected


def test_extract_hardware_from_name_matches_cluster_label():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "gb200-nv"])

    assert (
        success_rate.extract_hardware_from_name(
            "dsv4 fp4 cluster:b300-nv vllm | tp=8", patterns
        )
        == "b300-nv"
    )
    assert (
        success_rate.extract_hardware_from_name(
            "glm5 fp4 gb200-nv dynamo-sglang", patterns
        )
        == "gb200-nv"
    )


def test_extract_hardware_from_name_does_not_infer_broad_sku():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "h200-dgxc"])

    assert success_rate.extract_hardware_from_name("dsv4 fp4 b300 vllm", patterns) is None
    assert success_rate.extract_hardware_from_name("dsv4 fp8 h200 sglang", patterns) is None


@pytest.mark.parametrize(
    "job_name,expected",
    [
        ("model CLUSTER:GPU.A | tp=8", "gpu.a"),
        ("model gpuXa | tp=8", None),
        ("model othergpu.a | tp=8", None),
        ("model gpu.a2 | tp=8", None),
    ],
)
def test_hardware_matching_respects_case_boundaries_and_literal_punctuation(job_name, expected):
    patterns = success_rate.build_hardware_match_patterns(["gpu.a"])

    assert success_rate.extract_hardware_from_name(job_name, patterns) == expected
