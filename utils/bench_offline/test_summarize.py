from bench_offline.summarize import markdown


def test_summary_explains_offline_units():
    rendered = markdown(
        [
            {
                "concurrency": 8,
                "status": "success",
                "candidate": "wait30",
                "mean_token_tpot_ms": 10.0,
                "derived_output_tput_per_gpu": 100.0,
                "wall_output_tput_per_gpu": 90.0,
                "observed_tokens_per_step": 2.4,
                "effective_acceptance_rate": 0.4667,
                "raw_speculative_metrics_available": False,
                "acceptance_rate": None,
                "mean_ttft_ms": 1000.0,
                "huawei_published_dataset_token_tput_per_chip": 110.0,
                "huawei_estimated_token_tput_per_chip": 120.0,
                "b300_to_huawei_ratio": 0.833,
            }
        ]
    )
    assert "Token TPOT ms" in rendered
    assert "last token time - first token time" in rendered
    assert "raw accepted/proposed counters" in rendered.lower()
    assert "Huawei pub tok/s/chip" in rendered
