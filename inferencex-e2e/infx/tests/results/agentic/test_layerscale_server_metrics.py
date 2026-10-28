"""LayerScale's server metrics normalize to the shared agentic cache and KV fields."""

from __future__ import annotations

from infx.results.agentic.server_metrics import compute_server_metrics


def _counter(total: float) -> dict:
    return {"type": "counter", "series": [{"labels": {}, "stats": {"total": total}}]}


def _gauge(low: float, high: float, labels: dict | None = None) -> dict:
    return {"type": "gauge", "series": [{"labels": labels or {}, "stats": {"min": low, "max": high}}]}


def test_layerscale_metrics_give_gpu_hit_rate_peak_kv_usage_and_capacity() -> None:
    metrics = {
        "layerscale_prompt_tokens": _counter(1000),
        "layerscale_generated_tokens": _counter(200),
        "layerscale_prefix_cached_tokens": _counter(750),
        "layerscale_session_cached_tokens": _counter(50),
        "layerscale_kv_pages_total": _gauge(1000, 1000),
        "layerscale_kv_pages_free": _gauge(250, 900),
        "layerscale_build_info": _gauge(1, 1, {"page_size": "16"}),
    }

    flat, nested, warnings = compute_server_metrics(
        {"metrics": metrics}, framework="layerscale", records=[]
    )

    assert nested["adapter"] == "layerscale"
    assert flat["total_prompt_tokens"] == 1000
    assert flat["total_generation_tokens"] == 200
    assert flat["server_gpu_cache_hit_rate"] == 0.8
    assert flat["server_overall_cache_hit_rate"] == 0.8
    assert flat["gpu_kv_cache_usage_pct"] == 0.75
    assert nested["cache"]["cached_tokens_by_source"] == {"prefix": 750, "session": 50}
    assert nested["kv_cache"]["gpu_total_tokens"] == 16000
    assert warnings == []
