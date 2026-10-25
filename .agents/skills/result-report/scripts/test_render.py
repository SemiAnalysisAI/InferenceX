import copy
import unittest
from pathlib import Path

import extract
import fetch
import render

FIX = Path(__file__).resolve().parent / "fixtures"


def data_for(*names, baseline=None):
    cfgs = [c for n in names for c in fetch.local_configs(fetch.parse_source(str(FIX / n)))]
    return extract.build_data(cfgs, [], baseline=baseline, inputs=[str(FIX / n) for n in names], generated="2026-10-23")


def grid_point(arm, scenario, conc, out_tps, ttft_p99, state=None, measured=0.5):
    g = extract.parse_grid(scenario)
    p = {"rf": scenario, "source": "s", "arm": arm, "scenario": scenario, "grid": g, "conc": conc,
         "mode": g["kind"], "identity": {}, "notes": [], "cache": {"overall": measured, "source": "aggregate"},
         "cache_check": {"nominal": g["cache"] / 100, "measured": measured,
                         "mismatch": abs(measured - g["cache"] / 100) > 0.10},
         "metrics": extract._empty_metrics("aiperf"), "search": None}
    p["metrics"]["throughput"]["output_tps"] = out_tps
    p["metrics"]["ttft"]["p99"] = ttft_p99
    p["metrics"]["itl"]["p99"] = ttft_p99 / 10 if ttft_p99 is not None else None
    if state:
        p["search"] = {"state": state, "max_conc": conc, "sla": [], "sla_filters": [], "binding": None,
                       "breach_observed": None, "best_trials_conc": None, "probes": [], "infeasible_min": None}
    return p


def grid_data(points, arms):
    return {"title": "t", "generated": "d", "inputs": [], "baseline": arms[0], "notes": [], "bytes_fetched": 0,
            "findings": None, "arms": [{"label": a, "identity": {}, "sources": [], "config": {}} for a in arms],
            "points": points}


class FormatTest(unittest.TestCase):
    def test_num_and_pct(self):
        self.assertEqual((render.num(None), render.num(78369.2), render.num(49.16), render.num(0.357)),
                         ("—", "78,369", "49.2", "0.36"))
        self.assertEqual(render.pct(0.43174), "43.2%")

    def test_delta_tone_follows_direction(self):
        self.assertEqual(render.cell(110.0, 100.0, render.num, True), ("110 (+10.0%)", "good"))
        self.assertEqual(render.cell(110.0, 100.0, render.num, False), ("110 (+10.0%)", "bad"))
        self.assertEqual(render.cell(110.0, None, render.num, True), ("110", None))


class PointDeltaTest(unittest.TestCase):
    def test_cache_hit_delta_is_absolute_percentage_points(self):
        self.assertEqual(render.cell(0.20, 0.0005, render.pct, True, pp=(100, 1.0)), ("20.0% (+19.95 pp)", "good"))
        self.assertEqual(render.cell(0.0005, 0.20, render.pct, True, pp=(100, 1.0)), ("0.1% (-19.95 pp)", "bad"))

    def test_small_or_missing_pp_delta_is_hidden(self):
        self.assertEqual(render.cell(0.205, 0.20, render.pct, True, pp=(100, 1.0)), ("20.5%", None))
        self.assertEqual(render.cell(0.2, None, render.pct, True, pp=(100, 1.0)), ("20.0%", None))

    def test_error_rate_delta_in_pp_up_is_bad(self):
        self.assertEqual(render.cell(1.5, 0.2, render.err_pct, False, pp=(1, 0.1)), ("1.50% (+1.30 pp)", "bad"))
        self.assertEqual(render.cell(0.2, 1.5, render.err_pct, False, pp=(1, 0.1)), ("0.20% (-1.30 pp)", "good"))

    def test_report_cache_cells_use_pp(self):
        a = grid_point("a", "fixed-isl8k-osl1k-cache0", 8, 1.0, 1.0, measured=0.0005)
        b = grid_point("b", "fixed-isl8k-osl1k-cache0", 8, 1.0, 1.0, measured=0.20)
        md = render.to_markdown(render.build_blocks(grid_data([a, b], ["a", "b"])))
        self.assertIn("20.0% (+19.95 pp)", md)


class ReportTest(unittest.TestCase):
    def test_fixed_report_hides_empty_cache_columns(self):
        md = render.to_markdown(render.build_blocks(data_for("gh_fixed")))
        self.assertIn("## Key findings", md)
        self.assertIn(render.KEY_FINDINGS_PLACEHOLDER, md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertIn("Hit CPU (DRAM)", md)
        self.assertIn("43.2%", md)
        self.assertNotIn("Hit external (L3)", md)
        self.assertIn("### TTFT (ms)", md)
        self.assertIn("723,840", md)
        self.assertIn("validator dropped 397", md)

    def test_aggregate_only_shows_dash_p99_and_note(self):
        md = render.to_markdown(render.build_blocks(data_for("aggonly")))
        self.assertIn("p99 unavailable", md)
        ttft = md.split("### TTFT (ms)")[1].split("###")[0]
        self.assertIn("| — |", ttft)

    def test_raw_folder_has_no_server_metrics(self):
        md = render.to_markdown(render.build_blocks(data_for("raw_aiperf")))
        self.assertIn("No server metrics logged", md)

    def test_search_headline_and_probe_table(self):
        md = render.to_markdown(render.build_blocks(data_for("search_local")))
        self.assertIn("max CCU = **60**", md)
        self.assertIn("binding: ITL p90 < 50 ms (observed 50.85 at c61)", md)
        self.assertIn("highest objective at c48", md)
        self.assertIn("★ max CCU", md)
        self.assertIn("## Grid summary", md)
        self.assertIn("| ISL 8k · OSL 1k | 60 · 1,335 |", md)

    def test_two_arms_get_deltas_against_baseline(self):
        data = data_for("gh_fixed")
        twin = copy.deepcopy(data["points"][0])
        twin["arm"] = "B"
        twin["metrics"]["throughput"]["output_tps"] *= 1.1
        data["points"].append(twin)
        data["arms"].append({"label": "B", "identity": {}, "sources": [], "config": {}})
        md = render.to_markdown(render.build_blocks(data))
        self.assertIn("(+10.0%)", md)
        self.assertIn("(baseline)", md)

    def test_search_grid_matrix_marks(self):
        pts = [grid_point("A", "sweep-isl8k-osl1k-cache0", 64, 2000.0, None, state="ok"),
               grid_point("A", "sweep-isl8k-osl1k-cache50", 0, None, None, state="no_pass_in_range"),
               grid_point("A", "sweep-isl32k-osl1k-cache0", 16, 900.0, None, state="cliff")]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| ISL 8k · OSL 1k | 64 · 2,000 | ✗ |", md)
        self.assertIn("| ISL 32k · OSL 1k | 16* · 900 | — |", md)

    def test_fixed_grid_matrices_and_cache_check(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0, measured=0.02),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0, measured=0.03),
               grid_point("A", "fixed-isl8k-osl1k-cache50", 8, 600.0, 700.0, measured=0.30)]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| Scenario | Arm | c8 | c32 |", md)
        self.assertIn("| fixed-isl8k-osl1k-cache50 | A | 600 | — |", md)
        self.assertIn("⚠", md)
        self.assertIn("<details><summary>Scenario: fixed-isl8k-osl1k-cache0</summary>", md)


if __name__ == "__main__":
    unittest.main()


class HtmlTest(unittest.TestCase):
    def test_line_chart_has_series_and_threshold(self):
        svg = render.line_chart("ITL", {"A": [(8, 40.0), (32, 55.0)]}, "CCU", "ms", threshold=50.0)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("<polyline", svg)
        self.assertIn('stroke-dasharray="4 3"', svg)
        self.assertEqual(render.line_chart("x", {}, "a", "b"), "")

    def test_html_report_has_tables_tones_and_probe_chart(self):
        data = data_for("search_local")
        page = render.to_html(render.build_blocks(data, charts=True), data["title"])
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<table>", page)
        self.assertIn('class="good"', page)
        self.assertIn("<svg", page)
        self.assertIn("by probe CCU", page)

    def test_fixed_ladder_gets_three_charts(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0)]
        blocks = render.chart_blocks(grid_data(pts, ["A"]), pts)
        self.assertEqual(len(blocks), 3)


class SafetyTest(unittest.TestCase):
    def test_inline_escapes_raw_html_but_keeps_placeholder(self):
        out = render._inline("<!-- x --><script>alert(1)</script>")
        self.assertNotIn("<script>", out)
        self.assertEqual(render._inline(render.KEY_FINDINGS_PLACEHOLDER), render.KEY_FINDINGS_PLACEHOLDER)

    def test_inline_links_only_safe_urls(self):
        self.assertNotIn("<a", render._inline("[x](javascript:alert(1))"))
        self.assertNotIn("<a", render._inline("[x](data:text/html,hi)"))
        self.assertIn('<a href="https://a.b/c">x</a>', render._inline("[x](https://a.b/c)"))

    def test_legend_outside_plot_area(self):
        import re
        series = {f"arm-number-{i}": [(1, 1.0), (2, 2.0)] for i in range(8)}
        svg = render.line_chart("t", series, "x", "y")
        ys = [float(y) for y in re.findall(r'class="legend" x="[\d.]+" y="([\d.]+)"', svg)]
        self.assertEqual(len(ys), 8)
        plot_top = min(float(y) for y in re.findall(r'<line x1="60" x2="\d+" y1="([\d.]+)"', svg))
        self.assertTrue(all(y < plot_top for y in ys))


class FixRound1RenderTest(unittest.TestCase):
    def _two_configs(self, concs=(8, 8)):
        pts = []
        for rf, c in zip(("conc2_v4", "ccu8_x"), concs):
            p = grid_point("a", "fixed-isl8k-osl1k-cache0", c, 100.0, 5.0)
            p.update(rf=rf, scenario=None, grid=None, mode="fixed", cache_check=None)
            pts.append(p)
        return grid_data(pts, ["a"])

    def test_config_label_column_when_conc_is_shared(self):
        md = render.to_markdown(render.build_blocks(self._two_configs()))
        self.assertIn("| Arm | Config | CCU |", md)
        self.assertIn("conc2_v4", md)
        self.assertIn("| a | conc2_v4, ccu8_x |", md)  # Runs table lists the configs

    def test_no_config_column_when_conc_differs(self):
        md = render.to_markdown(render.build_blocks(self._two_configs((8, 16))))
        self.assertNotIn("Config |", md)

    def test_baseline_note_only_when_deltas_exist(self):
        d = grid_data([grid_point("a", "fixed-isl8k-osl1k-cache0", 8, 1.0, 1.0),
                       grid_point("b", "fixed-isl8k-osl1k-cache50", 8, 1.0, 1.0)], ["a", "b"])
        self.assertNotIn("Deltas are against", render.to_markdown(render.build_blocks(d)))
        d["points"][1]["scenario"] = "fixed-isl8k-osl1k-cache0"
        self.assertIn("Deltas are against", render.to_markdown(render.build_blocks(d)))

    def test_runs_table_truncates_long_config_lists(self):
        d = self._two_configs()
        d["points"] = [dict(d["points"][0], rf=f"cfg{i}") for i in range(21)]
        cell = render.runs_table(d).rows[0][1][0]
        self.assertIn("cfg0", cell)
        self.assertNotIn("cfg20", cell)
        self.assertIn("(+", cell)

    def test_search_matrix_dash_when_non_baseline_has_no_max_conc(self):
        a = grid_point("a", "sweep-isl8k-osl1k-cache0", 60, 1.0, 1.0, state="ok")
        b = grid_point("b", "sweep-isl8k-osl1k-cache0", 60, 1.0, 1.0, state="ok")
        b["search"]["max_conc"] = None
        md = render.to_markdown(render.build_blocks(grid_data([a, b], ["a", "b"])))
        self.assertIn("60 · 1.00 / —", md)

    def test_error_percent_header(self):
        md = render.to_markdown(render.build_blocks(self._two_configs()))
        self.assertIn("Error %", md)
        self.assertNotIn("| Errors", md)
