import unittest

from research.dsv41.window_diagnostics import progress_diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_uniform(self):
        r = [{"events": [[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]]}] * 2
        d = progress_diagnostics(r, 0, 4)
        self.assertEqual(d["whole"]["tokens_per_second"], 4)
        self.assertEqual(d["whole"]["coefficient_of_variation"], 0)
        self.assertEqual(
            [x["per_request_tokens"] for x in d["fixed_quarters"]], [[2, 2]] * 4
        )

    def test_stalled_subwindows(self):
        r = [
            {"events": [[0, 1], [1, 1], [2, 1], [3, 5], [4, 9]]},
            {"events": [[0, 1], [1, 5], [2, 9], [3, 9], [4, 9]]},
        ]
        d = progress_diagnostics(r, 0, 4)
        self.assertEqual(d["whole"]["coefficient_of_variation"], 0)
        self.assertEqual(
            [x["zero_progress_requests"] for x in d["fixed_quarters"]], [1] * 4
        )
        self.assertEqual(
            d["publication_status"], "held_pending_representativeness_review"
        )

    def test_nonmonotonic(self):
        with self.assertRaises(ValueError):
            progress_diagnostics([{"events": [[0, 2], [1, 1]]}], 0, 1)


if __name__ == "__main__":
    unittest.main()
