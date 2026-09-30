"""Count-based closure budgets; no machine-dependent wall-clock assertions."""
import itertools
import random
import unittest
from unittest.mock import patch

from arena import risk


class LargeFrontierBudgetTests(unittest.TestCase):
    def tearDown(self):
        risk._infer.cache_clear()

    def test_awkward_frontier_stops_at_deterministic_comparison_budget(self):
        rng = random.Random(8)
        truth = set(rng.sample(range(256), 60))
        safe = rng.sample(sorted(set(range(256)) - truth), 80)
        obs = {'size': 16, 'width': 16, 'height': 16, 'mine_count': 60,
               'known_safe': [[i // 16, i % 16] for i in safe],
               'revealed': [[i // 16, i % 16, len(set(risk.neighbors(i, 16)) & truth)] for i in safe]}
        first = risk.infer(obs)
        risk._infer.cache_clear()
        second = risk.infer(obs)
        self.assertEqual(first, second)
        p, status, detail = first
        self.assertEqual(detail['closure_comparisons'], risk.LARGE_BOARD_SUBSET_COMPARISONS)
        self.assertTrue(detail['closure_exhausted'])
        self.assertEqual(status, 'approximate_cutoff')
        self.assertTrue(all(0 <= value <= 1 for value in p))
        self.assertTrue(all(p[i] == 0 for i in safe))

    def test_exhausted_optional_closure_can_still_have_exact_posterior(self):
        width, height = 10, 9
        unknown = [r * width + c for r in range(3, 6) for c in range(3, 7)]
        truth = {unknown[i] for i in (0, 2, 7, 10)}
        safe = sorted(set(range(width * height)) - set(unknown))
        clues = [(2, 3), (2, 5)]
        def near(i, j):
            return i != j and abs(i // width - j // width) <= 1 and abs(i % width - j % width) <= 1
        obs = {'size': width, 'width': width, 'height': height, 'mine_count': 4,
               'known_safe': [[i // width, i % width] for i in safe],
               'revealed': [[r, c, sum(near(r * width + c, m) for m in truth)] for r, c in clues]}
        layouts = [set(values) for values in itertools.combinations(unknown, 4)
                   if all(sum(near(r * width + c, m) for m in values) == number
                          for r, c, number in obs['revealed'])]
        self.assertGreater(len(layouts), 1)
        risk._infer.cache_clear()
        with patch.object(risk, 'LARGE_BOARD_SUBSET_COMPARISONS', 1):
            p, status, detail = risk.infer(obs)
        self.assertTrue(detail['closure_exhausted'])
        self.assertEqual(status, 'exact_global_model_count')
        self.assertEqual(detail['models'], len(layouts))
        for i in range(width * height):
            self.assertAlmostEqual(p[i], sum(i in layout for layout in layouts) / len(layouts), places=12)

    def test_legacy_diagnostics_do_not_gain_new_fields(self):
        p, status, detail = risk.infer({'size': 9, 'mine_count': 10,
                                       'known_safe': [[0, 0], [8, 8]], 'revealed': []})
        self.assertEqual(status, 'exact_global_model_count')
        self.assertNotIn('closure_budget', detail)
        self.assertAlmostEqual(sum(p), 10.)


if __name__ == '__main__':
    unittest.main()
